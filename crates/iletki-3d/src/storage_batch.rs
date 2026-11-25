//! Persistent storage-buffer chunk transforms and retained per-viewport draw lists.
//!
//! Capability-gated to backends that support read-only vertex-stage storage buffers (Vulkan, DX12,
//! Metal and WebGPU core). OpenGL / WebGL2 keep the 256-entry uniform fallback path in
//! [`crate::face_batch`].

use crate::face_batch::{GeometryPage, PageKind, PaintPipelines, VertexRange};
use bytemuck::{Pod, Zeroable};
use iletki_core::Vec3;
use iletki_wgpu::wgpu;
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::ops::Range;
use std::sync::Arc;

const MIN_TRANSFORM_CAPACITY: usize = 256;
// Bound the extra vertex work bridged across clipped gaps in raster section mode.
const MAX_SECTION_GAP_ELEMENTS: u32 = 256;
// Safe range for Q32.32 signed differences without i32 overflow (+/- 536,870 km from anchor).
const MAX_Q32_COORD: f64 = 536_870_912.0;
// Merge nearby dirty transform writes (up to 16 entries = 512 B) to avoid many tiny queue writes
// without turning sparse strided edits into multi-megabyte uploads.
const DIRTY_SPAN_GAP: usize = 16;
// Threshold at which unordered masks pack into per-page multi-draw indirect buffers.
const MIN_INDIRECT_DRAWS: usize = 32;
const DRAW_INDEXED_STRIDE: u64 = 20;
const DRAW_NON_INDEXED_STRIDE: u64 = 16;

/// Per-viewport Q32.32 fixed-point camera origin relative to the scene anchor (`@group(1) @binding(1)`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub(crate) struct ViewCameraOrigin {
    pub(crate) cam_whole: [i32; 3],
    pub(crate) _pad0: u32,
    pub(crate) cam_frac: [u32; 3],
    pub(crate) _pad1: u32,
}

impl ViewCameraOrigin {
    pub(crate) fn from_vec3(v: Vec3) -> Self {
        let (wx, fx) = to_q32_32(v.x);
        let (wy, fy) = to_q32_32(v.y);
        let (wz, fz) = to_q32_32(v.z);
        Self {
            cam_whole: [wx, wy, wz],
            _pad0: 0,
            cam_frac: [fx, fy, fz],
            _pad1: 0,
        }
    }
}

/// Per-chunk persistent GPU transform entry (`@group(1) @binding(2)`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub(crate) struct ChunkTransformGpu {
    pub(crate) whole: [i32; 3],
    pub(crate) scale: f32,
    pub(crate) frac: [u32; 3],
    /// Bit 0 = visible (`1` = drawn, `0` = clipped by vertex shader); bit 1 = pickable.
    pub(crate) flags: u32,
}

impl ChunkTransformGpu {
    pub(crate) fn new(origin_rel: Vec3, visible: bool, pickable: bool) -> Self {
        let (wx, fx) = to_q32_32(origin_rel.x);
        let (wy, fy) = to_q32_32(origin_rel.y);
        let (wz, fz) = to_q32_32(origin_rel.z);
        Self {
            whole: [wx, wy, wz],
            scale: 1.0,
            frac: [fx, fy, fz],
            flags: u32::from(visible) | (u32::from(pickable) << 1),
        }
    }
}

#[inline]
fn in_q32_range(v: Vec3) -> bool {
    v.is_finite()
        && v.x.abs() < MAX_Q32_COORD
        && v.y.abs() < MAX_Q32_COORD
        && v.z.abs() < MAX_Q32_COORD
}

#[inline]
fn to_q32_32(v: f64) -> (i32, u32) {
    if !v.is_finite() || v.abs() >= MAX_Q32_COORD {
        return (0, 0);
    }
    let floor = v.floor();
    let scaled = ((v - floor) * 4_294_967_296.0).round();
    if scaled >= 4_294_967_296.0 {
        ((floor as i32).saturating_add(1), 0)
    } else if scaled <= 0.0 {
        (floor as i32, 0)
    } else {
        (floor as i32, scaled as u32)
    }
}

/// Returns whether `device` supports vertex-stage storage buffers for chunk transforms.
pub(crate) fn supports(device: &wgpu::Device) -> bool {
    let info = device.adapter_info();
    let limits = device.limits();
    info.backend != wgpu::Backend::Gl
        && limits.max_storage_buffers_per_shader_stage >= 1
        && limits.max_storage_buffer_binding_size >= (1024 * 1024)
        && limits.max_buffer_size >= (1024 * 1024)
}

pub(crate) fn bind_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("iletki-3d storage transforms layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(size_of::<ViewCameraOrigin>() as u64),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(size_of::<ChunkTransformGpu>() as u64),
                },
                count: None,
            },
        ],
    })
}

/// Persistent GPU storage buffer holding every chunk's Q32.32 fixed-point origin and visibility.
#[derive(Default)]
pub(crate) struct ChunkTransformTable {
    anchor: Option<Vec3>,
    out_of_range: bool,
    cpu: Vec<ChunkTransformGpu>,
    dirty: Vec<usize>,
    dirty_flags: Vec<bool>,
    buffer: Option<wgpu::Buffer>,
    capacity: usize,
    generation: u64,
}

impl ChunkTransformTable {
    pub(crate) fn anchor(&self) -> Vec3 {
        self.anchor.unwrap_or(Vec3::ZERO)
    }

    pub(crate) fn in_range_for_camera(&self, camera_pos: Vec3) -> bool {
        !self.out_of_range && in_q32_range(camera_pos - self.anchor())
    }

    pub(crate) fn mark_chunk(&mut self, index: usize, origin: Vec3, visible: bool, pickable: bool) {
        if self.anchor.is_none() && origin.is_finite() {
            self.anchor = Some(origin);
        }
        let rel = origin - self.anchor();
        if !in_q32_range(rel) {
            self.out_of_range = true;
        }
        self.set_raw(index, ChunkTransformGpu::new(rel, visible, pickable));
    }

    pub(crate) fn clear_chunk(&mut self, index: usize) {
        self.set_raw(index, ChunkTransformGpu::default());
    }

    fn set_raw(&mut self, index: usize, transform: ChunkTransformGpu) {
        if index >= self.cpu.len() {
            let new_len = index.saturating_add(1);
            self.cpu.resize(new_len, ChunkTransformGpu::default());
            self.dirty_flags.resize(new_len, false);
        }
        if let Some(slot) = self.cpu.get_mut(index) {
            *slot = transform;
        }
        if let Some(flag) = self.dirty_flags.get_mut(index)
            && !*flag
        {
            *flag = true;
            self.dirty.push(index);
        }
    }

    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    pub(crate) fn buffer(&self) -> Option<&wgpu::Buffer> {
        self.buffer.as_ref()
    }

    /// Synchronizes only modified chunk transforms to the GPU buffer.
    pub(crate) fn sync(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let needed = self.cpu.len().max(1);
        let reallocate = self.buffer.is_none() || self.capacity < needed;
        if reallocate {
            let capacity = needed.max(MIN_TRANSFORM_CAPACITY).next_power_of_two();
            let byte_size = (capacity as u64).saturating_mul(size_of::<ChunkTransformGpu>() as u64);
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("iletki-3d chunk transforms storage"),
                size: byte_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            if !self.cpu.is_empty() {
                queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&self.cpu));
            }
            self.buffer = Some(buffer);
            self.capacity = capacity;
            self.generation = self.generation.wrapping_add(1);
            for idx in self.dirty.drain(..) {
                if let Some(flag) = self.dirty_flags.get_mut(idx) {
                    *flag = false;
                }
            }
            return;
        }

        if self.dirty.is_empty() {
            return;
        }
        let Some(buffer) = &self.buffer else {
            return;
        };

        // Coalesce contiguous or nearby dirty spans into minimal queue.write_buffer calls.
        let gap_limit = if self.dirty.len() > 16 {
            DIRTY_SPAN_GAP
        } else {
            0
        };
        self.dirty.sort_unstable();
        let mut span_start: Option<usize> = None;
        let mut span_end: usize = 0;
        let stride = size_of::<ChunkTransformGpu>() as u64;

        let flush_span = |start: usize, end: usize| {
            if let Some(slice) = self.cpu.get(start..end) {
                queue.write_buffer(buffer, (start as u64) * stride, bytemuck::cast_slice(slice));
            }
        };

        for idx in self.dirty.drain(..) {
            if let Some(flag) = self.dirty_flags.get_mut(idx) {
                *flag = false;
            }
            match span_start {
                Some(_) if idx <= span_end.saturating_add(gap_limit) => {
                    span_end = idx + 1;
                }
                Some(start) => {
                    flush_span(start, span_end);
                    span_start = Some(idx);
                    span_end = idx + 1;
                }
                None => {
                    span_start = Some(idx);
                    span_end = idx + 1;
                }
            }
        }
        if let Some(start) = span_start {
            flush_span(start, span_end);
        }
    }
}

/// Per-viewport uniform + bind group referencing the shared [`ChunkTransformTable`].
pub(crate) struct StorageViewBinding {
    origin_buf: wgpu::Buffer,
    bind_group: Option<wgpu::BindGroup>,
    bound_generation: u64,
    last_origin: Option<ViewCameraOrigin>,
}

impl StorageViewBinding {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let origin_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("iletki-3d view camera origin"),
            size: size_of::<ViewCameraOrigin>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            origin_buf,
            bind_group: None,
            bound_generation: 0,
            last_origin: None,
        }
    }

    pub(crate) fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        table: &ChunkTransformTable,
        camera_pos: Vec3,
    ) {
        let origin = ViewCameraOrigin::from_vec3(camera_pos - table.anchor());
        let changed = self
            .last_origin
            .is_none_or(|prev| bytemuck::bytes_of(&prev) != bytemuck::bytes_of(&origin));
        if changed {
            queue.write_buffer(&self.origin_buf, 0, bytemuck::bytes_of(&origin));
            self.last_origin = Some(origin);
        }
        if (self.bind_group.is_none() || self.bound_generation != table.generation())
            && let Some(transforms_buf) = table.buffer()
        {
            self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("iletki-3d storage view binding"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.origin_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: transforms_buf.as_entire_binding(),
                    },
                ],
            }));
            self.bound_generation = table.generation();
        }
    }

    pub(crate) fn bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.bind_group.as_ref()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DrawIndexedIndirectArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DrawIndirectArgs {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}

struct IndirectPageGroup {
    geometry: Arc<GeometryPage>,
    indexed: bool,
    first_arg: u32,
    arg_count: u32,
}

struct RepeatedMaskGroup {
    geometry: Arc<GeometryPage>,
    elements: u32,
    indexed: bool,
    first_instance: u32,
    instance_count: u32,
}

#[derive(Default)]
struct IndirectMaskState {
    active: bool,
    direct: Vec<usize>,
    groups: Vec<IndirectPageGroup>,
    indexed_args: Vec<DrawIndexedIndirectArgs>,
    non_indexed_args: Vec<DrawIndirectArgs>,
    indexed_buf: Option<wgpu::Buffer>,
    indexed_capacity: usize,
    non_indexed_buf: Option<wgpu::Buffer>,
    non_indexed_capacity: usize,
    // Instanced mask groups for repeated geometry pages (1 draw call per page).
    rep_groups: Vec<RepeatedMaskGroup>,
    rep_entries: Vec<(u32, u32)>,
    rep_chunk_ids: Vec<u32>,
    rep_counts: Vec<u32>,
    rep_offsets: Vec<u32>,
    rep_buf: Option<wgpu::Buffer>,
    rep_capacity: usize,
    // Reusable scratch for zero-per-chunk-Arc::clone unordered mask builds.
    page_table: Vec<Arc<GeometryPage>>,
    page_index: HashMap<usize, u32>,
    raw_ranges: Vec<(u32, u32, u32)>,
}

impl IndirectMaskState {
    fn reset(&mut self) {
        self.active = false;
        self.direct.clear();
        self.groups.clear();
        self.indexed_args.clear();
        self.non_indexed_args.clear();
        self.rep_groups.clear();
        self.rep_entries.clear();
        self.rep_chunk_ids.clear();
        self.page_table.clear();
        self.page_index.clear();
        self.raw_ranges.clear();
    }
}

/// Retained draw list for storage-buffer packed geometry (zero per-frame uniform copies).
#[derive(Default)]
pub(crate) struct StorageDraws {
    commands: Vec<StorageDraw>,
    section_gap_end: Option<u32>,
    indirect: IndirectMaskState,
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    section_gap_vertices: usize,
}

enum StorageDraw {
    Direct(usize),
    Packed {
        geometry: Arc<GeometryPage>,
        elements: Range<u32>,
        /// `None` for regular vertex-stream draw; `Some(range)` for repeated instanced draw.
        instances: Option<Range<u32>>,
    },
}

impl StorageDraws {
    pub(crate) fn clear(&mut self) {
        self.commands.clear();
        self.section_gap_end = None;
        self.indirect.reset();
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        {
            self.section_gap_vertices = 0;
        }
    }

    pub(crate) fn has_packed(&self) -> bool {
        !self.indirect.rep_groups.is_empty()
            || (self.indirect.active && !self.indirect.groups.is_empty())
            || self
                .commands
                .iter()
                .any(|cmd| matches!(cmd, StorageDraw::Packed { .. }))
    }

    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    pub(crate) fn count(&self) -> usize {
        let base = if self.indirect.active {
            self.indirect
                .direct
                .len()
                .saturating_add(self.indirect.groups.len())
        } else {
            self.commands.len()
        };
        base.saturating_add(self.indirect.rep_groups.len())
    }

    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    pub(crate) fn section_gap_vertices(&self) -> usize {
        self.section_gap_vertices
    }

    /// Stages a clipped element gap after an active raster draw.
    fn stage_section_gap(&mut self, geometry: &VertexRange) {
        let VertexRange::Packed { page, elements, .. } = geometry else {
            return;
        };
        let Some(StorageDraw::Packed {
            geometry: previous_page,
            elements: previous,
            instances: None,
        }) = self.commands.last()
        else {
            return;
        };
        if self.section_gap_end.unwrap_or(previous.end) == elements.start
            && elements.end.saturating_sub(previous.end) <= MAX_SECTION_GAP_ELEMENTS
            && Arc::ptr_eq(previous_page, page)
        {
            self.section_gap_end = Some(elements.end);
        }
    }

    #[inline]
    pub(crate) fn push_color(
        &mut self,
        geometry: &VertexRange,
        draw_index: usize,
        clipped: bool,
        bridge_sections: bool,
        repeat: bool,
    ) {
        if !clipped {
            self.push_inner(geometry, draw_index, repeat);
        } else if bridge_sections {
            self.stage_section_gap(geometry);
        }
    }

    pub(crate) fn push(&mut self, geometry: &VertexRange, draw_index: usize) {
        self.push_inner(geometry, draw_index, false);
    }

    fn push_inner(&mut self, geometry: &VertexRange, draw_index: usize, repeat: bool) {
        self.indirect.active = false;
        let gap_end = self.section_gap_end.take();
        let VertexRange::Packed {
            page,
            elements,
            slot,
        } = geometry
        else {
            self.commands.push(StorageDraw::Direct(draw_index));
            return;
        };
        let repeated_elems = page.repeated_elements();
        if repeat && repeated_elems != 0 && page.repeated_chunk_ids().is_some() {
            let inst = *slot as u32;
            if let Some(StorageDraw::Packed {
                geometry: prev_page,
                instances: Some(range),
                ..
            }) = self.commands.last_mut()
                && Arc::ptr_eq(prev_page, page)
                && range.end == inst
            {
                range.end = inst.saturating_add(1);
                return;
            }
            self.commands.push(StorageDraw::Packed {
                geometry: page.clone(),
                elements: 0..repeated_elems,
                instances: Some(inst..inst.saturating_add(1)),
            });
            return;
        }

        if let Some(StorageDraw::Packed {
            geometry: prev_page,
            elements: range,
            instances: None,
        }) = self.commands.last_mut()
            && Arc::ptr_eq(prev_page, page)
            && gap_end.unwrap_or(range.end) == elements.start
        {
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            if let Some(end) = gap_end {
                self.section_gap_vertices = self
                    .section_gap_vertices
                    .saturating_add((end - range.end) as usize);
            }
            range.end = elements.end;
            return;
        }
        self.commands.push(StorageDraw::Packed {
            geometry: page.clone(),
            elements: elements.clone(),
            instances: None,
        });
    }

    /// Rebuilds an order-independent draw list directly from `(chunk_slot, geometry)` pairs without
    /// per-chunk `Arc::clone` refcount traffic, coalesces contiguous ranges per geometry page, and
    /// uploads per-page multi-draw indirect buffers when supported and worthwhile.
    pub(crate) fn rebuild_unordered<'a>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        items: impl Iterator<Item = (usize, &'a VertexRange)>,
    ) {
        self.clear();
        let mut last_ptr: usize = 0;
        let mut last_page_idx: u32 = 0;

        for (slot_index, geometry) in items {
            match geometry {
                VertexRange::Packed { page, elements, .. } => {
                    let is_repeated =
                        page.repeated_elements() != 0 && page.repeated_chunk_ids().is_some();
                    if !is_repeated && elements.end <= elements.start {
                        continue;
                    }
                    let ptr = Arc::as_ptr(page) as usize;
                    let page_idx = if ptr == last_ptr {
                        last_page_idx
                    } else if let Some(&existing) = self.indirect.page_index.get(&ptr) {
                        last_ptr = ptr;
                        last_page_idx = existing;
                        existing
                    } else {
                        let Ok(idx) = u32::try_from(self.indirect.page_table.len()) else {
                            continue;
                        };
                        self.indirect.page_table.push(page.clone());
                        self.indirect.page_index.insert(ptr, idx);
                        last_ptr = ptr;
                        last_page_idx = idx;
                        idx
                    };
                    if is_repeated {
                        if let Ok(slot_u32) = u32::try_from(slot_index) {
                            self.indirect.rep_entries.push((page_idx, slot_u32));
                        }
                    } else {
                        self.indirect
                            .raw_ranges
                            .push((page_idx, elements.start, elements.end));
                    }
                }
                VertexRange::Direct(_) | VertexRange::DirectIndexed { .. } => {
                    self.indirect.direct.push(slot_index);
                }
            }
        }

        // Pack repeated-page selections into one instanced draw per page using O(N) counting sort.
        self.pack_repeated_entries(device, queue, None);

        if self.indirect.raw_ranges.is_empty() {
            for &slot in &self.indirect.direct {
                self.commands.push(StorageDraw::Direct(slot));
            }
            self.indirect.direct.clear();
            self.indirect.page_table.clear();
            self.indirect.page_index.clear();
            return;
        }

        // Sort flat 12-byte Copy triples (page_idx, start, end) and coalesce adjacent ranges.
        self.indirect.raw_ranges.sort_unstable();
        let mut write = 0usize;
        for read in 0..self.indirect.raw_ranges.len() {
            let Some(&(page_idx, start, end)) = self.indirect.raw_ranges.get(read) else {
                continue;
            };
            if write > 0
                && let Some((prev_page, _, prev_end)) = self.indirect.raw_ranges.get_mut(write - 1)
                && *prev_page == page_idx
                && start <= *prev_end
            {
                *prev_end = (*prev_end).max(end);
            } else {
                if let Some(dst) = self.indirect.raw_ranges.get_mut(write) {
                    *dst = (page_idx, start, end);
                }
                write += 1;
            }
        }
        self.indirect.raw_ranges.truncate(write);

        let use_indirect = write >= MIN_INDIRECT_DRAWS
            && device
                .features()
                .contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT);
        if !use_indirect {
            for &slot in &self.indirect.direct {
                self.commands.push(StorageDraw::Direct(slot));
            }
            for &(page_idx, start, end) in &self.indirect.raw_ranges {
                if let Some(page) = self.indirect.page_table.get(page_idx as usize) {
                    self.commands.push(StorageDraw::Packed {
                        geometry: page.clone(),
                        elements: start..end,
                        instances: None,
                    });
                }
            }
            self.indirect.direct.clear();
            self.indirect.page_table.clear();
            self.indirect.page_index.clear();
            self.indirect.raw_ranges.clear();
            return;
        }

        let mut active_page: Option<(u32, usize)> = None;
        for &(page_idx, start, end) in &self.indirect.raw_ranges {
            let count = end.saturating_sub(start);
            if count == 0 {
                continue;
            }
            let Some(page) = self.indirect.page_table.get(page_idx as usize) else {
                continue;
            };
            let indexed = page.indices().is_some();
            let arg_index = if indexed {
                let Ok(idx) = u32::try_from(self.indirect.indexed_args.len()) else {
                    self.indirect.reset();
                    return;
                };
                self.indirect.indexed_args.push(DrawIndexedIndirectArgs {
                    index_count: count,
                    instance_count: 1,
                    first_index: start,
                    base_vertex: 0,
                    first_instance: 0,
                });
                idx
            } else {
                let Ok(idx) = u32::try_from(self.indirect.non_indexed_args.len()) else {
                    self.indirect.reset();
                    return;
                };
                self.indirect.non_indexed_args.push(DrawIndirectArgs {
                    vertex_count: count,
                    instance_count: 1,
                    first_vertex: start,
                    first_instance: 0,
                });
                idx
            };
            if let Some((prev_page, group_idx)) = active_page
                && prev_page == page_idx
                && let Some(group) = self.indirect.groups.get_mut(group_idx)
            {
                group.arg_count = group.arg_count.saturating_add(1);
            } else {
                let group_idx = self.indirect.groups.len();
                self.indirect.groups.push(IndirectPageGroup {
                    geometry: page.clone(),
                    indexed,
                    first_arg: arg_index,
                    arg_count: 1,
                });
                active_page = Some((page_idx, group_idx));
            }
        }
        self.indirect.page_table.clear();
        self.indirect.page_index.clear();
        self.indirect.raw_ranges.clear();

        if self.indirect.groups.is_empty() {
            for &slot in &self.indirect.direct {
                self.commands.push(StorageDraw::Direct(slot));
            }
            self.indirect.reset();
            return;
        }

        if !self.indirect.indexed_args.is_empty() {
            let needed = self.indirect.indexed_args.len();
            if self.indirect.indexed_buf.is_none() || self.indirect.indexed_capacity < needed {
                let capacity = needed.max(64).next_power_of_two();
                let byte_size = (capacity as u64).saturating_mul(DRAW_INDEXED_STRIDE);
                self.indirect.indexed_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("iletki-3d mask indirect indexed args"),
                    size: byte_size,
                    usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                self.indirect.indexed_capacity = capacity;
            }
            if let Some(buf) = &self.indirect.indexed_buf {
                queue.write_buffer(buf, 0, bytemuck::cast_slice(&self.indirect.indexed_args));
            }
        }

        if !self.indirect.non_indexed_args.is_empty() {
            let needed = self.indirect.non_indexed_args.len();
            if self.indirect.non_indexed_buf.is_none()
                || self.indirect.non_indexed_capacity < needed
            {
                let capacity = needed.max(64).next_power_of_two();
                let byte_size = (capacity as u64).saturating_mul(DRAW_NON_INDEXED_STRIDE);
                self.indirect.non_indexed_buf =
                    Some(device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("iletki-3d mask indirect args"),
                        size: byte_size,
                        usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }));
                self.indirect.non_indexed_capacity = capacity;
            }
            if let Some(buf) = &self.indirect.non_indexed_buf {
                queue.write_buffer(
                    buf,
                    0,
                    bytemuck::cast_slice(&self.indirect.non_indexed_args),
                );
            }
        }

        self.indirect.active = true;
    }

    /// Fast-path mask rebuild for pre-indexed `(repeated_page_idx, chunk_slot)` entries with zero
    /// `HashMap` lookups or per-chunk `GeometryPage` dereferences.
    pub(crate) fn rebuild_repeated_only(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pages: &[Arc<GeometryPage>],
        entries: impl Iterator<Item = (u32, u32)>,
    ) {
        self.clear();
        if pages.len() == 1
            && let Some(page) = pages.first()
        {
            self.indirect
                .rep_chunk_ids
                .extend(entries.map(|(_, slot)| slot));
            if let Ok(count) = u32::try_from(self.indirect.rep_chunk_ids.len())
                && count > 0
            {
                self.indirect.rep_groups.push(RepeatedMaskGroup {
                    geometry: page.clone(),
                    elements: page.repeated_elements(),
                    indexed: page.indices().is_some(),
                    first_instance: 0,
                    instance_count: count,
                });
                self.upload_repeated_instances(device, queue);
            }
            return;
        }
        self.indirect.rep_entries.extend(entries);
        self.pack_repeated_entries(device, queue, Some(pages));
    }

    fn upload_repeated_instances(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.indirect.rep_chunk_ids.is_empty() {
            return;
        }
        let needed = self.indirect.rep_chunk_ids.len();
        if self.indirect.rep_buf.is_none() || self.indirect.rep_capacity < needed {
            let capacity = needed.max(256).next_power_of_two();
            let byte_size = (capacity as u64).saturating_mul(4);
            self.indirect.rep_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("iletki-3d repeated mask instances"),
                size: byte_size,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.indirect.rep_capacity = capacity;
        }
        if let Some(buf) = &self.indirect.rep_buf {
            queue.write_buffer(buf, 0, bytemuck::cast_slice(&self.indirect.rep_chunk_ids));
        }
    }

    fn pack_repeated_entries(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        external_pages: Option<&[Arc<GeometryPage>]>,
    ) {
        if self.indirect.rep_entries.is_empty() {
            return;
        }
        let pages = external_pages.unwrap_or(&self.indirect.page_table);
        let num_pages = pages.len();
        self.indirect.rep_counts.clear();
        self.indirect.rep_counts.resize(num_pages, 0);
        for &(page_idx, _) in &self.indirect.rep_entries {
            if let Some(c) = self.indirect.rep_counts.get_mut(page_idx as usize) {
                *c = c.saturating_add(1);
            }
        }
        self.indirect.rep_offsets.clear();
        self.indirect.rep_offsets.resize(num_pages, 0);
        let mut running: u32 = 0;
        for (page_idx, &count) in self.indirect.rep_counts.iter().enumerate() {
            if let Some(off) = self.indirect.rep_offsets.get_mut(page_idx) {
                *off = running;
            }
            if count > 0
                && let Some(page) = pages.get(page_idx)
            {
                self.indirect.rep_groups.push(RepeatedMaskGroup {
                    geometry: page.clone(),
                    elements: page.repeated_elements(),
                    indexed: page.indices().is_some(),
                    first_instance: running,
                    instance_count: count,
                });
                running = running.saturating_add(count);
            }
        }
        self.indirect.rep_chunk_ids.resize(running as usize, 0);
        for &(page_idx, chunk_slot) in &self.indirect.rep_entries {
            if let Some(cursor) = self.indirect.rep_offsets.get_mut(page_idx as usize) {
                if let Some(dst) = self.indirect.rep_chunk_ids.get_mut(*cursor as usize) {
                    *dst = chunk_slot;
                }
                *cursor = cursor.saturating_add(1);
            }
        }
        self.indirect.rep_entries.clear();
        self.upload_repeated_instances(device, queue);
    }

    /// Replays the retained draw list into `pass` with a single `@group(1)` storage binding.
    pub(crate) fn paint(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        storage_bg: &wgpu::BindGroup,
        pipelines: &PaintPipelines<'_>,
        picking: bool,
        direct: impl FnMut(&mut wgpu::RenderPass<'_>, usize),
    ) {
        self.paint_inner(pass, storage_bg, pipelines, picking, picking, direct);
    }

    /// Replays the retained draw list binding only position (`@location(0)`) and chunk slot
    /// (`@location(3)`), avoiding normal/color vertex-stream binds and fetches in mask passes.
    pub(crate) fn paint_position_only(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        storage_bg: &wgpu::BindGroup,
        pipeline: &wgpu::RenderPipeline,
        repeated_pipeline: Option<&wgpu::RenderPipeline>,
        direct: impl FnMut(&mut wgpu::RenderPass<'_>, usize),
    ) {
        if !self.indirect.rep_groups.is_empty()
            && let (Some(rep_pl), Some(rep_buf)) =
                (repeated_pipeline, self.indirect.rep_buf.as_ref())
        {
            pass.set_bind_group(1, storage_bg, &[]);
            pass.set_pipeline(rep_pl);
            for group in &self.indirect.rep_groups {
                group.geometry.bind_position(pass);
                let byte_start = u64::from(group.first_instance).saturating_mul(4);
                let byte_end = u64::from(group.first_instance.saturating_add(group.instance_count))
                    .saturating_mul(4);
                pass.set_vertex_buffer(1, rep_buf.slice(byte_start..byte_end));
                if group.indexed {
                    if let Some(indices) = group.geometry.indices() {
                        pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint16);
                        pass.draw_indexed(0..group.elements, 0, 0..group.instance_count);
                    }
                } else {
                    pass.draw(0..group.elements, 0..group.instance_count);
                }
            }
        }
        let pipelines = PaintPipelines {
            smooth: pipeline,
            flat: None,
            repeated_smooth: None,
            repeated_flat: None,
            repeated_slots: None,
        };
        self.paint_inner(pass, storage_bg, &pipelines, true, false, direct);
    }

    fn paint_inner(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        storage_bg: &wgpu::BindGroup,
        pipelines: &PaintPipelines<'_>,
        position_only: bool,
        picking: bool,
        mut direct: impl FnMut(&mut wgpu::RenderPass<'_>, usize),
    ) {
        if self.indirect.active && !picking {
            for &index in &self.indirect.direct {
                direct(pass, index);
            }
            if !self.indirect.groups.is_empty() {
                pass.set_bind_group(1, storage_bg, &[]);
                let mut active_flat: Option<bool> = None;
                for group in &self.indirect.groups {
                    let use_flat =
                        group.geometry.kind() == PageKind::IndexedFlat && pipelines.flat.is_some();
                    if active_flat != Some(use_flat) {
                        let pipeline = if use_flat {
                            pipelines.flat.unwrap_or(pipelines.smooth)
                        } else {
                            pipelines.smooth
                        };
                        pass.set_pipeline(pipeline);
                        active_flat = Some(use_flat);
                    }
                    let slot_binding = if position_only {
                        group.geometry.bind_position(pass);
                        1
                    } else {
                        group.geometry.bind_vertices(pass)
                    };
                    pass.set_vertex_buffer(slot_binding, group.geometry.slots().slice(..));
                    if group.indexed {
                        if let (Some(indices), Some(buf)) =
                            (group.geometry.indices(), self.indirect.indexed_buf.as_ref())
                        {
                            pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint16);
                            let offset =
                                u64::from(group.first_arg).saturating_mul(DRAW_INDEXED_STRIDE);
                            pass.multi_draw_indexed_indirect(buf, offset, group.arg_count);
                        }
                    } else if let Some(buf) = self.indirect.non_indexed_buf.as_ref() {
                        let offset =
                            u64::from(group.first_arg).saturating_mul(DRAW_NON_INDEXED_STRIDE);
                        pass.multi_draw_indirect(buf, offset, group.arg_count);
                    }
                }
            }
            return;
        }

        let mut group_bound = false;
        let mut bound_page: Option<(*const GeometryPage, bool)> = None;

        for command in &self.commands {
            match command {
                StorageDraw::Direct(index) => {
                    group_bound = false;
                    bound_page = None;
                    direct(pass, *index);
                }
                StorageDraw::Packed {
                    geometry,
                    elements,
                    instances,
                } => {
                    if !group_bound {
                        pass.set_bind_group(1, storage_bg, &[]);
                        group_bound = true;
                    }
                    let is_repeated = instances.is_some();
                    let page_key = (Arc::as_ptr(geometry), is_repeated);
                    if bound_page != Some(page_key) {
                        let is_flat = geometry.kind() == PageKind::IndexedFlat;
                        let slots = if is_repeated {
                            let pipeline = if is_flat {
                                pipelines.repeated_flat.or(pipelines.repeated_smooth)
                            } else {
                                pipelines.repeated_smooth
                            };
                            let (Some(pipeline), Some(slots)) =
                                (pipeline, geometry.repeated_chunk_ids())
                            else {
                                continue;
                            };
                            pass.set_pipeline(pipeline);
                            slots
                        } else {
                            let pipeline = if is_flat {
                                pipelines.flat.unwrap_or(pipelines.smooth)
                            } else {
                                pipelines.smooth
                            };
                            pass.set_pipeline(pipeline);
                            geometry.slots()
                        };
                        let slot_binding = if position_only {
                            geometry.bind_position(pass);
                            1
                        } else {
                            geometry.bind_vertices(pass)
                        };
                        pass.set_vertex_buffer(slot_binding, slots.slice(..));
                        if picking {
                            let Some(picks) = geometry.picks() else {
                                continue;
                            };
                            pass.set_vertex_buffer(slot_binding + 1, picks.slice(..));
                        }
                        if let Some(indices) = geometry.indices() {
                            pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint16);
                        }
                        bound_page = Some(page_key);
                    }
                    let inst_range = instances.clone().unwrap_or(0..1);
                    if geometry.indices().is_some() {
                        pass.draw_indexed(elements.clone(), 0, inst_range);
                    } else {
                        pass.draw(elements.clone(), inst_range);
                    }
                }
            }
        }
    }
}
