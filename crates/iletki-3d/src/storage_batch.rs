//! Persistent storage-buffer chunk transforms and retained per-viewport draw lists.
//!
//! Capability-gated to backends that support read-only vertex-stage storage buffers (Vulkan, DX12,
//! Metal and WebGPU core). OpenGL / WebGL2 keep the 256-entry uniform fallback path in
//! [`crate::face_batch`].

use crate::face_batch::{GeometryPage, PageKind, PaintPipelines, VertexRange};
use bytemuck::{Pod, Zeroable};
use iletki_core::Vec3;
use iletki_wgpu::wgpu;
use std::num::NonZeroU64;
use std::ops::Range;
use std::sync::Arc;

const MIN_TRANSFORM_CAPACITY: usize = 256;
// Bound the extra vertex work bridged across clipped gaps in raster section mode.
const MAX_SECTION_GAP_ELEMENTS: u32 = 256;
// Safe range for Q32.32 signed differences without i32 overflow (+/- 536,870 km from anchor).
const MAX_Q32_COORD: f64 = 536_870_912.0;
// Merge nearby dirty transform writes (up to 256 entries = 8 KiB) to avoid many tiny queue writes.
const DIRTY_SPAN_GAP: usize = 256;

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

/// Retained draw list for storage-buffer packed geometry (zero per-frame uniform copies).
#[derive(Default)]
pub(crate) struct StorageDraws {
    commands: Vec<StorageDraw>,
    section_gap_end: Option<u32>,
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
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        {
            self.section_gap_vertices = 0;
        }
    }

    pub(crate) fn has_packed(&self) -> bool {
        self.commands
            .iter()
            .any(|cmd| matches!(cmd, StorageDraw::Packed { .. }))
    }

    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    pub(crate) fn count(&self) -> usize {
        self.commands.len()
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

    /// Replays the retained draw list into `pass` with a single `@group(1)` storage binding.
    pub(crate) fn paint(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        storage_bg: &wgpu::BindGroup,
        pipelines: &PaintPipelines<'_>,
        picking: bool,
        mut direct: impl FnMut(&mut wgpu::RenderPass<'_>, usize),
    ) {
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
                        let slot_binding = if picking {
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
