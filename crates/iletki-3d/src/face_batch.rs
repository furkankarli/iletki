//! Immutable vertex/index pages and per-viewport draw ranges.

use crate::{Vertex3D, scene::ObjectUniforms};
use bytemuck::{Pod, Zeroable};
use iletki_wgpu::wgpu::{self, util::DeviceExt};
use std::{num::NonZeroU64, ops::Range, sync::Arc};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShadeAttributes {
    normal: [f32; 3],
    color: [u8; 4],
}

// The matching WGSL uniform array is 4 KiB, within the WebGPU uniform binding limit.
pub(crate) const OBJECTS: usize = 256;
pub(crate) const MAX_VERTICES: usize = 65_536;
// Bound the extra vertex work in exchange for one fewer draw.
const MAX_SECTION_GAP_VERTICES: u32 = 256;
const UNIFORM_BLOCK_BYTES: usize = 1024 * 1024;
pub(crate) const OBJECT_BYTES: u64 = (OBJECTS * size_of::<ObjectUniforms>()) as u64;
const POSITION: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x3];

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PageKind {
    Unindexed,
    IndexedSmooth,
    IndexedFlat,
}

#[derive(Clone, Copy)]
pub(crate) enum VertexLayout {
    Interleaved,
    Separate,
}

impl VertexLayout {
    pub(crate) fn for_device(device: &wgpu::Device) -> Self {
        // Packed color draws use three attribute streams and one object-slot stream.
        if device.limits().max_vertex_buffers >= 4 {
            Self::Separate
        } else {
            Self::Interleaved
        }
    }

    pub(crate) fn buffers(self) -> Vec<Option<wgpu::VertexBufferLayout<'static>>> {
        use wgpu::{VertexBufferLayout, VertexStepMode::Vertex};
        const INTERLEAVED: [wgpu::VertexAttribute; 3] =
            wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Unorm8x4];
        const SHADE_ATTRS: [wgpu::VertexAttribute; 2] =
            wgpu::vertex_attr_array![1 => Float32x3, 2 => Unorm8x4];
        match self {
            Self::Interleaved => vec![Some(VertexBufferLayout {
                array_stride: size_of::<Vertex3D>() as u64,
                step_mode: Vertex,
                attributes: &INTERLEAVED,
            })],
            Self::Separate => vec![
                Some(VertexBufferLayout {
                    array_stride: size_of::<[f32; 3]>() as u64,
                    step_mode: Vertex,
                    attributes: &POSITION,
                }),
                Some(VertexBufferLayout {
                    array_stride: size_of::<ShadeAttributes>() as u64,
                    step_mode: Vertex,
                    attributes: &SHADE_ATTRS,
                }),
            ],
        }
    }

    pub(crate) fn position(self) -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: self.position_stride(),
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &POSITION,
        }
    }

    fn position_stride(self) -> u64 {
        match self {
            Self::Interleaved => size_of::<Vertex3D>() as u64,
            Self::Separate => size_of::<[f32; 3]>() as u64,
        }
    }
}

pub(crate) enum VertexBuffers {
    Interleaved(wgpu::Buffer),
    Separate([wgpu::Buffer; 2]),
}

impl VertexBuffers {
    pub(crate) fn upload(
        device: &wgpu::Device,
        layout: VertexLayout,
        vertices: &[Vertex3D],
    ) -> Self {
        let buffer = |label, contents| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::VERTEX,
            })
        };
        match layout {
            VertexLayout::Interleaved => Self::Interleaved(buffer(
                "iletki-3d mesh vertices",
                bytemuck::cast_slice(vertices),
            )),
            VertexLayout::Separate => {
                let mut positions = Vec::with_capacity(vertices.len());
                let mut attrs = Vec::with_capacity(vertices.len());
                for vertex in vertices {
                    positions.push(vertex.position);
                    attrs.push(ShadeAttributes {
                        normal: vertex.normal,
                        color: vertex.color,
                    });
                }
                Self::Separate([
                    buffer("iletki-3d positions", bytemuck::cast_slice(&positions)),
                    buffer("iletki-3d shade attrs", bytemuck::cast_slice(&attrs)),
                ])
            }
        }
    }

    fn bind_position(&self, pass: &mut wgpu::RenderPass<'_>, vertices: Option<&Range<u32>>) {
        let (buffer, layout) = match self {
            Self::Interleaved(buffer) => (buffer, VertexLayout::Interleaved),
            Self::Separate(buffers) => {
                let [positions, _] = buffers;
                (positions, VertexLayout::Separate)
            }
        };
        bind_vertices(pass, 0, buffer, layout.position_stride(), vertices);
    }

    /// Binds mesh attributes and returns the next available binding.
    fn bind(&self, pass: &mut wgpu::RenderPass<'_>, vertices: Option<&Range<u32>>) -> u32 {
        self.bind_position(pass, vertices);
        match self {
            Self::Interleaved(_) => 1,
            Self::Separate(buffers) => {
                let [_, attrs] = buffers;
                bind_vertices(
                    pass,
                    1,
                    attrs,
                    size_of::<ShadeAttributes>() as u64,
                    vertices,
                );
                2
            }
        }
    }
}

fn bind_vertices(
    pass: &mut wgpu::RenderPass<'_>,
    slot: u32,
    buffer: &wgpu::Buffer,
    stride: u64,
    vertices: Option<&Range<u32>>,
) {
    let slice = match vertices {
        Some(range) => buffer.slice(u64::from(range.start) * stride..u64::from(range.end) * stride),
        None => buffer.slice(..),
    };
    pass.set_vertex_buffer(slot, slice);
}

pub(crate) struct GeometryPage {
    vertices: VertexBuffers,
    indices: Option<wgpu::Buffer>,
    slots: wgpu::Buffer,
    picks: Option<wgpu::Buffer>,
    kind: PageKind,
    repeated_elements: u32,
    repeated_chunk_ids: Option<wgpu::Buffer>,
}

impl GeometryPage {
    pub(crate) fn kind(&self) -> PageKind {
        self.kind
    }

    pub(crate) fn indices(&self) -> Option<&wgpu::Buffer> {
        self.indices.as_ref()
    }

    pub(crate) fn slots(&self) -> &wgpu::Buffer {
        &self.slots
    }

    pub(crate) fn picks(&self) -> Option<&wgpu::Buffer> {
        self.picks.as_ref()
    }

    pub(crate) fn repeated_elements(&self) -> u32 {
        self.repeated_elements
    }

    pub(crate) fn repeated_chunk_ids(&self) -> Option<&wgpu::Buffer> {
        self.repeated_chunk_ids.as_ref()
    }

    pub(crate) fn bind_vertices(&self, pass: &mut wgpu::RenderPass<'_>) -> u32 {
        self.vertices.bind(pass, None)
    }

    pub(crate) fn bind_position(&self, pass: &mut wgpu::RenderPass<'_>) {
        self.vertices.bind_position(pass, None);
    }
}

pub(crate) enum VertexRange {
    // Keep buffer handles out of the chunk table that visibility scans every frame.
    Direct(Box<VertexBuffers>),
    DirectIndexed {
        vertices: Box<VertexBuffers>,
        indices: Box<wgpu::Buffer>,
        index_format: wgpu::IndexFormat,
        index_count: u32,
        flat: bool,
    },
    Packed {
        page: Arc<GeometryPage>,
        /// Vertex range for unindexed pages; index range for indexed pages.
        elements: Range<u32>,
        slot: usize,
    },
}

impl VertexRange {
    pub(crate) fn is_flat(&self) -> bool {
        match self {
            Self::Direct(_) => false,
            Self::DirectIndexed { flat, .. } => *flat,
            Self::Packed { page, .. } => page.kind == PageKind::IndexedFlat,
        }
    }

    pub(crate) fn is_direct(&self) -> bool {
        matches!(self, Self::Direct(_) | Self::DirectIndexed { .. })
    }

    pub(crate) fn repeats(&self) -> bool {
        matches!(self, Self::Packed { page, .. } if page.repeated_elements != 0)
    }

    pub(crate) fn packed_subrange(&self, range: Range<u32>) -> Option<Self> {
        let Self::Packed {
            page,
            elements,
            slot,
        } = self
        else {
            return None;
        };
        if range.start >= range.end || range.end > elements.end - elements.start {
            return None;
        }
        Some(Self::Packed {
            page: page.clone(),
            elements: elements.start + range.start..elements.start + range.end,
            slot: *slot,
        })
    }

    pub(crate) fn draw_direct(&self, pass: &mut wgpu::RenderPass<'_>, count: u32) {
        self.draw_direct_instance(pass, count, 0);
    }

    pub(crate) fn draw_direct_instance(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        count: u32,
        instance: u32,
    ) {
        let instances = instance..instance.saturating_add(1);
        match self {
            Self::Direct(buffers) => {
                buffers.bind(pass, None);
                pass.draw(0..count, instances);
            }
            Self::DirectIndexed {
                vertices,
                indices,
                index_format,
                index_count,
                ..
            } => {
                vertices.bind(pass, None);
                pass.set_index_buffer(indices.slice(..), *index_format);
                pass.draw_indexed(0..*index_count, 0, instances);
            }
            Self::Packed { page, elements, .. } => {
                if let Some(indices) = &page.indices {
                    page.vertices.bind(pass, None);
                    pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint16);
                    pass.draw_indexed(elements.clone(), 0, instances);
                } else {
                    page.vertices.bind(pass, Some(elements));
                    pass.draw(0..count, instances);
                }
            }
        }
    }

    pub(crate) fn draw_direct_pick(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        direct_picks: Option<&wgpu::Buffer>,
        count: u32,
    ) {
        match self {
            Self::Direct(buffers) => {
                let Some(picks) = direct_picks else {
                    return;
                };
                buffers.bind_position(pass, None);
                pass.set_vertex_buffer(1, picks.slice(..));
                pass.draw(0..count, 0..1);
            }
            Self::DirectIndexed {
                vertices,
                indices,
                index_format,
                index_count,
                ..
            } => {
                let Some(picks) = direct_picks else {
                    return;
                };
                vertices.bind_position(pass, None);
                pass.set_vertex_buffer(1, picks.slice(..));
                pass.set_index_buffer(indices.slice(..), *index_format);
                pass.draw_indexed(0..*index_count, 0, 0..1);
            }
            Self::Packed { page, elements, .. } => {
                let Some(picks) = &page.picks else {
                    return;
                };
                if let Some(indices) = &page.indices {
                    page.vertices.bind_position(pass, None);
                    pass.set_vertex_buffer(1, picks.slice(..));
                    pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint16);
                    pass.draw_indexed(elements.clone(), 0, 0..1);
                } else {
                    let stride = size_of::<[u32; 2]>() as u64;
                    page.vertices.bind_position(pass, Some(elements));
                    pass.set_vertex_buffer(
                        1,
                        picks.slice(
                            (u64::from(elements.start) * stride)
                                ..(u64::from(elements.end) * stride),
                        ),
                    );
                    pass.draw(0..count, 0..1);
                }
            }
        }
    }
}

pub(crate) fn upload_index_buffer_u32(device: &wgpu::Device, indices: &[u32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("iletki-3d direct indices"),
        contents: bytemuck::cast_slice(indices),
        usage: wgpu::BufferUsages::INDEX,
    })
}

pub(crate) struct PageBuilder {
    vertices: Vec<Vertex3D>,
    indices: Vec<u16>,
    slots: Vec<u32>,
    picks: Vec<[u32; 2]>,
    // (chunk_index, element_range, vertex_range)
    objects: Vec<(usize, Range<u32>, Range<u32>)>,
    kind: PageKind,
    repeatable: bool,
    global_slots: bool,
    layout: VertexLayout,
}

impl PageBuilder {
    pub(crate) fn new(
        layout: VertexLayout,
        kind: PageKind,
        repeatable: bool,
        global_slots: bool,
    ) -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            slots: Vec::new(),
            picks: Vec::new(),
            objects: Vec::new(),
            kind,
            repeatable,
            global_slots,
            layout,
        }
    }

    pub(crate) fn supports(device: &wgpu::Device) -> bool {
        let limits = device.limits();
        limits.max_uniform_buffer_binding_size >= OBJECT_BYTES
            && limits.max_buffer_size >= (MAX_VERTICES * size_of::<Vertex3D>()) as u64
    }

    pub(crate) fn can_pack(vertices: &[Vertex3D], primitive_vertices: usize) -> bool {
        !vertices.is_empty()
            && vertices.len() <= MAX_VERTICES
            && vertices.len().is_multiple_of(primitive_vertices)
    }

    fn repeated_elements(&self) -> u32 {
        let Some((_, first_elem, first_vert)) = self.objects.first() else {
            return 0;
        };
        if !self.repeatable || !(1..=64).contains(&first_elem.end) || self.objects.len() < 8 {
            return 0;
        }
        let prototype = self
            .vertices
            .get(..first_vert.end as usize)
            .unwrap_or_default();
        let proto_bytes = bytemuck::cast_slice::<_, u8>(prototype);
        if self.kind == PageKind::Unindexed {
            let identical = self.objects.iter().all(|(_, _, vertices)| {
                self.vertices
                    .get(vertices.start as usize..vertices.end as usize)
                    .is_some_and(|v| bytemuck::cast_slice::<_, u8>(v) == proto_bytes)
            });
            return if identical { first_elem.end } else { 0 };
        }
        let proto_idx = self
            .indices
            .get(..first_elem.end as usize)
            .unwrap_or_default();
        let identical = self.objects.iter().all(|(_, elems, verts)| {
            let verts_match = self
                .vertices
                .get(verts.start as usize..verts.end as usize)
                .is_some_and(|v| bytemuck::cast_slice::<_, u8>(v) == proto_bytes);
            if !verts_match {
                return false;
            }
            let Some(base) = u16::try_from(verts.start).ok() else {
                return false;
            };
            self.indices
                .get(elems.start as usize..elems.end as usize)
                .is_some_and(|idx| {
                    idx.len() == proto_idx.len()
                        && idx
                            .iter()
                            .zip(proto_idx)
                            .all(|(&a, &b)| a.checked_sub(base) == Some(b))
                })
        });
        if identical { first_elem.end } else { 0 }
    }

    pub(crate) fn push(
        &mut self,
        chunk: usize,
        vertices: &[Vertex3D],
        picks: Option<&[[u32; 2]]>,
        device: &wgpu::Device,
        uploaded: &mut Vec<(usize, VertexRange)>,
    ) {
        let max_objects = if self.global_slots {
            usize::MAX
        } else {
            OBJECTS
        };
        if self.objects.len() == max_objects || self.vertices.len() + vertices.len() > MAX_VERTICES
        {
            self.flush(device, uploaded);
        }
        let start = self.vertices.len() as u32;
        if let Some(picks) = picks {
            self.picks.resize(self.vertices.len(), [0, 0]);
            self.picks.extend_from_slice(picks);
        } else if !self.picks.is_empty() {
            self.picks
                .resize(self.vertices.len() + vertices.len(), [0, 0]);
        }
        self.vertices.extend_from_slice(vertices);
        let slot_id = if self.global_slots {
            chunk as u32
        } else {
            self.objects.len() as u32
        };
        self.slots.resize(self.vertices.len(), slot_id);
        let range = start..self.vertices.len() as u32;
        self.objects.push((chunk, range.clone(), range));
    }

    pub(crate) fn push_indexed(
        &mut self,
        chunk: usize,
        vertices: &[Vertex3D],
        indices: &[u32],
        picks: Option<&[[u32; 2]]>,
        device: &wgpu::Device,
        uploaded: &mut Vec<(usize, VertexRange)>,
    ) {
        let max_objects = if self.global_slots {
            usize::MAX
        } else {
            OBJECTS
        };
        if self.objects.len() == max_objects || self.vertices.len() + vertices.len() > MAX_VERTICES
        {
            self.flush(device, uploaded);
        }
        let vert_start = self.vertices.len() as u32;
        let idx_start = self.indices.len() as u32;
        if let Some(picks) = picks {
            self.picks.resize(self.vertices.len(), [0, 0]);
            self.picks.extend_from_slice(picks);
        } else if !self.picks.is_empty() {
            self.picks
                .resize(self.vertices.len() + vertices.len(), [0, 0]);
        }
        self.vertices.extend_from_slice(vertices);
        let slot_id = if self.global_slots {
            chunk as u32
        } else {
            self.objects.len() as u32
        };
        self.slots.resize(self.vertices.len(), slot_id);
        self.indices.reserve(indices.len());
        for &idx in indices {
            let page_idx = vert_start.saturating_add(idx).min(u32::from(u16::MAX)) as u16;
            self.indices.push(page_idx);
        }
        self.objects.push((
            chunk,
            idx_start..self.indices.len() as u32,
            vert_start..self.vertices.len() as u32,
        ));
    }

    pub(crate) fn flush(
        &mut self,
        device: &wgpu::Device,
        uploaded: &mut Vec<(usize, VertexRange)>,
    ) {
        if self.objects.is_empty() {
            return;
        }
        let make_buffer = |label, contents, usage| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage,
            })
        };
        let repeated_elements = self.repeated_elements();
        let repeated_chunk_ids = (self.global_slots && repeated_elements != 0).then(|| {
            let ids: Vec<u32> = self
                .objects
                .iter()
                .map(|(chunk, _, _)| *chunk as u32)
                .collect();
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("iletki-3d repeated chunk IDs"),
                contents: bytemuck::cast_slice(&ids),
                usage: wgpu::BufferUsages::VERTEX,
            })
        });
        let indices = if self.kind == PageKind::Unindexed {
            None
        } else {
            if !self.indices.len().is_multiple_of(2) {
                self.indices.push(0);
            }
            Some(make_buffer(
                "iletki-3d geometry indices",
                bytemuck::cast_slice(&self.indices),
                wgpu::BufferUsages::INDEX,
            ))
        };
        let page = Arc::new(GeometryPage {
            vertices: VertexBuffers::upload(device, self.layout, &self.vertices),
            indices,
            slots: make_buffer(
                "iletki-3d geometry slots",
                bytemuck::cast_slice(&self.slots),
                wgpu::BufferUsages::VERTEX,
            ),
            picks: (!self.picks.is_empty()).then(|| {
                make_buffer(
                    "iletki-3d geometry pick IDs",
                    bytemuck::cast_slice(&self.picks),
                    wgpu::BufferUsages::VERTEX,
                )
            }),
            kind: self.kind,
            repeated_elements,
            repeated_chunk_ids,
        });
        for (slot, (chunk, elements, _)) in self.objects.drain(..).enumerate() {
            uploaded.push((
                chunk,
                VertexRange::Packed {
                    page: page.clone(),
                    elements,
                    slot,
                },
            ));
        }
        self.vertices.clear();
        self.indices.clear();
        self.slots.clear();
        self.picks.clear();
    }
}

fn dynamic_offsets(limits: &wgpu::Limits) -> bool {
    limits.max_dynamic_uniform_buffers_per_pipeline_layout != 0
}

pub(crate) fn bind_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("iletki-3d geometry page transforms"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: dynamic_offsets(&device.limits()),
                min_binding_size: NonZeroU64::new(OBJECT_BYTES),
            },
            count: None,
        }],
    })
}

pub(crate) struct PaintPipelines<'a> {
    pub(crate) smooth: &'a wgpu::RenderPipeline,
    pub(crate) flat: Option<&'a wgpu::RenderPipeline>,
    pub(crate) repeated_smooth: Option<&'a wgpu::RenderPipeline>,
    pub(crate) repeated_flat: Option<&'a wgpu::RenderPipeline>,
    pub(crate) repeated_slots: Option<&'a wgpu::Buffer>,
}

struct ViewUniforms {
    buffer: wgpu::Buffer,
    binding: wgpu::BindGroup,
    capacity: usize,
}

#[derive(Default)]
pub(crate) struct PackedDraws {
    pages: Vec<[ObjectUniforms; OBJECTS]>,
    buffers: Vec<ViewUniforms>,
    staging: Vec<u8>,
    pages_per_buffer: usize,
    stride: usize,
    dynamic_offsets: bool,
    commands: Vec<Draw>,
    last_page: Option<Arc<GeometryPage>>,
    used_pages: usize,
    section_gap_end: Option<u32>,
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    section_gap_vertices: usize,
}

enum Draw {
    Direct(usize),
    Packed {
        geometry: Arc<GeometryPage>,
        uniforms: usize,
        elements: Range<u32>,
        instances: u32,
    },
}

impl PackedDraws {
    pub(crate) fn clear(&mut self) {
        self.commands.clear();
        self.last_page = None;
        self.used_pages = 0;
        self.section_gap_end = None;
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        {
            self.section_gap_vertices = 0;
        }
    }

    /// Stages fully clipped elements after an active raster draw. A matching retained range
    /// commits the gap, so trailing clipped geometry stays out of the draw.
    fn stage_section_gap(&mut self, geometry: &VertexRange, offset: ObjectUniforms) {
        let VertexRange::Packed {
            page,
            elements,
            slot,
        } = geometry
        else {
            return;
        };
        let Some(Draw::Packed {
            geometry: previous_page,
            uniforms,
            elements: previous,
            instances: 0,
        }) = self.commands.last()
        else {
            return;
        };
        if self.section_gap_end.unwrap_or(previous.end) == elements.start
            && elements.end - previous.end <= MAX_SECTION_GAP_VERTICES
            && Arc::ptr_eq(previous_page, page)
            && let Some(transform) = self.pages.get_mut(*uniforms).and_then(|p| p.get_mut(*slot))
        {
            *transform = offset;
            self.section_gap_end = Some(elements.end);
        }
    }

    /// Adds a color range, with short gaps staged only for the raster clipping pipeline.
    #[inline]
    pub(crate) fn push_color(
        &mut self,
        geometry: &VertexRange,
        offset: ObjectUniforms,
        draw_index: usize,
        clipped: bool,
        bridge_sections: bool,
        repeat: bool,
    ) {
        if !clipped {
            self.push_inner(geometry, offset, draw_index, repeat);
        } else if bridge_sections {
            self.stage_section_gap(geometry, offset);
        }
    }

    pub(crate) fn push(
        &mut self,
        geometry: &VertexRange,
        offset: ObjectUniforms,
        draw_index: usize,
    ) {
        self.push_inner(geometry, offset, draw_index, false);
    }

    fn push_inner(
        &mut self,
        geometry: &VertexRange,
        offset: ObjectUniforms,
        draw_index: usize,
        repeat: bool,
    ) {
        let gap_end = self.section_gap_end.take();
        let VertexRange::Packed {
            page,
            elements,
            slot,
        } = geometry
        else {
            self.commands.push(Draw::Direct(draw_index));
            return;
        };
        let repeat = repeat && page.repeated_elements != 0;
        let continuation = !repeat
            || matches!(self.commands.last(), Some(Draw::Packed { instances, .. }) if *instances != 0);
        if !continuation
            || !self
                .last_page
                .as_ref()
                .is_some_and(|prev| Arc::ptr_eq(prev, page))
        {
            if self.used_pages == self.pages.len() {
                self.pages.push(
                    [ObjectUniforms {
                        offset_rel: [0.0; 3],
                        scale: 1.0,
                    }; OBJECTS],
                );
            }
            self.used_pages += 1;
            self.last_page = Some(page.clone());
        }
        let uniforms = self.used_pages.saturating_sub(1);
        if repeat {
            if let Some(Draw::Packed {
                uniforms: previous,
                instances,
                ..
            }) = self.commands.last_mut()
                && *previous == uniforms
                && *instances != 0
                && let Some(transform) = self
                    .pages
                    .get_mut(uniforms)
                    .and_then(|p| p.get_mut(*instances as usize))
            {
                *transform = offset;
                *instances += 1;
                return;
            }
            if let Some(transform) = self.pages.get_mut(uniforms).and_then(|p| p.first_mut()) {
                *transform = offset;
                self.commands.push(Draw::Packed {
                    geometry: page.clone(),
                    uniforms,
                    elements: 0..page.repeated_elements,
                    instances: 1,
                });
            }
            return;
        }
        if let Some(uniforms) = self.pages.get_mut(uniforms)
            && let Some(transform) = uniforms.get_mut(*slot)
        {
            *transform = offset;
        }
        // Each merged element has staged uniforms for this viewport. The range ends at
        // retained geometry and preserves the existing draw order.
        if let Some(Draw::Packed {
            uniforms: prev,
            elements: range,
            instances: 0,
            ..
        }) = self.commands.last_mut()
            && *prev == uniforms
            && gap_end.unwrap_or(range.end) == elements.start
        {
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            if let Some(end) = gap_end {
                self.section_gap_vertices += (end - range.end) as usize;
            }
            range.end = elements.end;
            return;
        }
        self.commands.push(Draw::Packed {
            geometry: page.clone(),
            uniforms,
            elements: elements.clone(),
            instances: 0,
        });
    }

    pub(crate) fn upload(
        &mut self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        queue: &wgpu::Queue,
    ) {
        let limits = device.limits();
        let alignment = (limits.min_uniform_buffer_offset_alignment as usize).max(1);
        let page_bytes = OBJECT_BYTES as usize;
        self.stride = page_bytes.div_ceil(alignment) * alignment;
        self.dynamic_offsets = dynamic_offsets(&limits);
        self.pages_per_buffer = if self.dynamic_offsets {
            (UNIFORM_BLOCK_BYTES / self.stride).max(1)
        } else {
            1
        };

        for (index, pages) in self
            .pages
            .get(..self.used_pages)
            .unwrap_or_default()
            .chunks(self.pages_per_buffer)
            .enumerate()
        {
            if self
                .buffers
                .get(index)
                .is_none_or(|block| block.capacity < pages.len())
            {
                let capacity = pages.len().next_power_of_two().min(self.pages_per_buffer);
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("iletki-3d geometry transforms"),
                    size: ((capacity - 1) * self.stride + page_bytes) as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("iletki-3d geometry transform binding"),
                    layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &buffer,
                            offset: 0,
                            size: NonZeroU64::new(OBJECT_BYTES),
                        }),
                    }],
                });
                let block = ViewUniforms {
                    buffer,
                    binding,
                    capacity,
                };
                if let Some(existing) = self.buffers.get_mut(index) {
                    *existing = block;
                } else {
                    self.buffers.push(block);
                }
            }
            let Some(block) = self.buffers.get(index) else {
                continue;
            };
            let bytes = if self.stride == page_bytes || pages.len() == 1 {
                bytemuck::cast_slice(pages)
            } else {
                self.staging
                    .resize((pages.len() - 1) * self.stride + page_bytes, 0);
                for (dst, src) in self.staging.chunks_mut(self.stride).zip(pages) {
                    if let Some(page) = dst.get_mut(..page_bytes) {
                        page.copy_from_slice(bytemuck::cast_slice(src));
                    }
                }
                self.staging.as_slice()
            };
            queue.write_buffer(&block.buffer, 0, bytes);
        }
    }

    pub(crate) fn has_packed(&self) -> bool {
        self.used_pages != 0
    }

    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    pub(crate) fn counts(&self) -> (usize, usize) {
        (
            self.commands.len(),
            self.used_pages.div_ceil(self.pages_per_buffer.max(1)),
        )
    }

    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    pub(crate) fn section_gap_vertices(&self) -> usize {
        self.section_gap_vertices
    }

    pub(crate) fn paint(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        pipelines: &PaintPipelines<'_>,
        picking: bool,
        mut direct: impl FnMut(&mut wgpu::RenderPass<'_>, usize),
    ) {
        let mut bound: Option<(usize, bool)> = None;
        for command in &self.commands {
            match command {
                Draw::Direct(index) => {
                    bound = None;
                    direct(pass, *index);
                }
                Draw::Packed {
                    geometry,
                    uniforms,
                    elements,
                    instances,
                } => {
                    let is_repeated = *instances != 0;
                    if bound != Some((*uniforms, is_repeated)) {
                        let pages_per_buffer = self.pages_per_buffer.max(1);
                        let Some(block) = self.buffers.get(*uniforms / pages_per_buffer) else {
                            continue;
                        };
                        let is_flat = geometry.kind == PageKind::IndexedFlat;
                        let slots = if is_repeated {
                            let pipeline = if is_flat {
                                pipelines.repeated_flat.or(pipelines.repeated_smooth)
                            } else {
                                pipelines.repeated_smooth
                            };
                            let (Some(pipeline), Some(slots)) =
                                (pipeline, pipelines.repeated_slots)
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
                            &geometry.slots
                        };
                        let slot_binding = if picking {
                            geometry.vertices.bind_position(pass, None);
                            1
                        } else {
                            geometry.vertices.bind(pass, None)
                        };
                        pass.set_vertex_buffer(slot_binding, slots.slice(..));
                        let offset = [((*uniforms % pages_per_buffer) * self.stride) as u32];
                        let offsets = if self.dynamic_offsets {
                            offset.as_slice()
                        } else {
                            &[]
                        };
                        pass.set_bind_group(1, &block.binding, offsets);
                        if picking {
                            let Some(picks) = &geometry.picks else {
                                continue;
                            };
                            pass.set_vertex_buffer(slot_binding + 1, picks.slice(..));
                        }
                        if let Some(indices) = &geometry.indices {
                            pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint16);
                        }
                        bound = Some((*uniforms, is_repeated));
                    }
                    if geometry.indices.is_some() {
                        pass.draw_indexed(elements.clone(), 0, 0..(*instances).max(1));
                    } else {
                        pass.draw(elements.clone(), 0..(*instances).max(1));
                    }
                }
            }
        }
    }
}
