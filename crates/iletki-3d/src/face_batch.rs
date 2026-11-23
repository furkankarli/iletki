//! Immutable vertex pages and per-viewport draw ranges.

use crate::{Vertex3D, scene::ObjectUniforms};
use iletki_wgpu::wgpu::{self, util::DeviceExt};
use std::{num::NonZeroU64, ops::Range, sync::Arc};

// The matching WGSL uniform array is 4 KiB, within the WebGPU uniform binding limit.
const OBJECTS: usize = 256;
const MAX_VERTICES: usize = 65_536;
// Bound the extra vertex work in exchange for one fewer draw.
const MAX_SECTION_GAP_VERTICES: u32 = 256;
const UNIFORM_BLOCK_BYTES: usize = 1024 * 1024;
pub(crate) const OBJECT_BYTES: u64 = (OBJECTS * size_of::<ObjectUniforms>()) as u64;

pub(crate) struct GeometryPage {
    vertices: wgpu::Buffer,
    slots: wgpu::Buffer,
    picks: Option<wgpu::Buffer>,
}

pub(crate) enum VertexRange {
    Direct(wgpu::Buffer),
    Packed {
        page: Arc<GeometryPage>,
        vertices: Range<u32>,
        slot: usize,
    },
}

impl VertexRange {
    pub(crate) fn slice(&self) -> wgpu::BufferSlice<'_> {
        match self {
            Self::Direct(buffer) => buffer.slice(..),
            Self::Packed { page, vertices, .. } => {
                let stride = size_of::<Vertex3D>() as u64;
                page.vertices
                    .slice((u64::from(vertices.start) * stride)..(u64::from(vertices.end) * stride))
            }
        }
    }

    pub(crate) fn pick_slice(&self) -> Option<wgpu::BufferSlice<'_>> {
        let Self::Packed { page, vertices, .. } = self else {
            return None;
        };
        let stride = size_of::<[u32; 2]>() as u64;
        Some(
            page.picks
                .as_ref()?
                .slice((u64::from(vertices.start) * stride)..(u64::from(vertices.end) * stride)),
        )
    }
}

#[derive(Default)]
pub(crate) struct PageBuilder {
    vertices: Vec<Vertex3D>,
    slots: Vec<u32>,
    picks: Vec<[u32; 2]>,
    objects: Vec<(usize, Range<u32>)>,
}

impl PageBuilder {
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

    pub(crate) fn push(
        &mut self,
        chunk: usize,
        vertices: &[Vertex3D],
        picks: Option<&[[u32; 2]]>,
        device: &wgpu::Device,
        uploaded: &mut Vec<(usize, VertexRange)>,
    ) {
        if self.objects.len() == OBJECTS || self.vertices.len() + vertices.len() > MAX_VERTICES {
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
        self.slots
            .resize(self.vertices.len(), self.objects.len() as u32);
        self.objects
            .push((chunk, start..self.vertices.len() as u32));
    }

    pub(crate) fn flush(
        &mut self,
        device: &wgpu::Device,
        uploaded: &mut Vec<(usize, VertexRange)>,
    ) {
        if self.objects.is_empty() {
            return;
        }
        let make_buffer = |label, contents| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::VERTEX,
            })
        };
        let page = Arc::new(GeometryPage {
            vertices: make_buffer(
                "iletki-3d geometry page",
                bytemuck::cast_slice(&self.vertices),
            ),
            slots: make_buffer(
                "iletki-3d geometry slots",
                bytemuck::cast_slice(&self.slots),
            ),
            picks: (!self.picks.is_empty()).then(|| {
                make_buffer(
                    "iletki-3d geometry pick IDs",
                    bytemuck::cast_slice(&self.picks),
                )
            }),
        });
        for (slot, (chunk, vertices)) in self.objects.drain(..).enumerate() {
            uploaded.push((
                chunk,
                VertexRange::Packed {
                    page: page.clone(),
                    vertices,
                    slot,
                },
            ));
        }
        self.vertices.clear();
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
        vertices: Range<u32>,
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

    /// Stages fully clipped vertices after an active raster draw. A matching retained range
    /// commits the gap, so trailing clipped geometry stays out of the draw.
    fn stage_section_gap(&mut self, geometry: &VertexRange, offset: ObjectUniforms) {
        let VertexRange::Packed {
            page,
            vertices,
            slot,
        } = geometry
        else {
            return;
        };
        let Some(Draw::Packed {
            geometry: previous_page,
            uniforms,
            vertices: previous,
        }) = self.commands.last()
        else {
            return;
        };
        if self.section_gap_end.unwrap_or(previous.end) == vertices.start
            && vertices.end - previous.end <= MAX_SECTION_GAP_VERTICES
            && Arc::ptr_eq(previous_page, page)
            && let Some(transform) = self.pages.get_mut(*uniforms).and_then(|p| p.get_mut(*slot))
        {
            *transform = offset;
            self.section_gap_end = Some(vertices.end);
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
    ) {
        if !clipped {
            self.push(geometry, offset, draw_index);
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
        let gap_end = self.section_gap_end.take();
        let VertexRange::Packed {
            page,
            vertices,
            slot,
        } = geometry
        else {
            self.commands.push(Draw::Direct(draw_index));
            return;
        };
        if !self
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
        if let Some(uniforms) = self.pages.get_mut(uniforms)
            && let Some(transform) = uniforms.get_mut(*slot)
        {
            *transform = offset;
        }
        // Each merged vertex has staged uniforms for this viewport. The range ends at
        // retained geometry and preserves the existing draw order.
        if let Some(Draw::Packed {
            uniforms: prev,
            vertices: range,
            ..
        }) = self.commands.last_mut()
            && *prev == uniforms
            && gap_end.unwrap_or(range.end) == vertices.start
        {
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            if let Some(end) = gap_end {
                self.section_gap_vertices += (end - range.end) as usize;
            }
            range.end = vertices.end;
            return;
        }
        self.commands.push(Draw::Packed {
            geometry: page.clone(),
            uniforms,
            vertices: vertices.clone(),
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
        pipeline: &wgpu::RenderPipeline,
        picking: bool,
        mut direct: impl FnMut(&mut wgpu::RenderPass<'_>, usize),
    ) {
        let mut bound = None;
        for command in &self.commands {
            match command {
                Draw::Direct(index) => {
                    bound = None;
                    direct(pass, *index);
                }
                Draw::Packed {
                    geometry,
                    uniforms,
                    vertices,
                } => {
                    if bound != Some(*uniforms) {
                        let pages_per_buffer = self.pages_per_buffer.max(1);
                        let Some(block) = self.buffers.get(*uniforms / pages_per_buffer) else {
                            continue;
                        };
                        pass.set_pipeline(pipeline);
                        let offset = [((*uniforms % pages_per_buffer) * self.stride) as u32];
                        let offsets = if self.dynamic_offsets {
                            offset.as_slice()
                        } else {
                            &[]
                        };
                        pass.set_bind_group(1, &block.binding, offsets);
                        pass.set_vertex_buffer(0, geometry.vertices.slice(..));
                        pass.set_vertex_buffer(1, geometry.slots.slice(..));
                        if picking {
                            let Some(picks) = &geometry.picks else {
                                continue;
                            };
                            pass.set_vertex_buffer(2, picks.slice(..));
                        }
                        bound = Some(*uniforms);
                    }
                    pass.draw(vertices.clone(), 0..1);
                }
            }
        }
    }
}
