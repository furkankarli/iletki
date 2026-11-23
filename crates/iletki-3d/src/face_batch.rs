//! Immutable vertex pages and per-viewport draw ranges.

use crate::{Vertex3D, scene::ObjectUniforms};
use iletki_wgpu::wgpu::{self, util::DeviceExt};
use std::{num::NonZeroU64, ops::Range, sync::Arc};

// The matching WGSL uniform array is 4 KiB, within the WebGPU uniform binding limit.
const OBJECTS: usize = 256;
const MAX_VERTICES: usize = 65_536;
pub(crate) const OBJECT_BYTES: u64 = (OBJECTS * size_of::<ObjectUniforms>()) as u64;

pub(crate) struct GeometryPage {
    vertices: wgpu::Buffer,
    slots: wgpu::Buffer,
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
}

#[derive(Default)]
pub(crate) struct PageBuilder {
    vertices: Vec<Vertex3D>,
    slots: Vec<u32>,
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
        device: &wgpu::Device,
        uploaded: &mut Vec<(usize, VertexRange)>,
    ) {
        if self.objects.len() == OBJECTS || self.vertices.len() + vertices.len() > MAX_VERTICES {
            self.flush(device, uploaded);
        }
        let start = self.vertices.len() as u32;
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
    }
}

pub(crate) fn bind_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("iletki-3d geometry page transforms"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(OBJECT_BYTES),
            },
            count: None,
        }],
    })
}

struct ViewUniforms {
    buffer: wgpu::Buffer,
    binding: wgpu::BindGroup,
    transforms: Box<[ObjectUniforms; OBJECTS]>,
}

#[derive(Default)]
pub(crate) struct PackedDraws {
    pages: Vec<ViewUniforms>,
    commands: Vec<Draw>,
    last_page: Option<Arc<GeometryPage>>,
    used_pages: usize,
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
    }

    pub(crate) fn push(
        &mut self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        geometry: &VertexRange,
        offset: ObjectUniforms,
        draw_index: usize,
    ) {
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
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("iletki-3d geometry transform page"),
                    size: OBJECT_BYTES,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("iletki-3d geometry transform binding"),
                    layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffer.as_entire_binding(),
                    }],
                });
                self.pages.push(ViewUniforms {
                    buffer,
                    binding,
                    transforms: Box::new(
                        [ObjectUniforms {
                            offset_rel: [0.0; 3],
                            scale: 1.0,
                        }; OBJECTS],
                    ),
                });
            }
            self.used_pages += 1;
            self.last_page = Some(page.clone());
        }
        let uniforms = self.used_pages.saturating_sub(1);
        if let Some(uniforms) = self.pages.get_mut(uniforms)
            && let Some(transform) = uniforms.transforms.get_mut(*slot)
        {
            *transform = offset;
        }
        // Only merge adjacent visible ranges. This keeps primitive order and never revives
        // hidden or deleted objects when another object on the page is still live.
        if let Some(Draw::Packed {
            uniforms: prev,
            vertices: range,
            ..
        }) = self.commands.last_mut()
            && *prev == uniforms
            && range.end == vertices.start
        {
            range.end = vertices.end;
            return;
        }
        self.commands.push(Draw::Packed {
            geometry: page.clone(),
            uniforms,
            vertices: vertices.clone(),
        });
    }

    pub(crate) fn upload(&self, queue: &wgpu::Queue) {
        for page in self.pages.iter().take(self.used_pages) {
            queue.write_buffer(
                &page.buffer,
                0,
                bytemuck::cast_slice(page.transforms.as_slice()),
            );
        }
    }

    pub(crate) fn has_packed(&self) -> bool {
        self.used_pages != 0
    }

    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    pub(crate) fn counts(&self) -> (usize, usize) {
        (self.commands.len(), self.used_pages)
    }

    pub(crate) fn paint(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        pipeline: &wgpu::RenderPipeline,
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
                        let Some(page) = self.pages.get(*uniforms) else {
                            continue;
                        };
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(1, &page.binding, &[]);
                        pass.set_vertex_buffer(0, geometry.vertices.slice(..));
                        pass.set_vertex_buffer(1, geometry.slots.slice(..));
                        bound = Some(*uniforms);
                    }
                    pass.draw(vertices.clone(), 0..1);
                }
            }
        }
    }
}
