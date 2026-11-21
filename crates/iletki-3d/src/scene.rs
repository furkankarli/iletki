//! Camera-relative Reversed-Z 3D scene layer with automatic 3D frustum culling, 6 section clip
//! planes, *Shaded with Edges* depth-offset wireframe overlays, Jump-Flood selection outline masks,
//! 64-bit sub-object GPU picking and simultaneous multi-viewport sharing.

use crate::vertex::{Mesh3D, Vertex3D};
use bytemuck::{Pod, Zeroable};
use iletki_core::{Bounds3, DrawPhase, DrawPhaseMask, Vec3, ViewportId};
use iletki_wgpu::wgpu;
use iletki_wgpu::wgpu::util::DeviceExt;
use iletki_wgpu::{
    CAMERA_WGSL, DEPTH_FORMAT, OUTLINE_MASK_FORMAT, PICK_ID_FORMAT, PaintCtx, PrepareCtx,
    ViewportLayer,
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct ObjectUniforms {
    pub(crate) offset_rel: [f32; 3],
    pub(crate) scale: f32,
}

/// Identifier of a 3D mesh chunk stored inside [`Scene3DLayer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Chunk3DId(usize);

struct ViewportBinding {
    id: ViewportId,
    buf: wgpu::Buffer,
    bg: wgpu::BindGroup,
}

struct ChunkEntry {
    origin: Vec3,
    local_bounds: Bounds3,
    visible: bool,
    selected: bool,
    pending_mesh: Option<Mesh3D>,
    vertex_buf: Option<wgpu::Buffer>,
    vertex_count: u32,
    edge_buf: Option<wgpu::Buffer>,
    edge_count: u32,
    pick_buf: Option<wgpu::Buffer>,
    has_picks: bool,
    per_viewport: Vec<ViewportBinding>,
}

struct Pipelines {
    format: wgpu::TextureFormat,
    msaa: u32,
    object_layout: wgpu::BindGroupLayout,
    lit: wgpu::RenderPipeline,
    edge: wgpu::RenderPipeline,
    mask: wgpu::RenderPipeline,
    pick: wgpu::RenderPipeline,
}

/// Camera-relative Reversed-Z 3D scene layer implementing [`ViewportLayer`].
///
/// Each chunk keeps its own `f64` world origin; every frame `prepare` computes
/// `(chunk.origin - camera_pos).to_f32()` into a per-[`ViewportId`] uniform buffer so 3D geometry
/// stays jitter-free even hundreds of kilometres from `(0, 0, 0)` and can be shared safely across
/// simultaneous multi-viewport quad views in a single `CommandEncoder`.
pub struct Scene3DLayer {
    layer_id: u16,
    pipelines: Option<Pipelines>,
    chunks: Vec<Option<ChunkEntry>>,
}

impl std::fmt::Debug for Scene3DLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scene3DLayer")
            .field("layer_id", &self.layer_id)
            .field("chunks", &self.chunks.len())
            .finish()
    }
}

impl Scene3DLayer {
    /// Creates an empty 3D scene layer with identifier `layer_id`.
    #[must_use]
    pub fn new(layer_id: u16) -> Self {
        Self {
            layer_id,
            pipelines: None,
            chunks: Vec::new(),
        }
    }

    /// Inserts a 3D mesh chunk anchored at `mesh.origin` and returns its [`Chunk3DId`].
    pub fn insert(&mut self, mesh: Mesh3D) -> Chunk3DId {
        let entry = ChunkEntry {
            origin: mesh.origin,
            local_bounds: mesh.local_bounds,
            visible: true,
            selected: false,
            pending_mesh: Some(mesh),
            vertex_buf: None,
            vertex_count: 0,
            edge_buf: None,
            edge_count: 0,
            pick_buf: None,
            has_picks: false,
            per_viewport: Vec::new(),
        };
        if let Some(pos) = self.chunks.iter().position(Option::is_none) {
            if let Some(slot) = self.chunks.get_mut(pos) {
                *slot = Some(entry);
            }
            return Chunk3DId(pos);
        }
        let idx = self.chunks.len();
        self.chunks.push(Some(entry));
        Chunk3DId(idx)
    }

    /// Updates a chunk's world origin (for moving 3D objects without re-uploading vertices).
    pub fn set_origin(&mut self, id: Chunk3DId, origin: Vec3) {
        if origin.is_finite()
            && let Some(Some(entry)) = self.chunks.get_mut(id.0)
        {
            entry.origin = origin;
        }
    }

    /// Toggles visibility of a chunk.
    pub fn set_visible(&mut self, id: Chunk3DId, visible: bool) {
        if let Some(Some(entry)) = self.chunks.get_mut(id.0) {
            entry.visible = visible;
        }
    }

    /// Toggles Jump-Flood selection outline highlighting for a chunk.
    ///
    /// When any visible chunk is selected, this layer automatically participates in
    /// [`DrawPhase::OutlineMask`] so [`iletki_wgpu::ViewportHost`] renders a crisp screen-space
    /// Jump-Flood silhouette outline around it.
    pub fn set_selected(&mut self, id: Chunk3DId, selected: bool) {
        if let Some(Some(entry)) = self.chunks.get_mut(id.0) {
            entry.selected = selected;
        }
    }

    /// Returns whether `id` is currently marked as selected.
    #[must_use]
    pub fn is_selected(&self, id: Chunk3DId) -> bool {
        self.chunks
            .get(id.0)
            .and_then(|s| s.as_ref())
            .is_some_and(|e| e.selected)
    }

    /// Clears selection on all chunks.
    pub fn clear_selection(&mut self) {
        for chunk in self.chunks.iter_mut().flatten() {
            chunk.selected = false;
        }
    }

    /// Removes a chunk and frees its GPU buffers.
    pub fn remove(&mut self, id: Chunk3DId) {
        if let Some(slot) = self.chunks.get_mut(id.0) {
            *slot = None;
        }
    }

    fn ensure_pipelines<W>(&mut self, ctx: &PrepareCtx<'_, W>) {
        if self
            .pipelines
            .as_ref()
            .is_some_and(|p| p.format == ctx.color_format && p.msaa == ctx.msaa_samples)
        {
            return;
        }
        let wgsl = format!("{CAMERA_WGSL}\n{}", include_str!("shaders/mesh3d.wgsl"));
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("iletki-3d mesh shader"),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });
        let object_layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("iletki-3d object layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("iletki-3d mesh pipeline layout"),
                bind_group_layouts: &[Some(ctx.camera_bind_group_layout), Some(&object_layout)],
                immediate_size: 0,
            });
        let vert_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex3D>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![
                0 => Float32x3,
                1 => Float32x3,
                2 => Unorm8x4
            ],
        };
        let pick_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<[u32; 2]>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![3 => Uint32x2],
        };

        let lit = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("iletki-3d lit mesh"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_mesh"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Some(vert_layout.clone())],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_lit"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: ctx.color_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    // Keep back faces visible so section clip planes reveal solid interiors.
                    cull_mode: None,
                    ..wgpu::PrimitiveState::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: ctx.msaa_samples,
                    ..wgpu::MultisampleState::default()
                },
                multiview_mask: None,
                cache: None,
            });

        let edge = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("iletki-3d shaded edges"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_edge"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Some(vert_layout.clone())],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_edge"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: ctx.color_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    cull_mode: None,
                    ..wgpu::PrimitiveState::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: ctx.msaa_samples,
                    ..wgpu::MultisampleState::default()
                },
                multiview_mask: None,
                cache: None,
            });

        let mask = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("iletki-3d outline mask"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_mask"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Some(vert_layout.clone())],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_mask"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: OUTLINE_MASK_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..wgpu::PrimitiveState::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        let pick = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("iletki-3d pick mesh"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_mesh_pick"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Some(vert_layout), Some(pick_layout)],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_pick"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: PICK_ID_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..wgpu::PrimitiveState::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        for chunk in self.chunks.iter_mut().flatten() {
            chunk.per_viewport.clear();
        }
        self.pipelines = Some(Pipelines {
            format: ctx.color_format,
            msaa: ctx.msaa_samples,
            object_layout,
            lit,
            edge,
            mask,
            pick,
        });
    }
}

impl<W> ViewportLayer<W> for Scene3DLayer {
    fn id(&self) -> u16 {
        self.layer_id
    }

    fn phases(&self) -> DrawPhaseMask {
        let mut mask = DrawPhaseMask::OPAQUE;
        let any_picks = self.chunks.iter().flatten().any(|c| {
            c.visible
                && (c.has_picks
                    || c.pending_mesh
                        .as_ref()
                        .is_some_and(|m| !m.pick_ids.is_empty()))
        });
        if any_picks {
            mask = mask.union(DrawPhaseMask::PICKING);
        }
        let any_selected = self
            .chunks
            .iter()
            .flatten()
            .any(|c| c.visible && c.selected);
        if any_selected {
            mask = mask.union(DrawPhaseMask::OUTLINE_MASK);
        }
        mask
    }

    fn prepare(&mut self, ctx: &mut PrepareCtx<'_, W>) -> bool {
        self.ensure_pipelines(ctx);
        let Some(p) = &self.pipelines else {
            return false;
        };
        let cam_pos = ctx.camera.position_3d();
        let vp_id = ctx.viewport_id;

        for chunk in self.chunks.iter_mut().flatten() {
            if let Some(mesh) = chunk.pending_mesh.take() {
                if mesh.vertices.is_empty() {
                    chunk.vertex_buf = None;
                    chunk.vertex_count = 0;
                    chunk.pick_buf = None;
                    chunk.has_picks = false;
                } else {
                    chunk.vertex_count = u32::try_from(mesh.vertices.len()).unwrap_or(0);
                    chunk.vertex_buf = Some(ctx.device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("iletki-3d vertices"),
                            contents: bytemuck::cast_slice(&mesh.vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        },
                    ));
                    if mesh.pick_ids.len() == mesh.vertices.len() {
                        chunk.pick_buf = Some(ctx.device.create_buffer_init(
                            &wgpu::util::BufferInitDescriptor {
                                label: Some("iletki-3d pick ids"),
                                contents: bytemuck::cast_slice(&mesh.pick_ids),
                                usage: wgpu::BufferUsages::VERTEX,
                            },
                        ));
                        chunk.has_picks = true;
                    } else {
                        chunk.pick_buf = None;
                        chunk.has_picks = false;
                    }
                }
                if mesh.edges.is_empty() {
                    chunk.edge_buf = None;
                    chunk.edge_count = 0;
                } else {
                    chunk.edge_count = u32::try_from(mesh.edges.len()).unwrap_or(0);
                    chunk.edge_buf = Some(ctx.device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("iletki-3d edges"),
                            contents: bytemuck::cast_slice(&mesh.edges),
                            usage: wgpu::BufferUsages::VERTEX,
                        },
                    ));
                }
            }

            if !chunk.per_viewport.iter().any(|b| b.id == vp_id) {
                let buf = ctx
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("iletki-3d object uniforms"),
                        contents: bytemuck::bytes_of(&ObjectUniforms {
                            offset_rel: [0.0, 0.0, 0.0],
                            scale: 1.0,
                        }),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    });
                let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("iletki-3d object bg"),
                    layout: &p.object_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    }],
                });
                chunk
                    .per_viewport
                    .push(ViewportBinding { id: vp_id, buf, bg });
            }

            if let Some(binding) = chunk.per_viewport.iter().find(|b| b.id == vp_id) {
                let offset_rel = (chunk.origin - cam_pos).to_f32();
                let u = ObjectUniforms {
                    offset_rel,
                    scale: 1.0,
                };
                ctx.queue
                    .write_buffer(&binding.buf, 0, bytemuck::bytes_of(&u));
            }
        }
        false
    }

    fn paint(&self, phase: DrawPhase, pass: &mut wgpu::RenderPass<'_>, ctx: &PaintCtx<'_, W>) {
        let Some(p) = &self.pipelines else {
            return;
        };
        let frustum = ctx.camera.frustum(ctx.logical_size);
        let cam_pos = ctx.camera.position_3d();
        let vp_id = ctx.viewport_id;

        match phase {
            DrawPhase::Opaque => {
                // 1. Solid shaded triangles
                pass.set_pipeline(&p.lit);
                for chunk in self.chunks.iter().flatten() {
                    if !chunk.visible || chunk.vertex_count == 0 {
                        continue;
                    }
                    let wb = Bounds3::from_corners(
                        chunk.origin + chunk.local_bounds.min,
                        chunk.origin + chunk.local_bounds.max,
                    );
                    if !wb.is_empty() && !frustum.intersects_bounds_world(wb, cam_pos) {
                        continue;
                    }
                    let Some(binding) = chunk.per_viewport.iter().find(|b| b.id == vp_id) else {
                        continue;
                    };
                    if let Some(vbuf) = &chunk.vertex_buf {
                        pass.set_bind_group(1, &binding.bg, &[]);
                        pass.set_vertex_buffer(0, vbuf.slice(..));
                        pass.draw(0..chunk.vertex_count, 0..1);
                    }
                }
                // 2. Crisp depth-offset CAD wireframe edges (*Shaded with Edges*)
                let any_edges = self
                    .chunks
                    .iter()
                    .flatten()
                    .any(|c| c.visible && c.edge_count > 0);
                if any_edges {
                    pass.set_pipeline(&p.edge);
                    for chunk in self.chunks.iter().flatten() {
                        if !chunk.visible || chunk.edge_count == 0 {
                            continue;
                        }
                        let wb = Bounds3::from_corners(
                            chunk.origin + chunk.local_bounds.min,
                            chunk.origin + chunk.local_bounds.max,
                        );
                        if !wb.is_empty() && !frustum.intersects_bounds_world(wb, cam_pos) {
                            continue;
                        }
                        let Some(binding) = chunk.per_viewport.iter().find(|b| b.id == vp_id)
                        else {
                            continue;
                        };
                        if let Some(ebuf) = &chunk.edge_buf {
                            pass.set_bind_group(1, &binding.bg, &[]);
                            pass.set_vertex_buffer(0, ebuf.slice(..));
                            pass.draw(0..chunk.edge_count, 0..1);
                        }
                    }
                }
            }
            DrawPhase::OutlineMask => {
                pass.set_pipeline(&p.mask);
                for chunk in self.chunks.iter().flatten() {
                    if !chunk.visible || !chunk.selected || chunk.vertex_count == 0 {
                        continue;
                    }
                    let wb = Bounds3::from_corners(
                        chunk.origin + chunk.local_bounds.min,
                        chunk.origin + chunk.local_bounds.max,
                    );
                    if !wb.is_empty() && !frustum.intersects_bounds_world(wb, cam_pos) {
                        continue;
                    }
                    let Some(binding) = chunk.per_viewport.iter().find(|b| b.id == vp_id) else {
                        continue;
                    };
                    if let Some(vbuf) = &chunk.vertex_buf {
                        pass.set_bind_group(1, &binding.bg, &[]);
                        pass.set_vertex_buffer(0, vbuf.slice(..));
                        pass.draw(0..chunk.vertex_count, 0..1);
                    }
                }
            }
            DrawPhase::Picking => {
                pass.set_pipeline(&p.pick);
                for chunk in self.chunks.iter().flatten() {
                    if !chunk.visible || !chunk.has_picks || chunk.vertex_count == 0 {
                        continue;
                    }
                    let Some(binding) = chunk.per_viewport.iter().find(|b| b.id == vp_id) else {
                        continue;
                    };
                    if let (Some(vbuf), Some(pbuf)) = (&chunk.vertex_buf, &chunk.pick_buf) {
                        pass.set_bind_group(1, &binding.bg, &[]);
                        pass.set_vertex_buffer(0, vbuf.slice(..));
                        pass.set_vertex_buffer(1, pbuf.slice(..));
                        pass.draw(0..chunk.vertex_count, 0..1);
                    }
                }
            }
            _ => {}
        }
    }
}
