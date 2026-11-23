//! Camera-relative Reversed-Z 3D scene layer with automatic 3D frustum culling, 6 section clip
//! planes, *Shaded with Edges* depth-offset wireframe overlays, Jump-Flood selection outline masks,
//! 64-bit sub-object GPU picking and simultaneous multi-viewport sharing.

use crate::face_batch::{self, PageBuilder, VertexRange};
use crate::objects::{PickVolume, PreparedObject, ViewObjects};
use crate::section::SectionAa;
use crate::vertex::{Mesh3D, Vertex3D};
use bytemuck::{Pod, Zeroable};
use iletki_core::{Bounds3, DrawPhase, DrawPhaseMask, Vec3, Vec4};
use iletki_wgpu::wgpu;
use iletki_wgpu::wgpu::util::DeviceExt;
use iletki_wgpu::{
    CAMERA_WGSL, DEPTH_FORMAT, LayerRevision, OUTLINE_MASK_FORMAT, PICK_ID_FORMAT, PaintCtx,
    PrepareCtx, ViewportLayer,
};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_CHUNK_ID: AtomicU64 = AtomicU64::new(1);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct ObjectUniforms {
    pub(crate) offset_rel: [f32; 3],
    pub(crate) scale: f32,
}

/// Identifier of a 3D mesh chunk stored inside [`Scene3DLayer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Chunk3DId {
    index: usize,
    generation: u64,
}

struct ChunkEntry {
    generation: u64,
    origin: Vec3,
    local_bounds: Bounds3,
    visible: bool,
    selected: bool,
    pending_mesh: Option<Mesh3D>,
    vertex_buf: Option<VertexRange>,
    vertex_count: u32,
    edge_buf: Option<VertexRange>,
    edge_count: u32,
    pick_buf: Option<wgpu::Buffer>,
    has_picks: bool,
    pick_bounds_trusted: bool,
}

struct Pipelines {
    format: wgpu::TextureFormat,
    msaa: u32,
    object_layout: wgpu::BindGroupLayout,
    dynamic_uniforms: bool,
    lit: wgpu::RenderPipeline,
    packed: Option<PackedPipelines>,
    section: Option<SectionPipelines>,
    edge: wgpu::RenderPipeline,
    mask: wgpu::RenderPipeline,
    multisampled_mask: Option<wgpu::RenderPipeline>,
    pick: wgpu::RenderPipeline,
}

struct ColorPipelines {
    faces: wgpu::RenderPipeline,
    edges: wgpu::RenderPipeline,
}

struct PackedPipelines {
    layout: wgpu::BindGroupLayout,
    base: ColorPipelines,
    section: Option<ColorPipelines>,
}

struct SectionPipelines {
    lit: wgpu::RenderPipeline,
    edge: wgpu::RenderPipeline,
    multisampled_mask: wgpu::RenderPipeline,
}

/// Camera-relative Reversed-Z 3D scene layer implementing [`ViewportLayer`].
///
/// Each chunk keeps its own `f64` world origin. Preparation computes
/// `(chunk.origin - camera_pos).to_f32()` for visible chunks and uploads their uniforms in
/// per-[`iletki_core::ViewportId`] pages so 3D geometry
/// stays jitter-free even hundreds of kilometres from `(0, 0, 0)` and can be shared safely across
/// simultaneous multi-viewport quad views in a single `CommandEncoder`.
pub struct Scene3DLayer {
    layer_id: u16,
    revision: LayerRevision,
    pipelines: Option<Pipelines>,
    use_antialiased_section: bool,
    chunks: Vec<Option<ChunkEntry>>,
    views: Vec<ViewObjects>,
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    profile_batch_uniforms: bool,
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    profile_batch_geometry: bool,
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
            revision: LayerRevision::new(),
            pipelines: None,
            use_antialiased_section: false,
            chunks: Vec::new(),
            views: Vec::new(),
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            profile_batch_uniforms: true,
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            profile_batch_geometry: true,
        }
    }

    /// Sets uniform upload batching for A/B profiling. Visibility filtering still applies.
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    pub fn set_profile_batch_uniforms(&mut self, enabled: bool) {
        if self.profile_batch_uniforms != enabled {
            self.profile_batch_uniforms = enabled;
            self.views.clear();
            self.revision.mark_static();
        }
    }

    /// Controls shared geometry pages for A/B profiling. Set before uploading the scene.
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    pub fn set_profile_batch_geometry(&mut self, enabled: bool) {
        if self.profile_batch_geometry != enabled {
            self.profile_batch_geometry = enabled;
            self.revision.mark_static();
        }
    }

    /// Inserts a 3D mesh chunk anchored at `mesh.origin` and returns its [`Chunk3DId`].
    pub fn insert(&mut self, mesh: Mesh3D) -> Chunk3DId {
        self.revision.mark_static();
        let generation = NEXT_CHUNK_ID.fetch_add(1, Ordering::Relaxed);
        let entry = ChunkEntry {
            generation,
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
            pick_bounds_trusted: false,
        };
        if let Some((index, slot)) = self
            .chunks
            .iter_mut()
            .enumerate()
            .find(|(_, slot)| slot.is_none())
        {
            *slot = Some(entry);
            return Chunk3DId { index, generation };
        }
        let index = self.chunks.len();
        self.chunks.push(Some(entry));
        Chunk3DId { index, generation }
    }

    /// Updates a chunk's world origin (for moving 3D objects without re-uploading vertices).
    pub fn set_origin(&mut self, id: Chunk3DId, origin: Vec3) {
        if origin.is_finite()
            && let Some(entry) = self.chunk_mut(id)
        {
            entry.origin = origin;
            self.revision.mark_static();
        }
    }

    /// Toggles visibility of a chunk.
    pub fn set_visible(&mut self, id: Chunk3DId, visible: bool) {
        if let Some(entry) = self.chunk_mut(id) {
            entry.visible = visible;
            self.revision.mark_static();
        }
    }

    /// Toggles Jump-Flood selection outline highlighting for a chunk.
    ///
    /// When any visible chunk is selected, this layer automatically participates in
    /// [`DrawPhase::OutlineMask`] so [`iletki_wgpu::ViewportHost`] renders a crisp screen-space
    /// Jump-Flood silhouette outline around it.
    pub fn set_selected(&mut self, id: Chunk3DId, selected: bool) {
        if let Some(entry) = self.chunk_mut(id) {
            entry.selected = selected;
            self.revision.mark_dynamic();
        }
    }

    /// Returns whether `id` is currently marked as selected.
    #[must_use]
    pub fn is_selected(&self, id: Chunk3DId) -> bool {
        self.chunks
            .get(id.index)
            .and_then(|s| s.as_ref())
            .is_some_and(|entry| entry.generation == id.generation && entry.selected)
    }

    /// Clears selection on all chunks.
    pub fn clear_selection(&mut self) {
        self.revision.mark_dynamic();
        for chunk in self.chunks.iter_mut().flatten() {
            chunk.selected = false;
        }
    }

    /// Removes a chunk and releases its GPU resources. Other chunks can retain a shared page.
    pub fn remove(&mut self, id: Chunk3DId) {
        if let Some(slot) = self.chunks.get_mut(id.index)
            && slot
                .as_ref()
                .is_some_and(|entry| entry.generation == id.generation)
        {
            *slot = None;
            for view in &mut self.views {
                view.faces.clear();
                view.edges.clear();
            }
            self.revision.mark_static();
        }
    }

    fn chunk_mut(&mut self, id: Chunk3DId) -> Option<&mut ChunkEntry> {
        self.chunks
            .get_mut(id.index)?
            .as_mut()
            .filter(|entry| entry.generation == id.generation)
    }

    /// Releases per-viewport binding sets for closed viewports and returns the number released.
    ///
    /// The layer also prunes closed bindings before preparing a viewport.
    pub fn prune_closed_viewports(&mut self) -> usize {
        let previous = self.views.len();
        self.views.retain(|view| view.lifetime.is_alive());
        previous - self.views.len()
    }

    fn ensure_pipelines<W>(&mut self, ctx: &PrepareCtx<'_, W>) {
        let section_aa = SectionAa::for_viewport(ctx);
        if self.pipelines.as_ref().is_some_and(|p| {
            p.format == ctx.color_format
                && p.msaa == ctx.msaa_samples
                && (section_aa.is_none() || p.section.is_some())
        }) {
            return;
        }
        let wgsl = format!("{CAMERA_WGSL}\n{}", include_str!("shaders/mesh3d.wgsl"));
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("iletki-3d mesh shader"),
                source: wgpu::ShaderSource::Wgsl(wgsl.as_str().into()),
            });
        let dynamic_uniforms = ctx
            .device
            .limits()
            .max_dynamic_uniform_buffers_per_pipeline_layout
            > 0;
        let object_layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("iletki-3d object layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: dynamic_uniforms,
                        min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<
                            ObjectUniforms,
                        >()
                            as u64),
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

        let make_color = |shader: &wgpu::ShaderModule,
                          vertex,
                          fragment,
                          layout: &wgpu::PipelineLayout,
                          buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
                          edges: bool| {
            ctx.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(if edges {
                        "iletki-3d shaded edges"
                    } else {
                        "iletki-3d lit mesh"
                    }),
                    layout: Some(layout),
                    vertex: wgpu::VertexState {
                        module: shader,
                        entry_point: Some(vertex),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers,
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: shader,
                        entry_point: Some(fragment),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: ctx.color_format,
                            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: if edges {
                            wgpu::PrimitiveTopology::LineList
                        } else {
                            wgpu::PrimitiveTopology::TriangleList
                        },
                        // Keep back faces visible so section clip planes reveal solid interiors.
                        cull_mode: None,
                        ..wgpu::PrimitiveState::default()
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: DEPTH_FORMAT,
                        depth_write_enabled: Some(!edges),
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
                })
        };
        let face_layout = [Some(vert_layout.clone())];
        let lit = make_color(
            &shader,
            "vs_mesh",
            "fs_lit",
            &pipeline_layout,
            &face_layout,
            false,
        );
        let edge = make_color(
            &shader,
            "vs_edge",
            "fs_edge",
            &pipeline_layout,
            &face_layout,
            true,
        );
        let packed = PageBuilder::supports(ctx.device).then(|| {
            let layout = face_batch::bind_layout(ctx.device);
            let pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("iletki-3d packed geometry layout"),
                        bind_group_layouts: &[Some(ctx.camera_bind_group_layout), Some(&layout)],
                        immediate_size: 0,
                    });
            let source = format!("{wgsl}\n{}", include_str!("shaders/mesh3d_batched.wgsl"));
            let shader = ctx
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("iletki-3d packed geometry shader"),
                    source: wgpu::ShaderSource::Wgsl(source.as_str().into()),
                });
            let buffers = [
                Some(vert_layout.clone()),
                Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<u32>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![3 => Uint32],
                }),
            ];
            let pipelines =
                |shader: &wgpu::ShaderModule, face, face_fragment, edge, edge_fragment| {
                    ColorPipelines {
                        faces: make_color(
                            shader,
                            face,
                            face_fragment,
                            &pipeline_layout,
                            &buffers,
                            false,
                        ),
                        edges: make_color(
                            shader,
                            edge,
                            edge_fragment,
                            &pipeline_layout,
                            &buffers,
                            true,
                        ),
                    }
                };
            let base = pipelines(
                &shader,
                "vs_mesh_batched",
                "fs_lit",
                "vs_edge_batched",
                "fs_edge",
            );
            let section = section_aa.map(|aa| {
                let sampled = concat!(
                    include_str!("shaders/mesh3d_sampled.wgsl"),
                    "\n",
                    include_str!("shaders/mesh3d_batched_sampled.wgsl")
                );
                let raster = concat!(
                    include_str!("shaders/mesh3d_raster.wgsl"),
                    "\n",
                    include_str!("shaders/mesh3d_batched_raster.wgsl")
                );
                let shader = aa.shader(ctx.device, &source, sampled, raster);
                match aa {
                    SectionAa::Sampled => pipelines(
                        &shader,
                        "vs_mesh_batched_sampled",
                        "fs_lit_sampled",
                        "vs_edge_batched_sampled",
                        "fs_edge_sampled",
                    ),
                    SectionAa::Raster => pipelines(
                        &shader,
                        "vs_mesh_batched_raster",
                        "fs_lit_raster",
                        "vs_edge_batched_raster",
                        "fs_unlit",
                    ),
                }
            });
            PackedPipelines {
                layout,
                base,
                section,
            }
        });

        let make_mask = |shader: &wgpu::ShaderModule, vertex, fragment, count| {
            ctx.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("iletki-3d outline mask"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: shader,
                        entry_point: Some(vertex),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[Some(vert_layout.clone())],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: shader,
                        entry_point: Some(fragment),
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
                    multisample: wgpu::MultisampleState {
                        count,
                        ..Default::default()
                    },
                    multiview_mask: None,
                    cache: None,
                })
        };
        let mask = make_mask(&shader, "vs_mask", "fs_mask", 1);
        let multisampled_mask =
            (ctx.msaa_samples == 4).then(|| make_mask(&shader, "vs_mask", "fs_mask", 4));

        let section = section_aa.map(|aa| {
            let shader = aa.shader(
                ctx.device,
                &wgsl,
                include_str!("shaders/mesh3d_sampled.wgsl"),
                include_str!("shaders/mesh3d_raster.wgsl"),
            );
            match aa {
                SectionAa::Sampled => SectionPipelines {
                    lit: make_color(
                        &shader,
                        "vs_mesh_sampled",
                        "fs_lit_sampled",
                        &pipeline_layout,
                        &face_layout,
                        false,
                    ),
                    edge: make_color(
                        &shader,
                        "vs_edge_sampled",
                        "fs_edge_sampled",
                        &pipeline_layout,
                        &face_layout,
                        true,
                    ),
                    multisampled_mask: make_mask(&shader, "vs_mesh_sampled", "fs_mask_sampled", 4),
                },
                SectionAa::Raster => SectionPipelines {
                    lit: make_color(
                        &shader,
                        "vs_mesh_raster",
                        "fs_lit_raster",
                        &pipeline_layout,
                        &face_layout,
                        false,
                    ),
                    edge: make_color(
                        &shader,
                        "vs_edge_raster",
                        "fs_unlit",
                        &pipeline_layout,
                        &face_layout,
                        true,
                    ),
                    multisampled_mask: make_mask(&shader, "vs_mesh_raster", "fs_mask_unclipped", 4),
                },
            }
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

        self.views.clear();
        self.pipelines = Some(Pipelines {
            format: ctx.color_format,
            msaa: ctx.msaa_samples,
            object_layout,
            dynamic_uniforms,
            lit,
            packed,
            section,
            edge,
            mask,
            multisampled_mask,
            pick,
        });
    }

    fn paint_outline_mask<W>(
        &self,
        pipeline: &wgpu::RenderPipeline,
        pass: &mut wgpu::RenderPass<'_>,
        ctx: &PaintCtx<'_, W>,
    ) {
        let Some(p) = &self.pipelines else {
            return;
        };
        let Some(view) = self.views.iter().find(|view| view.id == ctx.viewport_id) else {
            return;
        };
        pass.set_pipeline(pipeline);
        for (index, object) in view.draws.iter().enumerate() {
            let Some(Some(chunk)) = self.chunks.get(object.chunk) else {
                continue;
            };
            if chunk.selected
                && let Some(vbuf) = &chunk.vertex_buf
                && view.bind(pass, index, p.dynamic_uniforms)
            {
                pass.set_vertex_buffer(0, vbuf.slice());
                pass.draw(0..chunk.vertex_count, 0..1);
            }
        }
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

    fn revision(&self) -> LayerRevision {
        self.revision
    }

    fn prepare(&mut self, ctx: &mut PrepareCtx<'_, W>) -> bool {
        self.prune_closed_viewports();
        self.ensure_pipelines(ctx);
        self.use_antialiased_section =
            ctx.supports_section_antialiasing() && !ctx.camera.clip_planes().is_empty();
        let Some(p) = &self.pipelines else {
            return false;
        };
        let cam_pos = ctx.camera.position_3d();
        let mut pending = false;
        let pack_geometry = p.packed.is_some();
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let pack_geometry = pack_geometry && self.profile_batch_geometry;
        let mut builder = PageBuilder::default();
        let mut uploaded_faces = Vec::new();
        let mut edge_builder = PageBuilder::default();
        let mut uploaded_edges = Vec::new();

        for (chunk_index, entry) in self.chunks.iter_mut().enumerate() {
            let Some(chunk) = entry else { continue };
            let Some(mesh) = &chunk.pending_mesh else {
                continue;
            };
            let pack = pack_geometry && PageBuilder::can_pack(&mesh.vertices, 3);
            let pack_edges = pack_geometry && PageBuilder::can_pack(&mesh.edges, 2);
            {
                let mut bytes = size_of_val(mesh.vertices.as_slice())
                    .saturating_add(size_of_val(mesh.edges.as_slice()));
                if pack {
                    bytes =
                        bytes.saturating_add(mesh.vertices.len().saturating_mul(size_of::<u32>()));
                }
                if pack_edges {
                    bytes = bytes.saturating_add(mesh.edges.len().saturating_mul(size_of::<u32>()));
                }
                if mesh.pick_ids.len() == mesh.vertices.len() {
                    bytes = bytes.saturating_add(size_of_val(mesh.pick_ids.as_slice()));
                }
                if !ctx.uploads.reserve(bytes) {
                    pending = true;
                    continue;
                }
            }
            if let Some(mesh) = chunk.pending_mesh.take() {
                self.revision.mark_static();
                chunk.pick_bounds_trusted = false;
                if mesh.vertices.is_empty() {
                    chunk.vertex_buf = None;
                    chunk.vertex_count = 0;
                    chunk.pick_buf = None;
                    chunk.has_picks = false;
                } else {
                    chunk.vertex_count = u32::try_from(mesh.vertices.len()).unwrap_or(0);
                    if pack {
                        builder.push(chunk_index, &mesh.vertices, ctx.device, &mut uploaded_faces);
                    } else {
                        chunk.vertex_buf = Some(VertexRange::Direct(
                            ctx.device
                                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                    label: Some("iletki-3d vertices"),
                                    contents: bytemuck::cast_slice(&mesh.vertices),
                                    usage: wgpu::BufferUsages::VERTEX,
                                }),
                        ));
                    }
                    if mesh.pick_ids.len() == mesh.vertices.len() {
                        chunk.pick_buf = Some(ctx.device.create_buffer_init(
                            &wgpu::util::BufferInitDescriptor {
                                label: Some("iletki-3d pick ids"),
                                contents: bytemuck::cast_slice(&mesh.pick_ids),
                                usage: wgpu::BufferUsages::VERTEX,
                            },
                        ));
                        chunk.has_picks = true;
                        chunk.pick_bounds_trusted = crate::objects::pick_bounds_cover_vertices(
                            chunk.local_bounds,
                            &mesh.vertices,
                        );
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
                    if pack_edges {
                        edge_builder.push(
                            chunk_index,
                            &mesh.edges,
                            ctx.device,
                            &mut uploaded_edges,
                        );
                    } else {
                        chunk.edge_buf = Some(VertexRange::Direct(ctx.device.create_buffer_init(
                            &wgpu::util::BufferInitDescriptor {
                                label: Some("iletki-3d edges"),
                                contents: bytemuck::cast_slice(&mesh.edges),
                                usage: wgpu::BufferUsages::VERTEX,
                            },
                        )));
                    }
                }
                ctx.uploads.finish_upload();
            }
        }
        builder.flush(ctx.device, &mut uploaded_faces);
        for (index, faces) in uploaded_faces {
            if let Some(Some(chunk)) = self.chunks.get_mut(index) {
                chunk.vertex_buf = Some(faces);
            }
        }

        edge_builder.flush(ctx.device, &mut uploaded_edges);
        for (index, edges) in uploaded_edges {
            if let Some(Some(chunk)) = self.chunks.get_mut(index) {
                chunk.edge_buf = Some(edges);
            }
        }

        let batch = p.dynamic_uniforms;
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let batch = batch && self.profile_batch_uniforms;
        let index = self
            .views
            .iter()
            .position(|view| view.id == ctx.viewport_id)
            .unwrap_or_else(|| {
                let index = self.views.len();
                self.views.push(ViewObjects::new(ctx, batch));
                index
            });
        let Some(view) = self.views.get_mut(index) else {
            return pending;
        };
        view.draws.clear();
        view.has_edges = false;
        view.faces.clear();
        view.edges.clear();
        view.direct_uniforms.clear();

        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let visibility = iletki_wgpu::profiling::scope("scene.visibility");
        let picking = ctx.picking_rect_px();
        let projection = ctx.camera.view_proj_relative(ctx.logical_size);
        let frustum = iletki_core::Frustum::from_view_proj_reverse_z(projection);
        let pick_volume = picking.map(|rect| PickVolume::new(projection, ctx.active_px, rect));
        for (index, chunk) in self.chunks.iter().enumerate() {
            let Some(chunk) = chunk else {
                continue;
            };
            if !chunk.visible || (chunk.vertex_count == 0 && chunk.edge_count == 0) {
                continue;
            }
            let offset_rel = (chunk.origin - cam_pos).to_f32();
            if let Some(volume) = &pick_volume {
                if !chunk.has_picks
                    || (chunk.pick_bounds_trusted
                        && !volume.intersects(chunk.local_bounds, offset_rel))
                {
                    continue;
                }
            } else {
                let bounds = Bounds3::from_corners(
                    chunk.origin + chunk.local_bounds.min,
                    chunk.origin + chunk.local_bounds.max,
                );
                if !bounds.is_empty() && !frustum.intersects_bounds_world(bounds, cam_pos) {
                    continue;
                }
            }
            view.has_edges |= chunk.edge_count != 0;
            view.draws.push(PreparedObject {
                chunk: index,
                direct_slot: None,
                uniforms: ObjectUniforms {
                    offset_rel,
                    scale: 1.0,
                },
            });
        }
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        drop(visibility);

        // Small visible subsets use direct draws. Each larger pass preserves chunk order
        // and stages aligned uniforms only for direct draws or a selected mask.
        let packed_layout = p
            .packed
            .as_ref()
            .filter(|_| pack_geometry && picking.is_none() && view.draws.len() >= 128)
            .map(|packed| &packed.layout);
        {
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            let _profile = iletki_wgpu::profiling::scope("scene.geometry_batch");
            for (index, object) in view.draws.iter_mut().enumerate() {
                let Some(Some(chunk)) = self.chunks.get(object.chunk) else {
                    continue;
                };
                let mut direct = packed_layout.is_none() || chunk.selected;
                if let Some(layout) = packed_layout {
                    for (geometry, draws) in [
                        (&chunk.vertex_buf, &mut view.faces),
                        (&chunk.edge_buf, &mut view.edges),
                    ] {
                        if let Some(geometry) = geometry {
                            direct |= matches!(geometry, VertexRange::Direct(_));
                            draws.push(ctx.device, layout, geometry, object.uniforms, index);
                        }
                    }
                }
                if direct {
                    object.direct_slot = Some(view.direct_uniforms.len());
                    view.direct_uniforms.push(object.uniforms);
                }
            }
            view.faces.upload(ctx.queue);
            view.edges.upload(ctx.queue);
        }
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        if iletki_wgpu::profiling::active() {
            iletki_wgpu::profiling::count("scene.scanned_slots", self.chunks.len());
            iletki_wgpu::profiling::count("scene.candidates", view.draws.len());
            let pass = if picking.is_some() {
                "scene.pick_draws"
            } else {
                "scene.opaque_draws"
            };
            let draws = if packed_layout.is_some() {
                let (faces, face_pages) = view.faces.counts();
                let (edges, edge_pages) = view.edges.counts();
                iletki_wgpu::profiling::count("scene.face_page_writes", face_pages);
                iletki_wgpu::profiling::count("scene.edge_page_writes", edge_pages);
                faces.saturating_add(edges)
            } else {
                view.draws
                    .iter()
                    .filter_map(|object| self.chunks.get(object.chunk)?.as_ref())
                    .map(|chunk| {
                        usize::from(chunk.vertex_buf.is_some())
                            + usize::from(picking.is_none() && chunk.edge_buf.is_some())
                    })
                    .sum()
            };
            iletki_wgpu::profiling::count(pass, draws);
        }
        view.upload(ctx.device, ctx.queue, &p.object_layout);
        pending
    }

    fn supports_multisampled_outline_mask(&self) -> bool {
        self.pipelines
            .as_ref()
            .is_some_and(|p| p.multisampled_mask.is_some())
    }

    fn paint_multisampled_outline_mask(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        ctx: &PaintCtx<'_, W>,
    ) {
        let Some(p) = &self.pipelines else {
            return;
        };
        let mask = p
            .section
            .as_ref()
            .filter(|_| self.use_antialiased_section)
            .map(|section| &section.multisampled_mask)
            .or(p.multisampled_mask.as_ref());
        if let Some(mask) = mask {
            self.paint_outline_mask(mask, pass, ctx);
        }
    }

    fn outline_mask_rect(&self, ctx: &PaintCtx<'_, W>) -> Option<[u32; 4]> {
        let projection = ctx.camera.view_proj_relative(ctx.logical_size);
        let camera_pos = ctx.camera.position_3d();
        let [width, height] = ctx.active_px.map(f64::from);
        let mut min = [width, height];
        let mut max = [0.0_f64; 2];
        let error_rows = [
            [
                projection.c0.x,
                projection.c1.x,
                projection.c2.x,
                projection.c3.x,
            ],
            [
                projection.c0.y,
                projection.c1.y,
                projection.c2.y,
                projection.c3.y,
            ],
            [
                projection.c0.w,
                projection.c1.w,
                projection.c2.w,
                projection.c3.w,
            ],
        ];
        for chunk in self
            .chunks
            .iter()
            .flatten()
            .filter(|chunk| chunk.selected && chunk.visible && chunk.vertex_count != 0)
        {
            let bounds = chunk.local_bounds;
            if bounds.is_empty() {
                return None;
            }
            let [ox, oy, oz] = (chunk.origin - camera_pos).to_f32();
            for x in [bounds.min.x, bounds.max.x] {
                for y in [bounds.min.y, bounds.max.y] {
                    for z in [bounds.min.z, bounds.max.z] {
                        // Use the same rounded local coordinates and origin as the vertex shader.
                        let point = Vec4::new(
                            f64::from(x as f32 + ox),
                            f64::from(y as f32 + oy),
                            f64::from(z as f32 + oz),
                            1.0,
                        );
                        let clip = projection.mul_vec4(point);
                        // Bound f32 uniform and dot-product rounding before the perspective divide.
                        let [x_error, y_error, w_error] = error_rows.map(|[a, b, c, d]| {
                            let magnitude = (a * point.x).abs()
                                + (b * point.y).abs()
                                + (c * point.z).abs()
                                + d.abs();
                            magnitude * (16.0 * f64::from(f32::EPSILON))
                                + f64::from(f32::MIN_POSITIVE)
                        });
                        if ![clip.x, clip.y, clip.z, clip.w]
                            .into_iter()
                            .all(f64::is_finite)
                            || clip.w <= w_error
                            || clip.w >= f64::from(f32::MAX) * 0.5
                            || x_error.max(y_error) > 1.0
                        {
                            return None;
                        }
                        let nx = clip.x / clip.w;
                        let ny = clip.y / clip.w;
                        let dx = (x_error + nx.abs() * w_error) / (clip.w - w_error);
                        let dy = (y_error + ny.abs() * w_error) / (clip.w - w_error);
                        let [left, top] = [
                            (nx - dx + 1.0) * width * 0.5,
                            (1.0 - ny - dy) * height * 0.5,
                        ];
                        let [right, bottom] = [
                            (nx + dx + 1.0) * width * 0.5,
                            (1.0 - ny + dy) * height * 0.5,
                        ];
                        let [l, t] = min;
                        let [r, b] = max;
                        min = [l.min(left), t.min(top)];
                        max = [r.max(right), b.max(bottom)];
                    }
                }
            }
        }
        let [left, top] = min.map(f64::floor);
        let [right, bottom] = max.map(f64::ceil);
        let x = left.clamp(0.0, width) as u32;
        let y = top.clamp(0.0, height) as u32;
        Some([
            x,
            y,
            (right.clamp(0.0, width) as u32).saturating_sub(x),
            (bottom.clamp(0.0, height) as u32).saturating_sub(y),
        ])
    }

    fn paint(&self, phase: DrawPhase, pass: &mut wgpu::RenderPass<'_>, ctx: &PaintCtx<'_, W>) {
        let Some(p) = &self.pipelines else {
            return;
        };
        let Some(view) = self.views.iter().find(|view| view.id == ctx.viewport_id) else {
            return;
        };

        let (lit, edge) = p
            .section
            .as_ref()
            .filter(|_| self.use_antialiased_section)
            .map_or((&p.lit, &p.edge), |section| (&section.lit, &section.edge));

        match phase {
            DrawPhase::Opaque => {
                let packed = p.packed.as_ref().map(|pipelines| {
                    pipelines
                        .section
                        .as_ref()
                        .filter(|_| self.use_antialiased_section)
                        .unwrap_or(&pipelines.base)
                });
                let passes = [
                    (lit, packed.map(|p| &p.faces), &view.faces, false),
                    (edge, packed.map(|p| &p.edges), &view.edges, true),
                ];
                for (direct_pipeline, packed_pipeline, draws, edges) in passes {
                    if edges && !view.has_edges {
                        continue;
                    }
                    let draw_direct = |pass: &mut wgpu::RenderPass<'_>, index: usize| {
                        let Some(object) = view.draws.get(index) else {
                            return;
                        };
                        let Some(Some(chunk)) = self.chunks.get(object.chunk) else {
                            return;
                        };
                        let (geometry, count) = if edges {
                            (&chunk.edge_buf, chunk.edge_count)
                        } else {
                            (&chunk.vertex_buf, chunk.vertex_count)
                        };
                        if let Some(buffer) = geometry
                            && view.bind(pass, index, p.dynamic_uniforms)
                        {
                            pass.set_vertex_buffer(0, buffer.slice());
                            pass.draw(0..count, 0..1);
                        }
                    };
                    if let Some(pipeline) = packed_pipeline
                        && draws.has_packed()
                    {
                        draws.paint(pass, pipeline, |pass, index| {
                            pass.set_pipeline(direct_pipeline);
                            draw_direct(pass, index);
                        });
                    } else {
                        pass.set_pipeline(direct_pipeline);
                        for index in 0..view.draws.len() {
                            draw_direct(pass, index);
                        }
                    }
                }
            }
            DrawPhase::OutlineMask => self.paint_outline_mask(&p.mask, pass, ctx),
            DrawPhase::Picking => {
                pass.set_pipeline(&p.pick);
                for (index, object) in view.draws.iter().enumerate() {
                    if let Some(Some(chunk)) = self.chunks.get(object.chunk)
                        && let (Some(vbuf), Some(pbuf)) = (&chunk.vertex_buf, &chunk.pick_buf)
                        && view.bind(pass, index, p.dynamic_uniforms)
                    {
                        pass.set_vertex_buffer(0, vbuf.slice());
                        pass.set_vertex_buffer(1, pbuf.slice(..));
                        pass.draw(0..chunk.vertex_count, 0..1);
                    }
                }
            }
            _ => {}
        }
    }
}
