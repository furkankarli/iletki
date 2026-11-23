//! Camera-relative Reversed-Z 3D scene layer with automatic 3D frustum culling, 6 section clip
//! planes, *Shaded with Edges* depth-offset wireframe overlays, Jump-Flood selection outline masks,
//! 64-bit sub-object GPU picking and simultaneous multi-viewport sharing.

use crate::face_batch::{self, PageBuilder, VertexRange};
use crate::objects::{
    PickVolume, PreparedObject, SectionGroup, SectionRelation, SectionVolume, ViewObjects,
    bounds_cover_vertices,
};
use crate::outline_bounds::OutlineUpdate;
use crate::section::SectionAa;
use crate::vertex::{Mesh3D, Vertex3D};
use bytemuck::{Pod, Zeroable};
use iletki_core::{Bounds3, DrawPhase, DrawPhaseMask, Vec3};
use iletki_wgpu::wgpu;
use iletki_wgpu::wgpu::util::DeviceExt;
use iletki_wgpu::{
    CAMERA_WGSL, DEPTH_FORMAT, LayerRevision, OUTLINE_MASK_FORMAT, PICK_ID_FORMAT, PaintCtx,
    PrepareCtx, ViewportLayer,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::{cmp::Reverse, collections::BinaryHeap};

static NEXT_CHUNK_ID: AtomicU64 = AtomicU64::new(1);
const SECTION_GROUP_SIZE: usize = 32;

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
    section_bounds_trusted: bool,
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
    pick: Option<wgpu::RenderPipeline>,
    mask: wgpu::RenderPipeline,
    multisampled_mask: Option<wgpu::RenderPipeline>,
    section_mask: Option<wgpu::RenderPipeline>,
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
    free_chunks: BinaryHeap<Reverse<usize>>,
    section_groups: Vec<Option<SectionGroup>>,
    views: Vec<ViewObjects>,
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    profile_batch_uniforms: bool,
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    profile_batch_geometry: bool,
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    profile_batch_picking: bool,
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    profile_section_culling: bool,
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    profile_section_groups: bool,
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
            free_chunks: BinaryHeap::new(),
            section_groups: Vec::new(),
            views: Vec::new(),
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            profile_batch_uniforms: true,
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            profile_batch_geometry: true,
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            profile_batch_picking: true,
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            profile_section_culling: true,
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            profile_section_groups: true,
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

    /// Switches the picking pass between grouped and per-chunk draws for A/B profiling.
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    pub fn set_profile_batch_picking(&mut self, enabled: bool) {
        self.profile_batch_picking = enabled;
    }

    /// Enables conservative CPU rejection in raster sections for A/B profiling.
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    pub fn set_profile_section_culling(&mut self, enabled: bool) {
        if self.profile_section_culling != enabled {
            self.profile_section_culling = enabled;
            self.revision.mark_static();
        }
    }

    /// Enables grouped section and frustum tests for A/B profiling.
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    pub fn set_profile_section_groups(&mut self, enabled: bool) {
        if self.profile_section_groups != enabled {
            self.profile_section_groups = enabled;
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
            section_bounds_trusted: false,
        };
        // Reuse the lowest vacant slot to preserve draw order, including equal-depth picks.
        if let Some(Reverse(index)) = self.free_chunks.pop()
            && let Some(slot @ None) = self.chunks.get_mut(index)
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
            // Motion inside the original envelope does not make its bounds stale.
            if entry.section_bounds_trusted
                && let Some(group) = self.section_groups.get_mut(id.index / SECTION_GROUP_SIZE)
                && group
                    .as_ref()
                    .is_some_and(|group| !group.contains_origin(origin))
            {
                *group = None;
            }
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
            self.free_chunks.push(Reverse(id.index));
            if let Some(group) = self.section_groups.get_mut(id.index / SECTION_GROUP_SIZE) {
                *group = None;
            }
            for view in &mut self.views {
                view.faces.clear();
                view.edges.clear();
                view.mask.clear();
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
        let make_pick = |shader: &wgpu::ShaderModule,
                         vertex,
                         layout: &wgpu::PipelineLayout,
                         buffers: &[Option<wgpu::VertexBufferLayout<'_>>]| {
            ctx.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("iletki-3d pick mesh"),
                    layout: Some(layout),
                    vertex: wgpu::VertexState {
                        module: shader,
                        entry_point: Some(vertex),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers,
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: shader,
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
                })
        };
        let pick = make_pick(
            &shader,
            "vs_mesh_pick",
            &pipeline_layout,
            &[Some(vert_layout.clone()), Some(pick_layout)],
        );
        let make_mask = |shader: &wgpu::ShaderModule,
                         vertex,
                         fragment,
                         count,
                         layout: &wgpu::PipelineLayout,
                         buffers: &[Option<wgpu::VertexBufferLayout<'_>>]| {
            ctx.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("iletki-3d outline mask"),
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
            let slot_layout = wgpu::VertexBufferLayout {
                array_stride: size_of::<u32>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![3 => Uint32],
            };
            let buffers = [Some(vert_layout.clone()), Some(slot_layout.clone())];
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
            let mask = make_mask(
                &shader,
                "vs_mask_batched",
                "fs_mask",
                1,
                &pipeline_layout,
                &buffers,
            );
            let multisampled_mask = (ctx.msaa_samples == 4).then(|| {
                make_mask(
                    &shader,
                    "vs_mask_batched",
                    "fs_mask",
                    4,
                    &pipeline_layout,
                    &buffers,
                )
            });
            let (section, section_mask) = section_aa.map_or((None, None), |aa| {
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
                let color = match aa {
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
                };
                let (vertex, fragment) = match aa {
                    SectionAa::Sampled => ("vs_mesh_batched_sampled", "fs_mask_sampled"),
                    SectionAa::Raster => ("vs_mesh_batched_raster", "fs_mask_unclipped"),
                };
                (
                    Some(color),
                    Some(make_mask(
                        &shader,
                        vertex,
                        fragment,
                        4,
                        &pipeline_layout,
                        &buffers,
                    )),
                )
            });
            let limits = ctx.device.limits();
            let pick =
                (limits.max_vertex_buffers >= 3 && limits.max_vertex_attributes >= 5).then(|| {
                    make_pick(
                        &shader,
                        "vs_pick_batched",
                        &pipeline_layout,
                        &[
                            Some(vert_layout.clone()),
                            Some(slot_layout),
                            Some(wgpu::VertexBufferLayout {
                                array_stride: size_of::<[u32; 2]>() as u64,
                                step_mode: wgpu::VertexStepMode::Vertex,
                                attributes: &wgpu::vertex_attr_array![4 => Uint32x2],
                            }),
                        ],
                    )
                });
            PackedPipelines {
                pick,
                layout,
                base,
                section,
                mask,
                multisampled_mask,
                section_mask,
            }
        });

        let mask = make_mask(
            &shader,
            "vs_mask",
            "fs_mask",
            1,
            &pipeline_layout,
            &face_layout,
        );
        let multisampled_mask = (ctx.msaa_samples == 4).then(|| {
            make_mask(
                &shader,
                "vs_mask",
                "fs_mask",
                4,
                &pipeline_layout,
                &face_layout,
            )
        });

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
                    multisampled_mask: make_mask(
                        &shader,
                        "vs_mesh_sampled",
                        "fs_mask_sampled",
                        4,
                        &pipeline_layout,
                        &face_layout,
                    ),
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
                    multisampled_mask: make_mask(
                        &shader,
                        "vs_mesh_raster",
                        "fs_mask_unclipped",
                        4,
                        &pipeline_layout,
                        &face_layout,
                    ),
                },
            }
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

    fn prepare_packed_mask<W>(
        ctx: &PrepareCtx<'_, W>,
        layout: &wgpu::BindGroupLayout,
        chunks: &[Option<ChunkEntry>],
        view: &mut ViewObjects,
    ) -> bool {
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let _scope = iletki_wgpu::profiling::scope("scene.outline_mask_prepare");
        let selected = view.draws.iter().enumerate().filter_map(|(index, object)| {
            let chunk = chunks.get(object.chunk)?.as_ref()?;
            if !chunk.selected {
                return None;
            }
            Some((index, object.uniforms, chunk.vertex_buf.as_ref()?))
        });
        if selected.clone().take(128).count() < 128 {
            return false;
        }
        view.mask.clear();
        for (index, uniforms, geometry) in selected {
            view.mask.push(geometry, uniforms, index);
        }
        view.mask.upload(ctx.device, layout, ctx.queue);
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        iletki_wgpu::profiling::count("scene.outline_mask_draws", view.mask.counts().0);
        true
    }

    fn paint_outline_mask<W>(
        &self,
        pipeline: &wgpu::RenderPipeline,
        packed: Option<&wgpu::RenderPipeline>,
        pass: &mut wgpu::RenderPass<'_>,
        ctx: &PaintCtx<'_, W>,
    ) {
        let Some(p) = &self.pipelines else {
            return;
        };
        let Some(view) = self
            .views
            .iter()
            .find(|view| view.id == ctx.viewport_id && !view.picking)
        else {
            return;
        };
        let draw_direct = |pass: &mut wgpu::RenderPass<'_>, index: usize| {
            if let Some(object) = view.draws.get(index)
                && let Some(Some(chunk)) = self.chunks.get(object.chunk)
                && chunk.selected
                && let Some(vertices) = &chunk.vertex_buf
                && view.bind(pass, index, p.dynamic_uniforms)
            {
                pass.set_vertex_buffer(0, vertices.slice());
                pass.draw(0..chunk.vertex_count, 0..1);
            }
        };
        if let Some(packed) = packed
            && view.mask.has_packed()
        {
            view.mask.paint(pass, packed, false, |pass, index| {
                pass.set_pipeline(pipeline);
                draw_direct(pass, index);
            });
        } else {
            pass.set_pipeline(pipeline);
            for index in 0..view.draws.len() {
                draw_direct(pass, index);
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
        let section = SectionAa::for_viewport(ctx).filter(|_| !ctx.camera.clip_planes().is_empty());
        self.use_antialiased_section = section.is_some();
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
                if pack || mesh.pick_ids.len() == mesh.vertices.len() {
                    // A shared face page can contain both pickable and unpickable chunks.
                    bytes = bytes
                        .saturating_add(mesh.vertices.len().saturating_mul(size_of::<[u32; 2]>()));
                }
                if !ctx.uploads.reserve(bytes) {
                    pending = true;
                    continue;
                }
            }
            if let Some(mesh) = chunk.pending_mesh.take() {
                self.revision.mark_static();
                if let Some(group) = self
                    .section_groups
                    .get_mut(chunk_index / SECTION_GROUP_SIZE)
                {
                    *group = None;
                }
                let face_bounds = bounds_cover_vertices(chunk.local_bounds, &mesh.vertices);
                chunk.pick_bounds_trusted = face_bounds;
                chunk.section_bounds_trusted =
                    face_bounds && bounds_cover_vertices(chunk.local_bounds, &mesh.edges);
                if mesh.vertices.is_empty() {
                    chunk.vertex_buf = None;
                    chunk.vertex_count = 0;
                    chunk.pick_buf = None;
                    chunk.has_picks = false;
                } else {
                    chunk.vertex_count = u32::try_from(mesh.vertices.len()).unwrap_or(0);
                    let picks = (mesh.pick_ids.len() == mesh.vertices.len())
                        .then_some(mesh.pick_ids.as_slice());
                    if pack {
                        builder.push(
                            chunk_index,
                            &mesh.vertices,
                            picks,
                            ctx.device,
                            &mut uploaded_faces,
                        );
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
                    chunk.pick_buf = picks.filter(|_| !pack).map(|ids| {
                        ctx.device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("iletki-3d pick ids"),
                                contents: bytemuck::cast_slice(ids),
                                usage: wgpu::BufferUsages::VERTEX,
                            })
                    });
                    chunk.has_picks = picks.is_some();
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
                            None,
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
        let picking = ctx.picking_rect_px();
        let index = self
            .views
            .iter()
            .position(|view| view.id == ctx.viewport_id && view.picking == picking.is_some())
            .unwrap_or_else(|| {
                let index = self.views.len();
                self.views
                    .push(ViewObjects::new(ctx, batch, picking.is_some()));
                index
            });
        let Some(view) = self.views.get_mut(index) else {
            return pending;
        };
        if picking.is_none() {
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            let _scope = iletki_wgpu::profiling::scope("scene.outline_bounds_prepare");
            let update = view.outline_bounds.update(
                ctx,
                self.revision,
                self.chunks.len(),
                self.chunks.iter().enumerate().filter_map(|(index, chunk)| {
                    let chunk = chunk.as_ref()?;
                    (chunk.selected && chunk.visible && chunk.vertex_count != 0).then_some((
                        index,
                        chunk.local_bounds,
                        chunk.origin,
                    ))
                }),
            );
            match update {
                OutlineUpdate::Unchanged => return pending,
                OutlineUpdate::Selection if pack_geometry && view.faces.has_packed() => {
                    if let Some(packed) = &p.packed
                        && Self::prepare_packed_mask(ctx, &packed.layout, &self.chunks, view)
                    {
                        // A dense mask uses the cached face ranges and its own uniform pages.
                        // Direct face draws already have their own transform slots.
                        return pending;
                    }
                }
                OutlineUpdate::Selection | OutlineUpdate::View => {}
            }
        }

        view.draws.clear();
        view.has_edges = false;
        view.faces.clear();
        view.edges.clear();
        view.mask.clear();
        view.direct_uniforms.clear();

        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let visibility = iletki_wgpu::profiling::scope("scene.visibility");
        let projection = ctx.camera.view_proj_relative(ctx.logical_size);
        let frustum = iletki_core::Frustum::from_view_proj_reverse_z(projection);
        let pick_volume = picking.map(|rect| PickVolume::new(projection, ctx.active_px, rect));
        let cull_sections = picking.is_none() && section == Some(SectionAa::Raster);
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let cull_sections = cull_sections && self.profile_section_culling;
        let section_volume = cull_sections.then(|| SectionVolume::new(ctx.camera));
        let group_sections = self.chunks.len() >= SECTION_GROUP_SIZE * 4;
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let group_sections = group_sections && self.profile_section_groups;
        let group_volume = section_volume.as_ref().filter(|_| group_sections);
        if group_volume.is_some() {
            self.section_groups
                .resize_with(self.chunks.len().div_ceil(SECTION_GROUP_SIZE), || None);
        }
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let profiling = iletki_wgpu::profiling::active();
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let (mut culled_chunks, mut culled_triangles, mut section_tests, mut frustum_tests) =
            (0, 0, 0, 0);
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let mut opaque_triangles = 0;
        for (group_index, chunks) in self.chunks.chunks(SECTION_GROUP_SIZE).enumerate() {
            let (relation, inside_frustum) = group_volume
                .and_then(|volume| {
                    let group = self
                        .section_groups
                        .get_mut(group_index)?
                        .get_or_insert_with(|| {
                            let mut bounds = SectionGroup::default();
                            for chunk in
                                chunks.iter().flatten().filter(|c| c.section_bounds_trusted)
                            {
                                bounds.include(chunk.origin, chunk.local_bounds);
                            }
                            bounds
                        });
                    Some((
                        group.section_relation(volume, cam_pos),
                        group.inside_frustum(&frustum, cam_pos),
                    ))
                })
                .unwrap_or((SectionRelation::Crossing, false));
            for (slot, chunk) in chunks.iter().enumerate() {
                let Some(chunk) = chunk else { continue };
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
                } else if !inside_frustum || !chunk.section_bounds_trusted {
                    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
                    if profiling {
                        frustum_tests += 1;
                    }
                    let bounds = Bounds3::from_corners(
                        chunk.origin + chunk.local_bounds.min,
                        chunk.origin + chunk.local_bounds.max,
                    );
                    if !bounds.is_empty() && !frustum.intersects_bounds_world(bounds, cam_pos) {
                        continue;
                    }
                }
                let section_culled = chunk.section_bounds_trusted
                    && match relation {
                        SectionRelation::Outside => true,
                        SectionRelation::Inside => false,
                        SectionRelation::Crossing => {
                            section_volume.as_ref().is_some_and(|volume| {
                                #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
                                if profiling {
                                    section_tests += 1;
                                }
                                volume.excludes(chunk.local_bounds, offset_rel)
                            })
                        }
                    };
                #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
                if profiling && picking.is_none() {
                    let triangles = chunk.vertex_count as usize / 3;
                    if section_culled {
                        culled_chunks += 1;
                        culled_triangles += triangles;
                    } else {
                        opaque_triangles += triangles;
                    }
                }
                view.has_edges |= chunk.edge_count != 0 && !section_culled;
                view.draws.push(PreparedObject {
                    chunk: group_index * SECTION_GROUP_SIZE + slot,
                    section_culled,
                    direct_slot: None,
                    uniforms: ObjectUniforms {
                        offset_rel,
                        scale: 1.0,
                    },
                });
            }
        }
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        drop(visibility);

        // Small visible subsets use direct draws. Each larger pass preserves chunk order
        // and stages aligned uniforms only for direct draws or a selected mask.
        let use_packed = pack_geometry && view.draws.len() >= 128;
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let use_packed = use_packed && (picking.is_none() || self.profile_batch_picking);
        let packed_layout = p
            .packed
            .as_ref()
            .filter(|packed| use_packed && (picking.is_none() || packed.pick.is_some()))
            .map(|packed| &packed.layout);
        let packed_mask = picking.is_none()
            && packed_layout
                .is_some_and(|layout| Self::prepare_packed_mask(ctx, layout, &self.chunks, view));
        {
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            let _profile = iletki_wgpu::profiling::scope("scene.geometry_batch");
            for (index, object) in view.draws.iter_mut().enumerate() {
                let Some(Some(chunk)) = self.chunks.get(object.chunk) else {
                    continue;
                };
                let mut direct = packed_layout.is_none()
                    || (picking.is_none() && chunk.selected && !packed_mask);
                if packed_layout.is_some() {
                    if let Some(geometry) = &chunk.vertex_buf {
                        // A direct face can still be selected even when its opaque draw is culled.
                        direct |= matches!(geometry, VertexRange::Direct(_));
                        if !object.section_culled {
                            view.faces.push(geometry, object.uniforms, index);
                        }
                    }
                    if picking.is_none()
                        && !object.section_culled
                        && let Some(geometry) = &chunk.edge_buf
                    {
                        direct |= matches!(geometry, VertexRange::Direct(_));
                        view.edges.push(geometry, object.uniforms, index);
                    }
                }
                if direct {
                    object.direct_slot = Some(view.direct_uniforms.len());
                    view.direct_uniforms.push(object.uniforms);
                }
            }
            if let Some(layout) = packed_layout {
                view.faces.upload(ctx.device, layout, ctx.queue);
                view.edges.upload(ctx.device, layout, ctx.queue);
            }
        }
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        if profiling {
            iletki_wgpu::profiling::count("scene.section_chunk_tests", section_tests);
            iletki_wgpu::profiling::count("scene.frustum_chunk_tests", frustum_tests);
            iletki_wgpu::profiling::count("scene.opaque_triangles", opaque_triangles);
            iletki_wgpu::profiling::count("scene.scanned_slots", self.chunks.len());
            iletki_wgpu::profiling::count("scene.candidates", view.draws.len());
            iletki_wgpu::profiling::count("scene.section_culled_chunks", culled_chunks);
            iletki_wgpu::profiling::count("scene.section_culled_triangles", culled_triangles);
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
                    .filter(|object| !object.section_culled)
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
            let packed = p.packed.as_ref().and_then(|packed| {
                packed
                    .section_mask
                    .as_ref()
                    .filter(|_| self.use_antialiased_section)
                    .or(packed.multisampled_mask.as_ref())
            });
            self.paint_outline_mask(mask, packed, pass, ctx);
        }
    }

    fn outline_mask_rect(&self, ctx: &PaintCtx<'_, W>) -> Option<[u32; 4]> {
        self.views
            .iter()
            .find(|view| view.id == ctx.viewport_id && !view.picking)?
            .outline_bounds
            .rect()
    }

    fn paint(&self, phase: DrawPhase, pass: &mut wgpu::RenderPass<'_>, ctx: &PaintCtx<'_, W>) {
        let Some(p) = &self.pipelines else {
            return;
        };
        let Some(view) = self.views.iter().find(|view| {
            view.id == ctx.viewport_id && view.picking == (phase == DrawPhase::Picking)
        }) else {
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
                        if object.section_culled {
                            return;
                        }
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
                        draws.paint(pass, pipeline, false, |pass, index| {
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
            DrawPhase::OutlineMask => {
                self.paint_outline_mask(
                    &p.mask,
                    p.packed.as_ref().map(|packed| &packed.mask),
                    pass,
                    ctx,
                );
            }
            DrawPhase::Picking => {
                let draw_direct = |pass: &mut wgpu::RenderPass<'_>, index: usize| {
                    if let Some(object) = view.draws.get(index)
                        && let Some(Some(chunk)) = self.chunks.get(object.chunk)
                        && let Some(vertices) = &chunk.vertex_buf
                        && let Some(ids) = chunk
                            .pick_buf
                            .as_ref()
                            .map(|buf| buf.slice(..))
                            .or_else(|| vertices.pick_slice())
                        && view.bind(pass, index, p.dynamic_uniforms)
                    {
                        pass.set_vertex_buffer(0, vertices.slice());
                        pass.set_vertex_buffer(1, ids);
                        pass.draw(0..chunk.vertex_count, 0..1);
                    }
                };
                if let Some(pipeline) = p.packed.as_ref().and_then(|packed| packed.pick.as_ref())
                    && view.faces.has_packed()
                {
                    view.faces.paint(pass, pipeline, true, |pass, index| {
                        pass.set_pipeline(&p.pick);
                        draw_direct(pass, index);
                    });
                } else {
                    pass.set_pipeline(&p.pick);
                    for index in 0..view.draws.len() {
                        draw_direct(pass, index);
                    }
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insertion_reuses_the_lowest_vacant_slot_and_rejects_stale_handles() {
        let mut scene = Scene3DLayer::new(1);
        let insert = |scene: &mut Scene3DLayer| scene.insert(Mesh3D::at_origin(Vec3::ZERO));
        let first = insert(&mut scene);
        let second = insert(&mut scene);
        let third = insert(&mut scene);
        scene.remove(third);
        scene.remove(first);
        scene.remove(first);

        let replacement_first = insert(&mut scene);
        scene.remove(first);
        assert_eq!(replacement_first.index, first.index);
        assert!(scene.chunk_mut(replacement_first).is_some());
        assert!(scene.chunk_mut(first).is_none());
        assert_eq!(insert(&mut scene).index, third.index);
        assert_eq!(insert(&mut scene).index, 3);
        assert!(scene.chunk_mut(second).is_some());
    }
}
