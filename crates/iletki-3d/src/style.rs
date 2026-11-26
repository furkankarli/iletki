//! CAD visual styles (`Shaded`, `ShadedWithEdges`, `Technical`, `Wireframe`, `XRay`) and
//! engineering diagnostic views (`Normals`, `Facets`, `Backfaces`, `Overdraw`, `Chunks`, `Depth`).
//!
//! Strictly opt-in: while [`VisualStyle::ShadedWithEdges`] or [`VisualStyle::Shaded`] is active
//! with [`DiagnosticView::None`], zero style shaders, pipelines or uniform buffers are allocated.

use crate::face_batch::VertexLayout;
use crate::gtao;
use crate::provoking;
use crate::section::SectionAa;
use bytemuck::{Pod, Zeroable};
use iletki_core::{Camera, Projection, Vec2};
use iletki_wgpu::{CAMERA_WGSL, PassHookCtx, PrepareCtx, wgpu};
use std::num::NonZeroU64;

/// High-level CAD/BIM visual presentation style for [`crate::Scene3DLayer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum VisualStyle {
    /// Lit CAD shading with uploaded mesh edges overlaid via Reversed-Z depth offset (default).
    #[default]
    ShadedWithEdges,
    /// Lit CAD shading without mesh edge overlays.
    Shaded,
    /// Technical hidden-line illustration: matte paper depth-occluding faces with dark CAD ink
    /// edges.
    Technical,
    /// Pure wireframe view drawing edges without opaque face fills (falls back to polygon line
    /// rasterization for edge-less meshes when the GPU supports `POLYGON_MODE_LINE`).
    Wireframe,
    /// See-through translucent X-Ray view routing all solid faces through MSAA-compatible
    /// Weighted Blended OIT with Fresnel silhouette emphasis.
    XRay,
}

/// Engineering diagnostic inspection view for [`crate::Scene3DLayer`].
///
/// When non-[`Self::None`], temporarily overrides [`VisualStyle`] shading to inspect geometry
/// normals, facets, triangle winding, fragment overdraw, chunk boundaries or Reversed-Z depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DiagnosticView {
    /// Standard rendering using the active [`VisualStyle`].
    #[default]
    None,
    /// World-space shading normals mapped from `[-1, 1]` to `[0, 1]` RGB.
    Normals,
    /// Flat geometric triangle normals computed via screen-space derivatives (`dpdx`/`dpdy`) to
    /// inspect raw tessellation facets.
    Facets,
    /// Highlights triangle orientation: cool blue for front-facing (`CCW`) triangles and bright
    /// red-orange for back-facing (`CW`) triangles.
    Backfaces,
    /// Additive pixel overdraw heatmap (`depth_compare = Always`, `depth_write = false`) where
    /// overlapping fragments accumulate from dark crimson to bright amber/yellow.
    Overdraw,
    /// Deterministic high-contrast per-chunk palette hashed from each chunk's slot ID.
    Chunks,
    /// Reversed-Z depth visualization (bright near camera → dark far).
    Depth,
}

impl DiagnosticView {
    #[inline]
    pub(crate) const fn is_active(self) -> bool {
        !matches!(self, Self::None)
    }

    #[inline]
    pub(crate) const fn mode_id(self) -> Option<u32> {
        match self {
            Self::None => None,
            Self::Normals => Some(1),
            Self::Facets => Some(2),
            Self::Backfaces => Some(3),
            Self::Overdraw => Some(4),
            Self::Chunks => Some(5),
            Self::Depth => Some(6),
        }
    }
}

/// GPU uniform controlling style/diagnostic mode and tint (`@group(2) @binding(0)`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, Default, PartialEq)]
pub(crate) struct StyleUniform {
    pub(crate) meta: [u32; 4],
    pub(crate) tint: [f32; 4],
}

impl StyleUniform {
    pub(crate) const TECHNICAL_PAPER: [f32; 4] = [0.94, 0.95, 0.97, 1.0];
    pub(crate) const TECHNICAL_INK: [f32; 4] = [0.10, 0.12, 0.16, 1.0];
    pub(crate) const WIREFRAME_INK: [f32; 4] = [0.78, 0.88, 0.98, 1.0];

    pub(crate) fn for_faces(
        visual_style: VisualStyle,
        diagnostic: DiagnosticView,
        has_clip_planes: bool,
    ) -> Self {
        let mode = diagnostic.mode_id().unwrap_or(match visual_style {
            VisualStyle::Technical => 0,
            _ => 0,
        });
        let tint = match visual_style {
            VisualStyle::Wireframe => Self::WIREFRAME_INK,
            _ => Self::TECHNICAL_PAPER,
        };
        Self {
            meta: [mode, u32::from(has_clip_planes), 0, 0],
            tint,
        }
    }

    pub(crate) fn for_edges(visual_style: VisualStyle, has_clip_planes: bool) -> Self {
        let tint = match visual_style {
            VisualStyle::Technical => Self::TECHNICAL_INK,
            _ => Self::WIREFRAME_INK,
        };
        Self {
            meta: [0, u32::from(has_clip_planes), 0, 0],
            tint,
        }
    }
}

/// GPU uniform for the fullscreen screen-space `Technical` silhouette & crease edge pass (64 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, Default, PartialEq)]
pub(crate) struct TechEdgeUniform {
    pub(crate) proj_xy: [f32; 2],
    pub(crate) depth_params: [f32; 2],
    pub(crate) active_px: [u32; 2],
    pub(crate) flags: u32,
    pub(crate) msaa_samples: u32,
    pub(crate) ink_color: [f32; 4],
    pub(crate) paper_color: [f32; 4],
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

impl TechEdgeUniform {
    const FLAG_CREASES: u32 = 4;

    /// `creases` enables screen-space crease detection; pass `false` when every chunk with faces
    /// has authored edges, which then own all creases.
    pub(crate) fn from_camera(
        camera: &Camera,
        logical_size: Vec2,
        active_px: [u32; 2],
        msaa_samples: u32,
        color_format: wgpu::TextureFormat,
        creases: bool,
    ) -> Option<Self> {
        let [aw, ah] = active_px;
        if aw < 2
            || ah < 2
            || !logical_size.is_finite()
            || logical_size.x <= 0.0
            || logical_size.y <= 0.0
        {
            return None;
        }
        let aspect = logical_size.x / logical_size.y;
        if !aspect.is_finite() || aspect <= 1e-6 {
            return None;
        }
        let (proj_xy, depth_params, flags) = match camera.projection {
            Projection::Perspective3D { fov_y, near, .. } => {
                let fov = fov_y.clamp(1e-4, std::f64::consts::PI - 1e-4);
                let tan_y = (0.5 * fov).tan();
                let tan_x = tan_y * aspect;
                let near_clamped = if near.is_finite() && near > 1e-9 {
                    near
                } else {
                    0.01
                };
                (
                    [tan_x as f32, tan_y as f32],
                    [near_clamped as f32, 0.0],
                    1u32,
                )
            }
            Projection::Ortho3D {
                view_height,
                near,
                far,
                ..
            } => {
                let half_h = (0.5 * view_height).max(1e-6);
                let half_w = half_h * aspect;
                let span = (far - near).max(1e-6);
                (
                    [half_w as f32, half_h as f32],
                    [far as f32, span as f32],
                    2u32,
                )
            }
            Projection::Ortho2D { .. } => return None,
        };
        if !proj_xy[0].is_finite()
            || !proj_xy[1].is_finite()
            || !depth_params[0].is_finite()
            || !depth_params[1].is_finite()
        {
            return None;
        }
        let to_target = |c: [f32; 4]| {
            if color_format.is_srgb() {
                [
                    srgb_to_linear(c[0]),
                    srgb_to_linear(c[1]),
                    srgb_to_linear(c[2]),
                    c[3],
                ]
            } else {
                c
            }
        };
        let flags = if creases {
            flags | Self::FLAG_CREASES
        } else {
            flags
        };
        Some(Self {
            proj_xy,
            depth_params,
            active_px,
            flags,
            msaa_samples: msaa_samples.max(1),
            ink_color: to_target(StyleUniform::TECHNICAL_INK),
            paper_color: to_target(StyleUniform::TECHNICAL_PAPER),
        })
    }
}

/// Lazily compiled fullscreen pipeline for automatic screen-space `Technical` edges.
pub(crate) struct TechnicalEdgePipeline {
    pub(crate) layout: wgpu::BindGroupLayout,
    pub(crate) pipeline: wgpu::RenderPipeline,
}

impl TechnicalEdgePipeline {
    fn build<W>(ctx: &PrepareCtx<'_, W>) -> Self {
        let device = ctx.device;
        let multisampled = ctx.msaa_samples > 1;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("iletki-3d technical edge layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(size_of::<TechEdgeUniform>() as u64),
                    },
                    count: None,
                },
            ],
        });
        let (depth_decl, depth_sample_load, resolve_edge_alpha) = if ctx.msaa_samples == 4 {
            (
                "var src_depth: texture_depth_multisampled_2d;",
                "return textureLoad(src_depth, clamped, sample_idx);",
                "let z0 = load_depth_sample(c, max_c, 0);\n\
                 let z1 = load_depth_sample(c, max_c, 1);\n\
                 let z2 = load_depth_sample(c, max_c, 2);\n\
                 let z3 = load_depth_sample(c, max_c, 3);\n\
                 if (max(max(z0, z1), max(z2, z3)) <= 0.0) {\n\
                     discard;\n\
                 }\n\
                 let edge_alpha = 0.25 * (\n\
                     eval_sample_edge(c, max_c, 0)\n\
                     + eval_sample_edge(c, max_c, 1)\n\
                     + eval_sample_edge(c, max_c, 2)\n\
                     + eval_sample_edge(c, max_c, 3)\n\
                 );",
            )
        } else if ctx.msaa_samples > 1 {
            (
                "var src_depth: texture_depth_multisampled_2d;",
                "return textureLoad(src_depth, clamped, sample_idx);",
                "let samples = max(i32(tech_u.msaa_samples), 1);\n\
                 var z_max = 0.0;\n\
                 for (var s: i32 = 0; s < samples; s = s + 1) {\n\
                     z_max = max(z_max, load_depth_sample(c, max_c, s));\n\
                 }\n\
                 if (z_max <= 0.0) {\n\
                     discard;\n\
                 }\n\
                 var sum_e = 0.0;\n\
                 for (var s: i32 = 0; s < samples; s = s + 1) {\n\
                     sum_e = sum_e + eval_sample_edge(c, max_c, s);\n\
                 }\n\
                 let edge_alpha = sum_e / f32(samples);",
            )
        } else {
            (
                "var src_depth: texture_depth_2d;",
                "return textureLoad(src_depth, clamped, 0);",
                "let edge_alpha = eval_sample_edge(c, max_c, 0);",
            )
        };
        let src = include_str!("shaders/technical_edges.wgsl")
            .replace("__DEPTH_DECL__", depth_decl)
            .replace("__DEPTH_SAMPLE_LOAD__", depth_sample_load)
            .replace("__RESOLVE_EDGE_ALPHA__", resolve_edge_alpha);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("iletki-3d technical edge shader"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("iletki-3d technical edge pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        // Ink never accumulates: where this overlay meets an authored edge line, the darker wins.
        let blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Min,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("iletki-3d technical edge overlay"),
            layout: Some(&pl_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_tech_edge"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_tech_edge"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: ctx.color_format,
                    blend: Some(blend),
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
                count: 1,
                ..wgpu::MultisampleState::default()
            },
            multiview_mask: None,
            cache: None,
        });
        Self { layout, pipeline }
    }
}

/// Per-viewport uniform buffers for face style, edge style and optional screen-space Technical
/// silhouette/crease overlays (`@group(2)` and `PostSceneResolved`).
pub(crate) struct ViewStyleUniform {
    face_buf: wgpu::Buffer,
    face_bg: wgpu::BindGroup,
    edge_buf: wgpu::Buffer,
    edge_bg: wgpu::BindGroup,
    last_face: Option<StyleUniform>,
    last_edge: Option<StyleUniform>,
    tech_edge_buf: Option<wgpu::Buffer>,
    last_tech_edge: Option<TechEdgeUniform>,
    tech_edge_active: bool,
    tech_edge_bg: Option<((u64, u32), wgpu::BindGroup)>,
}

impl ViewStyleUniform {
    pub(crate) fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> Self {
        let make = |label: &str| {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size_of::<StyleUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            });
            (buf, bg)
        };
        let (face_buf, face_bg) = make("iletki-3d style face uniform");
        let (edge_buf, edge_bg) = make("iletki-3d style edge uniform");
        Self {
            face_buf,
            face_bg,
            edge_buf,
            edge_bg,
            last_face: None,
            last_edge: None,
            tech_edge_buf: None,
            last_tech_edge: None,
            tech_edge_active: false,
            tech_edge_bg: None,
        }
    }

    pub(crate) fn update(&mut self, queue: &wgpu::Queue, face: StyleUniform, edge: StyleUniform) {
        if self.last_face != Some(face) {
            queue.write_buffer(&self.face_buf, 0, bytemuck::bytes_of(&face));
            self.last_face = Some(face);
        }
        if self.last_edge != Some(edge) {
            queue.write_buffer(&self.edge_buf, 0, bytemuck::bytes_of(&edge));
            self.last_edge = Some(edge);
        }
    }

    pub(crate) fn update_tech_edges(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        uniform: Option<TechEdgeUniform>,
    ) {
        let Some(u) = uniform else {
            self.tech_edge_active = false;
            return;
        };
        let buf = self.tech_edge_buf.get_or_insert_with(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("iletki-3d technical edge uniform"),
                size: size_of::<TechEdgeUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        if self.last_tech_edge != Some(u) {
            queue.write_buffer(buf, 0, bytemuck::bytes_of(&u));
            self.last_tech_edge = Some(u);
        }
        self.tech_edge_active = true;
    }

    #[inline]
    pub(crate) fn tech_edges_active(&self) -> bool {
        self.tech_edge_active && self.tech_edge_buf.is_some()
    }

    pub(crate) fn encode_tech_edges<W>(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &PassHookCtx<'_, W>,
        tech_pl: &TechnicalEdgePipeline,
        depth_sample_view: &wgpu::TextureView,
        resolved_color_view: &wgpu::TextureView,
    ) {
        let Some(buf) = self.tech_edge_buf.as_ref() else {
            return;
        };
        let key = (ctx.target_generation, ctx.msaa_samples);
        if self.tech_edge_bg.as_ref().is_none_or(|(k, _)| *k != key) {
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("iletki-3d technical edge bg"),
                layout: &tech_pl.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(depth_sample_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buf.as_entire_binding(),
                    },
                ],
            });
            self.tech_edge_bg = Some((key, bg));
        }
        let Some((_, bg)) = self.tech_edge_bg.as_ref() else {
            return;
        };
        let [aw, ah] = ctx.active_px;
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let timer = iletki_wgpu::profiling::pass("iletki-3d technical crease edges");
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let timestamp_writes = timer.as_ref().and_then(|t| t.render_writes());
        #[cfg(not(all(feature = "profiling", not(target_arch = "wasm32"))))]
        let timestamp_writes = None;
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("iletki-3d technical crease edges"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: resolved_color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_viewport(0.0, 0.0, aw as f32, ah as f32, 0.0, 1.0);
        pass.set_scissor_rect(0, 0, aw, ah);
        pass.set_pipeline(&tech_pl.pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.draw(0..3, 0..1);
    }

    pub(crate) fn face_bind_group(&self) -> &wgpu::BindGroup {
        &self.face_bg
    }

    pub(crate) fn edge_bind_group(&self) -> &wgpu::BindGroup {
        &self.edge_bg
    }
}

pub(crate) struct StylePipelineSet {
    pub(crate) opaque: wgpu::RenderPipeline,
    pub(crate) opaque_flat: Option<wgpu::RenderPipeline>,
    pub(crate) overdraw: wgpu::RenderPipeline,
    pub(crate) edge: wgpu::RenderPipeline,
    pub(crate) wireframe_line: Option<wgpu::RenderPipeline>,
    /// The same pipelines with 4x MSAA antialiased section clipping ([`SectionAa`]).
    section: Option<Box<StylePipelineSet>>,
}

impl StylePipelineSet {
    /// Returns the antialiased section variant while clip planes are active, when available.
    #[inline]
    pub(crate) fn variant(&self, antialiased_section: bool) -> &Self {
        match self.section.as_deref() {
            Some(section) if antialiased_section => section,
            _ => self,
        }
    }
}

struct StyleModules {
    shader: wgpu::ShaderModule,
    flat: Option<wgpu::ShaderModule>,
    section: Option<wgpu::ShaderModule>,
    flat_section: Option<wgpu::ShaderModule>,
}

/// WGSL vertex entry points `vs_style_{kind}[_section]` and `vs_style_edge_{kind}[_section]`
/// wrapping the `style_{kind}` and `style_edge_{kind}` helpers.
fn style_entry_points(kind: &str, params: &str, args: &str, section: bool) -> String {
    let (suffix, ret, open, close) = if section {
        ("_section", "StyleSectionOut", "style_section(", ")")
    } else {
        ("", "StyleOut", "", "")
    };
    format!(
        "@vertex\n\
         fn vs_style_{kind}{suffix}({params}) -> {ret} {{\n\
             return {open}style_{kind}({args}){close};\n\
         }}\n\
         @vertex\n\
         fn vs_style_edge_{kind}{suffix}({params}) -> {ret} {{\n\
             return {open}style_edge_{kind}({args}){close};\n\
         }}\n"
    )
}

/// Lazily compiled pipelines for `Technical`, `Wireframe`, and all [`DiagnosticView`] modes.
pub(crate) struct StylePipelines {
    color_format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
    msaa_samples: u32,
    has_storage: bool,
    has_packed: bool,
    has_section: bool,
    pub(crate) style_layout: wgpu::BindGroupLayout,
    pub(crate) direct: StylePipelineSet,
    pub(crate) packed: Option<StylePipelineSet>,
    pub(crate) storage: Option<StylePipelineSet>,
    pub(crate) storage_repeated: Option<StylePipelineSet>,
    pub(crate) tech_edges: Option<TechnicalEdgePipeline>,
}

impl StylePipelines {
    pub(crate) fn matches<W>(
        &self,
        ctx: &PrepareCtx<'_, W>,
        has_storage: bool,
        has_packed: bool,
    ) -> bool {
        self.color_format == ctx.color_format
            && self.depth_format == ctx.depth_format
            && self.msaa_samples == ctx.msaa_samples
            && self.has_storage == has_storage
            && self.has_packed == has_packed
            && (self.has_section || SectionAa::for_viewport(ctx).is_none())
    }

    pub(crate) fn build<W>(
        ctx: &PrepareCtx<'_, W>,
        vertex_layout: VertexLayout,
        object_layout: &wgpu::BindGroupLayout,
        packed_layout: Option<&wgpu::BindGroupLayout>,
        storage_layout: Option<&wgpu::BindGroupLayout>,
        has_flat_support: bool,
        use_either_flat: bool,
    ) -> Self {
        let style_layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("iletki-3d style uniform layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(size_of::<StyleUniform>() as u64),
                    },
                    count: None,
                }],
            });

        let has_polygon_line = ctx
            .device
            .features()
            .contains(wgpu::Features::POLYGON_MODE_LINE);
        let raw_base = format!(
            "{CAMERA_WGSL}\n{}\n{}",
            include_str!("shaders/mesh3d.wgsl"),
            include_str!("shaders/style3d.wgsl")
        );
        let base_wgsl = provoking::adapt_flat_sampling(&raw_base, use_either_flat);
        let flat_base_wgsl =
            has_flat_support.then(|| make_flat_style_wgsl(&raw_base, use_either_flat));
        // Antialiased section variants, used only while clip planes are active.
        let section_aa = SectionAa::for_viewport(ctx);
        let raw_section_base = section_aa.map(|aa| {
            aa.shader_source(
                &raw_base,
                include_str!("shaders/style3d_sampled.wgsl"),
                include_str!("shaders/style3d_raster.wgsl"),
            )
        });
        let section_base_wgsl = raw_section_base
            .as_deref()
            .map(|raw| provoking::adapt_flat_sampling(raw, use_either_flat));
        let flat_section_base_wgsl = raw_section_base
            .as_deref()
            .filter(|_| has_flat_support)
            .map(|raw| make_flat_style_wgsl(raw, use_either_flat));
        let create_module = |label: String, src: String| {
            ctx.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(&label),
                    source: wgpu::ShaderSource::Wgsl(src.into()),
                })
        };
        let make_modules = |kind: &str, helpers: &str, params: &str, args: &str| {
            let plain = format!(
                "{helpers}\n{}",
                style_entry_points(kind, params, args, false)
            );
            let section = format!(
                "{helpers}\n{}",
                style_entry_points(kind, params, args, true)
            );
            StyleModules {
                shader: create_module(
                    format!("iletki-3d {kind} style shader"),
                    format!("{base_wgsl}\n{plain}"),
                ),
                flat: flat_base_wgsl.as_ref().map(|base| {
                    create_module(
                        format!("iletki-3d {kind} flat style shader"),
                        format!("{base}\n{plain}"),
                    )
                }),
                section: section_base_wgsl.as_ref().map(|base| {
                    create_module(
                        format!("iletki-3d {kind} section style shader"),
                        format!("{base}\n{section}"),
                    )
                }),
                flat_section: flat_section_base_wgsl.as_ref().map(|base| {
                    create_module(
                        format!("iletki-3d {kind} flat section style shader"),
                        format!("{base}\n{section}"),
                    )
                }),
            }
        };

        let build_set = |label: &str,
                         shader: &wgpu::ShaderModule,
                         flat_shader: Option<&wgpu::ShaderModule>,
                         vs_face: &str,
                         vs_edge: &str,
                         entry_suffix: &str,
                         layout: &wgpu::PipelineLayout,
                         buffers: &[Option<wgpu::VertexBufferLayout<'_>>]|
         -> StylePipelineSet {
            let vs_face = format!("{vs_face}{entry_suffix}");
            let vs_edge = format!("{vs_edge}{entry_suffix}");
            let fs_face = format!("fs_style{entry_suffix}");
            let fs_edge = format!("fs_style_edge{entry_suffix}");
            let make_face = |module: &wgpu::ShaderModule,
                             suffix: &str,
                             blend: Option<wgpu::BlendState>,
                             depth_write: bool,
                             depth_compare: wgpu::CompareFunction,
                             polygon_mode: wgpu::PolygonMode,
                             fs_entry: &str| {
                ctx.device
                    .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: Some(&format!("{label} {suffix}")),
                        layout: Some(layout),
                        vertex: wgpu::VertexState {
                            module,
                            entry_point: Some(&vs_face),
                            compilation_options: wgpu::PipelineCompilationOptions::default(),
                            buffers,
                        },
                        fragment: Some(wgpu::FragmentState {
                            module,
                            entry_point: Some(fs_entry),
                            compilation_options: wgpu::PipelineCompilationOptions::default(),
                            targets: &[Some(wgpu::ColorTargetState {
                                format: ctx.color_format,
                                blend,
                                write_mask: wgpu::ColorWrites::ALL,
                            })],
                        }),
                        primitive: wgpu::PrimitiveState {
                            topology: wgpu::PrimitiveTopology::TriangleList,
                            polygon_mode,
                            cull_mode: None,
                            ..wgpu::PrimitiveState::default()
                        },
                        depth_stencil: Some(wgpu::DepthStencilState {
                            format: ctx.depth_format,
                            depth_write_enabled: Some(depth_write),
                            depth_compare: Some(depth_compare),
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

            let additive = wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            };

            let opaque = make_face(
                shader,
                "opaque",
                None,
                true,
                wgpu::CompareFunction::GreaterEqual,
                wgpu::PolygonMode::Fill,
                &fs_face,
            );
            let opaque_flat = flat_shader.map(|fs| {
                make_face(
                    fs,
                    "opaque_flat",
                    None,
                    true,
                    wgpu::CompareFunction::GreaterEqual,
                    wgpu::PolygonMode::Fill,
                    &fs_face,
                )
            });
            let overdraw = make_face(
                shader,
                "overdraw",
                Some(additive),
                false,
                wgpu::CompareFunction::Always,
                wgpu::PolygonMode::Fill,
                &fs_face,
            );
            let wireframe_line = has_polygon_line.then(|| {
                make_face(
                    shader,
                    "wireframe_line",
                    Some(wgpu::BlendState::ALPHA_BLENDING),
                    false,
                    wgpu::CompareFunction::GreaterEqual,
                    wgpu::PolygonMode::Line,
                    &fs_edge,
                )
            });
            let edge = ctx
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(&format!("{label} edge")),
                    layout: Some(layout),
                    vertex: wgpu::VertexState {
                        module: shader,
                        entry_point: Some(&vs_edge),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers,
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: shader,
                        entry_point: Some(&fs_edge),
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
                        format: ctx.depth_format,
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
            StylePipelineSet {
                opaque,
                opaque_flat,
                overdraw,
                edge,
                wireframe_line,
                section: None,
            }
        };
        let build_variants = |label: &str,
                              modules: &StyleModules,
                              vs_face: &str,
                              vs_edge: &str,
                              layout: &wgpu::PipelineLayout,
                              buffers: &[Option<wgpu::VertexBufferLayout<'_>>]|
         -> StylePipelineSet {
            let mut set = build_set(
                label,
                &modules.shader,
                modules.flat.as_ref(),
                vs_face,
                vs_edge,
                "",
                layout,
                buffers,
            );
            set.section = modules.section.as_ref().map(|section| {
                Box::new(build_set(
                    &format!("{label} section"),
                    section,
                    modules.flat_section.as_ref(),
                    vs_face,
                    vs_edge,
                    "_section",
                    layout,
                    buffers,
                ))
            });
            set
        };

        let face_buffers = vertex_layout.buffers();
        let slot_vertex = wgpu::VertexBufferLayout {
            array_stride: size_of::<u32>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![3 => Uint32],
        };
        let slot_instance = wgpu::VertexBufferLayout {
            array_stride: size_of::<u32>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![3 => Uint32],
        };
        let mut buffers_vertex_slot = face_buffers.clone();
        buffers_vertex_slot.push(Some(slot_vertex));
        let mut buffers_instance_slot = face_buffers.clone();
        buffers_instance_slot.push(Some(slot_instance));

        const INSTANCE_PARAMS: &str = "v: VertIn, @builtin(instance_index) inst: u32";
        const SLOT_PARAMS: &str = "v: VertIn, @location(3) slot: u32";
        let direct_modules = make_modules(
            "direct",
            "fn style_direct(v: VertIn, inst: u32) -> StyleOut {\n\
                 return style_vertex(v.position, v.normal, object, inst);\n\
             }\n\
             fn style_edge_direct(v: VertIn, inst: u32) -> StyleOut {\n\
                 return style_edge_vertex(v.position, v.normal, object, inst);\n\
             }\n",
            INSTANCE_PARAMS,
            "v, inst",
        );
        let direct_pl_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("iletki-3d direct style layout"),
                bind_group_layouts: &[
                    Some(ctx.camera_bind_group_layout),
                    Some(object_layout),
                    Some(&style_layout),
                ],
                immediate_size: 0,
            });
        let direct = build_variants(
            "iletki-3d direct style",
            &direct_modules,
            "vs_style_direct",
            "vs_style_edge_direct",
            &direct_pl_layout,
            &face_buffers,
        );

        let packed = packed_layout.map(|layout| {
            let helpers = format!(
                "{}\n\
                 fn style_batched(v: VertIn, slot: u32) -> StyleOut {{\n\
                     return style_vertex(v.position, v.normal, page_objects[slot], slot);\n\
                 }}\n\
                 fn style_edge_batched(v: VertIn, slot: u32) -> StyleOut {{\n\
                     return style_edge_vertex(v.position, v.normal, page_objects[slot], slot);\n\
                 }}\n",
                include_str!("shaders/mesh3d_batched.wgsl")
            );
            let modules = make_modules("batched", &helpers, SLOT_PARAMS, "v, slot");
            let pl_layout = ctx
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("iletki-3d packed style layout"),
                    bind_group_layouts: &[
                        Some(ctx.camera_bind_group_layout),
                        Some(layout),
                        Some(&style_layout),
                    ],
                    immediate_size: 0,
                });
            build_variants(
                "iletki-3d packed style",
                &modules,
                "vs_style_batched",
                "vs_style_edge_batched",
                &pl_layout,
                &buffers_vertex_slot,
            )
        });

        let (storage, storage_repeated) = match storage_layout {
            Some(layout) => {
                let helpers = format!(
                    "{}\n\
                     fn style_storage(v: VertIn, slot: u32) -> StyleOut {{\n\
                         let st = load_storage_transform(slot);\n\
                         var out = style_vertex(v.position, v.normal, st.placement, slot);\n\
                         if (!st.visible) {{\n\
                             out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);\n\
                         }}\n\
                         return out;\n\
                     }}\n\
                     fn style_edge_storage(v: VertIn, slot: u32) -> StyleOut {{\n\
                         let st = load_storage_transform(slot);\n\
                         var out = style_edge_vertex(v.position, v.normal, st.placement, slot);\n\
                         if (!st.visible) {{\n\
                             out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);\n\
                         }}\n\
                         return out;\n\
                     }}\n",
                    include_str!("shaders/mesh3d_storage.wgsl")
                );
                let modules = make_modules("storage", &helpers, SLOT_PARAMS, "v, slot");
                let pl_layout =
                    ctx.device
                        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: Some("iletki-3d storage style layout"),
                            bind_group_layouts: &[
                                Some(ctx.camera_bind_group_layout),
                                Some(layout),
                                Some(&style_layout),
                            ],
                            immediate_size: 0,
                        });
                let per_vert = build_variants(
                    "iletki-3d storage style",
                    &modules,
                    "vs_style_storage",
                    "vs_style_edge_storage",
                    &pl_layout,
                    &buffers_vertex_slot,
                );
                let per_inst = build_variants(
                    "iletki-3d repeated style",
                    &modules,
                    "vs_style_storage",
                    "vs_style_edge_storage",
                    &pl_layout,
                    &buffers_instance_slot,
                );
                (Some(per_vert), Some(per_inst))
            }
            None => (None, None),
        };

        let tech_edges =
            gtao::supports_depth_sampling(ctx.device, ctx.depth_format, ctx.msaa_samples)
                .then(|| TechnicalEdgePipeline::build(ctx));

        Self {
            color_format: ctx.color_format,
            depth_format: ctx.depth_format,
            msaa_samples: ctx.msaa_samples,
            has_storage: storage_layout.is_some(),
            has_packed: packed_layout.is_some(),
            has_section: section_aa.is_some(),
            style_layout,
            direct,
            packed,
            storage,
            storage_repeated,
            tech_edges,
        }
    }
}

fn make_flat_style_wgsl(raw_base: &str, use_either_flat: bool) -> String {
    let normalized = raw_base.replace("\r\n", "\n");
    let flat = normalized.replace(
        "@location(0) pos_rel: vec3<f32>,\n    @location(1) normal: vec3<f32>,\n    @location(2) @interpolate(flat) chunk_id: u32,",
        "@location(0) pos_rel: vec3<f32>,\n    @location(1) @interpolate(flat) normal: vec3<f32>,\n    @location(2) @interpolate(flat) chunk_id: u32,",
    );
    provoking::adapt_flat_sampling(&flat, use_either_flat)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_style_and_diagnostic_are_zero_tax() {
        assert_eq!(VisualStyle::default(), VisualStyle::ShadedWithEdges);
        assert_eq!(DiagnosticView::default(), DiagnosticView::None);
        assert!(!DiagnosticView::None.is_active());
        assert!(DiagnosticView::Normals.is_active());
        assert_eq!(size_of::<StyleUniform>(), 32);
        assert_eq!(size_of::<TechEdgeUniform>(), 64);
    }

    #[test]
    fn tech_edge_uniform_handles_perspective_ortho_and_2d() {
        use iletki_core::Vec3;
        let persp = Camera::perspective_3d(
            Vec3::new(0.0, 0.0, 8.0),
            Vec3::ZERO,
            Vec3::Y,
            std::f64::consts::FRAC_PI_3,
            0.1,
        );
        let u = TechEdgeUniform::from_camera(
            &persp,
            Vec2::new(640.0, 360.0),
            [640, 360],
            4,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            true,
        )
        .expect("perspective camera produces technical edge uniform");
        assert_eq!(u.flags, 1 | TechEdgeUniform::FLAG_CREASES);
        assert!(u.ink_color[0] < StyleUniform::TECHNICAL_INK[0]);

        let ortho = Camera::ortho_3d(
            Vec3::new(0.0, 0.0, 10.0),
            Vec3::ZERO,
            Vec3::Y,
            6.0,
            0.1,
            40.0,
        );
        let u_o = TechEdgeUniform::from_camera(
            &ortho,
            Vec2::new(320.0, 200.0),
            [320, 200],
            1,
            wgpu::TextureFormat::Rgba8Unorm,
            false,
        )
        .expect("ortho camera produces technical edge uniform");
        assert_eq!(u_o.flags, 2);
        assert_eq!(u_o.ink_color, StyleUniform::TECHNICAL_INK);
        assert_eq!(u_o.paper_color, StyleUniform::TECHNICAL_PAPER);

        let cam_2d = Camera::ortho_2d(Vec2::ZERO, 1.0);
        assert!(
            TechEdgeUniform::from_camera(
                &cam_2d,
                Vec2::new(200.0, 200.0),
                [200, 200],
                4,
                wgpu::TextureFormat::Rgba8Unorm,
                true,
            )
            .is_none()
        );
    }

    #[test]
    fn flat_style_wgsl_only_modifies_style_out_and_not_vert_in() {
        let base = format!(
            "{}\n{}",
            include_str!("shaders/mesh3d.wgsl"),
            include_str!("shaders/style3d.wgsl")
        );
        let variants = [
            ("plain", base.clone(), 1usize),
            (
                "raster",
                format!(
                    "{base}\n{}\n{}",
                    include_str!("shaders/raster_section.wgsl"),
                    include_str!("shaders/style3d_raster.wgsl")
                ),
                2usize,
            ),
            (
                "sampled",
                format!(
                    "{base}\n{}\n{}",
                    include_str!("shaders/sampled_section.wgsl"),
                    include_str!("shaders/style3d_sampled.wgsl")
                ),
                2usize,
            ),
        ];
        for (label, raw, expected_matches) in variants {
            for use_either in [false, true] {
                let flat = make_flat_style_wgsl(&raw, use_either);
                let interp = if use_either {
                    "@interpolate(flat, either)"
                } else {
                    "@interpolate(flat)"
                };
                let needle = format!(
                    "@location(0) pos_rel: vec3<f32>,\n    @location(1) {interp} normal: vec3<f32>,\n    @location(2) {interp} chunk_id: u32,"
                );
                assert_eq!(
                    flat.matches(&needle).count(),
                    expected_matches,
                    "{label}: StyleOut and section out normals must be flat (use_either={use_either})"
                );
                assert!(
                    flat.contains(
                        "struct VertIn {\n    @location(0) position: vec3<f32>,\n    @location(1) normal: vec3<f32>,\n    @location(2) color: vec4<f32>,"
                    ),
                    "{label}: VertIn must remain free of @interpolate attributes (use_either={use_either})"
                );
            }
        }
    }
}
