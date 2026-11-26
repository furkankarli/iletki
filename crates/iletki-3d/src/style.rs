//! CAD visual styles (`Shaded`, `ShadedWithEdges`, `Technical`, `Wireframe`, `XRay`) and
//! engineering diagnostic views (`Normals`, `Facets`, `Backfaces`, `Overdraw`, `Chunks`, `Depth`).
//!
//! Strictly opt-in: while [`VisualStyle::ShadedWithEdges`] or [`VisualStyle::Shaded`] is active
//! with [`DiagnosticView::None`], zero style shaders, pipelines or uniform buffers are allocated.

use crate::face_batch::VertexLayout;
use crate::provoking;
use bytemuck::{Pod, Zeroable};
use iletki_wgpu::{CAMERA_WGSL, PrepareCtx, wgpu};
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

/// Per-viewport uniform buffers for face style and edge style passes (`@group(2)`).
pub(crate) struct ViewStyleUniform {
    face_buf: wgpu::Buffer,
    face_bg: wgpu::BindGroup,
    edge_buf: wgpu::Buffer,
    edge_bg: wgpu::BindGroup,
    last_face: Option<StyleUniform>,
    last_edge: Option<StyleUniform>,
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
}

/// Lazily compiled pipelines for `Technical`, `Wireframe`, and all [`DiagnosticView`] modes.
pub(crate) struct StylePipelines {
    color_format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
    msaa_samples: u32,
    has_storage: bool,
    has_packed: bool,
    pub(crate) style_layout: wgpu::BindGroupLayout,
    pub(crate) direct: StylePipelineSet,
    pub(crate) packed: Option<StylePipelineSet>,
    pub(crate) storage: Option<StylePipelineSet>,
    pub(crate) storage_repeated: Option<StylePipelineSet>,
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

        let build_set = |label: &str,
                         shader: &wgpu::ShaderModule,
                         flat_shader: Option<&wgpu::ShaderModule>,
                         vs_face: &str,
                         vs_edge: &str,
                         layout: &wgpu::PipelineLayout,
                         buffers: &[Option<wgpu::VertexBufferLayout<'_>>]|
         -> StylePipelineSet {
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
                            entry_point: Some(vs_face),
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
                "fs_style",
            );
            let opaque_flat = flat_shader.map(|fs| {
                make_face(
                    fs,
                    "opaque_flat",
                    None,
                    true,
                    wgpu::CompareFunction::GreaterEqual,
                    wgpu::PolygonMode::Fill,
                    "fs_style",
                )
            });
            let overdraw = make_face(
                shader,
                "overdraw",
                Some(additive),
                false,
                wgpu::CompareFunction::Always,
                wgpu::PolygonMode::Fill,
                "fs_style",
            );
            let wireframe_line = has_polygon_line.then(|| {
                make_face(
                    shader,
                    "wireframe_line",
                    Some(wgpu::BlendState::ALPHA_BLENDING),
                    false,
                    wgpu::CompareFunction::GreaterEqual,
                    wgpu::PolygonMode::Line,
                    "fs_style_edge",
                )
            });
            let edge = ctx
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(&format!("{label} edge")),
                    layout: Some(layout),
                    vertex: wgpu::VertexState {
                        module: shader,
                        entry_point: Some(vs_edge),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers,
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: shader,
                        entry_point: Some("fs_style_edge"),
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
            }
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

        let direct_suffix = "\
             @vertex\n\
             fn vs_style_direct(v: VertIn, @builtin(instance_index) inst: u32) -> StyleOut {\n\
                 return style_vertex(v.position, v.normal, object, inst);\n\
             }\n\
             @vertex\n\
             fn vs_style_edge_direct(v: VertIn, @builtin(instance_index) inst: u32) -> StyleOut {\n\
                 return style_edge_vertex(v.position, v.normal, object, inst);\n\
             }\n";
        let direct_src = format!("{base_wgsl}\n{direct_suffix}");
        let direct_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("iletki-3d direct style shader"),
                source: wgpu::ShaderSource::Wgsl(direct_src.into()),
            });
        let direct_flat_shader = flat_base_wgsl.as_ref().map(|fb| {
            let src = format!("{fb}\n{direct_suffix}");
            ctx.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("iletki-3d direct flat style shader"),
                    source: wgpu::ShaderSource::Wgsl(src.into()),
                })
        });
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
        let direct = build_set(
            "iletki-3d direct style",
            &direct_shader,
            direct_flat_shader.as_ref(),
            "vs_style_direct",
            "vs_style_edge_direct",
            &direct_pl_layout,
            &face_buffers,
        );

        let packed = packed_layout.map(|layout| {
            let suffix = format!(
                "{}\n\
                 @vertex\n\
                 fn vs_style_batched(v: VertIn, @location(3) slot: u32) -> StyleOut {{\n\
                     return style_vertex(v.position, v.normal, page_objects[slot], slot);\n\
                 }}\n\
                 @vertex\n\
                 fn vs_style_edge_batched(v: VertIn, @location(3) slot: u32) -> StyleOut {{\n\
                     return style_edge_vertex(v.position, v.normal, page_objects[slot], slot);\n\
                 }}\n",
                include_str!("shaders/mesh3d_batched.wgsl")
            );
            let src = format!("{base_wgsl}\n{suffix}");
            let shader = ctx
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("iletki-3d packed style shader"),
                    source: wgpu::ShaderSource::Wgsl(src.into()),
                });
            let flat_shader = flat_base_wgsl.as_ref().map(|fb| {
                let fsrc = format!("{fb}\n{suffix}");
                ctx.device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("iletki-3d packed flat style shader"),
                        source: wgpu::ShaderSource::Wgsl(fsrc.into()),
                    })
            });
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
            build_set(
                "iletki-3d packed style",
                &shader,
                flat_shader.as_ref(),
                "vs_style_batched",
                "vs_style_edge_batched",
                &pl_layout,
                &buffers_vertex_slot,
            )
        });

        let (storage, storage_repeated) = match storage_layout {
            Some(layout) => {
                let suffix = format!(
                    "{}\n\
                     @vertex\n\
                     fn vs_style_storage(v: VertIn, @location(3) slot: u32) -> StyleOut {{\n\
                         let st = load_storage_transform(slot);\n\
                         var out = style_vertex(v.position, v.normal, st.placement, slot);\n\
                         if (!st.visible) {{\n\
                             out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);\n\
                         }}\n\
                         return out;\n\
                     }}\n\
                     @vertex\n\
                     fn vs_style_edge_storage(v: VertIn, @location(3) slot: u32) -> StyleOut {{\n\
                         let st = load_storage_transform(slot);\n\
                         var out = style_edge_vertex(v.position, v.normal, st.placement, slot);\n\
                         if (!st.visible) {{\n\
                             out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);\n\
                         }}\n\
                         return out;\n\
                     }}\n",
                    include_str!("shaders/mesh3d_storage.wgsl")
                );
                let src = format!("{base_wgsl}\n{suffix}");
                let shader = ctx
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("iletki-3d storage style shader"),
                        source: wgpu::ShaderSource::Wgsl(src.into()),
                    });
                let flat_shader = flat_base_wgsl.as_ref().map(|fb| {
                    let fsrc = format!("{fb}\n{suffix}");
                    ctx.device
                        .create_shader_module(wgpu::ShaderModuleDescriptor {
                            label: Some("iletki-3d storage flat style shader"),
                            source: wgpu::ShaderSource::Wgsl(fsrc.into()),
                        })
                });
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
                let per_vert = build_set(
                    "iletki-3d storage style",
                    &shader,
                    flat_shader.as_ref(),
                    "vs_style_storage",
                    "vs_style_edge_storage",
                    &pl_layout,
                    &buffers_vertex_slot,
                );
                let per_inst = build_set(
                    "iletki-3d repeated style",
                    &shader,
                    flat_shader.as_ref(),
                    "vs_style_storage",
                    "vs_style_edge_storage",
                    &pl_layout,
                    &buffers_instance_slot,
                );
                (Some(per_vert), Some(per_inst))
            }
            None => (None, None),
        };

        Self {
            color_format: ctx.color_format,
            depth_format: ctx.depth_format,
            msaa_samples: ctx.msaa_samples,
            has_storage: storage_layout.is_some(),
            has_packed: packed_layout.is_some(),
            style_layout,
            direct,
            packed,
            storage,
            storage_repeated,
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
    }

    #[test]
    fn flat_style_wgsl_only_modifies_style_out_and_not_vert_in() {
        let raw = format!(
            "{}\n{}",
            include_str!("shaders/mesh3d.wgsl"),
            include_str!("shaders/style3d.wgsl")
        );
        for use_either in [false, true] {
            let flat = make_flat_style_wgsl(&raw, use_either);
            let interp = if use_either {
                "@interpolate(flat, either)"
            } else {
                "@interpolate(flat)"
            };
            assert!(
                flat.contains(&format!(
                    "@location(0) pos_rel: vec3<f32>,\n    @location(1) {interp} normal: vec3<f32>,\n    @location(2) {interp} chunk_id: u32,"
                )),
                "StyleOut normal must be flat (use_either={use_either})"
            );
            assert!(
                flat.contains(
                    "struct VertIn {\n    @location(0) position: vec3<f32>,\n    @location(1) normal: vec3<f32>,\n    @location(2) color: vec4<f32>,"
                ),
                "VertIn must remain free of @interpolate attributes (use_either={use_either})"
            );
        }
    }
}
