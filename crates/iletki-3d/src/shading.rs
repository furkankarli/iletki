//! Decoupled 3D shading pipeline registry for [`crate::Scene3DLayer`].
//!
//! Isolates GPU shader compilation, render-pipeline variants (smooth, provoking-vertex flat,
//! unclipped, raster/sampled section AA, repeated instancing, MRT color-frame selection, and
//! persistent [`crate::Material3D`] shading), and depth-format adaptation (`Depth32Float` vs.
//! `Depth32FloatStencil8`) from scene geometry storage and culling.

use crate::face_batch::{self, PageBuilder, VertexLayout};
use crate::material;
use crate::provoking::{self, FlatSupport};
use crate::scene::ObjectUniforms;
use crate::section::SectionAa;
use crate::storage_batch;
use iletki_wgpu::wgpu;
use iletki_wgpu::wgpu::util::DeviceExt;
use iletki_wgpu::{
    CAMERA_WGSL, DEPTH_FORMAT, OUTLINE_MASK_FORMAT, PICK_ID_FORMAT, PrepareCtx, SELECTION_ID_FORMAT,
};

pub(crate) struct ColorPipelines {
    pub(crate) faces: wgpu::RenderPipeline,
    pub(crate) edges: wgpu::RenderPipeline,
}

pub(crate) struct FaceShadingSet {
    pub(crate) base: wgpu::RenderPipeline,
    pub(crate) unclipped: wgpu::RenderPipeline,
    pub(crate) flat_base: Option<wgpu::RenderPipeline>,
    pub(crate) flat_unclipped: Option<wgpu::RenderPipeline>,
    pub(crate) section: Option<wgpu::RenderPipeline>,
    pub(crate) flat_section: Option<wgpu::RenderPipeline>,
}

impl FaceShadingSet {
    pub(crate) fn select(
        &self,
        has_clip_planes: bool,
        antialiased_section: bool,
    ) -> (&wgpu::RenderPipeline, Option<&wgpu::RenderPipeline>) {
        if !has_clip_planes {
            (&self.unclipped, self.flat_unclipped.as_ref())
        } else {
            let smooth = self
                .section
                .as_ref()
                .filter(|_| antialiased_section)
                .unwrap_or(&self.base);
            let flat = self
                .flat_section
                .as_ref()
                .filter(|_| antialiased_section)
                .or(self.flat_base.as_ref());
            (smooth, flat)
        }
    }
}

pub(crate) struct SectionPipelines {
    pub(crate) lit: wgpu::RenderPipeline,
    pub(crate) lit_flat: Option<wgpu::RenderPipeline>,
    pub(crate) edge: wgpu::RenderPipeline,
    pub(crate) multisampled_mask: wgpu::RenderPipeline,
}

pub(crate) struct SectionSelectPipelines {
    pub(crate) lit: wgpu::RenderPipeline,
    pub(crate) lit_flat: Option<wgpu::RenderPipeline>,
    pub(crate) edge: wgpu::RenderPipeline,
}

pub(crate) struct RepeatedPipelines {
    pub(crate) base: ColorPipelines,
    pub(crate) unclipped: ColorPipelines,
    pub(crate) flat_base: Option<wgpu::RenderPipeline>,
    pub(crate) flat_unclipped: Option<wgpu::RenderPipeline>,
    pub(crate) section: Option<ColorPipelines>,
    pub(crate) flat_section: Option<wgpu::RenderPipeline>,
    pub(crate) slots: wgpu::Buffer,
    pub(crate) mask: Option<wgpu::RenderPipeline>,
    pub(crate) multisampled_mask: Option<wgpu::RenderPipeline>,
    pub(crate) section_mask: Option<wgpu::RenderPipeline>,
}

pub(crate) struct RepeatedSelectPipelines {
    pub(crate) base: ColorPipelines,
    pub(crate) unclipped: ColorPipelines,
    pub(crate) flat_base: Option<wgpu::RenderPipeline>,
    pub(crate) flat_unclipped: Option<wgpu::RenderPipeline>,
    pub(crate) section: Option<ColorPipelines>,
    pub(crate) flat_section: Option<wgpu::RenderPipeline>,
}

pub(crate) struct PackedPipelines {
    pub(crate) layout: wgpu::BindGroupLayout,
    pub(crate) storage: bool,
    pub(crate) base: ColorPipelines,
    pub(crate) unclipped: ColorPipelines,
    pub(crate) flat_base: Option<wgpu::RenderPipeline>,
    pub(crate) flat_unclipped: Option<wgpu::RenderPipeline>,
    pub(crate) section: Option<ColorPipelines>,
    pub(crate) flat_section: Option<wgpu::RenderPipeline>,
    pub(crate) pipeline_layout: wgpu::PipelineLayout,
    pub(crate) shader: wgpu::ShaderModule,
    pub(crate) flat_shader: Option<wgpu::ShaderModule>,
    pub(crate) section_shader: Option<(SectionAa, wgpu::ShaderModule)>,
    pub(crate) flat_section_shader: Option<(SectionAa, wgpu::ShaderModule)>,
    pub(crate) repeated: Option<RepeatedPipelines>,
    pub(crate) pick: Option<wgpu::RenderPipeline>,
    pub(crate) pick_unclipped: Option<wgpu::RenderPipeline>,
    pub(crate) mask: wgpu::RenderPipeline,
    pub(crate) multisampled_mask: Option<wgpu::RenderPipeline>,
    pub(crate) section_mask: Option<wgpu::RenderPipeline>,
}

pub(crate) struct StorageSelectPipelines {
    pub(crate) base: ColorPipelines,
    pub(crate) unclipped: ColorPipelines,
    pub(crate) flat_base: Option<wgpu::RenderPipeline>,
    pub(crate) flat_unclipped: Option<wgpu::RenderPipeline>,
    pub(crate) section: Option<ColorPipelines>,
    pub(crate) flat_section: Option<wgpu::RenderPipeline>,
    pub(crate) shader: wgpu::ShaderModule,
    pub(crate) flat_shader: Option<wgpu::ShaderModule>,
    pub(crate) section_shader: Option<(SectionAa, wgpu::ShaderModule)>,
    pub(crate) flat_section_shader: Option<(SectionAa, wgpu::ShaderModule)>,
    pub(crate) repeated: Option<RepeatedSelectPipelines>,
}

pub(crate) struct SelectPipelines {
    pub(crate) lit: wgpu::RenderPipeline,
    pub(crate) lit_unclipped: wgpu::RenderPipeline,
    pub(crate) lit_flat: Option<wgpu::RenderPipeline>,
    pub(crate) lit_flat_unclipped: Option<wgpu::RenderPipeline>,
    pub(crate) edge: wgpu::RenderPipeline,
    pub(crate) edge_unclipped: wgpu::RenderPipeline,
    pub(crate) section: Option<SectionSelectPipelines>,
    pub(crate) storage: Option<StorageSelectPipelines>,
}

/// Lazily compiled persistent-material storage pipelines (`@group(1) @binding(3)` SSBO).
pub(crate) struct StorageMaterialPipelines {
    pub(crate) layout: wgpu::BindGroupLayout,
    pub(crate) pipeline_layout: wgpu::PipelineLayout,
    pub(crate) base: ColorPipelines,
    pub(crate) unclipped: ColorPipelines,
    pub(crate) flat_base: Option<wgpu::RenderPipeline>,
    pub(crate) flat_unclipped: Option<wgpu::RenderPipeline>,
    pub(crate) section: Option<ColorPipelines>,
    pub(crate) flat_section: Option<wgpu::RenderPipeline>,
    shader: wgpu::ShaderModule,
    flat_shader: Option<wgpu::ShaderModule>,
    section_shader: Option<(SectionAa, wgpu::ShaderModule)>,
    flat_section_shader: Option<(SectionAa, wgpu::ShaderModule)>,
    pub(crate) repeated: Option<RepeatedSelectPipelines>,
    pub(crate) select: Option<StorageMaterialSelectPipelines>,
}

pub(crate) struct StorageMaterialSelectPipelines {
    pub(crate) base: ColorPipelines,
    pub(crate) unclipped: ColorPipelines,
    pub(crate) flat_base: Option<wgpu::RenderPipeline>,
    pub(crate) flat_unclipped: Option<wgpu::RenderPipeline>,
    pub(crate) section: Option<ColorPipelines>,
    pub(crate) flat_section: Option<wgpu::RenderPipeline>,
    shader: wgpu::ShaderModule,
    flat_shader: Option<wgpu::ShaderModule>,
    section_shader: Option<(SectionAa, wgpu::ShaderModule)>,
    flat_section_shader: Option<(SectionAa, wgpu::ShaderModule)>,
    pub(crate) repeated: Option<RepeatedSelectPipelines>,
}

/// Lazily compiled shading pipelines for scenes with custom [`crate::Material3D`] records.
///
/// Never created while every chunk uses [`crate::MaterialId::DEFAULT`].
pub(crate) struct MaterialPipelines {
    pub(crate) direct_material_layout: wgpu::BindGroupLayout,
    direct_pipeline_layout: wgpu::PipelineLayout,
    pub(crate) direct: FaceShadingSet,
    pub(crate) direct_select: Option<FaceShadingSet>,
    pub(crate) storage: Option<StorageMaterialPipelines>,
}

/// Complete shading pipeline set cached for a `(color_format, depth_format, msaa, storage)` tuple.
pub(crate) struct ScenePipelines {
    pub(crate) format: wgpu::TextureFormat,
    pub(crate) depth_format: wgpu::TextureFormat,
    pub(crate) msaa: u32,
    pub(crate) vertex_layout: VertexLayout,
    pub(crate) flat_support: Option<FlatSupport>,
    pub(crate) object_layout: wgpu::BindGroupLayout,
    pub(crate) pipeline_layout: wgpu::PipelineLayout,
    pub(crate) dynamic_uniforms: bool,
    pub(crate) lit: wgpu::RenderPipeline,
    pub(crate) lit_unclipped: wgpu::RenderPipeline,
    pub(crate) lit_flat: Option<wgpu::RenderPipeline>,
    pub(crate) lit_flat_unclipped: Option<wgpu::RenderPipeline>,
    pub(crate) packed: Option<PackedPipelines>,
    pub(crate) storage: Option<PackedPipelines>,
    pub(crate) select: Option<SelectPipelines>,
    pub(crate) materials: Option<MaterialPipelines>,
    pub(crate) section: Option<SectionPipelines>,
    pub(crate) edge: wgpu::RenderPipeline,
    pub(crate) edge_unclipped: wgpu::RenderPipeline,
    pub(crate) mask: wgpu::RenderPipeline,
    pub(crate) multisampled_mask: Option<wgpu::RenderPipeline>,
    pub(crate) pick: wgpu::RenderPipeline,
    pub(crate) pick_unclipped: wgpu::RenderPipeline,
}

fn color_select_pipeline<W>(
    ctx: &PrepareCtx<'_, W>,
    shader: &wgpu::ShaderModule,
    vertex: &str,
    fragment: &str,
    layout: &wgpu::PipelineLayout,
    buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
    edges: bool,
) -> wgpu::RenderPipeline {
    ctx.device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(if edges {
                "iletki-3d shaded edges (selection MRT)"
            } else {
                "iletki-3d lit mesh (selection MRT)"
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
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: ctx.color_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: SELECTION_ID_FORMAT,
                        blend: None,
                        write_mask: if edges {
                            wgpu::ColorWrites::empty()
                        } else {
                            wgpu::ColorWrites::ALL
                        },
                    }),
                ],
            }),
            primitive: wgpu::PrimitiveState {
                topology: if edges {
                    wgpu::PrimitiveTopology::LineList
                } else {
                    wgpu::PrimitiveTopology::TriangleList
                },
                cull_mode: None,
                ..wgpu::PrimitiveState::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: ctx.depth_format,
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
}

fn color_pipeline<W>(
    ctx: &PrepareCtx<'_, W>,
    shader: &wgpu::ShaderModule,
    vertex: &str,
    fragment: &str,
    layout: &wgpu::PipelineLayout,
    buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
    edges: bool,
) -> wgpu::RenderPipeline {
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
                format: ctx.depth_format,
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
}

fn mask_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    vertex: &str,
    fragment: &str,
    count: u32,
    layout: &wgpu::PipelineLayout,
    buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
}

fn pick_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    vertex: &str,
    fragment: &str,
    layout: &wgpu::PipelineLayout,
    buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
            entry_point: Some(fragment),
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
}

impl StorageSelectPipelines {
    fn color<W>(
        ctx: &PrepareCtx<'_, W>,
        shader: &wgpu::ShaderModule,
        layout: &wgpu::PipelineLayout,
        buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
        section: Option<SectionAa>,
        unclipped: bool,
    ) -> ColorPipelines {
        let (face, face_fragment, edge, edge_fragment) = match section {
            None if unclipped => (
                "vs_mesh_storage_select",
                "fs_lit_select_unclipped",
                "vs_edge_storage_select",
                "fs_unlit_select",
            ),
            None => (
                "vs_mesh_storage_select",
                "fs_lit_select",
                "vs_edge_storage_select",
                "fs_edge_select",
            ),
            Some(SectionAa::Sampled) => (
                "vs_mesh_storage_select_sampled",
                "fs_lit_select_sampled",
                "vs_edge_storage_select_sampled",
                "fs_edge_select_sampled",
            ),
            Some(SectionAa::Raster) => (
                "vs_mesh_storage_select_raster",
                "fs_lit_select_raster",
                "vs_edge_storage_select_raster",
                "fs_unlit_select_raster",
            ),
        };
        ColorPipelines {
            faces: color_select_pipeline(ctx, shader, face, face_fragment, layout, buffers, false),
            edges: color_select_pipeline(ctx, shader, edge, edge_fragment, layout, buffers, true),
        }
    }

    fn face_only<W>(
        ctx: &PrepareCtx<'_, W>,
        shader: &wgpu::ShaderModule,
        layout: &wgpu::PipelineLayout,
        buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
        section: Option<SectionAa>,
        unclipped: bool,
    ) -> wgpu::RenderPipeline {
        let (face, face_fragment) = match section {
            None if unclipped => ("vs_mesh_storage_select", "fs_lit_select_unclipped"),
            None => ("vs_mesh_storage_select", "fs_lit_select"),
            Some(SectionAa::Sampled) => ("vs_mesh_storage_select_sampled", "fs_lit_select_sampled"),
            Some(SectionAa::Raster) => ("vs_mesh_storage_select_raster", "fs_lit_select_raster"),
        };
        color_select_pipeline(ctx, shader, face, face_fragment, layout, buffers, false)
    }

    pub(crate) fn ensure_repeated<W>(
        &mut self,
        ctx: &PrepareCtx<'_, W>,
        vertex_layout: VertexLayout,
        pipeline_layout: &wgpu::PipelineLayout,
    ) {
        if self.repeated.is_some() {
            return;
        }
        let mut buffers = vertex_layout.buffers();
        buffers.push(Some(wgpu::VertexBufferLayout {
            array_stride: size_of::<u32>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![3 => Uint32],
        }));
        let color = |shader, section, unclipped| {
            Self::color(ctx, shader, pipeline_layout, &buffers, section, unclipped)
        };
        let face_only = |shader, section, unclipped| {
            Self::face_only(ctx, shader, pipeline_layout, &buffers, section, unclipped)
        };
        self.repeated = Some(RepeatedSelectPipelines {
            base: color(&self.shader, None, false),
            unclipped: color(&self.shader, None, true),
            flat_base: self
                .flat_shader
                .as_ref()
                .map(|shader| face_only(shader, None, false)),
            flat_unclipped: self
                .flat_shader
                .as_ref()
                .map(|shader| face_only(shader, None, true)),
            section: self
                .section_shader
                .as_ref()
                .map(|(aa, shader)| color(shader, Some(*aa), false)),
            flat_section: self
                .flat_section_shader
                .as_ref()
                .map(|(aa, shader)| face_only(shader, Some(*aa), false)),
        });
    }

    pub(crate) fn select_color(
        &self,
        has_clip_planes: bool,
        antialiased_section: bool,
    ) -> (&ColorPipelines, Option<&wgpu::RenderPipeline>) {
        if !has_clip_planes {
            (&self.unclipped, self.flat_unclipped.as_ref())
        } else {
            let color = self
                .section
                .as_ref()
                .filter(|_| antialiased_section)
                .unwrap_or(&self.base);
            let flat = self
                .flat_section
                .as_ref()
                .filter(|_| antialiased_section)
                .or(self.flat_base.as_ref());
            (color, flat)
        }
    }

    pub(crate) fn select_repeated(
        &self,
        has_clip_planes: bool,
        antialiased_section: bool,
    ) -> Option<(&ColorPipelines, Option<&wgpu::RenderPipeline>)> {
        let repeated = self.repeated.as_ref()?;
        if !has_clip_planes {
            Some((&repeated.unclipped, repeated.flat_unclipped.as_ref()))
        } else {
            let color = repeated
                .section
                .as_ref()
                .filter(|_| antialiased_section)
                .unwrap_or(&repeated.base);
            let flat = repeated
                .flat_section
                .as_ref()
                .filter(|_| antialiased_section)
                .or(repeated.flat_base.as_ref());
            Some((color, flat))
        }
    }
}

impl StorageMaterialPipelines {
    fn color<W>(
        ctx: &PrepareCtx<'_, W>,
        shader: &wgpu::ShaderModule,
        layout: &wgpu::PipelineLayout,
        buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
        section: Option<SectionAa>,
        unclipped: bool,
    ) -> ColorPipelines {
        let (face, face_fragment, edge, edge_fragment) = match section {
            None if unclipped => (
                "vs_mesh_storage_mat",
                "fs_lit_storage_mat_unclipped",
                "vs_edge_storage",
                "fs_unlit",
            ),
            None => (
                "vs_mesh_storage_mat",
                "fs_lit_storage_mat",
                "vs_edge_storage",
                "fs_edge",
            ),
            Some(SectionAa::Sampled) => (
                "vs_mesh_storage_mat_sampled",
                "fs_lit_storage_mat_sampled",
                "vs_edge_storage_sampled",
                "fs_edge_sampled",
            ),
            Some(SectionAa::Raster) => (
                "vs_mesh_storage_mat_raster",
                "fs_lit_storage_mat_unclipped",
                "vs_edge_storage_raster",
                "fs_unlit",
            ),
        };
        ColorPipelines {
            faces: color_pipeline(ctx, shader, face, face_fragment, layout, buffers, false),
            edges: color_pipeline(ctx, shader, edge, edge_fragment, layout, buffers, true),
        }
    }

    fn face_only<W>(
        ctx: &PrepareCtx<'_, W>,
        shader: &wgpu::ShaderModule,
        layout: &wgpu::PipelineLayout,
        buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
        section: Option<SectionAa>,
        unclipped: bool,
    ) -> wgpu::RenderPipeline {
        let (face, face_fragment) = match section {
            None if unclipped => ("vs_mesh_storage_mat", "fs_lit_storage_mat_unclipped"),
            None => ("vs_mesh_storage_mat", "fs_lit_storage_mat"),
            Some(SectionAa::Sampled) => {
                ("vs_mesh_storage_mat_sampled", "fs_lit_storage_mat_sampled")
            }
            Some(SectionAa::Raster) => {
                ("vs_mesh_storage_mat_raster", "fs_lit_storage_mat_unclipped")
            }
        };
        color_pipeline(ctx, shader, face, face_fragment, layout, buffers, false)
    }

    pub(crate) fn ensure_repeated<W>(
        &mut self,
        ctx: &PrepareCtx<'_, W>,
        vertex_layout: VertexLayout,
    ) {
        if self.repeated.is_none() {
            let mut buffers = vertex_layout.buffers();
            buffers.push(Some(wgpu::VertexBufferLayout {
                array_stride: size_of::<u32>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &wgpu::vertex_attr_array![3 => Uint32],
            }));
            let color = |shader, section, unclipped| {
                Self::color(
                    ctx,
                    shader,
                    &self.pipeline_layout,
                    &buffers,
                    section,
                    unclipped,
                )
            };
            let face_only = |shader, section, unclipped| {
                Self::face_only(
                    ctx,
                    shader,
                    &self.pipeline_layout,
                    &buffers,
                    section,
                    unclipped,
                )
            };
            self.repeated = Some(RepeatedSelectPipelines {
                base: color(&self.shader, None, false),
                unclipped: color(&self.shader, None, true),
                flat_base: self
                    .flat_shader
                    .as_ref()
                    .map(|sh| face_only(sh, None, false)),
                flat_unclipped: self
                    .flat_shader
                    .as_ref()
                    .map(|sh| face_only(sh, None, true)),
                section: self
                    .section_shader
                    .as_ref()
                    .map(|(aa, sh)| color(sh, Some(*aa), false)),
                flat_section: self
                    .flat_section_shader
                    .as_ref()
                    .map(|(aa, sh)| face_only(sh, Some(*aa), false)),
            });
        }
        if let Some(sel) = self.select.as_mut() {
            sel.ensure_repeated(ctx, vertex_layout, &self.pipeline_layout);
        }
    }

    pub(crate) fn select_color(
        &self,
        has_clip_planes: bool,
        antialiased_section: bool,
    ) -> (&ColorPipelines, Option<&wgpu::RenderPipeline>) {
        if !has_clip_planes {
            (&self.unclipped, self.flat_unclipped.as_ref())
        } else {
            let color = self
                .section
                .as_ref()
                .filter(|_| antialiased_section)
                .unwrap_or(&self.base);
            let flat = self
                .flat_section
                .as_ref()
                .filter(|_| antialiased_section)
                .or(self.flat_base.as_ref());
            (color, flat)
        }
    }

    pub(crate) fn select_repeated(
        &self,
        has_clip_planes: bool,
        antialiased_section: bool,
    ) -> Option<(&ColorPipelines, Option<&wgpu::RenderPipeline>)> {
        let repeated = self.repeated.as_ref()?;
        if !has_clip_planes {
            Some((&repeated.unclipped, repeated.flat_unclipped.as_ref()))
        } else {
            let color = repeated
                .section
                .as_ref()
                .filter(|_| antialiased_section)
                .unwrap_or(&repeated.base);
            let flat = repeated
                .flat_section
                .as_ref()
                .filter(|_| antialiased_section)
                .or(repeated.flat_base.as_ref());
            Some((color, flat))
        }
    }
}

impl StorageMaterialSelectPipelines {
    fn color<W>(
        ctx: &PrepareCtx<'_, W>,
        shader: &wgpu::ShaderModule,
        layout: &wgpu::PipelineLayout,
        buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
        section: Option<SectionAa>,
        unclipped: bool,
    ) -> ColorPipelines {
        let (face, face_fragment, edge, edge_fragment) = match section {
            None if unclipped => (
                "vs_mesh_storage_mat_select",
                "fs_lit_storage_mat_select_unclipped",
                "vs_edge_storage_select",
                "fs_unlit_select",
            ),
            None => (
                "vs_mesh_storage_mat_select",
                "fs_lit_storage_mat_select",
                "vs_edge_storage_select",
                "fs_edge_select",
            ),
            Some(SectionAa::Sampled) => (
                "vs_mesh_storage_mat_select_sampled",
                "fs_lit_storage_mat_select_sampled",
                "vs_edge_storage_select_sampled",
                "fs_edge_select_sampled",
            ),
            Some(SectionAa::Raster) => (
                "vs_mesh_storage_mat_select_raster",
                "fs_lit_storage_mat_select_unclipped",
                "vs_edge_storage_select_raster",
                "fs_unlit_select_raster",
            ),
        };
        ColorPipelines {
            faces: color_select_pipeline(ctx, shader, face, face_fragment, layout, buffers, false),
            edges: color_select_pipeline(ctx, shader, edge, edge_fragment, layout, buffers, true),
        }
    }

    fn face_only<W>(
        ctx: &PrepareCtx<'_, W>,
        shader: &wgpu::ShaderModule,
        layout: &wgpu::PipelineLayout,
        buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
        section: Option<SectionAa>,
        unclipped: bool,
    ) -> wgpu::RenderPipeline {
        let (face, face_fragment) = match section {
            None if unclipped => (
                "vs_mesh_storage_mat_select",
                "fs_lit_storage_mat_select_unclipped",
            ),
            None => ("vs_mesh_storage_mat_select", "fs_lit_storage_mat_select"),
            Some(SectionAa::Sampled) => (
                "vs_mesh_storage_mat_select_sampled",
                "fs_lit_storage_mat_select_sampled",
            ),
            Some(SectionAa::Raster) => (
                "vs_mesh_storage_mat_select_raster",
                "fs_lit_storage_mat_select_unclipped",
            ),
        };
        color_select_pipeline(ctx, shader, face, face_fragment, layout, buffers, false)
    }

    fn ensure_repeated<W>(
        &mut self,
        ctx: &PrepareCtx<'_, W>,
        vertex_layout: VertexLayout,
        pipeline_layout: &wgpu::PipelineLayout,
    ) {
        if self.repeated.is_some() {
            return;
        }
        let mut buffers = vertex_layout.buffers();
        buffers.push(Some(wgpu::VertexBufferLayout {
            array_stride: size_of::<u32>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![3 => Uint32],
        }));
        let color = |shader, section, unclipped| {
            Self::color(ctx, shader, pipeline_layout, &buffers, section, unclipped)
        };
        let face_only = |shader, section, unclipped| {
            Self::face_only(ctx, shader, pipeline_layout, &buffers, section, unclipped)
        };
        self.repeated = Some(RepeatedSelectPipelines {
            base: color(&self.shader, None, false),
            unclipped: color(&self.shader, None, true),
            flat_base: self
                .flat_shader
                .as_ref()
                .map(|sh| face_only(sh, None, false)),
            flat_unclipped: self
                .flat_shader
                .as_ref()
                .map(|sh| face_only(sh, None, true)),
            section: self
                .section_shader
                .as_ref()
                .map(|(aa, sh)| color(sh, Some(*aa), false)),
            flat_section: self
                .flat_section_shader
                .as_ref()
                .map(|(aa, sh)| face_only(sh, Some(*aa), false)),
        });
    }

    pub(crate) fn select_color(
        &self,
        has_clip_planes: bool,
        antialiased_section: bool,
    ) -> (&ColorPipelines, Option<&wgpu::RenderPipeline>) {
        if !has_clip_planes {
            (&self.unclipped, self.flat_unclipped.as_ref())
        } else {
            let color = self
                .section
                .as_ref()
                .filter(|_| antialiased_section)
                .unwrap_or(&self.base);
            let flat = self
                .flat_section
                .as_ref()
                .filter(|_| antialiased_section)
                .or(self.flat_base.as_ref());
            (color, flat)
        }
    }

    pub(crate) fn select_repeated(
        &self,
        has_clip_planes: bool,
        antialiased_section: bool,
    ) -> Option<(&ColorPipelines, Option<&wgpu::RenderPipeline>)> {
        let repeated = self.repeated.as_ref()?;
        if !has_clip_planes {
            Some((&repeated.unclipped, repeated.flat_unclipped.as_ref()))
        } else {
            let color = repeated
                .section
                .as_ref()
                .filter(|_| antialiased_section)
                .unwrap_or(&repeated.base);
            let flat = repeated
                .flat_section
                .as_ref()
                .filter(|_| antialiased_section)
                .or(repeated.flat_base.as_ref());
            Some((color, flat))
        }
    }
}

impl PackedPipelines {
    fn color<W>(
        ctx: &PrepareCtx<'_, W>,
        shader: &wgpu::ShaderModule,
        layout: &wgpu::PipelineLayout,
        buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
        section: Option<SectionAa>,
        unclipped: bool,
        storage: bool,
    ) -> ColorPipelines {
        let (face, face_fragment, edge, edge_fragment) = match (storage, section) {
            (false, None) if unclipped => (
                "vs_mesh_batched",
                "fs_lit_unclipped",
                "vs_edge_batched",
                "fs_unlit",
            ),
            (false, None) => ("vs_mesh_batched", "fs_lit", "vs_edge_batched", "fs_edge"),
            (false, Some(SectionAa::Sampled)) => (
                "vs_mesh_batched_sampled",
                "fs_lit_sampled",
                "vs_edge_batched_sampled",
                "fs_edge_sampled",
            ),
            (false, Some(SectionAa::Raster)) => (
                "vs_mesh_batched_raster",
                "fs_lit_raster",
                "vs_edge_batched_raster",
                "fs_unlit",
            ),
            (true, None) if unclipped => (
                "vs_mesh_storage",
                "fs_lit_unclipped",
                "vs_edge_storage",
                "fs_unlit",
            ),
            (true, None) => ("vs_mesh_storage", "fs_lit", "vs_edge_storage", "fs_edge"),
            (true, Some(SectionAa::Sampled)) => (
                "vs_mesh_storage_sampled",
                "fs_lit_sampled",
                "vs_edge_storage_sampled",
                "fs_edge_sampled",
            ),
            (true, Some(SectionAa::Raster)) => (
                "vs_mesh_storage_raster",
                "fs_lit_raster",
                "vs_edge_storage_raster",
                "fs_unlit",
            ),
        };
        ColorPipelines {
            faces: color_pipeline(ctx, shader, face, face_fragment, layout, buffers, false),
            edges: color_pipeline(ctx, shader, edge, edge_fragment, layout, buffers, true),
        }
    }

    fn face_only<W>(
        ctx: &PrepareCtx<'_, W>,
        shader: &wgpu::ShaderModule,
        layout: &wgpu::PipelineLayout,
        buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
        section: Option<SectionAa>,
        unclipped: bool,
        storage: bool,
    ) -> wgpu::RenderPipeline {
        let (face, face_fragment) = match (storage, section) {
            (false, None) if unclipped => ("vs_mesh_batched", "fs_lit_unclipped"),
            (false, None) => ("vs_mesh_batched", "fs_lit"),
            (false, Some(SectionAa::Sampled)) => ("vs_mesh_batched_sampled", "fs_lit_sampled"),
            (false, Some(SectionAa::Raster)) => ("vs_mesh_batched_raster", "fs_lit_raster"),
            (true, None) if unclipped => ("vs_mesh_storage", "fs_lit_unclipped"),
            (true, None) => ("vs_mesh_storage", "fs_lit"),
            (true, Some(SectionAa::Sampled)) => ("vs_mesh_storage_sampled", "fs_lit_sampled"),
            (true, Some(SectionAa::Raster)) => ("vs_mesh_storage_raster", "fs_lit_raster"),
        };
        color_pipeline(ctx, shader, face, face_fragment, layout, buffers, false)
    }

    pub(crate) fn ensure_repeated<W>(
        &mut self,
        ctx: &PrepareCtx<'_, W>,
        vertex_layout: VertexLayout,
    ) {
        if self.repeated.is_some() {
            return;
        }
        let mut buffers = vertex_layout.buffers();
        buffers.push(Some(wgpu::VertexBufferLayout {
            array_stride: size_of::<u32>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![3 => Uint32],
        }));
        let storage = self.storage;
        let color = |shader, section, unclipped| {
            Self::color(
                ctx,
                shader,
                &self.pipeline_layout,
                &buffers,
                section,
                unclipped,
                storage,
            )
        };
        let face_only = |shader, section, unclipped| {
            Self::face_only(
                ctx,
                shader,
                &self.pipeline_layout,
                &buffers,
                section,
                unclipped,
                storage,
            )
        };
        let slots = (0..face_batch::OBJECTS as u32).collect::<Vec<_>>();
        let (mask, multisampled_mask, section_mask) = if storage {
            let mask_buffers = [
                Some(vertex_layout.position()),
                Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<u32>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![3 => Uint32],
                }),
            ];
            let m1 = mask_pipeline(
                ctx.device,
                &self.shader,
                "vs_mask_storage",
                "fs_mask",
                1,
                &self.pipeline_layout,
                &mask_buffers,
            );
            let m4 = (ctx.msaa_samples == 4).then(|| {
                mask_pipeline(
                    ctx.device,
                    &self.shader,
                    "vs_mask_storage",
                    "fs_mask",
                    4,
                    &self.pipeline_layout,
                    &mask_buffers,
                )
            });
            let ms = self.section_shader.as_ref().map(|(aa, shader)| {
                let (vertex, fragment) = match aa {
                    SectionAa::Sampled => ("vs_mask_storage_sampled", "fs_mask_sampled_pos"),
                    SectionAa::Raster => ("vs_mask_storage_raster", "fs_mask_unclipped"),
                };
                mask_pipeline(
                    ctx.device,
                    shader,
                    vertex,
                    fragment,
                    4,
                    &self.pipeline_layout,
                    &mask_buffers,
                )
            });
            (Some(m1), m4, ms)
        } else {
            (None, None, None)
        };
        self.repeated = Some(RepeatedPipelines {
            base: color(&self.shader, None, false),
            unclipped: color(&self.shader, None, true),
            flat_base: self
                .flat_shader
                .as_ref()
                .map(|shader| face_only(shader, None, false)),
            flat_unclipped: self
                .flat_shader
                .as_ref()
                .map(|shader| face_only(shader, None, true)),
            section: self
                .section_shader
                .as_ref()
                .map(|(aa, shader)| color(shader, Some(*aa), false)),
            flat_section: self
                .flat_section_shader
                .as_ref()
                .map(|(aa, shader)| face_only(shader, Some(*aa), false)),
            slots: ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("iletki-3d repeated geometry slots"),
                    contents: bytemuck::cast_slice(&slots),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
            mask,
            multisampled_mask,
            section_mask,
        });
    }

    pub(crate) fn select_color(
        &self,
        has_clip_planes: bool,
        antialiased_section: bool,
    ) -> (&ColorPipelines, Option<&wgpu::RenderPipeline>) {
        if !has_clip_planes {
            (&self.unclipped, self.flat_unclipped.as_ref())
        } else {
            let color = self
                .section
                .as_ref()
                .filter(|_| antialiased_section)
                .unwrap_or(&self.base);
            let flat = self
                .flat_section
                .as_ref()
                .filter(|_| antialiased_section)
                .or(self.flat_base.as_ref());
            (color, flat)
        }
    }

    pub(crate) fn select_repeated(
        &self,
        has_clip_planes: bool,
        antialiased_section: bool,
    ) -> Option<(
        &ColorPipelines,
        Option<&wgpu::RenderPipeline>,
        &wgpu::Buffer,
    )> {
        let repeated = self.repeated.as_ref()?;
        if !has_clip_planes {
            Some((
                &repeated.unclipped,
                repeated.flat_unclipped.as_ref(),
                &repeated.slots,
            ))
        } else {
            let color = repeated
                .section
                .as_ref()
                .filter(|_| antialiased_section)
                .unwrap_or(&repeated.base);
            let flat = repeated
                .flat_section
                .as_ref()
                .filter(|_| antialiased_section)
                .or(repeated.flat_base.as_ref());
            Some((color, flat, &repeated.slots))
        }
    }
}

impl ScenePipelines {
    pub(crate) fn matches<W>(
        &self,
        ctx: &PrepareCtx<'_, W>,
        section_aa: Option<SectionAa>,
        want_storage: bool,
    ) -> bool {
        self.format == ctx.color_format
            && self.depth_format == ctx.depth_format
            && self.msaa == ctx.msaa_samples
            && (section_aa.is_none() || self.section.is_some())
            && self.storage.is_some() == want_storage
    }

    pub(crate) fn build<W>(ctx: &PrepareCtx<'_, W>, want_storage: bool) -> Self {
        let section_aa = SectionAa::for_viewport(ctx);
        let flat_support = provoking::detect_flat_support(ctx.device, ctx.queue);
        let use_either = flat_support.is_some_and(|s| s.use_either)
            || ctx.device.adapter_info().backend == wgpu::Backend::Gl;
        let wgsl = provoking::adapt_flat_sampling(
            &format!("{CAMERA_WGSL}\n{}", include_str!("shaders/mesh3d.wgsl")),
            use_either,
        );
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("iletki-3d mesh shader"),
                source: wgpu::ShaderSource::Wgsl(wgsl.as_str().into()),
            });
        let flat_wgsl = flat_support
            .as_ref()
            .map(|_| provoking::make_flat_color_wgsl(&wgsl, use_either));
        let flat_shader = flat_wgsl.as_ref().map(|source| {
            ctx.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("iletki-3d flat mesh shader"),
                    source: wgpu::ShaderSource::Wgsl(source.as_str().into()),
                })
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
        let vertex_layout = VertexLayout::for_device(ctx.device);
        let pick_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<[u32; 2]>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![3 => Uint32x2],
        };

        let face_layout = vertex_layout.buffers();
        let lit = color_pipeline(
            ctx,
            &shader,
            "vs_mesh",
            "fs_lit",
            &pipeline_layout,
            &face_layout,
            false,
        );
        let lit_unclipped = color_pipeline(
            ctx,
            &shader,
            "vs_mesh",
            "fs_lit_unclipped",
            &pipeline_layout,
            &face_layout,
            false,
        );
        let lit_flat = flat_shader.as_ref().map(|shader| {
            color_pipeline(
                ctx,
                shader,
                "vs_mesh",
                "fs_lit",
                &pipeline_layout,
                &face_layout,
                false,
            )
        });
        let lit_flat_unclipped = flat_shader.as_ref().map(|shader| {
            color_pipeline(
                ctx,
                shader,
                "vs_mesh",
                "fs_lit_unclipped",
                &pipeline_layout,
                &face_layout,
                false,
            )
        });
        let edge = color_pipeline(
            ctx,
            &shader,
            "vs_edge",
            "fs_edge",
            &pipeline_layout,
            &face_layout,
            true,
        );
        let edge_unclipped = color_pipeline(
            ctx,
            &shader,
            "vs_edge",
            "fs_unlit",
            &pipeline_layout,
            &face_layout,
            true,
        );
        let pick_buffers = [Some(vertex_layout.position()), Some(pick_layout)];
        let pick = pick_pipeline(
            ctx.device,
            &shader,
            "vs_mesh_pick",
            "fs_pick",
            &pipeline_layout,
            &pick_buffers,
        );
        let pick_unclipped = pick_pipeline(
            ctx.device,
            &shader,
            "vs_mesh_pick",
            "fs_pick_unclipped",
            &pipeline_layout,
            &pick_buffers,
        );

        let build_packed = |is_storage: bool| -> PackedPipelines {
            let layout = if is_storage {
                storage_batch::bind_layout(ctx.device)
            } else {
                face_batch::bind_layout(ctx.device)
            };
            let pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some(if is_storage {
                            "iletki-3d storage geometry layout"
                        } else {
                            "iletki-3d packed geometry layout"
                        }),
                        bind_group_layouts: &[Some(ctx.camera_bind_group_layout), Some(&layout)],
                        immediate_size: 0,
                    });
            let extra_wgsl = if is_storage {
                include_str!("shaders/mesh3d_storage.wgsl")
            } else {
                include_str!("shaders/mesh3d_batched.wgsl")
            };
            let source = format!("{wgsl}\n{extra_wgsl}");
            let shader = ctx
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(if is_storage {
                        "iletki-3d storage geometry shader"
                    } else {
                        "iletki-3d packed geometry shader"
                    }),
                    source: wgpu::ShaderSource::Wgsl(source.as_str().into()),
                });
            let packed_flat_source = flat_wgsl.as_ref().map(|fw| format!("{fw}\n{extra_wgsl}"));
            let packed_flat_shader = packed_flat_source.as_ref().map(|src| {
                ctx.device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(if is_storage {
                            "iletki-3d storage flat geometry shader"
                        } else {
                            "iletki-3d packed flat geometry shader"
                        }),
                        source: wgpu::ShaderSource::Wgsl(src.as_str().into()),
                    })
            });
            let slot_layout = wgpu::VertexBufferLayout {
                array_stride: size_of::<u32>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![3 => Uint32],
            };
            let mut buffers = face_layout.clone();
            buffers.push(Some(slot_layout.clone()));
            let base = PackedPipelines::color(
                ctx,
                &shader,
                &pipeline_layout,
                &buffers,
                None,
                false,
                is_storage,
            );
            let unclipped = PackedPipelines::color(
                ctx,
                &shader,
                &pipeline_layout,
                &buffers,
                None,
                true,
                is_storage,
            );
            let flat_base = packed_flat_shader.as_ref().map(|sh| {
                PackedPipelines::face_only(
                    ctx,
                    sh,
                    &pipeline_layout,
                    &buffers,
                    None,
                    false,
                    is_storage,
                )
            });
            let flat_unclipped = packed_flat_shader.as_ref().map(|sh| {
                PackedPipelines::face_only(
                    ctx,
                    sh,
                    &pipeline_layout,
                    &buffers,
                    None,
                    true,
                    is_storage,
                )
            });
            let mask_vs = if is_storage {
                "vs_mask_storage"
            } else {
                "vs_mask_batched"
            };
            let mask_buffers = if is_storage {
                vec![Some(vertex_layout.position()), Some(slot_layout.clone())]
            } else {
                buffers.clone()
            };
            let mask = mask_pipeline(
                ctx.device,
                &shader,
                mask_vs,
                "fs_mask",
                1,
                &pipeline_layout,
                &mask_buffers,
            );
            let multisampled_mask = (ctx.msaa_samples == 4).then(|| {
                mask_pipeline(
                    ctx.device,
                    &shader,
                    mask_vs,
                    "fs_mask",
                    4,
                    &pipeline_layout,
                    &mask_buffers,
                )
            });
            let sampled_extra = if is_storage {
                concat!(
                    include_str!("shaders/mesh3d_sampled.wgsl"),
                    "\n",
                    include_str!("shaders/mesh3d_storage_sampled.wgsl")
                )
            } else {
                concat!(
                    include_str!("shaders/mesh3d_sampled.wgsl"),
                    "\n",
                    include_str!("shaders/mesh3d_batched_sampled.wgsl")
                )
            };
            let raster_extra = if is_storage {
                concat!(
                    include_str!("shaders/mesh3d_raster.wgsl"),
                    "\n",
                    include_str!("shaders/mesh3d_storage_raster.wgsl")
                )
            } else {
                concat!(
                    include_str!("shaders/mesh3d_raster.wgsl"),
                    "\n",
                    include_str!("shaders/mesh3d_batched_raster.wgsl")
                )
            };
            let section_shader = section_aa.map(|aa| {
                (
                    aa,
                    aa.shader(ctx.device, &source, sampled_extra, raster_extra),
                )
            });
            let flat_section_shader = section_aa.and_then(|aa| {
                let base_src = packed_flat_source.as_ref()?;
                let raw = aa.shader_source(base_src, sampled_extra, raster_extra);
                let flat_src = provoking::make_flat_color_wgsl(&raw, use_either);
                let module = ctx
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(if is_storage {
                            "iletki-3d storage flat section shader"
                        } else {
                            "iletki-3d packed flat section shader"
                        }),
                        source: wgpu::ShaderSource::Wgsl(flat_src.into()),
                    });
                Some((aa, module))
            });
            let section = section_shader.as_ref().map(|(aa, shader)| {
                PackedPipelines::color(
                    ctx,
                    shader,
                    &pipeline_layout,
                    &buffers,
                    Some(*aa),
                    false,
                    is_storage,
                )
            });
            let flat_section = flat_section_shader.as_ref().map(|(aa, shader)| {
                PackedPipelines::face_only(
                    ctx,
                    shader,
                    &pipeline_layout,
                    &buffers,
                    Some(*aa),
                    false,
                    is_storage,
                )
            });
            let section_mask = section_shader.as_ref().map(|(aa, shader)| {
                let (vertex, fragment) = match (is_storage, aa) {
                    (false, SectionAa::Sampled) => ("vs_mesh_batched_sampled", "fs_mask_sampled"),
                    (false, SectionAa::Raster) => ("vs_mesh_batched_raster", "fs_mask_unclipped"),
                    (true, SectionAa::Sampled) => {
                        ("vs_mask_storage_sampled", "fs_mask_sampled_pos")
                    }
                    (true, SectionAa::Raster) => ("vs_mask_storage_raster", "fs_mask_unclipped"),
                };
                mask_pipeline(
                    ctx.device,
                    shader,
                    vertex,
                    fragment,
                    4,
                    &pipeline_layout,
                    &mask_buffers,
                )
            });
            let limits = ctx.device.limits();
            let pick_vs = if is_storage {
                "vs_pick_storage"
            } else {
                "vs_pick_batched"
            };
            let (pick, pick_unclipped) =
                if limits.max_vertex_buffers >= 3 && limits.max_vertex_attributes >= 5 {
                    let id_layout = wgpu::VertexBufferLayout {
                        array_stride: size_of::<[u32; 2]>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![4 => Uint32x2],
                    };
                    let pick_buffers = [
                        Some(vertex_layout.position()),
                        Some(slot_layout),
                        Some(id_layout),
                    ];
                    (
                        Some(pick_pipeline(
                            ctx.device,
                            &shader,
                            pick_vs,
                            "fs_pick",
                            &pipeline_layout,
                            &pick_buffers,
                        )),
                        Some(pick_pipeline(
                            ctx.device,
                            &shader,
                            pick_vs,
                            "fs_pick_unclipped",
                            &pipeline_layout,
                            &pick_buffers,
                        )),
                    )
                } else {
                    (None, None)
                };
            PackedPipelines {
                pick,
                pick_unclipped,
                layout,
                storage: is_storage,
                base,
                unclipped,
                flat_base,
                flat_unclipped,
                section,
                flat_section,
                pipeline_layout,
                shader,
                flat_shader: packed_flat_shader,
                section_shader,
                flat_section_shader,
                repeated: None,
                mask,
                multisampled_mask,
                section_mask,
            }
        };
        let storage = want_storage.then(|| build_packed(true));
        let packed =
            (!want_storage && PageBuilder::supports(ctx.device)).then(|| build_packed(false));

        let mask = mask_pipeline(
            ctx.device,
            &shader,
            "vs_mask",
            "fs_mask",
            1,
            &pipeline_layout,
            &face_layout,
        );
        let multisampled_mask = (ctx.msaa_samples == 4).then(|| {
            mask_pipeline(
                ctx.device,
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
            let flat_sec_shader = flat_wgsl.as_ref().map(|fw| {
                let raw = aa.shader_source(
                    fw,
                    include_str!("shaders/mesh3d_sampled.wgsl"),
                    include_str!("shaders/mesh3d_raster.wgsl"),
                );
                let flat_src = provoking::make_flat_color_wgsl(&raw, use_either);
                ctx.device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("iletki-3d flat section shader"),
                        source: wgpu::ShaderSource::Wgsl(flat_src.into()),
                    })
            });
            match aa {
                SectionAa::Sampled => SectionPipelines {
                    lit: color_pipeline(
                        ctx,
                        &shader,
                        "vs_mesh_sampled",
                        "fs_lit_sampled",
                        &pipeline_layout,
                        &face_layout,
                        false,
                    ),
                    lit_flat: flat_sec_shader.as_ref().map(|sh| {
                        color_pipeline(
                            ctx,
                            sh,
                            "vs_mesh_sampled",
                            "fs_lit_sampled",
                            &pipeline_layout,
                            &face_layout,
                            false,
                        )
                    }),
                    edge: color_pipeline(
                        ctx,
                        &shader,
                        "vs_edge_sampled",
                        "fs_edge_sampled",
                        &pipeline_layout,
                        &face_layout,
                        true,
                    ),
                    multisampled_mask: mask_pipeline(
                        ctx.device,
                        &shader,
                        "vs_mesh_sampled",
                        "fs_mask_sampled",
                        4,
                        &pipeline_layout,
                        &face_layout,
                    ),
                },
                SectionAa::Raster => SectionPipelines {
                    lit: color_pipeline(
                        ctx,
                        &shader,
                        "vs_mesh_raster",
                        "fs_lit_raster",
                        &pipeline_layout,
                        &face_layout,
                        false,
                    ),
                    lit_flat: flat_sec_shader.as_ref().map(|sh| {
                        color_pipeline(
                            ctx,
                            sh,
                            "vs_mesh_raster",
                            "fs_lit_raster",
                            &pipeline_layout,
                            &face_layout,
                            false,
                        )
                    }),
                    edge: color_pipeline(
                        ctx,
                        &shader,
                        "vs_edge_raster",
                        "fs_unlit",
                        &pipeline_layout,
                        &face_layout,
                        true,
                    ),
                    multisampled_mask: mask_pipeline(
                        ctx.device,
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

        Self {
            format: ctx.color_format,
            depth_format: ctx.depth_format,
            msaa: ctx.msaa_samples,
            vertex_layout,
            flat_support,
            object_layout,
            pipeline_layout,
            dynamic_uniforms,
            lit,
            lit_unclipped,
            lit_flat,
            lit_flat_unclipped,
            packed,
            storage,
            select: None,
            materials: None,
            section,
            edge,
            edge_unclipped,
            mask,
            multisampled_mask,
            pick,
            pick_unclipped,
        }
    }

    pub(crate) fn ensure_select<W>(&mut self, ctx: &PrepareCtx<'_, W>) {
        let section_aa = SectionAa::for_viewport(ctx);
        if self
            .select
            .as_ref()
            .is_some_and(|sel| section_aa.is_none() || sel.section.is_some())
        {
            return;
        }
        let use_either = self.flat_support.is_some_and(|s| s.use_either)
            || ctx.device.adapter_info().backend == wgpu::Backend::Gl;
        let base_wgsl = provoking::adapt_flat_sampling(
            &format!(
                "{CAMERA_WGSL}\n{}\n{}",
                include_str!("shaders/mesh3d.wgsl"),
                include_str!("shaders/mesh3d_select.wgsl")
            ),
            use_either,
        );
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("iletki-3d selection MRT shader"),
                source: wgpu::ShaderSource::Wgsl(base_wgsl.as_str().into()),
            });
        let flat_wgsl = self
            .flat_support
            .as_ref()
            .map(|_| provoking::make_flat_color_wgsl(&base_wgsl, use_either));
        let flat_shader = flat_wgsl.as_ref().map(|source| {
            ctx.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("iletki-3d flat selection MRT shader"),
                    source: wgpu::ShaderSource::Wgsl(source.as_str().into()),
                })
        });
        let face_layout = self.vertex_layout.buffers();
        let lit = color_select_pipeline(
            ctx,
            &shader,
            "vs_mesh_select",
            "fs_lit_select",
            &self.pipeline_layout,
            &face_layout,
            false,
        );
        let lit_unclipped = color_select_pipeline(
            ctx,
            &shader,
            "vs_mesh_select",
            "fs_lit_select_unclipped",
            &self.pipeline_layout,
            &face_layout,
            false,
        );
        let lit_flat = flat_shader.as_ref().map(|sh| {
            color_select_pipeline(
                ctx,
                sh,
                "vs_mesh_select",
                "fs_lit_select",
                &self.pipeline_layout,
                &face_layout,
                false,
            )
        });
        let lit_flat_unclipped = flat_shader.as_ref().map(|sh| {
            color_select_pipeline(
                ctx,
                sh,
                "vs_mesh_select",
                "fs_lit_select_unclipped",
                &self.pipeline_layout,
                &face_layout,
                false,
            )
        });
        let edge = color_select_pipeline(
            ctx,
            &shader,
            "vs_edge_select",
            "fs_edge_select",
            &self.pipeline_layout,
            &face_layout,
            true,
        );
        let edge_unclipped = color_select_pipeline(
            ctx,
            &shader,
            "vs_edge_select",
            "fs_unlit_select",
            &self.pipeline_layout,
            &face_layout,
            true,
        );

        let sampled_direct = concat!(
            include_str!("shaders/mesh3d_sampled.wgsl"),
            "\n",
            include_str!("shaders/mesh3d_select_sampled.wgsl")
        );
        let raster_direct = concat!(
            include_str!("shaders/mesh3d_raster.wgsl"),
            "\n",
            include_str!("shaders/mesh3d_select_raster.wgsl")
        );
        let section = section_aa.map(|aa| {
            let sec_shader = aa.shader(ctx.device, &base_wgsl, sampled_direct, raster_direct);
            let flat_sec_shader = flat_wgsl.as_ref().map(|fw| {
                let raw = aa.shader_source(fw, sampled_direct, raster_direct);
                let flat_src = provoking::make_flat_color_wgsl(&raw, use_either);
                ctx.device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("iletki-3d flat section selection MRT shader"),
                        source: wgpu::ShaderSource::Wgsl(flat_src.into()),
                    })
            });
            match aa {
                SectionAa::Sampled => SectionSelectPipelines {
                    lit: color_select_pipeline(
                        ctx,
                        &sec_shader,
                        "vs_mesh_select_sampled",
                        "fs_lit_select_sampled",
                        &self.pipeline_layout,
                        &face_layout,
                        false,
                    ),
                    lit_flat: flat_sec_shader.as_ref().map(|sh| {
                        color_select_pipeline(
                            ctx,
                            sh,
                            "vs_mesh_select_sampled",
                            "fs_lit_select_sampled",
                            &self.pipeline_layout,
                            &face_layout,
                            false,
                        )
                    }),
                    edge: color_select_pipeline(
                        ctx,
                        &sec_shader,
                        "vs_edge_select_sampled",
                        "fs_edge_select_sampled",
                        &self.pipeline_layout,
                        &face_layout,
                        true,
                    ),
                },
                SectionAa::Raster => SectionSelectPipelines {
                    lit: color_select_pipeline(
                        ctx,
                        &sec_shader,
                        "vs_mesh_select_raster",
                        "fs_lit_select_raster",
                        &self.pipeline_layout,
                        &face_layout,
                        false,
                    ),
                    lit_flat: flat_sec_shader.as_ref().map(|sh| {
                        color_select_pipeline(
                            ctx,
                            sh,
                            "vs_mesh_select_raster",
                            "fs_lit_select_raster",
                            &self.pipeline_layout,
                            &face_layout,
                            false,
                        )
                    }),
                    edge: color_select_pipeline(
                        ctx,
                        &sec_shader,
                        "vs_edge_select_raster",
                        "fs_unlit_select_raster",
                        &self.pipeline_layout,
                        &face_layout,
                        true,
                    ),
                },
            }
        });

        let storage = self.storage.as_ref().map(|storage_pl| {
            let storage_src = format!(
                "{base_wgsl}\n{}\n{}",
                include_str!("shaders/mesh3d_storage.wgsl"),
                include_str!("shaders/mesh3d_storage_select.wgsl")
            );
            let shader = ctx
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("iletki-3d storage selection MRT shader"),
                    source: wgpu::ShaderSource::Wgsl(storage_src.as_str().into()),
                });
            let flat_storage_src = flat_wgsl.as_ref().map(|fw| {
                format!(
                    "{fw}\n{}\n{}",
                    include_str!("shaders/mesh3d_storage.wgsl"),
                    include_str!("shaders/mesh3d_storage_select.wgsl")
                )
            });
            let flat_shader = flat_storage_src.as_ref().map(|src| {
                ctx.device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("iletki-3d storage flat selection MRT shader"),
                        source: wgpu::ShaderSource::Wgsl(src.as_str().into()),
                    })
            });
            let mut buffers = face_layout.clone();
            buffers.push(Some(wgpu::VertexBufferLayout {
                array_stride: size_of::<u32>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![3 => Uint32],
            }));
            let base = StorageSelectPipelines::color(
                ctx,
                &shader,
                &storage_pl.pipeline_layout,
                &buffers,
                None,
                false,
            );
            let unclipped = StorageSelectPipelines::color(
                ctx,
                &shader,
                &storage_pl.pipeline_layout,
                &buffers,
                None,
                true,
            );
            let flat_base = flat_shader.as_ref().map(|sh| {
                StorageSelectPipelines::face_only(
                    ctx,
                    sh,
                    &storage_pl.pipeline_layout,
                    &buffers,
                    None,
                    false,
                )
            });
            let flat_unclipped = flat_shader.as_ref().map(|sh| {
                StorageSelectPipelines::face_only(
                    ctx,
                    sh,
                    &storage_pl.pipeline_layout,
                    &buffers,
                    None,
                    true,
                )
            });
            let sampled_extra = concat!(
                include_str!("shaders/mesh3d_sampled.wgsl"),
                "\n",
                include_str!("shaders/mesh3d_select_sampled.wgsl"),
                "\n",
                include_str!("shaders/mesh3d_storage_select_sampled.wgsl")
            );
            let raster_extra = concat!(
                include_str!("shaders/mesh3d_raster.wgsl"),
                "\n",
                include_str!("shaders/mesh3d_select_raster.wgsl"),
                "\n",
                include_str!("shaders/mesh3d_storage_select_raster.wgsl")
            );
            let section_shader = section_aa.map(|aa| {
                (
                    aa,
                    aa.shader(ctx.device, &storage_src, sampled_extra, raster_extra),
                )
            });
            let flat_section_shader = section_aa.and_then(|aa| {
                let base_src = flat_storage_src.as_ref()?;
                let raw = aa.shader_source(base_src, sampled_extra, raster_extra);
                let flat_src = provoking::make_flat_color_wgsl(&raw, use_either);
                let module = ctx
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("iletki-3d storage flat section selection MRT shader"),
                        source: wgpu::ShaderSource::Wgsl(flat_src.into()),
                    });
                Some((aa, module))
            });
            let section = section_shader.as_ref().map(|(aa, sh)| {
                StorageSelectPipelines::color(
                    ctx,
                    sh,
                    &storage_pl.pipeline_layout,
                    &buffers,
                    Some(*aa),
                    false,
                )
            });
            let flat_section = flat_section_shader.as_ref().map(|(aa, sh)| {
                StorageSelectPipelines::face_only(
                    ctx,
                    sh,
                    &storage_pl.pipeline_layout,
                    &buffers,
                    Some(*aa),
                    false,
                )
            });
            StorageSelectPipelines {
                base,
                unclipped,
                flat_base,
                flat_unclipped,
                section,
                flat_section,
                shader,
                flat_shader,
                section_shader,
                flat_section_shader,
                repeated: None,
            }
        });

        self.select = Some(SelectPipelines {
            lit,
            lit_unclipped,
            lit_flat,
            lit_flat_unclipped,
            edge,
            edge_unclipped,
            section,
            storage,
        });
    }

    /// Lazily compiles material-aware pipelines (direct `@group(2)` + storage `@group(1) @binding(3)`
    /// SSBO, plus MRT selection variants when requested). Never called while only default material
    /// `0` is in use.
    pub(crate) fn ensure_materials<W>(&mut self, ctx: &PrepareCtx<'_, W>, need_select: bool) {
        let section_aa = SectionAa::for_viewport(ctx);
        let use_either = self.flat_support.is_some_and(|s| s.use_either)
            || ctx.device.adapter_info().backend == wgpu::Backend::Gl;
        let face_layout = self.vertex_layout.buffers();
        let need_build = self
            .materials
            .as_ref()
            .is_none_or(|m| section_aa.is_some() && m.direct.section.is_none());

        if need_build {
            let direct_material_layout =
                material::direct_bind_layout(ctx.device, self.dynamic_uniforms);
            let direct_pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("iletki-3d direct material pipeline layout"),
                        bind_group_layouts: &[
                            Some(ctx.camera_bind_group_layout),
                            Some(&self.object_layout),
                            Some(&direct_material_layout),
                        ],
                        immediate_size: 0,
                    });
            let direct_wgsl = provoking::adapt_flat_sampling(
                &format!(
                    "{CAMERA_WGSL}\n{}\n{}\n{}",
                    include_str!("shaders/mesh3d.wgsl"),
                    include_str!("shaders/material3d.wgsl"),
                    include_str!("shaders/mesh3d_material_direct.wgsl")
                ),
                use_either,
            );
            let direct_shader = ctx
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("iletki-3d direct material shader"),
                    source: wgpu::ShaderSource::Wgsl(direct_wgsl.as_str().into()),
                });
            let direct_flat_wgsl = self
                .flat_support
                .as_ref()
                .map(|_| provoking::make_flat_color_wgsl(&direct_wgsl, use_either));
            let direct_flat_shader = direct_flat_wgsl.as_ref().map(|src| {
                ctx.device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("iletki-3d direct flat material shader"),
                        source: wgpu::ShaderSource::Wgsl(src.as_str().into()),
                    })
            });
            let direct_sampled_extra = concat!(
                include_str!("shaders/mesh3d_sampled.wgsl"),
                "\n",
                include_str!("shaders/mesh3d_material_direct_sampled.wgsl")
            );
            let direct_raster_extra = include_str!("shaders/mesh3d_raster.wgsl");
            let direct_sec_shader = section_aa.map(|aa| {
                (
                    aa,
                    aa.shader(
                        ctx.device,
                        &direct_wgsl,
                        direct_sampled_extra,
                        direct_raster_extra,
                    ),
                )
            });
            let direct_flat_sec_shader = section_aa.and_then(|aa| {
                let fw = direct_flat_wgsl.as_ref()?;
                let raw = aa.shader_source(fw, direct_sampled_extra, direct_raster_extra);
                let flat_src = provoking::make_flat_color_wgsl(&raw, use_either);
                let module = ctx
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("iletki-3d direct flat section material shader"),
                        source: wgpu::ShaderSource::Wgsl(flat_src.into()),
                    });
                Some((aa, module))
            });
            let direct_sec_entry = |aa: SectionAa| match aa {
                SectionAa::Sampled => ("vs_mesh_sampled", "fs_lit_mat_sampled"),
                SectionAa::Raster => ("vs_mesh_raster", "fs_lit_mat_unclipped"),
            };
            let direct = FaceShadingSet {
                base: color_pipeline(
                    ctx,
                    &direct_shader,
                    "vs_mesh",
                    "fs_lit_mat",
                    &direct_pipeline_layout,
                    &face_layout,
                    false,
                ),
                unclipped: color_pipeline(
                    ctx,
                    &direct_shader,
                    "vs_mesh",
                    "fs_lit_mat_unclipped",
                    &direct_pipeline_layout,
                    &face_layout,
                    false,
                ),
                flat_base: direct_flat_shader.as_ref().map(|sh| {
                    color_pipeline(
                        ctx,
                        sh,
                        "vs_mesh",
                        "fs_lit_mat",
                        &direct_pipeline_layout,
                        &face_layout,
                        false,
                    )
                }),
                flat_unclipped: direct_flat_shader.as_ref().map(|sh| {
                    color_pipeline(
                        ctx,
                        sh,
                        "vs_mesh",
                        "fs_lit_mat_unclipped",
                        &direct_pipeline_layout,
                        &face_layout,
                        false,
                    )
                }),
                section: direct_sec_shader.as_ref().map(|(aa, sh)| {
                    let (vs, fs) = direct_sec_entry(*aa);
                    color_pipeline(
                        ctx,
                        sh,
                        vs,
                        fs,
                        &direct_pipeline_layout,
                        &face_layout,
                        false,
                    )
                }),
                flat_section: direct_flat_sec_shader.as_ref().map(|(aa, sh)| {
                    let (vs, fs) = direct_sec_entry(*aa);
                    color_pipeline(
                        ctx,
                        sh,
                        vs,
                        fs,
                        &direct_pipeline_layout,
                        &face_layout,
                        false,
                    )
                }),
            };

            let storage = self.storage.is_some().then(|| {
                let layout = storage_batch::bind_layout_with_materials(ctx.device);
                let pipeline_layout =
                    ctx.device
                        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: Some("iletki-3d storage material pipeline layout"),
                            bind_group_layouts: &[
                                Some(ctx.camera_bind_group_layout),
                                Some(&layout),
                            ],
                            immediate_size: 0,
                        });
                let storage_wgsl = provoking::adapt_flat_sampling(
                    &format!(
                        "{CAMERA_WGSL}\n{}\n{}\n{}\n{}",
                        include_str!("shaders/mesh3d.wgsl"),
                        include_str!("shaders/material3d.wgsl"),
                        include_str!("shaders/mesh3d_storage.wgsl"),
                        include_str!("shaders/mesh3d_material_storage.wgsl")
                    ),
                    use_either,
                );
                let shader = ctx
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("iletki-3d storage material shader"),
                        source: wgpu::ShaderSource::Wgsl(storage_wgsl.as_str().into()),
                    });
                let flat_wgsl = self
                    .flat_support
                    .as_ref()
                    .map(|_| provoking::make_flat_color_wgsl(&storage_wgsl, use_either));
                let flat_shader = flat_wgsl.as_ref().map(|src| {
                    ctx.device
                        .create_shader_module(wgpu::ShaderModuleDescriptor {
                            label: Some("iletki-3d storage flat material shader"),
                            source: wgpu::ShaderSource::Wgsl(src.as_str().into()),
                        })
                });
                let mut buffers = face_layout.clone();
                buffers.push(Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<u32>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![3 => Uint32],
                }));
                let base = StorageMaterialPipelines::color(
                    ctx,
                    &shader,
                    &pipeline_layout,
                    &buffers,
                    None,
                    false,
                );
                let unclipped = StorageMaterialPipelines::color(
                    ctx,
                    &shader,
                    &pipeline_layout,
                    &buffers,
                    None,
                    true,
                );
                let flat_base = flat_shader.as_ref().map(|sh| {
                    StorageMaterialPipelines::face_only(
                        ctx,
                        sh,
                        &pipeline_layout,
                        &buffers,
                        None,
                        false,
                    )
                });
                let flat_unclipped = flat_shader.as_ref().map(|sh| {
                    StorageMaterialPipelines::face_only(
                        ctx,
                        sh,
                        &pipeline_layout,
                        &buffers,
                        None,
                        true,
                    )
                });
                let sampled_extra = concat!(
                    include_str!("shaders/mesh3d_sampled.wgsl"),
                    "\n",
                    include_str!("shaders/mesh3d_storage_sampled.wgsl"),
                    "\n",
                    include_str!("shaders/mesh3d_material_storage_sampled.wgsl")
                );
                let raster_extra = concat!(
                    include_str!("shaders/mesh3d_raster.wgsl"),
                    "\n",
                    include_str!("shaders/mesh3d_storage_raster.wgsl"),
                    "\n",
                    include_str!("shaders/mesh3d_material_storage_raster.wgsl")
                );
                let section_shader = section_aa.map(|aa| {
                    (
                        aa,
                        aa.shader(ctx.device, &storage_wgsl, sampled_extra, raster_extra),
                    )
                });
                let flat_section_shader = section_aa.and_then(|aa| {
                    let fw = flat_wgsl.as_ref()?;
                    let raw = aa.shader_source(fw, sampled_extra, raster_extra);
                    let flat_src = provoking::make_flat_color_wgsl(&raw, use_either);
                    let module = ctx
                        .device
                        .create_shader_module(wgpu::ShaderModuleDescriptor {
                            label: Some("iletki-3d storage flat section material shader"),
                            source: wgpu::ShaderSource::Wgsl(flat_src.into()),
                        });
                    Some((aa, module))
                });
                let section = section_shader.as_ref().map(|(aa, sh)| {
                    StorageMaterialPipelines::color(
                        ctx,
                        sh,
                        &pipeline_layout,
                        &buffers,
                        Some(*aa),
                        false,
                    )
                });
                let flat_section = flat_section_shader.as_ref().map(|(aa, sh)| {
                    StorageMaterialPipelines::face_only(
                        ctx,
                        sh,
                        &pipeline_layout,
                        &buffers,
                        Some(*aa),
                        false,
                    )
                });
                StorageMaterialPipelines {
                    layout,
                    pipeline_layout,
                    base,
                    unclipped,
                    flat_base,
                    flat_unclipped,
                    section,
                    flat_section,
                    shader,
                    flat_shader,
                    section_shader,
                    flat_section_shader,
                    repeated: None,
                    select: None,
                }
            });

            self.materials = Some(MaterialPipelines {
                direct_material_layout,
                direct_pipeline_layout,
                direct,
                direct_select: None,
                storage,
            });
        }

        if !need_select {
            return;
        }
        let Some(mats) = self.materials.as_mut() else {
            return;
        };
        if mats.direct_select.is_none() {
            let direct_sel_wgsl = provoking::adapt_flat_sampling(
                &format!(
                    "{CAMERA_WGSL}\n{}\n{}\n{}\n{}\n{}",
                    include_str!("shaders/mesh3d.wgsl"),
                    include_str!("shaders/material3d.wgsl"),
                    include_str!("shaders/mesh3d_select.wgsl"),
                    include_str!("shaders/mesh3d_material_direct.wgsl"),
                    include_str!("shaders/mesh3d_material_direct_select.wgsl")
                ),
                use_either,
            );
            let shader = ctx
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("iletki-3d direct material selection MRT shader"),
                    source: wgpu::ShaderSource::Wgsl(direct_sel_wgsl.as_str().into()),
                });
            let flat_wgsl = self
                .flat_support
                .as_ref()
                .map(|_| provoking::make_flat_color_wgsl(&direct_sel_wgsl, use_either));
            let flat_shader = flat_wgsl.as_ref().map(|src| {
                ctx.device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("iletki-3d direct flat material selection MRT shader"),
                        source: wgpu::ShaderSource::Wgsl(src.as_str().into()),
                    })
            });
            let sampled_extra = concat!(
                include_str!("shaders/mesh3d_sampled.wgsl"),
                "\n",
                include_str!("shaders/mesh3d_select_sampled.wgsl"),
                "\n",
                include_str!("shaders/mesh3d_material_direct_select_sampled.wgsl")
            );
            let raster_extra = concat!(
                include_str!("shaders/mesh3d_raster.wgsl"),
                "\n",
                include_str!("shaders/mesh3d_select_raster.wgsl")
            );
            let sec_shader = section_aa.map(|aa| {
                (
                    aa,
                    aa.shader(ctx.device, &direct_sel_wgsl, sampled_extra, raster_extra),
                )
            });
            let flat_sec_shader = section_aa.and_then(|aa| {
                let fw = flat_wgsl.as_ref()?;
                let raw = aa.shader_source(fw, sampled_extra, raster_extra);
                let flat_src = provoking::make_flat_color_wgsl(&raw, use_either);
                let module = ctx
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("iletki-3d direct flat section material selection MRT shader"),
                        source: wgpu::ShaderSource::Wgsl(flat_src.into()),
                    });
                Some((aa, module))
            });
            let sec_entry = |aa: SectionAa| match aa {
                SectionAa::Sampled => ("vs_mesh_select_sampled", "fs_lit_select_mat_sampled"),
                SectionAa::Raster => ("vs_mesh_select_raster", "fs_lit_select_mat_unclipped"),
            };
            mats.direct_select = Some(FaceShadingSet {
                base: color_select_pipeline(
                    ctx,
                    &shader,
                    "vs_mesh_select",
                    "fs_lit_select_mat",
                    &mats.direct_pipeline_layout,
                    &face_layout,
                    false,
                ),
                unclipped: color_select_pipeline(
                    ctx,
                    &shader,
                    "vs_mesh_select",
                    "fs_lit_select_mat_unclipped",
                    &mats.direct_pipeline_layout,
                    &face_layout,
                    false,
                ),
                flat_base: flat_shader.as_ref().map(|sh| {
                    color_select_pipeline(
                        ctx,
                        sh,
                        "vs_mesh_select",
                        "fs_lit_select_mat",
                        &mats.direct_pipeline_layout,
                        &face_layout,
                        false,
                    )
                }),
                flat_unclipped: flat_shader.as_ref().map(|sh| {
                    color_select_pipeline(
                        ctx,
                        sh,
                        "vs_mesh_select",
                        "fs_lit_select_mat_unclipped",
                        &mats.direct_pipeline_layout,
                        &face_layout,
                        false,
                    )
                }),
                section: sec_shader.as_ref().map(|(aa, sh)| {
                    let (vs, fs) = sec_entry(*aa);
                    color_select_pipeline(
                        ctx,
                        sh,
                        vs,
                        fs,
                        &mats.direct_pipeline_layout,
                        &face_layout,
                        false,
                    )
                }),
                flat_section: flat_sec_shader.as_ref().map(|(aa, sh)| {
                    let (vs, fs) = sec_entry(*aa);
                    color_select_pipeline(
                        ctx,
                        sh,
                        vs,
                        fs,
                        &mats.direct_pipeline_layout,
                        &face_layout,
                        false,
                    )
                }),
            });
        }

        if let Some(st) = mats.storage.as_mut()
            && st.select.is_none()
        {
            let storage_sel_wgsl = provoking::adapt_flat_sampling(
                &format!(
                    "{CAMERA_WGSL}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
                    include_str!("shaders/mesh3d.wgsl"),
                    include_str!("shaders/material3d.wgsl"),
                    include_str!("shaders/mesh3d_select.wgsl"),
                    include_str!("shaders/mesh3d_storage.wgsl"),
                    include_str!("shaders/mesh3d_storage_select.wgsl"),
                    include_str!("shaders/mesh3d_material_storage.wgsl"),
                    include_str!("shaders/mesh3d_material_storage_select.wgsl")
                ),
                use_either,
            );
            let shader = ctx
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("iletki-3d storage material selection MRT shader"),
                    source: wgpu::ShaderSource::Wgsl(storage_sel_wgsl.as_str().into()),
                });
            let flat_wgsl = self
                .flat_support
                .as_ref()
                .map(|_| provoking::make_flat_color_wgsl(&storage_sel_wgsl, use_either));
            let flat_shader = flat_wgsl.as_ref().map(|src| {
                ctx.device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("iletki-3d storage flat material selection MRT shader"),
                        source: wgpu::ShaderSource::Wgsl(src.as_str().into()),
                    })
            });
            let mut buffers = face_layout;
            buffers.push(Some(wgpu::VertexBufferLayout {
                array_stride: size_of::<u32>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![3 => Uint32],
            }));
            let base = StorageMaterialSelectPipelines::color(
                ctx,
                &shader,
                &st.pipeline_layout,
                &buffers,
                None,
                false,
            );
            let unclipped = StorageMaterialSelectPipelines::color(
                ctx,
                &shader,
                &st.pipeline_layout,
                &buffers,
                None,
                true,
            );
            let flat_base = flat_shader.as_ref().map(|sh| {
                StorageMaterialSelectPipelines::face_only(
                    ctx,
                    sh,
                    &st.pipeline_layout,
                    &buffers,
                    None,
                    false,
                )
            });
            let flat_unclipped = flat_shader.as_ref().map(|sh| {
                StorageMaterialSelectPipelines::face_only(
                    ctx,
                    sh,
                    &st.pipeline_layout,
                    &buffers,
                    None,
                    true,
                )
            });
            let sampled_extra = concat!(
                include_str!("shaders/mesh3d_sampled.wgsl"),
                "\n",
                include_str!("shaders/mesh3d_select_sampled.wgsl"),
                "\n",
                include_str!("shaders/mesh3d_storage_select_sampled.wgsl"),
                "\n",
                include_str!("shaders/mesh3d_material_storage_select_sampled.wgsl")
            );
            let raster_extra = concat!(
                include_str!("shaders/mesh3d_raster.wgsl"),
                "\n",
                include_str!("shaders/mesh3d_select_raster.wgsl"),
                "\n",
                include_str!("shaders/mesh3d_storage_select_raster.wgsl"),
                "\n",
                include_str!("shaders/mesh3d_material_storage_select_raster.wgsl")
            );
            let section_shader = section_aa.map(|aa| {
                (
                    aa,
                    aa.shader(ctx.device, &storage_sel_wgsl, sampled_extra, raster_extra),
                )
            });
            let flat_section_shader = section_aa.and_then(|aa| {
                let fw = flat_wgsl.as_ref()?;
                let raw = aa.shader_source(fw, sampled_extra, raster_extra);
                let flat_src = provoking::make_flat_color_wgsl(&raw, use_either);
                let module = ctx
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("iletki-3d storage flat section material selection MRT shader"),
                        source: wgpu::ShaderSource::Wgsl(flat_src.into()),
                    });
                Some((aa, module))
            });
            let section = section_shader.as_ref().map(|(aa, sh)| {
                StorageMaterialSelectPipelines::color(
                    ctx,
                    sh,
                    &st.pipeline_layout,
                    &buffers,
                    Some(*aa),
                    false,
                )
            });
            let flat_section = flat_section_shader.as_ref().map(|(aa, sh)| {
                StorageMaterialSelectPipelines::face_only(
                    ctx,
                    sh,
                    &st.pipeline_layout,
                    &buffers,
                    Some(*aa),
                    false,
                )
            });
            st.select = Some(StorageMaterialSelectPipelines {
                base,
                unclipped,
                flat_base,
                flat_unclipped,
                section,
                flat_section,
                shader,
                flat_shader,
                section_shader,
                flat_section_shader,
                repeated: None,
            });
        }
    }
}
