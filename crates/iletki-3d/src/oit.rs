//! MSAA-compatible Weighted Blended Order-Independent Transparency (WBOIT).
//!
//! Uses an `Rgba16Float` weighted-color accumulation target and an `R8Unorm` multiplicative
//! revealage target (with hardware MSAA resolve when `msaa_samples > 1`), followed by a single
//! fullscreen composite pass over the opaque scene color target.
//!
//! Strictly opt-in: when no translucent/ghost/X-Ray pass is active, zero OIT textures, zero OIT
//! pipelines and zero extra render passes are created.

use crate::face_batch::{PackedDraws, VertexLayout};
use crate::material::MaterialGpu;
use crate::provoking;
use crate::scene::ObjectUniforms;
use bytemuck::{Pod, Zeroable};
use iletki_wgpu::{CAMERA_WGSL, PrepareCtx, wgpu};
use std::num::NonZeroU64;

pub(crate) const OIT_ACCUM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub(crate) const OIT_REVEAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;
pub(crate) const DEFAULT_XRAY_OPACITY: f32 = 0.28;

const OBJECT_BYTES: usize = size_of::<ObjectUniforms>();
const DIRECT_META_BYTES: usize = size_of::<OitDirectUniform>();
const DIRECT_PAGE_BYTES: usize = 256 * 1024;

/// Global OIT parameters for storage-buffer translucent draws (`@group(2) @binding(0)`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, Default, PartialEq)]
pub(crate) struct OitGlobalUniform {
    pub(crate) xray_opacity: f32,
    /// Bit 0 = xray_mode; bit 1 = has_clip_planes.
    pub(crate) flags: u32,
    pub(crate) _pad: [u32; 2],
}

/// Per-draw OIT parameters for direct and packed fallback draws (`@group(2) @binding(0)`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, Default, PartialEq)]
pub(crate) struct OitDirectUniform {
    pub(crate) material: MaterialGpu,
    pub(crate) chunk_opacity: f32,
    /// Bit 0 = chunk_ghost.
    pub(crate) chunk_flags: u32,
    pub(crate) xray_opacity: f32,
    /// Bit 0 = xray_mode; bit 1 = has_clip_planes.
    pub(crate) global_flags: u32,
}

pub(crate) struct OitStorageVariant {
    pub(crate) smooth: wgpu::RenderPipeline,
    pub(crate) flat: Option<wgpu::RenderPipeline>,
    pub(crate) repeated_smooth: wgpu::RenderPipeline,
    pub(crate) repeated_flat: Option<wgpu::RenderPipeline>,
}

/// Lazily compiled WBOIT accumulation and composite pipelines.
pub(crate) struct OitPipelines {
    color_format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
    msaa_samples: u32,
    has_storage: bool,
    has_storage_materials: bool,
    has_packed: bool,
    pub(crate) global_layout: wgpu::BindGroupLayout,
    pub(crate) direct_meta_layout: wgpu::BindGroupLayout,
    pub(crate) composite_layout: wgpu::BindGroupLayout,
    pub(crate) direct_smooth: wgpu::RenderPipeline,
    pub(crate) direct_flat: Option<wgpu::RenderPipeline>,
    pub(crate) packed_smooth: Option<wgpu::RenderPipeline>,
    pub(crate) packed_flat: Option<wgpu::RenderPipeline>,
    pub(crate) storage: Option<OitStorageVariant>,
    pub(crate) storage_materials: Option<OitStorageVariant>,
    pub(crate) composite: wgpu::RenderPipeline,
}

impl OitPipelines {
    pub(crate) fn matches<W>(
        &self,
        ctx: &PrepareCtx<'_, W>,
        has_storage: bool,
        has_storage_materials: bool,
        has_packed: bool,
    ) -> bool {
        self.color_format == ctx.color_format
            && self.depth_format == ctx.depth_format
            && self.msaa_samples == ctx.msaa_samples
            && self.has_storage == has_storage
            && (!has_storage_materials || self.has_storage_materials)
            && self.has_packed == has_packed
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build<W>(
        ctx: &PrepareCtx<'_, W>,
        vertex_layout: VertexLayout,
        object_layout: &wgpu::BindGroupLayout,
        packed_layout: Option<&wgpu::BindGroupLayout>,
        storage_layout: Option<&wgpu::BindGroupLayout>,
        storage_material_layout: Option<&wgpu::BindGroupLayout>,
        dynamic_uniforms: bool,
        has_flat_support: bool,
        use_either_flat: bool,
    ) -> Self {
        let global_layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("iletki-3d oit global layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(size_of::<OitGlobalUniform>() as u64),
                    },
                    count: None,
                }],
            });
        let direct_meta_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("iletki-3d oit direct meta layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: dynamic_uniforms,
                            min_binding_size: NonZeroU64::new(DIRECT_META_BYTES as u64),
                        },
                        count: None,
                    }],
                });
        let composite_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("iletki-3d oit composite layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                    ],
                });

        let accum_blend = wgpu::BlendState {
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
        let reveal_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::OneMinusSrc,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        };

        let make_accum =
            |label: &str,
             shader: &wgpu::ShaderModule,
             vs_entry: &str,
             fs_entry: &str,
             layout: &wgpu::PipelineLayout,
             buffers: &[Option<wgpu::VertexBufferLayout<'_>>]| {
                ctx.device
                    .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: Some(label),
                        layout: Some(layout),
                        vertex: wgpu::VertexState {
                            module: shader,
                            entry_point: Some(vs_entry),
                            compilation_options: wgpu::PipelineCompilationOptions::default(),
                            buffers,
                        },
                        fragment: Some(wgpu::FragmentState {
                            module: shader,
                            entry_point: Some(fs_entry),
                            compilation_options: wgpu::PipelineCompilationOptions::default(),
                            targets: &[
                                Some(wgpu::ColorTargetState {
                                    format: OIT_ACCUM_FORMAT,
                                    blend: Some(accum_blend),
                                    write_mask: wgpu::ColorWrites::ALL,
                                }),
                                Some(wgpu::ColorTargetState {
                                    format: OIT_REVEAL_FORMAT,
                                    blend: Some(reveal_blend),
                                    write_mask: wgpu::ColorWrites::ALL,
                                }),
                            ],
                        }),
                        primitive: wgpu::PrimitiveState {
                            topology: wgpu::PrimitiveTopology::TriangleList,
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
                    })
            };

        let raw_base = format!(
            "{CAMERA_WGSL}\n{}\n{}\n{}",
            include_str!("shaders/mesh3d.wgsl"),
            include_str!("shaders/material3d.wgsl"),
            include_str!("shaders/oit_accum.wgsl")
        );
        let base_wgsl = provoking::adapt_flat_sampling(&raw_base, use_either_flat);
        let flat_base_wgsl =
            has_flat_support.then(|| provoking::make_flat_color_wgsl(&base_wgsl, use_either_flat));

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

        // Direct OIT pipelines.
        let direct_suffix = "\
            struct OitDirectUniform {\n\
                material: MaterialGpu,\n\
                chunk_opacity: f32,\n\
                chunk_flags: u32,\n\
                xray_opacity: f32,\n\
                global_flags: u32,\n\
            };\n\
            @group(2) @binding(0)\n\
            var<uniform> oit_direct: OitDirectUniform;\n\
            @fragment\n\
            fn fs_oit_direct(in: ColorOut, @builtin(front_facing) is_front: bool) -> OitFragOut {\n\
                return evaluate_oit(\n\
                    in.clip_pos.z, in.pos_rel, in.normal, in.color, is_front,\n\
                    oit_direct.material, oit_direct.chunk_opacity,\n\
                    (oit_direct.chunk_flags & 1u) != 0u,\n\
                    oit_direct.xray_opacity, oit_direct.global_flags,\n\
                );\n\
            }\n";
        let direct_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("iletki-3d direct oit shader"),
                source: wgpu::ShaderSource::Wgsl(format!("{base_wgsl}\n{direct_suffix}").into()),
            });
        let direct_flat_shader = flat_base_wgsl.as_ref().map(|fb| {
            ctx.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("iletki-3d direct flat oit shader"),
                    source: wgpu::ShaderSource::Wgsl(format!("{fb}\n{direct_suffix}").into()),
                })
        });
        let direct_pl_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("iletki-3d direct oit layout"),
                bind_group_layouts: &[
                    Some(ctx.camera_bind_group_layout),
                    Some(object_layout),
                    Some(&direct_meta_layout),
                ],
                immediate_size: 0,
            });
        let direct_smooth = make_accum(
            "iletki-3d direct oit",
            &direct_shader,
            "vs_mesh",
            "fs_oit_direct",
            &direct_pl_layout,
            &face_buffers,
        );
        let direct_flat = direct_flat_shader.as_ref().map(|sh| {
            make_accum(
                "iletki-3d direct flat oit",
                sh,
                "vs_mesh",
                "fs_oit_direct",
                &direct_pl_layout,
                &face_buffers,
            )
        });

        // Packed fallback OIT pipelines (used on GL/WebGL2 for uniform-material X-Ray/vertex-alpha batches).
        let (packed_smooth, packed_flat) = match packed_layout {
            Some(layout) => {
                let suffix = format!(
                    "{}\n{direct_suffix}",
                    include_str!("shaders/mesh3d_batched.wgsl")
                );
                let shader = ctx
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("iletki-3d packed oit shader"),
                        source: wgpu::ShaderSource::Wgsl(format!("{base_wgsl}\n{suffix}").into()),
                    });
                let flat_sh = flat_base_wgsl.as_ref().map(|fb| {
                    ctx.device
                        .create_shader_module(wgpu::ShaderModuleDescriptor {
                            label: Some("iletki-3d packed flat oit shader"),
                            source: wgpu::ShaderSource::Wgsl(format!("{fb}\n{suffix}").into()),
                        })
                });
                let pl_layout =
                    ctx.device
                        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: Some("iletki-3d packed oit layout"),
                            bind_group_layouts: &[
                                Some(ctx.camera_bind_group_layout),
                                Some(layout),
                                Some(&direct_meta_layout),
                            ],
                            immediate_size: 0,
                        });
                let smooth = make_accum(
                    "iletki-3d packed oit",
                    &shader,
                    "vs_mesh_batched",
                    "fs_oit_direct",
                    &pl_layout,
                    &buffers_vertex_slot,
                );
                let flat = flat_sh.as_ref().map(|sh| {
                    make_accum(
                        "iletki-3d packed flat oit",
                        sh,
                        "vs_mesh_batched",
                        "fs_oit_direct",
                        &pl_layout,
                        &buffers_vertex_slot,
                    )
                });
                (Some(smooth), flat)
            }
            None => (None, None),
        };

        let build_storage_variant = |layout: &wgpu::BindGroupLayout,
                                     with_materials: bool|
         -> OitStorageVariant {
            let mat_decl = if with_materials {
                "@group(1) @binding(3)\nvar<storage, read> materials: array<MaterialGpu>;\n"
            } else {
                ""
            };
            let mat_fetch = if with_materials {
                "let mat_idx = min(in.chunk_flags >> 8u, arrayLength(&materials) - 1u);\n\
                 let mat = materials[mat_idx];\n"
            } else {
                "let mat = oit_default_material();\n"
            };
            let suffix = format!(
                "{}\n{mat_decl}{}",
                include_str!("shaders/mesh3d_storage.wgsl"),
                storage_oit_suffix_wgsl(mat_fetch),
            );
            let smooth_src =
                provoking::adapt_flat_sampling(&format!("{base_wgsl}\n{suffix}"), use_either_flat);
            let shader = ctx
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("iletki-3d storage oit shader"),
                    source: wgpu::ShaderSource::Wgsl(smooth_src.as_str().into()),
                });
            let flat_sh = has_flat_support.then(|| {
                let flat_src = provoking::make_flat_color_wgsl(&smooth_src, use_either_flat);
                ctx.device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("iletki-3d storage flat oit shader"),
                        source: wgpu::ShaderSource::Wgsl(flat_src.into()),
                    })
            });
            let pl_layout = ctx
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("iletki-3d storage oit layout"),
                    bind_group_layouts: &[
                        Some(ctx.camera_bind_group_layout),
                        Some(layout),
                        Some(&global_layout),
                    ],
                    immediate_size: 0,
                });
            OitStorageVariant {
                smooth: make_accum(
                    "iletki-3d storage oit",
                    &shader,
                    "vs_oit_storage",
                    "fs_oit_storage",
                    &pl_layout,
                    &buffers_vertex_slot,
                ),
                flat: flat_sh.as_ref().map(|sh| {
                    make_accum(
                        "iletki-3d storage flat oit",
                        sh,
                        "vs_oit_storage",
                        "fs_oit_storage",
                        &pl_layout,
                        &buffers_vertex_slot,
                    )
                }),
                repeated_smooth: make_accum(
                    "iletki-3d repeated oit",
                    &shader,
                    "vs_oit_storage",
                    "fs_oit_storage",
                    &pl_layout,
                    &buffers_instance_slot,
                ),
                repeated_flat: flat_sh.as_ref().map(|sh| {
                    make_accum(
                        "iletki-3d repeated flat oit",
                        sh,
                        "vs_oit_storage",
                        "fs_oit_storage",
                        &pl_layout,
                        &buffers_instance_slot,
                    )
                }),
            }
        };

        let storage = storage_layout.map(|l| build_storage_variant(l, false));
        let storage_materials = storage_material_layout.map(|l| build_storage_variant(l, true));

        // Composite fullscreen pipeline.
        let comp_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("iletki-3d oit composite shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/oit_composite.wgsl").into()),
            });
        let comp_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("iletki-3d oit composite layout"),
                bind_group_layouts: &[Some(&composite_layout)],
                immediate_size: 0,
            });
        let composite = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("iletki-3d oit composite"),
                layout: Some(&comp_layout),
                vertex: wgpu::VertexState {
                    module: &comp_shader,
                    entry_point: Some("vs_oit_composite"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &comp_shader,
                    entry_point: Some("fs_oit_composite"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: ctx.color_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
                    count: ctx.msaa_samples,
                    ..wgpu::MultisampleState::default()
                },
                multiview_mask: None,
                cache: None,
            });

        Self {
            color_format: ctx.color_format,
            depth_format: ctx.depth_format,
            msaa_samples: ctx.msaa_samples,
            has_storage: storage_layout.is_some(),
            has_storage_materials: storage_material_layout.is_some(),
            has_packed: packed_layout.is_some(),
            global_layout,
            direct_meta_layout,
            composite_layout,
            direct_smooth,
            direct_flat,
            packed_smooth,
            packed_flat,
            storage,
            storage_materials,
            composite,
        }
    }
}

/// Per-viewport lazily allocated WBOIT accumulation + revealage textures (`allocated_px`).
pub(crate) struct ViewOitTargets {
    allocated_px: [u32; 2],
    msaa_samples: u32,
    accum_msaa: Option<wgpu::TextureView>,
    accum_resolved: wgpu::TextureView,
    reveal_msaa: Option<wgpu::TextureView>,
    reveal_resolved: wgpu::TextureView,
    composite_bg: wgpu::BindGroup,
}

impl ViewOitTargets {
    pub(crate) fn ensure<'a>(
        slot: &'a mut Option<Self>,
        device: &wgpu::Device,
        composite_layout: &wgpu::BindGroupLayout,
        allocated_px: [u32; 2],
        msaa_samples: u32,
    ) -> Option<&'a Self> {
        let needs_alloc = slot
            .as_ref()
            .is_none_or(|t| t.allocated_px != allocated_px || t.msaa_samples != msaa_samples);
        if needs_alloc {
            let [w, h] = [allocated_px[0].max(1), allocated_px[1].max(1)];
            let extent = wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            };
            let make_tex = |label, format, samples, usage| {
                device
                    .create_texture(&wgpu::TextureDescriptor {
                        label: Some(label),
                        size: extent,
                        mip_level_count: 1,
                        sample_count: samples,
                        dimension: wgpu::TextureDimension::D2,
                        format,
                        usage,
                        view_formats: &[],
                    })
                    .create_view(&wgpu::TextureViewDescriptor::default())
            };
            let resolved_usage =
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
            let accum_resolved = make_tex(
                "iletki-3d oit accum resolved",
                OIT_ACCUM_FORMAT,
                1,
                resolved_usage,
            );
            let reveal_resolved = make_tex(
                "iletki-3d oit reveal resolved",
                OIT_REVEAL_FORMAT,
                1,
                resolved_usage,
            );
            let (accum_msaa, reveal_msaa) = if msaa_samples > 1 {
                (
                    Some(make_tex(
                        "iletki-3d oit accum msaa",
                        OIT_ACCUM_FORMAT,
                        msaa_samples,
                        wgpu::TextureUsages::RENDER_ATTACHMENT,
                    )),
                    Some(make_tex(
                        "iletki-3d oit reveal msaa",
                        OIT_REVEAL_FORMAT,
                        msaa_samples,
                        wgpu::TextureUsages::RENDER_ATTACHMENT,
                    )),
                )
            } else {
                (None, None)
            };
            let composite_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("iletki-3d oit composite binding"),
                layout: composite_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&accum_resolved),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&reveal_resolved),
                    },
                ],
            });
            *slot = Some(Self {
                allocated_px,
                msaa_samples,
                accum_msaa,
                accum_resolved,
                reveal_msaa,
                reveal_resolved,
                composite_bg,
            });
        }
        slot.as_ref()
    }

    pub(crate) fn accum_views(&self) -> (&wgpu::TextureView, Option<&wgpu::TextureView>) {
        match &self.accum_msaa {
            Some(msaa) => (msaa, Some(&self.accum_resolved)),
            None => (&self.accum_resolved, None),
        }
    }

    pub(crate) fn reveal_views(&self) -> (&wgpu::TextureView, Option<&wgpu::TextureView>) {
        match &self.reveal_msaa {
            Some(msaa) => (msaa, Some(&self.reveal_resolved)),
            None => (&self.reveal_resolved, None),
        }
    }

    pub(crate) fn composite_bind_group(&self) -> &wgpu::BindGroup {
        &self.composite_bg
    }
}

struct OitUniformPage {
    capacity: usize,
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

/// Per-viewport OIT uniform state and direct/fallback draw lists.
pub(crate) struct ViewOitState {
    pub(crate) active: bool,
    pub(crate) use_storage: bool,
    pub(crate) use_materials: bool,
    global_buf: wgpu::Buffer,
    global_bg: wgpu::BindGroup,
    last_global: Option<OitGlobalUniform>,
    pub(crate) targets: Option<ViewOitTargets>,
    pub(crate) direct_chunks: Vec<usize>,
    direct_objects: Vec<ObjectUniforms>,
    direct_metas: Vec<OitDirectUniform>,
    object_pages: Vec<OitUniformPage>,
    object_staging: Vec<u8>,
    object_stride: usize,
    object_slots_per_page: usize,
    meta_pages: Vec<OitUniformPage>,
    meta_staging: Vec<u8>,
    meta_stride: usize,
    meta_slots_per_page: usize,
    pub(crate) fallback_packed: PackedDraws,
    packed_meta_buf: wgpu::Buffer,
    packed_meta_bg: wgpu::BindGroup,
    last_packed_meta: Option<OitDirectUniform>,
}

impl ViewOitState {
    pub(crate) fn new(
        device: &wgpu::Device,
        pipelines: &OitPipelines,
        dynamic_uniforms: bool,
    ) -> Self {
        let limits = device.limits();
        let alignment = (limits.min_uniform_buffer_offset_alignment as usize).max(1);
        let object_stride = OBJECT_BYTES.div_ceil(alignment) * alignment;
        let meta_stride = DIRECT_META_BYTES.div_ceil(alignment) * alignment;
        let bytes = usize::try_from(limits.max_buffer_size.min(DIRECT_PAGE_BYTES as u64))
            .unwrap_or(DIRECT_META_BYTES);
        let object_slots_per_page = if dynamic_uniforms {
            bytes.saturating_sub(OBJECT_BYTES) / object_stride + 1
        } else {
            1
        };
        let meta_slots_per_page = if dynamic_uniforms {
            bytes.saturating_sub(DIRECT_META_BYTES) / meta_stride + 1
        } else {
            1
        };

        let global_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("iletki-3d oit global uniform"),
            size: size_of::<OitGlobalUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let global_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("iletki-3d oit global binding"),
            layout: &pipelines.global_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: global_buf.as_entire_binding(),
            }],
        });
        let packed_meta_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("iletki-3d oit packed meta uniform"),
            size: DIRECT_META_BYTES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let packed_meta_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("iletki-3d oit packed meta binding"),
            layout: &pipelines.direct_meta_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &packed_meta_buf,
                    offset: 0,
                    size: NonZeroU64::new(DIRECT_META_BYTES as u64),
                }),
            }],
        });

        Self {
            active: false,
            use_storage: false,
            use_materials: false,
            global_buf,
            global_bg,
            last_global: None,
            targets: None,
            direct_chunks: Vec::new(),
            direct_objects: Vec::new(),
            direct_metas: Vec::new(),
            object_pages: Vec::new(),
            object_staging: Vec::new(),
            object_stride,
            object_slots_per_page,
            meta_pages: Vec::new(),
            meta_staging: Vec::new(),
            meta_stride,
            meta_slots_per_page,
            fallback_packed: PackedDraws::default(),
            packed_meta_buf,
            packed_meta_bg,
            last_packed_meta: None,
        }
    }

    pub(crate) fn clear_draws(&mut self) {
        self.direct_chunks.clear();
        self.direct_objects.clear();
        self.direct_metas.clear();
        self.fallback_packed.clear();
    }

    pub(crate) fn update_global(&mut self, queue: &wgpu::Queue, global: OitGlobalUniform) {
        if self.last_global != Some(global) {
            queue.write_buffer(&self.global_buf, 0, bytemuck::bytes_of(&global));
            self.last_global = Some(global);
        }
    }

    pub(crate) fn update_packed_meta(&mut self, queue: &wgpu::Queue, meta: OitDirectUniform) {
        if self.last_packed_meta != Some(meta) {
            queue.write_buffer(&self.packed_meta_buf, 0, bytemuck::bytes_of(&meta));
            self.last_packed_meta = Some(meta);
        }
    }

    pub(crate) fn global_bind_group(&self) -> &wgpu::BindGroup {
        &self.global_bg
    }

    pub(crate) fn bind_packed_meta(&self, pass: &mut wgpu::RenderPass<'_>, dynamic: bool) {
        if dynamic {
            pass.set_bind_group(2, &self.packed_meta_bg, &[0]);
        } else {
            pass.set_bind_group(2, &self.packed_meta_bg, &[]);
        }
    }

    pub(crate) fn push_direct(
        &mut self,
        chunk_index: usize,
        object: ObjectUniforms,
        meta: OitDirectUniform,
    ) {
        self.direct_chunks.push(chunk_index);
        self.direct_objects.push(object);
        self.direct_metas.push(meta);
    }

    pub(crate) fn upload_direct(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        object_layout: &wgpu::BindGroupLayout,
        meta_layout: &wgpu::BindGroupLayout,
    ) {
        upload_paged(
            device,
            queue,
            object_layout,
            &self.direct_objects,
            &mut self.object_pages,
            &mut self.object_staging,
            self.object_stride,
            self.object_slots_per_page,
            "iletki-3d oit direct object page",
        );
        upload_paged(
            device,
            queue,
            meta_layout,
            &self.direct_metas,
            &mut self.meta_pages,
            &mut self.meta_staging,
            self.meta_stride,
            self.meta_slots_per_page,
            "iletki-3d oit direct meta page",
        );
    }

    pub(crate) fn bind_direct(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        direct_idx: usize,
        dynamic: bool,
    ) -> bool {
        let obj_slots = self.object_slots_per_page.max(1);
        let meta_slots = self.meta_slots_per_page.max(1);
        let Some(obj_page) = self.object_pages.get(direct_idx / obj_slots) else {
            return false;
        };
        let Some(meta_page) = self.meta_pages.get(direct_idx / meta_slots) else {
            return false;
        };
        let obj_off = (direct_idx % obj_slots) * self.object_stride;
        let meta_off = (direct_idx % meta_slots) * self.meta_stride;
        if dynamic {
            pass.set_bind_group(1, &obj_page.bind_group, &[obj_off as u32]);
            pass.set_bind_group(2, &meta_page.bind_group, &[meta_off as u32]);
        } else {
            pass.set_bind_group(1, &obj_page.bind_group, &[]);
            pass.set_bind_group(2, &meta_page.bind_group, &[]);
        }
        true
    }
}

#[allow(clippy::too_many_arguments)]
fn upload_paged<T: Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    items: &[T],
    pages: &mut Vec<OitUniformPage>,
    staging: &mut Vec<u8>,
    stride: usize,
    slots_per_page: usize,
    label: &str,
) {
    if items.is_empty() {
        return;
    }
    let elem_bytes = size_of::<T>();
    let per_page = slots_per_page.max(1);
    for (index, chunk) in items.chunks(per_page).enumerate() {
        let bytes_for = |slots: usize| (slots.saturating_sub(1)) * stride + elem_bytes;
        let page = match pages.get(index) {
            Some(page) if page.capacity >= chunk.len() => page,
            _ => {
                let capacity = chunk.len().next_power_of_two().min(per_page);
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: bytes_for(capacity) as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &buffer,
                            offset: 0,
                            size: NonZeroU64::new(elem_bytes as u64),
                        }),
                    }],
                });
                let page = OitUniformPage {
                    capacity,
                    buffer,
                    bind_group,
                };
                if let Some(slot) = pages.get_mut(index) {
                    *slot = page;
                    slot
                } else {
                    pages.push(page);
                    let Some(page) = pages.last() else {
                        continue;
                    };
                    page
                }
            }
        };
        let bytes = if let [only] = chunk {
            bytemuck::bytes_of(only)
        } else {
            staging.resize(bytes_for(chunk.len()), 0);
            for (slot, obj) in staging.chunks_mut(stride).zip(chunk) {
                if let Some(payload) = slot.get_mut(..elem_bytes) {
                    payload.copy_from_slice(bytemuck::bytes_of(obj));
                }
            }
            staging.as_slice()
        };
        queue.write_buffer(&page.buffer, 0, bytes);
    }
}

fn storage_oit_suffix_wgsl(mat_fetch: &str) -> String {
    r#"struct OitGlobalUniform {
    xray_opacity: f32,
    flags: u32,
    _pad0: u32,
    _pad1: u32,
};
@group(2) @binding(0)
var<uniform> oit_global: OitGlobalUniform;
struct OitStorageOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) pos_rel: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) @interpolate(flat) chunk_flags: u32,
};
@vertex
fn vs_oit_storage(v: VertIn, @location(3) slot: u32) -> OitStorageOut {
    let t = chunk_transforms[slot];
    let offset_rel = sub_q32_32(t.whole, t.frac, view_origin.cam_whole, view_origin.cam_frac);
    let base = mesh_vertex_with_object(v, ObjectUniforms(offset_rel, t.scale));
    var clip = base.clip_pos;
    if ((t.flags & 1u) == 0u) {
        clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    }
    return OitStorageOut(clip, base.pos_rel, base.normal, base.color, t.flags);
}
@fragment
fn fs_oit_storage(in: OitStorageOut, @builtin(front_facing) is_front: bool) -> OitFragOut {
    __MAT_FETCH__
    let chunk_opacity = unpack_chunk_opacity(in.chunk_flags);
    let chunk_ghost = (in.chunk_flags & 4u) != 0u;
    return evaluate_oit(
        in.clip_pos.z, in.pos_rel, in.normal, in.color, is_front,
        mat, chunk_opacity, chunk_ghost,
        oit_global.xray_opacity, oit_global.flags,
    );
}
"#
    .replace("__MAT_FETCH__", mat_fetch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage_batch::pack_chunk_opacity;

    #[test]
    fn oit_uniform_layouts_are_16_byte_aligned() {
        assert_eq!(size_of::<OitGlobalUniform>(), 16);
        assert_eq!(size_of::<OitDirectUniform>(), 64);
    }

    #[test]
    fn storage_oit_flat_wgsl_marks_oit_storage_out_flat_without_touching_vert_in() {
        let src = format!(
            "{}\n{}",
            include_str!("shaders/mesh3d.wgsl"),
            storage_oit_suffix_wgsl("let mat = oit_default_material();\n")
        );
        for use_either in [false, true] {
            let flat = provoking::make_flat_color_wgsl(&src, use_either);
            let interp = if use_either {
                "@interpolate(flat, either)"
            } else {
                "@interpolate(flat)"
            };
            assert!(
                flat.contains(&format!(
                    "struct OitStorageOut {{\n    @builtin(position) clip_pos: vec4<f32>,\n    @location(0) pos_rel: vec3<f32>,\n    @location(1) {interp} normal: vec3<f32>,\n    @location(2) {interp} color: vec4<f32>,\n    @location(3) {interp} chunk_flags: u32,"
                )),
                "OitStorageOut must get flat normal and color (use_either={use_either})"
            );
            assert!(
                flat.contains(
                    "struct VertIn {\n    @location(0) position: vec3<f32>,\n    @location(1) normal: vec3<f32>,\n    @location(2) color: vec4<f32>,"
                ),
                "VertIn must remain untouched (use_either={use_either})"
            );
        }
    }

    #[test]
    fn chunk_opacity_5bit_packing_preserves_exact_tenths_and_guards_hostile_values() {
        let unpack = |code: u32| -> f32 {
            if code == 0 {
                1.0
            } else {
                (code - 1) as f32 / 30.0
            }
        };
        assert_eq!(pack_chunk_opacity(1.0), 0);
        assert_eq!(unpack(pack_chunk_opacity(1.0)), 1.0);
        assert_eq!(pack_chunk_opacity(f32::NAN), 0);
        assert_eq!(pack_chunk_opacity(f32::INFINITY), 0);
        assert_eq!(unpack(pack_chunk_opacity(0.0)), 0.0);
        assert_eq!(unpack(pack_chunk_opacity(-5.0)), 0.0);
        for tenths in 1..=9 {
            let v = tenths as f32 / 10.0;
            let round = unpack(pack_chunk_opacity(v));
            assert!((round - v).abs() < 1e-6, "v={v} round={round}");
        }
    }
}
