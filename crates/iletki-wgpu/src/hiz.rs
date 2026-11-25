//! Conservative Reversed-Z min-depth Hi-Z pyramid for two-phase GPU occlusion culling.
//!
//! Uses quarter-resolution power-of-two mip 0 dimensions (`ceil(w/4) x ceil(h/4)`, padded with
//! `1.0` neutral identity for `min()` in Reversed-Z `[0.0, 1.0]`). Pass 0 uses a coalesced
//! `16x16` workgroup (one screen pixel per thread, unrolled 4x MSAA loads) reducing three pyramid
//! levels (`4x4`, `2x2`, `1x1`) in LDS in one pass.

use crate::target::DEPTH_FORMAT;
use bytemuck::{Pod, Zeroable};
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};
use wgpu::util::DeviceExt;

static NEXT_HIZ_GENERATION: AtomicU64 = AtomicU64::new(1);

// Shared 16x16 -> 8x8 -> 4x4 (dst0) -> 2x2 (dst1) -> 1x1 (dst2) LDS reduction for Pass 0.
const REDUCE_16X16_QUARTER: &str = r#"
var<workgroup> tile16: array<f32, 256>;

fn reduce_depth_16x16(lid: vec2<u32>, wgid: vec2<u32>, pixel_d: f32) {
    tile16[lid.y * 16u + lid.x] = pixel_d;
    workgroupBarrier();

    if (lid.x < 8u && lid.y < 8u) {
        let sx = lid.x * 2u;
        let sy = lid.y * 2u;
        tile8[lid.y * 8u + lid.x] = min(
            min(tile16[sy * 16u + sx], tile16[sy * 16u + sx + 1u]),
            min(tile16[(sy + 1u) * 16u + sx], tile16[(sy + 1u) * 16u + sx + 1u]),
        );
    }
    workgroupBarrier();

    let write_count = params.dst3_and_meta.z;
    if (lid.x < 4u && lid.y < 4u) {
        let sx = lid.x * 2u;
        let sy = lid.y * 2u;
        let v0 = min(
            min(tile8[sy * 8u + sx], tile8[sy * 8u + sx + 1u]),
            min(tile8[(sy + 1u) * 8u + sx], tile8[(sy + 1u) * 8u + sx + 1u]),
        );
        let out0 = wgid * 4u + lid;
        let dst0 = params.src_and_dst0.zw;
        let in0 = out0.x < dst0.x && out0.y < dst0.y;
        if (in0) {
            textureStore(dst0_tex, vec2<i32>(out0), vec4<f32>(v0, 0.0, 0.0, 0.0));
        }
        tile4[lid.y * 4u + lid.x] = select(1.0, v0, in0);
    }
    workgroupBarrier();

    if (lid.x < 2u && lid.y < 2u) {
        let sx = lid.x * 2u;
        let sy = lid.y * 2u;
        let v1 = min(
            min(tile4[sy * 4u + sx], tile4[sy * 4u + sx + 1u]),
            min(tile4[(sy + 1u) * 4u + sx], tile4[(sy + 1u) * 4u + sx + 1u]),
        );
        let out1 = wgid * 2u + lid;
        let dst1 = params.dst1_and_dst2.xy;
        let in1 = out1.x < dst1.x && out1.y < dst1.y;
        if (write_count >= 2u && in1) {
            textureStore(dst1_tex, vec2<i32>(out1), vec4<f32>(v1, 0.0, 0.0, 0.0));
        }
        tile2[lid.y * 2u + lid.x] = select(1.0, v1, in1);
    }
    workgroupBarrier();

    if (lid.x == 0u && lid.y == 0u && write_count >= 3u) {
        let v2 = min(min(tile2[0], tile2[1]), min(tile2[2], tile2[3]));
        let dst2 = params.dst1_and_dst2.zw;
        if (wgid.x < dst2.x && wgid.y < dst2.y) {
            textureStore(dst2_tex, vec2<i32>(wgid), vec4<f32>(v2, 0.0, 0.0, 0.0));
        }
    }
}
"#;

const EARLY_OUT_QUARTER: &str = r#"
    let src = params.src_and_dst0.xy;
    if (wgid.x * 16u >= src.x || wgid.y * 16u >= src.y) {
        if (lid.x == 0u && lid.y == 0u && params.dst3_and_meta.z >= 3u) {
            let dst2 = params.dst1_and_dst2.zw;
            if (wgid.x < dst2.x && wgid.y < dst2.y) {
                textureStore(dst2_tex, vec2<i32>(wgid.xy), vec4<f32>(1.0, 0.0, 0.0, 0.0));
            }
        }
        return;
    }
"#;

const DEPTH_1X_BODY: &str = r#"
    var d = 1.0;
    if (gid.x < src.x && gid.y < src.y) {
        d = textureLoad(src_depth, vec2<i32>(gid.xy), 0);
    }
    reduce_depth_16x16(lid.xy, wgid.xy, d);
"#;

const DEPTH_4X_BODY: &str = r#"
    var d = 1.0;
    if (gid.x < src.x && gid.y < src.y) {
        let c = vec2<i32>(gid.xy);
        let d0 = textureLoad(src_depth, c, 0);
        let d1 = textureLoad(src_depth, c, 1);
        let d2 = textureLoad(src_depth, c, 2);
        let d3 = textureLoad(src_depth, c, 3);
        d = min(min(d0, d1), min(d2, d3));
    }
    reduce_depth_16x16(lid.xy, wgid.xy, d);
"#;

const DEPTH_MS_BODY: &str = r#"
    let samples = i32(params.dst3_and_meta.w);
    var d = 1.0;
    if (gid.x < src.x && gid.y < src.y) {
        let c = vec2<i32>(gid.xy);
        var m = textureLoad(src_depth, c, 0);
        for (var s: i32 = 1; s < samples; s = s + 1) {
            m = min(m, textureLoad(src_depth, c, s));
        }
        d = m;
    }
    reduce_depth_16x16(lid.xy, wgid.xy, d);
"#;

const DOWNSAMPLE_SUFFIX: &str = r#"
@group(0) @binding(1)
var src_mip: texture_2d<f32>;

@compute @workgroup_size(8, 8, 1)
fn hiz_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let src = params.src_and_dst0.xy;
    let dst0 = params.src_and_dst0.zw;
    var d = 1.0;
    if (gid.x < dst0.x && gid.y < dst0.y) {
        let sx0 = gid.x * 2u;
        let sy0 = gid.y * 2u;
        let sx1 = min(sx0 + 1u, src.x - 1u);
        let sy1 = min(sy0 + 1u, src.y - 1u);
        let d00 = textureLoad(src_mip, vec2<i32>(i32(sx0), i32(sy0)), 0).r;
        let d10 = textureLoad(src_mip, vec2<i32>(i32(sx1), i32(sy0)), 0).r;
        let d01 = textureLoad(src_mip, vec2<i32>(i32(sx0), i32(sy1)), 0).r;
        let d11 = textureLoad(src_mip, vec2<i32>(i32(sx1), i32(sy1)), 0).r;
        d = min(min(d00, d10), min(d01, d11));
    }
    reduce_and_store_8x8(gid.xy, lid.xy, wgid.xy, d);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HizParams {
    src_and_dst0: [u32; 4],
    dst1_and_dst2: [u32; 4],
    dst3_and_meta: [u32; 4],
}

/// Returns whether `device` supports building a compute Reversed-Z Hi-Z pyramid from the viewport
/// depth target at `msaa_samples`.
pub(crate) fn supports_hiz(device: &wgpu::Device, msaa_samples: u32) -> bool {
    if device.adapter_info().backend == wgpu::Backend::Gl {
        return false;
    }
    let limits = device.limits();
    if limits.max_compute_workgroups_per_dimension < 256
        || limits.max_compute_invocations_per_workgroup < 256
        || limits.max_compute_workgroup_size_x < 16
        || limits.max_compute_workgroup_size_y < 16
        || limits.max_storage_textures_per_shader_stage < 4
    {
        return false;
    }
    let enabled = device.features();
    let r32 = wgpu::TextureFormat::R32Float.guaranteed_format_features(enabled);
    let depth = DEPTH_FORMAT.guaranteed_format_features(enabled);
    r32.allowed_usages
        .contains(wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING)
        && depth
            .allowed_usages
            .contains(wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING)
        && depth.flags.sample_count_supported(msaa_samples)
}

struct HizDispatch {
    is_depth: bool,
    bind_group: wgpu::BindGroup,
    workgroups: [u32; 2],
}

/// Per-viewport cached Hi-Z pyramid texture and pre-bound multi-mip compute dispatches.
pub(crate) struct HiZPyramid {
    generation: u64,
    active_px: [u32; 2],
    msaa_samples: u32,
    target_generation: u64,
    size: [u32; 2],
    mip_count: u32,
    full_view: wgpu::TextureView,
    depth_pipeline: wgpu::ComputePipeline,
    downsample_pipeline: wgpu::ComputePipeline,
    passes: Vec<HizDispatch>,
}

impl HiZPyramid {
    pub(crate) fn ensure<'a>(
        slot: &'a mut Option<Self>,
        device: &wgpu::Device,
        depth_view: &wgpu::TextureView,
        active_px: [u32; 2],
        msaa_samples: u32,
        target_generation: u64,
    ) -> &'a Self {
        let needs_rebuild = slot.as_ref().is_none_or(|hiz| {
            hiz.active_px != active_px
                || hiz.msaa_samples != msaa_samples
                || hiz.target_generation != target_generation
        });
        if needs_rebuild {
            *slot = Some(Self::new(
                device,
                depth_view,
                active_px,
                msaa_samples,
                target_generation,
            ));
        }
        slot.get_or_insert_with(|| {
            Self::new(
                device,
                depth_view,
                active_px,
                msaa_samples,
                target_generation,
            )
        })
    }

    fn new(
        device: &wgpu::Device,
        depth_view: &wgpu::TextureView,
        active_px: [u32; 2],
        msaa_samples: u32,
        target_generation: u64,
    ) -> Self {
        let max_dim = device.limits().max_texture_dimension_2d.max(1);
        let aw = active_px[0].max(1).min(max_dim);
        let ah = active_px[1].max(1).min(max_dim);
        let w0 = aw.div_ceil(4).max(1).next_power_of_two().min(max_dim);
        let h0 = ah.div_ceil(4).max(1).next_power_of_two().min(max_dim);
        let mip_count = w0.max(h0).trailing_zeros().saturating_add(1);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("iletki hiz pyramid"),
            size: wgpu::Extent3d {
                width: w0,
                height: h0,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let full_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("iletki hiz full view"),
            base_mip_level: 0,
            mip_level_count: Some(mip_count),
            ..wgpu::TextureViewDescriptor::default()
        });
        let mip_views: Vec<wgpu::TextureView> = (0..mip_count)
            .map(|mip| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("iletki hiz mip view"),
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    ..wgpu::TextureViewDescriptor::default()
                })
            })
            .collect();

        // Three tiny 1x1 dummy storage targets so unused slots in the final pass never alias.
        let [dummy0, dummy1, dummy2]: [wgpu::TextureView; 3] = std::array::from_fn(|_| {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("iletki hiz dummy mip"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });
            tex.create_view(&wgpu::TextureViewDescriptor::default())
        });

        let multisampled = msaa_samples > 1;
        let depth_layout = create_layout(
            device,
            "iletki hiz depth layout",
            wgpu::TextureSampleType::Depth,
            multisampled,
        );
        let downsample_layout = create_layout(
            device,
            "iletki hiz downsample layout",
            wgpu::TextureSampleType::Float { filterable: false },
            false,
        );
        let depth_suffix = build_depth_suffix(msaa_samples);
        let depth_pipeline = create_pipeline(
            device,
            "iletki hiz depth pipeline",
            &depth_layout,
            &depth_suffix,
        );
        let downsample_pipeline = create_pipeline(
            device,
            "iletki hiz downsample pipeline",
            &downsample_layout,
            DOWNSAMPLE_SUFFIX,
        );

        let mip_dims = |mip: u32| -> [u32; 2] { [(w0 >> mip).max(1), (h0 >> mip).max(1)] };

        let mut passes = Vec::new();
        let mut base_mip = 0u32;
        while base_mip < mip_count {
            let is_depth = base_mip == 0;
            let step_mips = if is_depth { 3 } else { 4 };
            let write_count = (mip_count - base_mip).min(step_mips);
            let src_size = if is_depth {
                [aw, ah]
            } else {
                mip_dims(base_mip - 1)
            };
            let dst0 = mip_dims(base_mip);
            let dst1 = if write_count >= 2 {
                mip_dims(base_mip + 1)
            } else {
                [1, 1]
            };
            let dst2 = if write_count >= 3 {
                mip_dims(base_mip + 2)
            } else {
                [1, 1]
            };
            let dst3 = if write_count >= 4 {
                mip_dims(base_mip + 3)
            } else {
                [1, 1]
            };
            let params = HizParams {
                src_and_dst0: [src_size[0], src_size[1], dst0[0], dst0[1]],
                dst1_and_dst2: [dst1[0], dst1[1], dst2[0], dst2[1]],
                dst3_and_meta: [dst3[0], dst3[1], write_count, msaa_samples],
            };
            let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("iletki hiz params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let src_view = if is_depth {
                depth_view
            } else if let Some(prev) = mip_views.get((base_mip - 1) as usize) {
                prev
            } else {
                break;
            };
            let Some(v0) = mip_views.get(base_mip as usize) else {
                break;
            };
            let v1 = mip_views.get((base_mip + 1) as usize).unwrap_or(&dummy0);
            let v2 = mip_views.get((base_mip + 2) as usize).unwrap_or(&dummy1);
            let v3 = if write_count >= 4 {
                mip_views.get((base_mip + 3) as usize).unwrap_or(&dummy2)
            } else {
                &dummy2
            };
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("iletki hiz pass bind group"),
                layout: if is_depth {
                    &depth_layout
                } else {
                    &downsample_layout
                },
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(src_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(v0),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(v1),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(v2),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(v3),
                    },
                ],
            });
            let tile_per_wg = if is_depth { 4 } else { 8 };
            passes.push(HizDispatch {
                is_depth,
                bind_group,
                workgroups: [dst0[0].div_ceil(tile_per_wg), dst0[1].div_ceil(tile_per_wg)],
            });
            base_mip = base_mip.saturating_add(step_mips);
        }

        Self {
            generation: NEXT_HIZ_GENERATION.fetch_add(1, Ordering::Relaxed),
            active_px,
            msaa_samples,
            target_generation,
            size: [w0, h0],
            mip_count,
            full_view,
            depth_pipeline,
            downsample_pipeline,
            passes,
        }
    }

    pub(crate) fn full_view(&self) -> &wgpu::TextureView {
        &self.full_view
    }

    pub(crate) const fn size(&self) -> [u32; 2] {
        self.size
    }

    pub(crate) const fn mip_count(&self) -> u32 {
        self.mip_count
    }

    pub(crate) const fn generation(&self) -> u64 {
        self.generation
    }

    pub(crate) fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        for (idx, dispatch) in self.passes.iter().enumerate() {
            let gpu_timer = if idx == 0 {
                crate::trace::pass("iletki hiz depth")
            } else {
                crate::trace::pass("iletki hiz mip")
            };
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(if dispatch.is_depth {
                    "iletki hiz depth"
                } else {
                    "iletki hiz mip"
                }),
                timestamp_writes: gpu_timer.as_ref().and_then(|t| t.compute_writes()),
            });
            pass.set_pipeline(if dispatch.is_depth {
                &self.depth_pipeline
            } else {
                &self.downsample_pipeline
            });
            pass.set_bind_group(0, &dispatch.bind_group, &[]);
            pass.dispatch_workgroups(dispatch.workgroups[0], dispatch.workgroups[1], 1);
        }
    }
}

fn build_depth_suffix(msaa_samples: u32) -> String {
    let tex_decl = if msaa_samples > 1 {
        "@group(0) @binding(1)\nvar src_depth: texture_depth_multisampled_2d;"
    } else {
        "@group(0) @binding(1)\nvar src_depth: texture_depth_2d;"
    };
    let body = match msaa_samples {
        0 | 1 => DEPTH_1X_BODY,
        4 => DEPTH_4X_BODY,
        _ => DEPTH_MS_BODY,
    };
    format!(
        "{tex_decl}\n{REDUCE_16X16_QUARTER}\n@compute @workgroup_size(16, 16, 1)\nfn hiz_main(\n    @builtin(global_invocation_id) gid: vec3<u32>,\n    @builtin(local_invocation_id) lid: vec3<u32>,\n    @builtin(workgroup_id) wgid: vec3<u32>,\n) {{\n{EARLY_OUT_QUARTER}\n{body}\n}}"
    )
}

fn create_layout(
    device: &wgpu::Device,
    label: &str,
    src_sample_type: wgpu::TextureSampleType,
    src_multisampled: bool,
) -> wgpu::BindGroupLayout {
    let storage_entry = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu::TextureFormat::R32Float,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(size_of::<HizParams>() as u64),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: src_sample_type,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: src_multisampled,
                },
                count: None,
            },
            storage_entry(2),
            storage_entry(3),
            storage_entry(4),
            storage_entry(5),
        ],
    })
}

fn create_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    suffix: &str,
) -> wgpu::ComputePipeline {
    let wgsl = format!("{}\n{suffix}", include_str!("shaders/hiz.wgsl"));
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("hiz_main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}
