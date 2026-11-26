//! Full-resolution depth-only Ground-Truth Ambient Occlusion (XeGTAO-inspired) with 5-level
//! weighted depth MIP pyramid, edge-symmetrized spatial denoise and 4× MSAA edge-aware composite.
//!
//! Runs inside [`iletki_wgpu::PassHookStage::PostSceneResolved`] directly over the resolved 1×
//! scene color buffer using the already-sampleable Reversed-Z depth target:
//!
//! 1. **Full-res depth prefilter + 5-level weighted MIP pyramid (`R32Float`)**: extracts linear
//!    view-space depth at exact `f32` precision (no `fp16` overflow at large CAD/UTM distances)
//!    and builds a halo-free weighted depth pyramid (`mip 0..4`) so large architectural sampling
//!    radii stay L1-cache friendly without even/odd half-resolution staircase aliasing.
//! 2. **Full-res XeGTAO main pass (`R8Unorm` AO + `R8Unorm` packed edges MRT)**: computes
//!    slope-adjusted 4-neighbor edges, edge-weighted 4-quadrant normals, zero-correlation 4×4
//!    Bayer-Latin stratified + R1 golden-ratio horizon integration across the depth pyramid, and
//!    packs visibility with `1.5×` headroom.
//! 3. **Two-pass full-res edge-chained À-Trous denoise (`R8Unorm` ping-pong)**: stride-1 and
//!    stride-2 `[1, 2, 1] ⊗ [1, 2, 1]` passes gated by chained 1-pixel symmetric edges form an
//!    exact 7×7 B-spline that annihilates the 4×4 stratified pattern on smooth static (non-TAA)
//!    CAD surfaces without bleeding across sharp corners or section-cut lips.
//! 4. **Full-res 4× MSAA-aware multi-bounce composite**: fast single-texel multiply on interior
//!    pixels (~96%) and per-subsample 3×3 depth-matched coverage averaging on MSAA silhouette and
//!    section-cut pixels (~4%), preserving full 4× MSAA edge smoothness on every row.
//!
//! Strictly opt-in: when [`GtaoConfig::enabled`] is `false` (default) or the active backend does
//! not expose sampleable depth (OpenGL), zero GTAO textures, pipelines or passes are allocated.

use bytemuck::{Pod, Zeroable};
use iletki_core::{Camera, Projection, Vec2};
use iletki_wgpu::{PassHookCtx, PrepareCtx, wgpu};
use std::num::NonZeroU64;

const GTAO_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
const GTAO_AO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;
const MAX_DEPTH_MIPS: usize = 5;

/// Configuration for opt-in full-resolution depth-only Ground-Truth Ambient Occlusion (GTAO).
///
/// Disabled by default ([`GtaoConfig::default`]) so default CAD viewports incur zero extra VRAM
/// and zero GPU passes until enabled via [`crate::Scene3DLayer::set_gtao`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GtaoConfig {
    /// Enables full-resolution depth-only GTAO (`false` by default).
    pub enabled: bool,
    /// World-space occlusion sampling radius (`> 0.0`, default `0.5`).
    pub radius: f32,
    /// Occlusion darkening intensity (`0.0..=3.0`, default `1.0`).
    pub intensity: f32,
    /// Occlusion contrast exponent (`0.25..=4.0`, default `1.25`).
    pub power: f32,
    /// Maximum screen-space sampling radius in full-resolution pixels (`8.0..=320.0`, default
    /// `200.0`), sampled through the 5-level depth MIP pyramid to keep texture fetches cache-hot.
    pub max_screen_radius_px: f32,
    /// Azimuthal horizon slices per pixel (`1..=6`, default `3`).
    pub slices: u32,
    /// Horizon march steps per side of each slice (`2..=8`, default `4`).
    pub steps_per_side: u32,
}

impl Default for GtaoConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            radius: 0.5,
            intensity: 1.0,
            power: 1.25,
            max_screen_radius_px: 200.0,
            slices: 3,
            steps_per_side: 4,
        }
    }
}

impl GtaoConfig {
    /// Returns an enabled GTAO configuration with balanced CAD/BIM defaults.
    #[must_use]
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            ..Self::default()
        }
    }

    /// Returns an enabled GTAO configuration with a custom world-space sampling `radius`.
    #[must_use]
    pub fn with_radius(radius: f32) -> Self {
        Self {
            enabled: true,
            radius,
            ..Self::default()
        }
        .sanitized()
    }

    /// Clamps every parameter into a finite, safe operating range.
    #[must_use]
    pub fn sanitized(self) -> Self {
        let radius = if self.radius.is_finite() && self.radius > 0.0 {
            self.radius.clamp(1e-3, 1e6)
        } else {
            0.5
        };
        let intensity = if self.intensity.is_finite() {
            self.intensity.clamp(0.0, 3.0)
        } else {
            1.0
        };
        let power = if self.power.is_finite() {
            self.power.clamp(0.25, 4.0)
        } else {
            1.2
        };
        let max_screen_radius_px = if self.max_screen_radius_px.is_finite() {
            self.max_screen_radius_px.clamp(8.0, 320.0)
        } else {
            200.0
        };
        Self {
            enabled: self.enabled,
            radius,
            intensity,
            power,
            max_screen_radius_px,
            slices: self.slices.clamp(1, 6),
            steps_per_side: self.steps_per_side.clamp(2, 8),
        }
    }
}

/// GPU uniform block shared by all GTAO passes (64 bytes, 16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, Default, PartialEq)]
pub(crate) struct GtaoUniform {
    pub(crate) proj_xy: [f32; 2],
    pub(crate) depth_params: [f32; 2],
    pub(crate) active_px: [u32; 2],
    pub(crate) mip_count: u32,
    pub(crate) msaa_samples: u32,
    pub(crate) radius: f32,
    pub(crate) intensity: f32,
    pub(crate) power: f32,
    pub(crate) max_screen_radius_px: f32,
    pub(crate) slices: u32,
    pub(crate) steps_per_side: u32,
    pub(crate) flags: u32,
    pub(crate) _pad: u32,
}

fn depth_mip_count(size: [u32; 2]) -> u32 {
    let max_dim = size[0].max(size[1]).max(1);
    max_dim
        .ilog2()
        .saturating_add(1)
        .min(MAX_DEPTH_MIPS as u32)
        .max(1)
}

impl GtaoUniform {
    /// Builds GTAO projection and sampling parameters from `camera`, or returns `None` for 2D
    /// cameras and degenerate viewports.
    pub(crate) fn from_camera(
        camera: &Camera,
        logical_size: Vec2,
        active_px: [u32; 2],
        msaa_samples: u32,
        config: GtaoConfig,
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
        let cfg = config.sanitized();
        if !cfg.enabled || cfg.intensity <= 0.0 {
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
        Some(Self {
            proj_xy,
            depth_params,
            active_px,
            mip_count: depth_mip_count(active_px),
            msaa_samples: msaa_samples.max(1),
            radius: cfg.radius,
            intensity: cfg.intensity,
            power: cfg.power,
            max_screen_radius_px: cfg.max_screen_radius_px,
            slices: cfg.slices,
            steps_per_side: cfg.steps_per_side,
            flags,
            _pad: 0,
        })
    }
}

/// Returns whether `device` exposes sampleable depth for `depth_format` and `msaa_samples`.
pub(crate) fn supports_depth_sampling(
    device: &wgpu::Device,
    depth_format: wgpu::TextureFormat,
    msaa_samples: u32,
) -> bool {
    if device.adapter_info().backend == wgpu::Backend::Gl {
        return false;
    }
    let enabled = device.features();
    if !enabled.contains(depth_format.required_features()) {
        return false;
    }
    let depth = depth_format.guaranteed_format_features(enabled);
    let rw_sample = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
    depth.allowed_usages.contains(rw_sample) && depth.flags.sample_count_supported(msaa_samples)
}

/// Returns whether `device` can run full-res depth-only GTAO for `depth_format` and `msaa_samples`.
pub(crate) fn supports_gtao(
    device: &wgpu::Device,
    depth_format: wgpu::TextureFormat,
    msaa_samples: u32,
) -> bool {
    if !supports_depth_sampling(device, depth_format, msaa_samples) {
        return false;
    }
    let enabled = device.features();
    let r32f = GTAO_DEPTH_FORMAT.guaranteed_format_features(enabled);
    let r8 = GTAO_AO_FORMAT.guaranteed_format_features(enabled);
    let rw_sample = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
    r32f.allowed_usages.contains(rw_sample) && r8.allowed_usages.contains(rw_sample)
}

/// Lazily compiled pipelines for the full-resolution XeGTAO pipeline.
pub(crate) struct GtaoPipelines {
    color_format: wgpu::TextureFormat,
    msaa_samples: u32,
    pub(crate) prefilter0_layout: wgpu::BindGroupLayout,
    pub(crate) prefilter_mip_layout: wgpu::BindGroupLayout,
    pub(crate) ao_layout: wgpu::BindGroupLayout,
    pub(crate) denoise_layout: wgpu::BindGroupLayout,
    pub(crate) apply_layout: wgpu::BindGroupLayout,
    pub(crate) prefilter0: wgpu::RenderPipeline,
    pub(crate) prefilter_mip: wgpu::RenderPipeline,
    pub(crate) ao: wgpu::RenderPipeline,
    pub(crate) denoise1: wgpu::RenderPipeline,
    pub(crate) denoise2: wgpu::RenderPipeline,
    pub(crate) apply: wgpu::RenderPipeline,
}

impl GtaoPipelines {
    pub(crate) fn matches<W>(&self, ctx: &PrepareCtx<'_, W>) -> bool {
        self.color_format == ctx.color_format && self.msaa_samples == ctx.msaa_samples
    }

    pub(crate) fn build<W>(ctx: &PrepareCtx<'_, W>) -> Self {
        let device = ctx.device;
        let multisampled = ctx.msaa_samples > 1;
        let uniform_size = NonZeroU64::new(size_of::<GtaoUniform>() as u64);

        let depth_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled,
            },
            count: None,
        };
        let f32_tex_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let uniform_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: uniform_size,
            },
            count: None,
        };

        let prefilter0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("iletki-3d gtao prefilter0 layout"),
            entries: &[depth_entry(0), uniform_entry(1)],
        });
        let prefilter_mip_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("iletki-3d gtao prefilter mip layout"),
                entries: &[f32_tex_entry(0), uniform_entry(1)],
            });
        let ao_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("iletki-3d gtao ao layout"),
            entries: &[f32_tex_entry(0), uniform_entry(1)],
        });
        let denoise_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("iletki-3d gtao denoise layout"),
            entries: &[f32_tex_entry(0), f32_tex_entry(1), uniform_entry(2)],
        });
        let apply_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("iletki-3d gtao apply layout"),
            entries: &[
                depth_entry(0),
                f32_tex_entry(1),
                f32_tex_entry(2),
                uniform_entry(3),
            ],
        });

        let common = include_str!("shaders/gtao_common.wgsl");
        let prefilter0_src = format!("{common}\n{}", prefilter_mip0_wgsl(ctx.msaa_samples));
        let prefilter_mip_src = format!("{common}\n{PREFILTER_MIP_N_WGSL}");
        let ao_src = format!("{common}\n{}", include_str!("shaders/gtao_ao.wgsl"));
        let denoise_src = format!("{common}\n{}", include_str!("shaders/gtao_blur.wgsl"));
        let apply_src = format!("{common}\n{}", apply_wgsl(ctx.msaa_samples));

        let make_fullscreen =
            |label: &str,
             layout: &wgpu::BindGroupLayout,
             source: &str,
             fs_entry: &str,
             targets: &[Option<wgpu::ColorTargetState>]| {
                let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
                let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(label),
                    bind_group_layouts: &[Some(layout)],
                    immediate_size: 0,
                });
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pl_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_gtao"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some(fs_entry),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets,
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
                })
            };

        let target_plain = |format| {
            Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })
        };

        let mul_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::Src,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };

        let prefilter0 = make_fullscreen(
            "iletki-3d gtao prefilter mip0",
            &prefilter0_layout,
            &prefilter0_src,
            "fs_gtao_prefilter0",
            &[target_plain(GTAO_DEPTH_FORMAT)],
        );
        let prefilter_mip = make_fullscreen(
            "iletki-3d gtao prefilter mip",
            &prefilter_mip_layout,
            &prefilter_mip_src,
            "fs_gtao_prefilter_mip",
            &[target_plain(GTAO_DEPTH_FORMAT)],
        );
        let ao = make_fullscreen(
            "iletki-3d gtao main",
            &ao_layout,
            &ao_src,
            "fs_gtao_ao",
            &[target_plain(GTAO_AO_FORMAT), target_plain(GTAO_AO_FORMAT)],
        );
        let denoise1 = make_fullscreen(
            "iletki-3d gtao denoise 1",
            &denoise_layout,
            &denoise_src,
            "fs_gtao_denoise_pass1",
            &[target_plain(GTAO_AO_FORMAT)],
        );
        let denoise2 = make_fullscreen(
            "iletki-3d gtao denoise 2",
            &denoise_layout,
            &denoise_src,
            "fs_gtao_denoise_pass2",
            &[target_plain(GTAO_AO_FORMAT)],
        );
        let apply = make_fullscreen(
            "iletki-3d gtao apply",
            &apply_layout,
            &apply_src,
            "fs_gtao_apply",
            &[Some(wgpu::ColorTargetState {
                format: ctx.color_format,
                blend: Some(mul_blend),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        );

        Self {
            color_format: ctx.color_format,
            msaa_samples: ctx.msaa_samples,
            prefilter0_layout,
            prefilter_mip_layout,
            ao_layout,
            denoise_layout,
            apply_layout,
            prefilter0,
            prefilter_mip,
            ao,
            denoise1,
            denoise2,
            apply,
        }
    }
}

fn prefilter_mip0_wgsl(msaa_samples: u32) -> &'static str {
    if msaa_samples == 4 {
        r#"
@group(0) @binding(0)
var src_depth: texture_depth_multisampled_2d;

@group(0) @binding(1)
var<uniform> gtao: GtaoUniform;

@fragment
fn fs_gtao_prefilter0(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let max_c = vec2<i32>(max(i32(gtao.active_px.x) - 1, 0), max(i32(gtao.active_px.y) - 1, 0));
    let c = clamp(vec2<i32>(frag_pos.xy), vec2<i32>(0, 0), max_c);
    let d0 = textureLoad(src_depth, c, 0);
    let d1 = textureLoad(src_depth, c, 1);
    let d2 = textureLoad(src_depth, c, 2);
    let d3 = textureLoad(src_depth, c, 3);
    // Select the 2nd-largest (majority-surface) MSAA depth sample so a single foreground lip
    // subsample never dilates over a 3/4-covered cavity pixel; keep 1-subsample sky silhouettes.
    let a1 = max(d0, d1);
    let b1 = max(d2, d3);
    let mid_hi = min(a1, b1);
    let z = select(mid_hi, max(a1, b1), mid_hi <= 0.0);
    if (z <= 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    return vec4<f32>(gtao_linear_depth(gtao, z), 0.0, 0.0, 1.0);
}
"#
    } else if msaa_samples > 1 {
        r#"
@group(0) @binding(0)
var src_depth: texture_depth_multisampled_2d;

@group(0) @binding(1)
var<uniform> gtao: GtaoUniform;

@fragment
fn fs_gtao_prefilter0(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let max_c = vec2<i32>(max(i32(gtao.active_px.x) - 1, 0), max(i32(gtao.active_px.y) - 1, 0));
    let c = clamp(vec2<i32>(frag_pos.xy), vec2<i32>(0, 0), max_c);
    let samples = max(i32(gtao.msaa_samples), 1);
    var z = textureLoad(src_depth, c, 0);
    for (var s: i32 = 1; s < samples; s = s + 1) {
        z = max(z, textureLoad(src_depth, c, s));
    }
    if (z <= 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    return vec4<f32>(gtao_linear_depth(gtao, z), 0.0, 0.0, 1.0);
}
"#
    } else {
        r#"
@group(0) @binding(0)
var src_depth: texture_depth_2d;

@group(0) @binding(1)
var<uniform> gtao: GtaoUniform;

@fragment
fn fs_gtao_prefilter0(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let max_c = vec2<i32>(max(i32(gtao.active_px.x) - 1, 0), max(i32(gtao.active_px.y) - 1, 0));
    let c = clamp(vec2<i32>(frag_pos.xy), vec2<i32>(0, 0), max_c);
    let z = textureLoad(src_depth, c, 0);
    if (z <= 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    return vec4<f32>(gtao_linear_depth(gtao, z), 0.0, 0.0, 1.0);
}
"#
    }
}

// XeGTAO weighted depth MIP downsample (skips sky pixels <= 0.0 and weights children by proximity
// to the farthest foreground depth within 0.75 * effectRadius to prevent silhouette halos).
const PREFILTER_MIP_N_WGSL: &str = r#"
@group(0) @binding(0)
var src_mip: texture_2d<f32>;

@group(0) @binding(1)
var<uniform> gtao: GtaoUniform;

@fragment
fn fs_gtao_prefilter_mip(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let src_dims = vec2<i32>(textureDimensions(src_mip, 0));
    let max_s = max(src_dims - vec2<i32>(1, 1), vec2<i32>(0, 0));
    let base = vec2<i32>(frag_pos.xy) * 2;

    let d0 = textureLoad(src_mip, clamp(base + vec2<i32>(0, 0), vec2<i32>(0, 0), max_s), 0).r;
    let d1 = textureLoad(src_mip, clamp(base + vec2<i32>(1, 0), vec2<i32>(0, 0), max_s), 0).r;
    let d2 = textureLoad(src_mip, clamp(base + vec2<i32>(0, 1), vec2<i32>(0, 0), max_s), 0).r;
    let d3 = textureLoad(src_mip, clamp(base + vec2<i32>(1, 1), vec2<i32>(0, 0), max_s), 0).r;

    let max_d = max(max(d0, d1), max(d2, d3));
    if (max_d <= 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    let eff_r = max(0.75 * gtao.radius * 1.457, 1e-4);
    let falloff_range = 0.615 * eff_r;
    let falloff_from = eff_r * (1.0 - 0.615);
    let fmul = -1.0 / max(falloff_range, 1e-5);
    let fadd = falloff_from / max(falloff_range, 1e-5) + 1.0;

    let w0 = select(0.0, clamp((max_d - d0) * fmul + fadd, 0.0, 1.0), d0 > 0.0);
    let w1 = select(0.0, clamp((max_d - d1) * fmul + fadd, 0.0, 1.0), d1 > 0.0);
    let w2 = select(0.0, clamp((max_d - d2) * fmul + fadd, 0.0, 1.0), d2 > 0.0);
    let w3 = select(0.0, clamp((max_d - d3) * fmul + fadd, 0.0, 1.0), d3 > 0.0);
    let w_sum = w0 + w1 + w2 + w3;
    let out_d = select(max_d, (w0 * d0 + w1 * d1 + w2 * d2 + w3 * d3) / w_sum, w_sum > 1e-5);
    return vec4<f32>(out_d, 0.0, 0.0, 1.0);
}
"#;

fn apply_wgsl(msaa_samples: u32) -> String {
    let (depth_decl, resolve_body) = if msaa_samples == 4 {
        (
            "var src_depth: texture_depth_multisampled_2d;",
            r#"let d0 = textureLoad(src_depth, c, 0);
    let d1 = textureLoad(src_depth, c, 1);
    let d2 = textureLoad(src_depth, c, 2);
    let d3 = textureLoad(src_depth, c, 3);
    let z_max = max(max(d0, d1), max(d2, d3));
    if (z_max <= 0.0) {
        discard;
    }
    let z_min = min(min(d0, d1), min(d2, d3));
    var factor = 1.0;
    // Fast path (~96% interior pixels): single full-res texel read, zero upsampling.
    if (z_min > 0.0 && (z_max - z_min) <= z_max * 0.003) {
        factor = shape_ao(textureLoad(ao_final_tex, c, 0).r);
    } else {
        // MSAA edge path (~4% silhouettes & cut lips): match each subsample's depth against the
        // full-resolution 3x3 neighborhood so every row (even & odd) gets identical 4x coverage.
        let f0 = select(1.0, eval_msaa_subsample(c, max_c, d0), d0 > 0.0);
        let f1 = select(1.0, eval_msaa_subsample(c, max_c, d1), d1 > 0.0);
        let f2 = select(1.0, eval_msaa_subsample(c, max_c, d2), d2 > 0.0);
        let f3 = select(1.0, eval_msaa_subsample(c, max_c, d3), d3 > 0.0);
        factor = 0.25 * (f0 + f1 + f2 + f3);
    }"#,
        )
    } else if msaa_samples > 1 {
        (
            "var src_depth: texture_depth_multisampled_2d;",
            r#"let samples = max(i32(gtao.msaa_samples), 1);
    var z_max = 0.0;
    var z_min = 2.0;
    for (var s: i32 = 0; s < samples; s = s + 1) {
        let ds = textureLoad(src_depth, c, s);
        z_max = max(z_max, ds);
        z_min = min(z_min, ds);
    }
    if (z_max <= 0.0) {
        discard;
    }
    var factor = 1.0;
    if (z_min > 0.0 && (z_max - z_min) <= z_max * 0.003) {
        factor = shape_ao(textureLoad(ao_final_tex, c, 0).r);
    } else {
        var f_sum = 0.0;
        for (var s: i32 = 0; s < samples; s = s + 1) {
            let ds = textureLoad(src_depth, c, s);
            f_sum = f_sum + select(1.0, eval_msaa_subsample(c, max_c, ds), ds > 0.0);
        }
        factor = f_sum / f32(samples);
    }"#,
        )
    } else {
        (
            "var src_depth: texture_depth_2d;",
            r#"let z_full = textureLoad(src_depth, c, 0);
    if (z_full <= 0.0) {
        discard;
    }
    let factor = shape_ao(textureLoad(ao_final_tex, c, 0).r);"#,
        )
    };

    r#"
@group(0) @binding(0)
__DEPTH_DECL__

@group(0) @binding(1)
var ao_final_tex: texture_2d<f32>;

@group(0) @binding(2)
var depth_mip0_tex: texture_2d<f32>;

@group(0) @binding(3)
var<uniform> gtao: GtaoUniform;

fn shape_ao(raw_ao: f32) -> f32 {
    let clamped = clamp(raw_ao, 0.0, 1.0);
    // Geometric-series multi-bounce approximation (Jimenez 2016): preserves warm/vibrant indirect
    // inter-reflection inside CAD corners instead of crushing lit faces into muddy grey.
    let mb_ao = clamped / max(1.0 - 0.35 * (1.0 - clamped), 1e-4);
    let shaped = pow(mb_ao, gtao.power);
    return clamp(1.0 - (1.0 - shaped) * gtao.intensity, 0.0, 1.0);
}

fn eval_msaa_subsample(c: vec2<i32>, max_c: vec2<i32>, z_sub: f32) -> f32 {
    let d_sub = gtao_linear_depth(gtao, z_sub);
    let is_persp = (gtao.flags & 1u) != 0u;
    let sigma = select(
        max(gtao.radius * 0.08, 1e-4),
        max(min(d_sub * 0.012, gtao.radius * 0.10), 1e-4),
        is_persp,
    );
    let inv_sigma = 1.0 / sigma;

    var ao_sum = 0.0;
    var w_sum = 0.0;
    var best_err = 1e30;
    var best_ao = textureLoad(ao_final_tex, c, 0).r;

    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let nc = clamp(c + vec2<i32>(dx, dy), vec2<i32>(0, 0), max_c);
            let d_n = textureLoad(depth_mip0_tex, nc, 0).r;
            if (d_n <= 0.0) {
                continue;
            }
            let err = abs(d_n - d_sub);
            let ao_n = textureLoad(ao_final_tex, nc, 0).r;
            if (err < best_err) {
                best_err = err;
                best_ao = ao_n;
            }
            let dz = err * inv_sigma;
            let sp = select(select(0.5, 0.25, dx != 0 && dy != 0), 1.0, dx == 0 && dy == 0);
            let w = sp * exp(-dz * dz * 4.0);
            ao_sum = ao_sum + ao_n * w;
            w_sum = w_sum + w;
        }
    }

    let raw = select(best_ao, clamp(ao_sum / w_sum, 0.0, 1.0), w_sum > 1e-5);
    return shape_ao(raw);
}

@fragment
fn fs_gtao_apply(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let max_c = vec2<i32>(
        max(i32(gtao.active_px.x) - 1, 0),
        max(i32(gtao.active_px.y) - 1, 0),
    );
    let c = clamp(vec2<i32>(frag_pos.xy), vec2<i32>(0, 0), max_c);
    __RESOLVE_BODY__
    return vec4<f32>(factor, factor, factor, 1.0);
}
"#
    .replace("__DEPTH_DECL__", depth_decl)
    .replace("__RESOLVE_BODY__", resolve_body)
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct GtaoTargetKey {
    allocated_px: [u32; 2],
    msaa_samples: u32,
    target_generation: u64,
}

/// Per-viewport block-aligned full-resolution XeGTAO targets and cached bind groups.
pub(crate) struct ViewGtaoTargets {
    key: GtaoTargetKey,
    mip_count: u32,
    depth_mip_views: Vec<wgpu::TextureView>,
    depth_mip_bgs: Vec<wgpu::BindGroup>,
    ao_a_view: wgpu::TextureView,
    ao_b_view: wgpu::TextureView,
    edges_view: wgpu::TextureView,
    prefilter0_bg: wgpu::BindGroup,
    ao_bg: wgpu::BindGroup,
    denoise1_bg: wgpu::BindGroup,
    denoise2_bg: wgpu::BindGroup,
    apply_bg: wgpu::BindGroup,
}

impl ViewGtaoTargets {
    pub(crate) fn ensure<'a, W>(
        slot: &'a mut Option<Self>,
        ctx: &PassHookCtx<'_, W>,
        pipelines: &GtaoPipelines,
        uniform_buf: &wgpu::Buffer,
        depth_sample_view: &wgpu::TextureView,
    ) -> &'a Self {
        let key = GtaoTargetKey {
            allocated_px: ctx.allocated_px,
            msaa_samples: ctx.msaa_samples,
            target_generation: ctx.target_generation,
        };
        if slot.as_ref().is_none_or(|t| t.key != key) {
            *slot = Some(Self::new(
                ctx.device,
                pipelines,
                uniform_buf,
                depth_sample_view,
                key,
            ));
        }
        slot.get_or_insert_with(|| {
            Self::new(ctx.device, pipelines, uniform_buf, depth_sample_view, key)
        })
    }

    fn new(
        device: &wgpu::Device,
        pipelines: &GtaoPipelines,
        uniform_buf: &wgpu::Buffer,
        depth_sample_view: &wgpu::TextureView,
        key: GtaoTargetKey,
    ) -> Self {
        let w = key.allocated_px[0].max(1);
        let h = key.allocated_px[1].max(1);
        let extent = wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        };
        let mip_count = depth_mip_count([w, h]);
        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;

        let depth_pyramid = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("iletki-3d gtao depth pyramid"),
            size: extent,
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: GTAO_DEPTH_FORMAT,
            usage,
            view_formats: &[],
        });
        let depth_all_view = depth_pyramid.create_view(&wgpu::TextureViewDescriptor::default());
        let mut depth_mip_views = Vec::with_capacity(mip_count as usize);
        for level in 0..mip_count {
            depth_mip_views.push(depth_pyramid.create_view(&wgpu::TextureViewDescriptor {
                label: Some("iletki-3d gtao depth mip"),
                base_mip_level: level,
                mip_level_count: Some(1),
                ..wgpu::TextureViewDescriptor::default()
            }));
        }
        let mut depth_mip_bgs = Vec::with_capacity(mip_count.saturating_sub(1) as usize);
        for parent in 0..mip_count.saturating_sub(1) as usize {
            if let Some(parent_view) = depth_mip_views.get(parent) {
                depth_mip_bgs.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("iletki-3d gtao prefilter mip bg"),
                    layout: &pipelines.prefilter_mip_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(parent_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: uniform_buf.as_entire_binding(),
                        },
                    ],
                }));
            }
        }

        let make_full = |label: &str, format: wgpu::TextureFormat| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: extent,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };

        let ao_a_view = make_full("iletki-3d gtao ao a", GTAO_AO_FORMAT);
        let ao_b_view = make_full("iletki-3d gtao ao b", GTAO_AO_FORMAT);
        let edges_view = make_full("iletki-3d gtao edges", GTAO_AO_FORMAT);

        let prefilter0_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("iletki-3d gtao prefilter0 bg"),
            layout: &pipelines.prefilter0_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(depth_sample_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });
        let ao_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("iletki-3d gtao main bg"),
            layout: &pipelines.ao_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&depth_all_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });
        let denoise1_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("iletki-3d gtao denoise 1 bg"),
            layout: &pipelines.denoise_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&ao_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&edges_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });
        let denoise2_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("iletki-3d gtao denoise 2 bg"),
            layout: &pipelines.denoise_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&ao_b_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&edges_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });
        let mip0_view = depth_mip_views.first().unwrap_or(&depth_all_view);
        let apply_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("iletki-3d gtao apply bg"),
            layout: &pipelines.apply_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(depth_sample_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&ao_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(mip0_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        Self {
            key,
            mip_count,
            depth_mip_views,
            depth_mip_bgs,
            ao_a_view,
            ao_b_view,
            edges_view,
            prefilter0_bg,
            ao_bg,
            denoise1_bg,
            denoise2_bg,
            apply_bg,
        }
    }

    pub(crate) fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipelines: &GtaoPipelines,
        active_px: [u32; 2],
        resolved_color_view: &wgpu::TextureView,
    ) {
        let [aw, ah] = active_px;
        let run_single = |encoder: &mut wgpu::CommandEncoder,
                          label: &'static str,
                          target: &wgpu::TextureView,
                          vp_px: [u32; 2],
                          pipeline: &wgpu::RenderPipeline,
                          bg: &wgpu::BindGroup| {
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            let gpu_timer = iletki_wgpu::profiling::pass(label);
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            let timestamp_writes = gpu_timer.as_ref().and_then(|t| t.render_writes());
            #[cfg(not(all(feature = "profiling", not(target_arch = "wasm32"))))]
            let timestamp_writes = None;
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_viewport(0.0, 0.0, vp_px[0] as f32, vp_px[1] as f32, 0.0, 1.0);
            pass.set_scissor_rect(0, 0, vp_px[0], vp_px[1]);
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.draw(0..3, 0..1);
        };

        // 1. Full-res linear depth prefilter (mip 0) + weighted MIP chain (mips 1..mip_count).
        if let Some(mip0) = self.depth_mip_views.first() {
            run_single(
                encoder,
                "iletki-3d gtao prefilter mip0",
                mip0,
                [aw, ah],
                &pipelines.prefilter0,
                &self.prefilter0_bg,
            );
        }
        let active_mips = depth_mip_count(active_px).min(self.mip_count);
        for level in 1..active_mips as usize {
            let Some(dst_view) = self.depth_mip_views.get(level) else {
                break;
            };
            let Some(bg) = self.depth_mip_bgs.get(level - 1) else {
                break;
            };
            let mw = (self.key.allocated_px[0] >> level).max(1);
            let mh = (self.key.allocated_px[1] >> level).max(1);
            let vw = aw.div_ceil(1u32 << level).min(mw).max(1);
            let vh = ah.div_ceil(1u32 << level).min(mh).max(1);
            run_single(
                encoder,
                "iletki-3d gtao prefilter mip",
                dst_view,
                [vw, vh],
                &pipelines.prefilter_mip,
                bg,
            );
        }

        // 2. Full-res XeGTAO main pass (MRT: raw AO -> ao_a, packed 4-neighbor edges -> edges).
        {
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            let ao_timer = iletki_wgpu::profiling::pass("iletki-3d gtao horizon");
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            let ao_ts = ao_timer.as_ref().and_then(|t| t.render_writes());
            #[cfg(not(all(feature = "profiling", not(target_arch = "wasm32"))))]
            let ao_ts = None;
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("iletki-3d gtao horizon"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.ao_a_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.edges_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: None,
                timestamp_writes: ao_ts,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_viewport(0.0, 0.0, aw as f32, ah as f32, 0.0, 1.0);
            pass.set_scissor_rect(0, 0, aw, ah);
            pass.set_pipeline(&pipelines.ao);
            pass.set_bind_group(0, &self.ao_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // 3. Two-pass full-res isotropic 3x3 edge-symmetrized XeGTAO denoise (ao_a -> ao_b -> ao_a).
        run_single(
            encoder,
            "iletki-3d gtao denoise 1",
            &self.ao_b_view,
            [aw, ah],
            &pipelines.denoise1,
            &self.denoise1_bg,
        );
        run_single(
            encoder,
            "iletki-3d gtao denoise 2",
            &self.ao_a_view,
            [aw, ah],
            &pipelines.denoise2,
            &self.denoise2_bg,
        );

        // 4. Full-res 4x MSAA edge-aware multi-bounce composite onto resolved_color_view.
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let apply_timer = iletki_wgpu::profiling::pass("iletki-3d gtao apply");
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let apply_ts = apply_timer.as_ref().and_then(|t| t.render_writes());
        #[cfg(not(all(feature = "profiling", not(target_arch = "wasm32"))))]
        let apply_ts = None;
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("iletki-3d gtao apply"),
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
            timestamp_writes: apply_ts,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_viewport(0.0, 0.0, aw as f32, ah as f32, 0.0, 1.0);
        pass.set_scissor_rect(0, 0, aw, ah);
        pass.set_pipeline(&pipelines.apply);
        pass.set_bind_group(0, &self.apply_bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// Per-viewport GTAO uniform and lazy full-res XeGTAO targets.
pub(crate) struct ViewGtaoState {
    pub(crate) active: bool,
    pub(crate) uniform_buf: wgpu::Buffer,
    last_uniform: Option<GtaoUniform>,
    pub(crate) targets: Option<ViewGtaoTargets>,
}

impl ViewGtaoState {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("iletki-3d gtao uniform"),
            size: size_of::<GtaoUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            active: false,
            uniform_buf,
            last_uniform: None,
            targets: None,
        }
    }

    pub(crate) fn update(&mut self, queue: &wgpu::Queue, uniform: GtaoUniform) {
        if self.last_uniform != Some(uniform) {
            queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniform));
            self.last_uniform = Some(uniform);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iletki_core::Vec3;

    #[test]
    fn default_config_is_disabled_and_sanitizes_hostile_inputs() {
        let def = GtaoConfig::default();
        assert!(!def.enabled);
        let hostile = GtaoConfig {
            enabled: true,
            radius: f32::NAN,
            intensity: f32::INFINITY,
            power: -10.0,
            max_screen_radius_px: f32::NEG_INFINITY,
            slices: 99,
            steps_per_side: 0,
        }
        .sanitized();
        assert!(hostile.enabled);
        assert_eq!(hostile.radius, 0.5);
        assert_eq!(hostile.intensity, 1.0);
        assert_eq!(hostile.power, 0.25);
        assert_eq!(hostile.max_screen_radius_px, 200.0);
        assert_eq!(hostile.slices, 6);
        assert_eq!(hostile.steps_per_side, 2);
    }

    #[test]
    fn gtao_uniform_is_16_byte_aligned_and_handles_perspective_ortho_and_2d() {
        assert_eq!(size_of::<GtaoUniform>(), 64);
        let cfg = GtaoConfig::with_radius(0.75);

        let persp = Camera::perspective_3d(
            Vec3::new(0.0, 0.0, 10.0),
            Vec3::ZERO,
            Vec3::Y,
            std::f64::consts::FRAC_PI_2,
            0.25,
        );
        let u_p = GtaoUniform::from_camera(&persp, Vec2::new(800.0, 400.0), [800, 400], 4, cfg)
            .expect("perspective camera produces GTAO uniform");
        assert_eq!(u_p.flags, 1);
        assert!((u_p.proj_xy[1] - 1.0).abs() < 1e-5);
        assert!((u_p.proj_xy[0] - 2.0).abs() < 1e-5);
        assert!((u_p.depth_params[0] - 0.25).abs() < 1e-6);
        assert_eq!(u_p.mip_count, 5);

        let ortho = Camera::ortho_3d(
            Vec3::new(0.0, 0.0, 10.0),
            Vec3::ZERO,
            Vec3::Y,
            6.0,
            0.5,
            40.5,
        );
        let u_o = GtaoUniform::from_camera(&ortho, Vec2::new(300.0, 200.0), [301, 199], 1, cfg)
            .expect("ortho 3D camera produces GTAO uniform");
        assert_eq!(u_o.flags, 2);
        assert!((u_o.proj_xy[1] - 3.0).abs() < 1e-5);
        assert!((u_o.proj_xy[0] - 4.5).abs() < 1e-5);
        assert!((u_o.depth_params[0] - 40.5).abs() < 1e-5);
        assert!((u_o.depth_params[1] - 40.0).abs() < 1e-5);
        assert_eq!(u_o.mip_count, 5);

        let cam_2d = Camera::ortho_2d(Vec2::ZERO, 1.0);
        assert!(
            GtaoUniform::from_camera(&cam_2d, Vec2::new(200.0, 200.0), [200, 200], 4, cfg)
                .is_none()
        );
        assert!(GtaoUniform::from_camera(&persp, Vec2::new(1.0, 1.0), [1, 1], 4, cfg).is_none());
    }
}
