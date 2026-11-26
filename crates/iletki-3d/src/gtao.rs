//! Half-resolution depth-only Ground-Truth Ambient Occlusion (GTAO) with bilateral upsample.
//!
//! Runs inside [`iletki_wgpu::PassHookStage::PostSceneResolved`] directly over the resolved 1×
//! scene color buffer using the already-sampleable Reversed-Z depth target:
//!
//! 1. **Half-res depth downsample (`R32Uint` bitcast `f32`)**: resolves MSAA/1× depth at exact
//!    IEEE-754 `f32` precision (no fp16 overflow at large CAD/UTM distances).
//! 2. **Half-res GTAO (`R8Unorm`)**: 4-neighbor minimum-discontinuity normal reconstruction and
//!    normalized horizon-based cosine-weighted visibility integration (supports both infinite
//!    Reversed-Z `Perspective3D` and linear Reversed-Z `Ortho3D`).
//! 3. **Half-res 3×3 bilateral blur (`R8Unorm`)**: depth-aware spatial denoise preserving sharp
//!    CAD edges and creases.
//! 4. **Full-res bilateral upsample + multiplicative composite**: depth-weighted 2×2 upsample
//!    blended directly onto `resolved_color_view` (`Zero` / `Src`), discarding background pixels
//!    so clear-color pixels remain bit-for-bit untouched.
//!
//! Strictly opt-in: when [`GtaoConfig::enabled`] is `false` (default) or the active backend does
//! not expose sampleable depth (OpenGL), zero GTAO textures, pipelines or passes are allocated.

use bytemuck::{Pod, Zeroable};
use iletki_core::{Camera, Projection, Vec2};
use iletki_wgpu::{PassHookCtx, PrepareCtx, wgpu};
use std::num::NonZeroU64;

const GTAO_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;
const GTAO_AO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

/// Configuration for opt-in half-resolution depth-only Ground-Truth Ambient Occlusion (GTAO).
///
/// Disabled by default ([`GtaoConfig::default`]) so default CAD viewports incur zero extra VRAM
/// and zero GPU passes until enabled via [`crate::Scene3DLayer::set_gtao`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GtaoConfig {
    /// Enables half-resolution depth-only GTAO (`false` by default).
    pub enabled: bool,
    /// World-space occlusion sampling radius (`> 0.0`, default `0.5`).
    pub radius: f32,
    /// Occlusion darkening intensity (`0.0..=3.0`, default `1.0`).
    pub intensity: f32,
    /// Occlusion contrast exponent (`0.25..=4.0`, default `1.5`).
    pub power: f32,
    /// Maximum screen-space sampling radius in full-resolution pixels (`8.0..=128.0`, default
    /// `48.0`), bounding texture-cache footprint when zoomed close to surfaces.
    pub max_screen_radius_px: f32,
    /// Azimuthal horizon slices per half-resolution pixel (`1..=4`, default `2`).
    pub slices: u32,
    /// Horizon march steps per side of each slice (`2..=6`, default `3`).
    pub steps_per_side: u32,
}

impl Default for GtaoConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            radius: 0.5,
            intensity: 1.0,
            power: 1.5,
            max_screen_radius_px: 48.0,
            slices: 2,
            steps_per_side: 3,
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
            1.5
        };
        let max_screen_radius_px = if self.max_screen_radius_px.is_finite() {
            self.max_screen_radius_px.clamp(8.0, 128.0)
        } else {
            48.0
        };
        Self {
            enabled: self.enabled,
            radius,
            intensity,
            power,
            max_screen_radius_px,
            slices: self.slices.clamp(1, 4),
            steps_per_side: self.steps_per_side.clamp(2, 6),
        }
    }
}

/// GPU uniform block shared by all 4 GTAO passes (64 bytes, 16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, Default, PartialEq)]
pub(crate) struct GtaoUniform {
    pub(crate) proj_xy: [f32; 2],
    pub(crate) depth_params: [f32; 2],
    pub(crate) active_px: [u32; 2],
    pub(crate) half_px: [u32; 2],
    pub(crate) radius: f32,
    pub(crate) intensity: f32,
    pub(crate) power: f32,
    pub(crate) max_half_radius_px: f32,
    pub(crate) slices: u32,
    pub(crate) steps_per_side: u32,
    pub(crate) flags: u32,
    pub(crate) msaa_samples: u32,
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
            half_px: [aw.div_ceil(2).max(1), ah.div_ceil(2).max(1)],
            radius: cfg.radius,
            intensity: cfg.intensity,
            power: cfg.power,
            max_half_radius_px: (cfg.max_screen_radius_px * 0.5).clamp(4.0, 64.0),
            slices: cfg.slices,
            steps_per_side: cfg.steps_per_side,
            flags,
            msaa_samples: msaa_samples.max(1),
        })
    }
}

/// Returns whether `device` can run half-res depth-only GTAO for `depth_format` and `msaa_samples`.
pub(crate) fn supports_gtao(
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
    let r32u = GTAO_DEPTH_FORMAT.guaranteed_format_features(enabled);
    let r8 = GTAO_AO_FORMAT.guaranteed_format_features(enabled);
    let rw_sample = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
    depth.allowed_usages.contains(rw_sample)
        && depth.flags.sample_count_supported(msaa_samples)
        && r32u.allowed_usages.contains(rw_sample)
        && r8.allowed_usages.contains(rw_sample)
}

/// Lazily compiled pipelines for the 4-pass half-resolution GTAO pipeline.
pub(crate) struct GtaoPipelines {
    color_format: wgpu::TextureFormat,
    msaa_samples: u32,
    pub(crate) downsample_layout: wgpu::BindGroupLayout,
    pub(crate) ao_layout: wgpu::BindGroupLayout,
    pub(crate) blur_layout: wgpu::BindGroupLayout,
    pub(crate) upsample_layout: wgpu::BindGroupLayout,
    pub(crate) downsample: wgpu::RenderPipeline,
    pub(crate) ao: wgpu::RenderPipeline,
    pub(crate) blur: wgpu::RenderPipeline,
    pub(crate) upsample: wgpu::RenderPipeline,
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
        let u32_tex_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Uint,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
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

        let downsample_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("iletki-3d gtao downsample layout"),
            entries: &[depth_entry(0), uniform_entry(1)],
        });
        let ao_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("iletki-3d gtao ao layout"),
            entries: &[u32_tex_entry(0), uniform_entry(1)],
        });
        let blur_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("iletki-3d gtao blur layout"),
            entries: &[f32_tex_entry(0), u32_tex_entry(1), uniform_entry(2)],
        });
        let upsample_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("iletki-3d gtao upsample layout"),
            entries: &[
                depth_entry(0),
                f32_tex_entry(1),
                u32_tex_entry(2),
                uniform_entry(3),
            ],
        });

        let common = include_str!("shaders/gtao_common.wgsl");
        let downsample_src = format!("{common}\n{}", downsample_wgsl(ctx.msaa_samples));
        let ao_src = format!("{common}\n{}", include_str!("shaders/gtao_ao.wgsl"));
        let blur_src = format!("{common}\n{}", include_str!("shaders/gtao_blur.wgsl"));
        let upsample_src = format!("{common}\n{}", upsample_wgsl(ctx.msaa_samples));

        let make_fullscreen = |label: &str,
                               layout: &wgpu::BindGroupLayout,
                               source: &str,
                               fs_entry: &str,
                               format: wgpu::TextureFormat,
                               blend: Option<wgpu::BlendState>| {
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
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend,
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

        let downsample = make_fullscreen(
            "iletki-3d gtao downsample",
            &downsample_layout,
            &downsample_src,
            "fs_gtao_downsample",
            GTAO_DEPTH_FORMAT,
            None,
        );
        let ao = make_fullscreen(
            "iletki-3d gtao horizon",
            &ao_layout,
            &ao_src,
            "fs_gtao_ao",
            GTAO_AO_FORMAT,
            None,
        );
        let blur = make_fullscreen(
            "iletki-3d gtao bilateral blur",
            &blur_layout,
            &blur_src,
            "fs_gtao_blur",
            GTAO_AO_FORMAT,
            None,
        );
        let upsample = make_fullscreen(
            "iletki-3d gtao bilateral upsample",
            &upsample_layout,
            &upsample_src,
            "fs_gtao_upsample",
            ctx.color_format,
            Some(mul_blend),
        );

        Self {
            color_format: ctx.color_format,
            msaa_samples: ctx.msaa_samples,
            downsample_layout,
            ao_layout,
            blur_layout,
            upsample_layout,
            downsample,
            ao,
            blur,
            upsample,
        }
    }
}

fn downsample_wgsl(msaa_samples: u32) -> &'static str {
    if msaa_samples == 4 {
        r#"
@group(0) @binding(0)
var src_depth: texture_depth_multisampled_2d;

@group(0) @binding(1)
var<uniform> gtao: GtaoUniform;

@fragment
fn fs_gtao_downsample(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<u32> {
    let hcoord = vec2<i32>(frag_pos.xy);
    let max_full = vec2<i32>(max(i32(gtao.active_px.x) - 1, 0), max(i32(gtao.active_px.y) - 1, 0));
    let c = clamp(hcoord * 2, vec2<i32>(0, 0), max_full);
    let d0 = textureLoad(src_depth, c, 0);
    let d1 = textureLoad(src_depth, c, 1);
    let d2 = textureLoad(src_depth, c, 2);
    let d3 = textureLoad(src_depth, c, 3);
    let d = max(max(d0, d1), max(d2, d3));
    return vec4<u32>(bitcast<u32>(d), 0u, 0u, 0u);
}
"#
    } else if msaa_samples > 1 {
        r#"
@group(0) @binding(0)
var src_depth: texture_depth_multisampled_2d;

@group(0) @binding(1)
var<uniform> gtao: GtaoUniform;

@fragment
fn fs_gtao_downsample(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<u32> {
    let hcoord = vec2<i32>(frag_pos.xy);
    let max_full = vec2<i32>(max(i32(gtao.active_px.x) - 1, 0), max(i32(gtao.active_px.y) - 1, 0));
    let c = clamp(hcoord * 2, vec2<i32>(0, 0), max_full);
    let samples = max(i32(gtao.msaa_samples), 1);
    var d = textureLoad(src_depth, c, 0);
    for (var s: i32 = 1; s < samples; s = s + 1) {
        d = max(d, textureLoad(src_depth, c, s));
    }
    return vec4<u32>(bitcast<u32>(d), 0u, 0u, 0u);
}
"#
    } else {
        r#"
@group(0) @binding(0)
var src_depth: texture_depth_2d;

@group(0) @binding(1)
var<uniform> gtao: GtaoUniform;

@fragment
fn fs_gtao_downsample(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<u32> {
    let hcoord = vec2<i32>(frag_pos.xy);
    let max_full = vec2<i32>(max(i32(gtao.active_px.x) - 1, 0), max(i32(gtao.active_px.y) - 1, 0));
    let c = clamp(hcoord * 2, vec2<i32>(0, 0), max_full);
    let d = textureLoad(src_depth, c, 0);
    return vec4<u32>(bitcast<u32>(d), 0u, 0u, 0u);
}
"#
    }
}

fn upsample_wgsl(msaa_samples: u32) -> String {
    let (depth_decl, depth_load) = if msaa_samples == 4 {
        (
            "var src_depth: texture_depth_multisampled_2d;",
            "let d0 = textureLoad(src_depth, c, 0);\n\
             let d1 = textureLoad(src_depth, c, 1);\n\
             let d2 = textureLoad(src_depth, c, 2);\n\
             let d3 = textureLoad(src_depth, c, 3);\n\
             let z_full = max(max(d0, d1), max(d2, d3));",
        )
    } else if msaa_samples > 1 {
        (
            "var src_depth: texture_depth_multisampled_2d;",
            "let samples = max(i32(gtao.msaa_samples), 1);\n\
             var z_full = textureLoad(src_depth, c, 0);\n\
             for (var s: i32 = 1; s < samples; s = s + 1) {\n\
                 z_full = max(z_full, textureLoad(src_depth, c, s));\n\
             }",
        )
    } else {
        (
            "var src_depth: texture_depth_2d;",
            "let z_full = textureLoad(src_depth, c, 0);",
        )
    };
    r#"
@group(0) @binding(0)
__DEPTH_DECL__

@group(0) @binding(1)
var half_ao_tex: texture_2d<f32>;

@group(0) @binding(2)
var half_depth_tex: texture_2d<u32>;

@group(0) @binding(3)
var<uniform> gtao: GtaoUniform;

@fragment
fn fs_gtao_upsample(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let c = vec2<i32>(frag_pos.xy);
    __DEPTH_LOAD__
    if (z_full <= 0.0) {
        discard;
    }

    let d_full = gtao_linear_depth(gtao, z_full);
    let is_persp = (gtao.flags & 1u) != 0u;
    let depth_sigma = select(
        max(gtao.radius * 0.25, 1e-4),
        max(d_full * 0.02, 1e-4),
        is_persp,
    );
    let inv_sigma = 1.0 / depth_sigma;

    let max_h = vec2<i32>(max(i32(gtao.half_px.x) - 1, 0), max(i32(gtao.half_px.y) - 1, 0));
    let hf = (vec2<f32>(frag_pos.xy) - vec2<f32>(0.5)) * 0.5;
    let base = vec2<i32>(floor(hf));
    let frac = clamp(hf - vec2<f32>(base), vec2<f32>(0.0), vec2<f32>(1.0));

    let c00 = clamp(base, vec2<i32>(0, 0), max_h);
    let c10 = clamp(base + vec2<i32>(1, 0), vec2<i32>(0, 0), max_h);
    let c01 = clamp(base + vec2<i32>(0, 1), vec2<i32>(0, 0), max_h);
    let c11 = clamp(base + vec2<i32>(1, 1), vec2<i32>(0, 0), max_h);

    let coords = array<vec2<i32>, 4>(c00, c10, c01, c11);
    let bweights = array<f32, 4>(
        (1.0 - frac.x) * (1.0 - frac.y),
        frac.x * (1.0 - frac.y),
        (1.0 - frac.x) * frac.y,
        frac.x * frac.y,
    );

    var ao_sum = 0.0;
    var w_sum = 0.0;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let hc = coords[i];
        let z_h = bitcast<f32>(textureLoad(half_depth_tex, hc, 0).r);
        if (z_h <= 0.0) {
            continue;
        }
        let d_h = gtao_linear_depth(gtao, z_h);
        let dz = abs(d_h - d_full) * inv_sigma;
        let w = (bweights[i] + 1e-4) / (1.0 + dz * dz * 6.0);
        ao_sum = ao_sum + textureLoad(half_ao_tex, hc, 0).r * w;
        w_sum = w_sum + w;
    }

    let raw_ao = select(1.0, clamp(ao_sum / w_sum, 0.0, 1.0), w_sum > 1e-6);
    let shaped = pow(raw_ao, gtao.power);
    let factor = clamp(1.0 - (1.0 - shaped) * gtao.intensity, 0.0, 1.0);
    return vec4<f32>(factor, factor, factor, 1.0);
}
"#
    .replace("__DEPTH_DECL__", depth_decl)
    .replace("__DEPTH_LOAD__", depth_load)
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct GtaoTargetKey {
    allocated_px: [u32; 2],
    msaa_samples: u32,
    target_generation: u64,
}

/// Per-viewport block-aligned half-res GTAO targets and cached bind groups.
pub(crate) struct ViewGtaoTargets {
    key: GtaoTargetKey,
    half_depth_view: wgpu::TextureView,
    half_ao_raw_view: wgpu::TextureView,
    half_ao_blur_view: wgpu::TextureView,
    downsample_bg: wgpu::BindGroup,
    ao_bg: wgpu::BindGroup,
    blur_bg: wgpu::BindGroup,
    upsample_bg: wgpu::BindGroup,
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
        let half_w = key.allocated_px[0].div_ceil(2).max(1);
        let half_h = key.allocated_px[1].div_ceil(2).max(1);
        let half_extent = wgpu::Extent3d {
            width: half_w,
            height: half_h,
            depth_or_array_layers: 1,
        };
        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;

        let make_view = |label: &str, format: wgpu::TextureFormat| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: half_extent,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };

        let half_depth_view = make_view("iletki-3d gtao half depth", GTAO_DEPTH_FORMAT);
        let half_ao_raw_view = make_view("iletki-3d gtao half ao raw", GTAO_AO_FORMAT);
        let half_ao_blur_view = make_view("iletki-3d gtao half ao blur", GTAO_AO_FORMAT);

        let downsample_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("iletki-3d gtao downsample bg"),
            layout: &pipelines.downsample_layout,
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
            label: Some("iletki-3d gtao ao bg"),
            layout: &pipelines.ao_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&half_depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });
        let blur_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("iletki-3d gtao blur bg"),
            layout: &pipelines.blur_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&half_ao_raw_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&half_depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });
        let upsample_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("iletki-3d gtao upsample bg"),
            layout: &pipelines.upsample_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(depth_sample_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&half_ao_blur_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&half_depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        Self {
            key,
            half_depth_view,
            half_ao_raw_view,
            half_ao_blur_view,
            downsample_bg,
            ao_bg,
            blur_bg,
            upsample_bg,
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
        let hw = aw.div_ceil(2).max(1);
        let hh = ah.div_ceil(2).max(1);

        let run_half = |encoder: &mut wgpu::CommandEncoder,
                        label: &'static str,
                        target: &wgpu::TextureView,
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
            pass.set_viewport(0.0, 0.0, hw as f32, hh as f32, 0.0, 1.0);
            pass.set_scissor_rect(0, 0, hw, hh);
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.draw(0..3, 0..1);
        };

        run_half(
            encoder,
            "iletki-3d gtao downsample",
            &self.half_depth_view,
            &pipelines.downsample,
            &self.downsample_bg,
        );
        run_half(
            encoder,
            "iletki-3d gtao horizon",
            &self.half_ao_raw_view,
            &pipelines.ao,
            &self.ao_bg,
        );
        run_half(
            encoder,
            "iletki-3d gtao bilateral blur",
            &self.half_ao_blur_view,
            &pipelines.blur,
            &self.blur_bg,
        );

        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let upsample_timer = iletki_wgpu::profiling::pass("iletki-3d gtao bilateral upsample");
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let upsample_ts = upsample_timer.as_ref().and_then(|t| t.render_writes());
        #[cfg(not(all(feature = "profiling", not(target_arch = "wasm32"))))]
        let upsample_ts = None;
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("iletki-3d gtao bilateral upsample"),
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
            timestamp_writes: upsample_ts,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_viewport(0.0, 0.0, aw as f32, ah as f32, 0.0, 1.0);
        pass.set_scissor_rect(0, 0, aw, ah);
        pass.set_pipeline(&pipelines.upsample);
        pass.set_bind_group(0, &self.upsample_bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// Per-viewport GTAO uniform and lazy half-res targets.
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
        assert_eq!(hostile.max_screen_radius_px, 48.0);
        assert_eq!(hostile.slices, 4);
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
        assert_eq!(u_p.half_px, [400, 200]);

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
        assert_eq!(u_o.half_px, [151, 100]);

        let cam_2d = Camera::ortho_2d(Vec2::ZERO, 1.0);
        assert!(
            GtaoUniform::from_camera(&cam_2d, Vec2::new(200.0, 200.0), [200, 200], 4, cfg)
                .is_none()
        );
        assert!(GtaoUniform::from_camera(&persp, Vec2::new(1.0, 1.0), [1, 1], 4, cfg).is_none());
    }
}
