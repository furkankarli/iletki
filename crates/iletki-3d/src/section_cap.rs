//! GPU stencil section caps with solid fill, anti-aliased hatch and cross-hatch patterns.
//!
//! When [`SectionCapConfig::enabled`] is `false` (the default), zero stencil buffers, zero shader
//! modules, zero render pipelines and zero extra render passes are allocated or executed.
//!
//! When enabled and at least one active camera clip plane faces the viewer, each visible clip plane
//! runs a two-pass stencil fill inside [`iletki_wgpu::PassHookStage::PostOpaque`]:
//! 1. **Pass A (Stencil Mark):** clears stencil to `0` and rasterizes all visible opaque geometry
//!    position-only, clipped strictly by plane `i` (never by other planes `j != i`, keeping opposing
//!    parallel slab cuts accurate) with two-sided stencil wrap (`DecrementWrap` front,
//!    `IncrementWrap` back) and `depth_compare = Always`, `depth_write = false`. Uses hardware
//!    `CLIP_DISTANCES` with `fragment: None` when supported, or a zero-color-target discard
//!    fallback otherwise.
//! 2. **Pass B (Cap Quad Fill):** draws a single 6-vertex plane-aligned quad anchored in `f64`
//!    world space, clipped against all other active planes `j != i`, testing `stencil != 0` and
//!    Reversed-Z `depth >= scene_depth` while writing exact cap depth so downstream transparent
//!    layers (such as [`crate::InfiniteGrid3DLayer`]) depth-test accurately against the cut cap.

use crate::face_batch::{PackedDraws, VertexLayout};
use crate::provoking;
use crate::scene::ObjectUniforms;
use bytemuck::{Pod, Zeroable};
use iletki_core::{Bounds3, Camera, MAX_CLIP_PLANES, Plane, Projection, Vec3, Vec4};
use iletki_wgpu::{CAMERA_WGSL, PrepareCtx, wgpu};
use std::num::NonZeroU64;

const OBJECT_BYTES: usize = size_of::<ObjectUniforms>();
const DIRECT_PAGE_BYTES: usize = 256 * 1024;
const DEFAULT_COLOR: [f32; 4] = [0.78, 0.78, 0.76, 1.0];
const DEFAULT_HATCH_COLOR: [f32; 4] = [0.22, 0.24, 0.27, 1.0];
const DEFAULT_HATCH_SPACING: f32 = 0.25;
const DEFAULT_HATCH_WIDTH_PX: f32 = 1.5;
const DEFAULT_HATCH_ANGLE_DEG: f32 = 45.0;

/// Fill pattern rendered on solid cross-sections sliced by active camera clip planes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SectionCapPattern {
    /// Uniform solid fill color.
    #[default]
    Solid,
    /// World-locked anti-aliased parallel CAD hatch lines over the base fill color.
    Hatch,
    /// World-locked anti-aliased perpendicular cross-hatch lines over the base fill color.
    CrossHatch,
}

impl SectionCapPattern {
    #[inline]
    const fn to_u32(self) -> u32 {
        match self {
            Self::Solid => 0,
            Self::Hatch => 1,
            Self::CrossHatch => 2,
        }
    }
}

/// Configuration for GPU stencil section caps on [`crate::Scene3DLayer`].
///
/// Disabled by default (`enabled: false`) so scenes without section caps pay zero VRAM and zero
/// frame-time overhead.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SectionCapConfig {
    /// Enables stencil section capping when camera clip planes are active.
    pub enabled: bool,
    /// Base sRGB fill color `[r, g, b, a]` in `0.0..=1.0`.
    pub color: [f32; 4],
    /// Cap shading pattern (`Solid`, `Hatch` or `CrossHatch`).
    pub pattern: SectionCapPattern,
    /// sRGB hatch line color `[r, g, b, a]` in `0.0..=1.0` (used for `Hatch` and `CrossHatch`).
    pub hatch_color: [f32; 4],
    /// Distance between parallel hatch lines in world units (must be positive and finite).
    pub hatch_spacing: f32,
    /// Logical hatch line width in points (scaled by viewport DPI).
    pub hatch_width_px: f32,
    /// In-plane hatch angle in degrees (default `45.0`).
    pub hatch_angle_deg: f32,
}

impl Default for SectionCapConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            color: DEFAULT_COLOR,
            pattern: SectionCapPattern::Solid,
            hatch_color: DEFAULT_HATCH_COLOR,
            hatch_spacing: DEFAULT_HATCH_SPACING,
            hatch_width_px: DEFAULT_HATCH_WIDTH_PX,
            hatch_angle_deg: DEFAULT_HATCH_ANGLE_DEG,
        }
    }
}

impl SectionCapConfig {
    /// Creates an enabled solid-color section cap configuration.
    #[must_use]
    pub fn solid(color: [f32; 4]) -> Self {
        Self {
            enabled: true,
            color,
            pattern: SectionCapPattern::Solid,
            ..Self::default()
        }
    }

    /// Creates an enabled hatched section cap configuration with parallel CAD hatch lines.
    #[must_use]
    pub fn hatch(color: [f32; 4], hatch_color: [f32; 4], spacing: f32) -> Self {
        Self {
            enabled: true,
            color,
            pattern: SectionCapPattern::Hatch,
            hatch_color,
            hatch_spacing: spacing,
            ..Self::default()
        }
    }

    /// Creates an enabled cross-hatched section cap configuration.
    #[must_use]
    pub fn cross_hatch(color: [f32; 4], hatch_color: [f32; 4], spacing: f32) -> Self {
        Self {
            enabled: true,
            color,
            pattern: SectionCapPattern::CrossHatch,
            hatch_color,
            hatch_spacing: spacing,
            ..Self::default()
        }
    }

    /// Returns a copy with all numeric fields clamped and guarded against NaN/Inf/zero.
    #[must_use]
    pub(crate) fn sanitized(self) -> Self {
        let sanitize_rgba = |[r, g, b, a]: [f32; 4], [fr, fg, fb, fa]: [f32; 4]| -> [f32; 4] {
            let clean = |v: f32, fallback: f32| {
                if v.is_finite() {
                    v.clamp(0.0, 1.0)
                } else {
                    fallback
                }
            };
            [clean(r, fr), clean(g, fg), clean(b, fb), clean(a, fa)]
        };
        let hatch_spacing = if self.hatch_spacing.is_finite() && self.hatch_spacing > 1e-6 {
            self.hatch_spacing
        } else {
            DEFAULT_HATCH_SPACING
        };
        let hatch_width_px = if self.hatch_width_px.is_finite() {
            self.hatch_width_px.clamp(0.5, 64.0)
        } else {
            DEFAULT_HATCH_WIDTH_PX
        };
        let hatch_angle_deg = if self.hatch_angle_deg.is_finite() {
            self.hatch_angle_deg.rem_euclid(360.0)
        } else {
            DEFAULT_HATCH_ANGLE_DEG
        };
        Self {
            enabled: self.enabled,
            color: sanitize_rgba(self.color, DEFAULT_COLOR),
            pattern: self.pattern,
            hatch_color: sanitize_rgba(self.hatch_color, DEFAULT_HATCH_COLOR),
            hatch_spacing,
            hatch_width_px,
            hatch_angle_deg,
        }
    }
}

/// GPU uniform describing a single camera-relative clip plane for Pass A (`@group(2) @binding(0)`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default, Debug, PartialEq)]
pub(crate) struct CapPlaneUniform {
    pub(crate) plane: [f32; 4],
}

/// GPU uniform describing a plane-aligned cap quad and hatch parameters (`@group(1) @binding(0)`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default, Debug, PartialEq)]
pub(crate) struct CapQuadUniform {
    pub(crate) origin_extent: [f32; 4],
    pub(crate) axis_u_inv_spacing: [f32; 4],
    pub(crate) axis_v_width_px: [f32; 4],
    pub(crate) color: [f32; 4],
    pub(crate) hatch_color: [f32; 4],
    pub(crate) hatch_phase: [f32; 4],
    pub(crate) meta: [u32; 4],
}

/// Builds a deterministic orthonormal basis `(u, v)` on a plane with unit normal `normal` such that
/// `u.cross(v) == -normal` (CCW front-facing from the clipped side), rotated in-plane by
/// `angle_rad`.
pub(crate) fn plane_basis(normal: Vec3, angle_rad: f64) -> Option<(Vec3, Vec3)> {
    let n = normal.normalized()?;
    let ref_axis = if n.z.abs() < 0.9 { Vec3::Z } else { Vec3::Y };
    let u0 = ref_axis.cross(n).normalized()?;
    let v0 = u0.cross(n).normalized()?;
    let (sin_a, cos_a) = angle_rad.sin_cos();
    let u = (u0 * cos_a + v0 * sin_a).normalized()?;
    let v = (-u0 * sin_a + v0 * cos_a).normalized()?;
    Some((u, v))
}

/// Returns whether a normalized clip plane's cap face is visible from `camera`.
pub(crate) fn is_cap_plane_visible(camera: &Camera, plane: Plane) -> bool {
    match camera.projection {
        Projection::Ortho2D { .. } => false,
        Projection::Perspective3D { position, .. } => {
            // Kept side is signed_distance >= 0; the viewer sees the cut face when standing on the
            // clipped side (strictly negative signed distance).
            let dist = plane.signed_distance(position);
            dist.is_finite() && dist < -1e-6
        }
        Projection::Ortho3D { .. } => {
            // Parallel orthographic view rays enter the kept half-space iff forward · normal > 0.
            let dot = camera.forward_3d().dot(plane.normal);
            dot.is_finite() && dot > 1e-6
        }
    }
}

/// Computes camera-relative uniforms for every viewer-facing active section clip plane (`0..=6`).
pub(crate) fn compute_visible_cap_planes(
    camera: &Camera,
    scene_bounds: Option<Bounds3>,
    untrusted_bounds: bool,
    dpi_scale: f32,
    config: SectionCapConfig,
) -> Vec<(CapPlaneUniform, CapQuadUniform)> {
    if !config.enabled || camera.clip_planes().is_empty() {
        return Vec::new();
    }
    if matches!(camera.projection, Projection::Ortho2D { .. }) {
        return Vec::new();
    }
    let cfg = config.sanitized();
    let cam_pos = camera.position_3d();
    if !cam_pos.is_finite() {
        return Vec::new();
    }
    let dpi = if dpi_scale.is_finite() && dpi_scale > 0.0 {
        dpi_scale.clamp(0.25, 16.0)
    } else {
        1.0
    };
    let spacing = f64::from(cfg.hatch_spacing);
    let inv_spacing = (1.0 / spacing) as f32;
    let width_px = (cfg.hatch_width_px * dpi).clamp(0.5, 128.0);
    let angle_rad = f64::from(cfg.hatch_angle_deg).to_radians();

    let (anchor_center, base_radius) = match scene_bounds {
        Some(bounds) if !bounds.is_empty() && bounds.min.is_finite() && bounds.max.is_finite() => {
            let radius = (bounds.max - bounds.min).length() * 0.5;
            match bounds.center() {
                Some(center) if center.is_finite() && radius.is_finite() => (center, radius),
                _ => (cam_pos, 1_000.0),
            }
        }
        _ => (cam_pos, 1_000.0),
    };

    let mut result = Vec::with_capacity(camera.clip_planes().len().min(MAX_CLIP_PLANES));
    for (idx, raw_plane) in camera
        .clip_planes()
        .iter()
        .take(MAX_CLIP_PLANES)
        .enumerate()
    {
        let Some(plane) = Plane::from_vec4(Vec4::from_vec3(raw_plane.normal, raw_plane.d)) else {
            continue;
        };
        if !is_cap_plane_visible(camera, plane) {
            continue;
        }
        let Some((u, v)) = plane_basis(plane.normal, angle_rad) else {
            continue;
        };

        // Project scene anchor center orthogonally onto the clip plane in exact f64.
        let dist_anchor = plane.signed_distance(anchor_center);
        let origin_world = anchor_center - plane.normal * dist_anchor;
        let origin_rel_f64 = origin_world - cam_pos;
        if !origin_world.is_finite() || !origin_rel_f64.is_finite() {
            continue;
        }
        let origin_rel = origin_rel_f64.to_f32();
        if !origin_rel.into_iter().all(f32::is_finite) {
            continue;
        }

        // Compute conservative half-extent guaranteed to cover all projected scene geometry.
        let mut extent_f64 = (base_radius * 1.25).max(1.0);
        if untrusted_bounds || scene_bounds.is_none() {
            let cam_dist = origin_rel_f64.length();
            extent_f64 = extent_f64.max((cam_dist * 16.0).max(10_000.0));
        }
        // Keep extent strictly within finite f32 range.
        let extent = extent_f64.min(1e12) as f32;

        // Exact f64 world-locked hatch phase reduced modulo 1.0 before f32 conversion (zero UTM jitter).
        let phase_u = (origin_world.dot(u) / spacing).rem_euclid(1.0) as f32;
        let phase_v = (origin_world.dot(v) / spacing).rem_euclid(1.0) as f32;

        // Pass the original camera-relative plane coefficients matching `iletki_camera.clip_planes[idx]`
        // so Pass A clips bit-identically with the scene color pass.
        let plane_rel = raw_plane.to_camera_relative_f32(cam_pos);
        if !plane_rel.into_iter().all(f32::is_finite) {
            continue;
        }

        let [ox, oy, oz] = origin_rel;
        let plane_uniform = CapPlaneUniform { plane: plane_rel };
        let quad_uniform = CapQuadUniform {
            origin_extent: [ox, oy, oz, extent],
            axis_u_inv_spacing: [u.x as f32, u.y as f32, u.z as f32, inv_spacing],
            axis_v_width_px: [v.x as f32, v.y as f32, v.z as f32, width_px],
            color: cfg.color,
            hatch_color: cfg.hatch_color,
            hatch_phase: [phase_u, phase_v, 0.0, 0.0],
            meta: [idx as u32, cfg.pattern.to_u32(), 0, 0],
        };
        result.push((plane_uniform, quad_uniform));
    }
    result
}

/// Lazily compiled GPU pipelines for two-sided stencil marking and plane-aligned cap quad filling.
///
/// Only constructed when [`SectionCapConfig::enabled`] is `true` and `depth_format` has a stencil
/// aspect (`Depth32FloatStencil8`).
pub(crate) struct SectionCapPipelines {
    color_format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
    msaa_samples: u32,
    has_storage: bool,
    has_packed: bool,
    pub(crate) plane_layout: wgpu::BindGroupLayout,
    pub(crate) quad_layout: wgpu::BindGroupLayout,
    pub(crate) direct_stencil: wgpu::RenderPipeline,
    pub(crate) packed_stencil: Option<wgpu::RenderPipeline>,
    pub(crate) storage_stencil: Option<wgpu::RenderPipeline>,
    pub(crate) storage_repeated_stencil: Option<wgpu::RenderPipeline>,
    pub(crate) cap_quad: wgpu::RenderPipeline,
}

impl SectionCapPipelines {
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
        use_either_flat: bool,
    ) -> Self {
        let use_clip_distances = ctx
            .device
            .features()
            .contains(wgpu::Features::CLIP_DISTANCES);
        let plane_layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("iletki-3d section cap plane layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(size_of::<CapPlaneUniform>() as u64),
                    },
                    count: None,
                }],
            });
        let quad_layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("iletki-3d section cap quad layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(size_of::<CapQuadUniform>() as u64),
                    },
                    count: None,
                }],
            });

        let base_wgsl = provoking::adapt_flat_sampling(
            &format!("{CAMERA_WGSL}\n{}", include_str!("shaders/mesh3d.wgsl")),
            use_either_flat,
        );
        let stencil_prefix = if use_clip_distances {
            "enable clip_distances;\n"
        } else {
            ""
        };
        let stencil_helper = if use_clip_distances {
            include_str!("shaders/section_cap_stencil_raster.wgsl")
        } else {
            include_str!("shaders/section_cap_stencil_discard.wgsl")
        };

        let make_stencil_pipeline =
            |label: &str,
             shader: &wgpu::ShaderModule,
             vs_entry: &str,
             layout: &wgpu::PipelineLayout,
             buffers: &[Option<wgpu::VertexBufferLayout<'_>>]| {
                let fragment = (!use_clip_distances).then_some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some("fs_cap_stencil"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[],
                });
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
                        fragment,
                        primitive: wgpu::PrimitiveState {
                            topology: wgpu::PrimitiveTopology::TriangleList,
                            front_face: wgpu::FrontFace::Ccw,
                            cull_mode: None,
                            ..wgpu::PrimitiveState::default()
                        },
                        depth_stencil: Some(wgpu::DepthStencilState {
                            format: ctx.depth_format,
                            depth_write_enabled: Some(false),
                            depth_compare: Some(wgpu::CompareFunction::Always),
                            stencil: wgpu::StencilState {
                                front: wgpu::StencilFaceState {
                                    compare: wgpu::CompareFunction::Always,
                                    fail_op: wgpu::StencilOperation::Keep,
                                    depth_fail_op: wgpu::StencilOperation::Keep,
                                    pass_op: wgpu::StencilOperation::DecrementWrap,
                                },
                                back: wgpu::StencilFaceState {
                                    compare: wgpu::CompareFunction::Always,
                                    fail_op: wgpu::StencilOperation::Keep,
                                    depth_fail_op: wgpu::StencilOperation::Keep,
                                    pass_op: wgpu::StencilOperation::IncrementWrap,
                                },
                                read_mask: 0xFF,
                                write_mask: 0xFF,
                            },
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

        // Direct stencil pipeline (@group(1) = object_layout, @group(2) = plane_layout).
        let direct_src = format!(
            "{stencil_prefix}{base_wgsl}\n{stencil_helper}\n\
             @vertex\n\
             fn vs_cap_stencil_direct(@location(0) position: vec3<f32>) -> CapStencilOut {{\n\
                 return cap_stencil_vertex(position, object);\n\
             }}\n"
        );
        let direct_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("iletki-3d direct section cap stencil shader"),
                source: wgpu::ShaderSource::Wgsl(direct_src.into()),
            });
        let direct_pipeline_layout =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("iletki-3d direct section cap stencil layout"),
                    bind_group_layouts: &[
                        Some(ctx.camera_bind_group_layout),
                        Some(object_layout),
                        Some(&plane_layout),
                    ],
                    immediate_size: 0,
                });
        let pos_only = [Some(vertex_layout.position())];
        let direct_stencil = make_stencil_pipeline(
            "iletki-3d direct section cap stencil",
            &direct_shader,
            "vs_cap_stencil_direct",
            &direct_pipeline_layout,
            &pos_only,
        );

        let slot_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: size_of::<u32>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![3 => Uint32],
        };
        let slot_instance_layout = wgpu::VertexBufferLayout {
            array_stride: size_of::<u32>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![3 => Uint32],
        };
        let pos_and_slot_vertex = [Some(vertex_layout.position()), Some(slot_vertex_layout)];
        let pos_and_slot_instance = [Some(vertex_layout.position()), Some(slot_instance_layout)];

        // Packed fallback stencil pipeline (@group(1) = packed_layout, @group(2) = plane_layout).
        let packed_stencil = packed_layout.map(|layout| {
            let src = format!(
                "{stencil_prefix}{base_wgsl}\n{}\n{stencil_helper}\n\
                 @vertex\n\
                 fn vs_cap_stencil_batched(@location(0) position: vec3<f32>, @location(3) slot: u32) -> CapStencilOut {{\n\
                     return cap_stencil_vertex(position, page_objects[slot]);\n\
                 }}\n",
                include_str!("shaders/mesh3d_batched.wgsl")
            );
            let shader = ctx
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("iletki-3d packed section cap stencil shader"),
                    source: wgpu::ShaderSource::Wgsl(src.into()),
                });
            let pl_layout = ctx
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("iletki-3d packed section cap stencil layout"),
                    bind_group_layouts: &[
                        Some(ctx.camera_bind_group_layout),
                        Some(layout),
                        Some(&plane_layout),
                    ],
                    immediate_size: 0,
                });
            make_stencil_pipeline(
                "iletki-3d packed section cap stencil",
                &shader,
                "vs_cap_stencil_batched",
                &pl_layout,
                &pos_and_slot_vertex,
            )
        });

        // Storage & repeated-instance stencil pipelines (@group(1) = storage_layout, @group(2) = plane_layout).
        let (storage_stencil, storage_repeated_stencil) = match storage_layout {
            Some(layout) => {
                let src = format!(
                    "{stencil_prefix}{base_wgsl}\n{}\n{stencil_helper}\n\
                     @vertex\n\
                     fn vs_cap_stencil_storage(@location(0) position: vec3<f32>, @location(3) slot: u32) -> CapStencilOut {{\n\
                         let st = load_storage_transform(slot);\n\
                         if (!st.visible) {{\n\
                             return cap_stencil_hidden();\n\
                         }}\n\
                         return cap_stencil_vertex(position, st.placement);\n\
                     }}\n",
                    include_str!("shaders/mesh3d_storage.wgsl")
                );
                let shader = ctx
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("iletki-3d storage section cap stencil shader"),
                        source: wgpu::ShaderSource::Wgsl(src.into()),
                    });
                let pl_layout =
                    ctx.device
                        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: Some("iletki-3d storage section cap stencil layout"),
                            bind_group_layouts: &[
                                Some(ctx.camera_bind_group_layout),
                                Some(layout),
                                Some(&plane_layout),
                            ],
                            immediate_size: 0,
                        });
                let per_vert = make_stencil_pipeline(
                    "iletki-3d storage section cap stencil",
                    &shader,
                    "vs_cap_stencil_storage",
                    &pl_layout,
                    &pos_and_slot_vertex,
                );
                let per_inst = make_stencil_pipeline(
                    "iletki-3d repeated section cap stencil",
                    &shader,
                    "vs_cap_stencil_storage",
                    &pl_layout,
                    &pos_and_slot_instance,
                );
                (Some(per_vert), Some(per_inst))
            }
            None => (None, None),
        };

        // Pass B: Plane-aligned cap quad pipeline.
        let quad_src = format!(
            "{CAMERA_WGSL}\n{}",
            include_str!("shaders/section_cap.wgsl")
        );
        let quad_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("iletki-3d section cap quad shader"),
                source: wgpu::ShaderSource::Wgsl(quad_src.into()),
            });
        let quad_pipeline_layout =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("iletki-3d section cap quad layout"),
                    bind_group_layouts: &[Some(ctx.camera_bind_group_layout), Some(&quad_layout)],
                    immediate_size: 0,
                });
        let cap_quad = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("iletki-3d section cap quad"),
                layout: Some(&quad_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &quad_shader,
                    entry_point: Some("vs_cap_quad"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &quad_shader,
                    entry_point: Some("fs_cap_quad"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: ctx.color_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    ..wgpu::PrimitiveState::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: ctx.depth_format,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                    stencil: wgpu::StencilState {
                        front: wgpu::StencilFaceState {
                            compare: wgpu::CompareFunction::NotEqual,
                            fail_op: wgpu::StencilOperation::Keep,
                            depth_fail_op: wgpu::StencilOperation::Keep,
                            pass_op: wgpu::StencilOperation::Keep,
                        },
                        back: wgpu::StencilFaceState {
                            compare: wgpu::CompareFunction::NotEqual,
                            fail_op: wgpu::StencilOperation::Keep,
                            depth_fail_op: wgpu::StencilOperation::Keep,
                            pass_op: wgpu::StencilOperation::Keep,
                        },
                        read_mask: 0xFF,
                        write_mask: 0x00,
                    },
                    bias: wgpu::DepthBiasState::default(),
                }),
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
            has_packed: packed_layout.is_some(),
            plane_layout,
            quad_layout,
            direct_stencil,
            packed_stencil,
            storage_stencil,
            storage_repeated_stencil,
            cap_quad,
        }
    }
}

struct DirectCapUniformPage {
    capacity: usize,
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

/// Per-viewport GPU uniforms and fallback draw state for section caps.
///
/// Stored per [`iletki_core::ViewportId`] inside `ViewObjects` so multiple viewports sharing one
/// [`crate::Scene3DLayer`] in a single `CommandEncoder` never clobber each other's cap planes.
pub(crate) struct ViewSectionCaps {
    plane_bufs: [wgpu::Buffer; MAX_CLIP_PLANES],
    plane_bgs: [wgpu::BindGroup; MAX_CLIP_PLANES],
    quad_bufs: [wgpu::Buffer; MAX_CLIP_PLANES],
    quad_bgs: [wgpu::BindGroup; MAX_CLIP_PLANES],
    visible_count: usize,
    pub(crate) use_storage: bool,
    pub(crate) direct_chunks: Vec<usize>,
    direct_uniforms: Vec<ObjectUniforms>,
    direct_pages: Vec<DirectCapUniformPage>,
    direct_staging: Vec<u8>,
    direct_stride: usize,
    direct_slots_per_page: usize,
    pub(crate) fallback_packed: PackedDraws,
}

impl ViewSectionCaps {
    pub(crate) fn new(
        device: &wgpu::Device,
        pipelines: &SectionCapPipelines,
        dynamic_uniforms: bool,
    ) -> Self {
        let limits = device.limits();
        let alignment = (limits.min_uniform_buffer_offset_alignment as usize).max(1);
        let direct_stride = OBJECT_BYTES.div_ceil(alignment) * alignment;
        let bytes = usize::try_from(limits.max_buffer_size.min(DIRECT_PAGE_BYTES as u64))
            .unwrap_or(OBJECT_BYTES);
        let direct_slots_per_page = if dynamic_uniforms {
            bytes.saturating_sub(OBJECT_BYTES) / direct_stride + 1
        } else {
            1
        };

        let plane_bufs: [wgpu::Buffer; MAX_CLIP_PLANES] = std::array::from_fn(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("iletki-3d section cap plane uniform"),
                size: size_of::<CapPlaneUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        let plane_bgs: [wgpu::BindGroup; MAX_CLIP_PLANES] = plane_bufs.each_ref().map(|buf| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("iletki-3d section cap plane binding"),
                layout: &pipelines.plane_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            })
        });
        let quad_bufs: [wgpu::Buffer; MAX_CLIP_PLANES] = std::array::from_fn(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("iletki-3d section cap quad uniform"),
                size: size_of::<CapQuadUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        let quad_bgs: [wgpu::BindGroup; MAX_CLIP_PLANES] = quad_bufs.each_ref().map(|buf| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("iletki-3d section cap quad binding"),
                layout: &pipelines.quad_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            })
        });

        Self {
            plane_bufs,
            plane_bgs,
            quad_bufs,
            quad_bgs,
            visible_count: 0,
            use_storage: false,
            direct_chunks: Vec::new(),
            direct_uniforms: Vec::new(),
            direct_pages: Vec::new(),
            direct_staging: Vec::new(),
            direct_stride,
            direct_slots_per_page,
            fallback_packed: PackedDraws::default(),
        }
    }

    pub(crate) fn clear_active(&mut self) {
        self.visible_count = 0;
        self.direct_chunks.clear();
        self.direct_uniforms.clear();
        self.fallback_packed.clear();
    }

    pub(crate) fn is_active(&self) -> bool {
        self.visible_count > 0
    }

    pub(crate) fn visible_count(&self) -> usize {
        self.visible_count
    }

    pub(crate) fn plane_bind_group(&self, slot: usize) -> Option<&wgpu::BindGroup> {
        self.plane_bgs.get(slot)
    }

    pub(crate) fn quad_bind_group(&self, slot: usize) -> Option<&wgpu::BindGroup> {
        self.quad_bgs.get(slot)
    }

    pub(crate) fn upload_planes(
        &mut self,
        queue: &wgpu::Queue,
        planes: &[(CapPlaneUniform, CapQuadUniform)],
    ) {
        let count = planes.len().min(MAX_CLIP_PLANES);
        self.visible_count = count;
        for (slot, (plane_u, quad_u)) in planes.iter().take(count).enumerate() {
            if let Some(buf) = self.plane_bufs.get(slot) {
                queue.write_buffer(buf, 0, bytemuck::bytes_of(plane_u));
            }
            if let Some(buf) = self.quad_bufs.get(slot) {
                queue.write_buffer(buf, 0, bytemuck::bytes_of(quad_u));
            }
        }
    }

    pub(crate) fn clear_draws(&mut self) {
        self.direct_chunks.clear();
        self.direct_uniforms.clear();
        self.fallback_packed.clear();
    }

    pub(crate) fn push_direct(&mut self, chunk_index: usize, uniforms: ObjectUniforms) {
        self.direct_chunks.push(chunk_index);
        self.direct_uniforms.push(uniforms);
    }

    pub(crate) fn upload_direct_uniforms(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        object_layout: &wgpu::BindGroupLayout,
    ) {
        if self.direct_uniforms.is_empty() {
            return;
        }
        let stride = self.direct_stride;
        let slots_per_page = self.direct_slots_per_page.max(1);
        for (index, objects) in self.direct_uniforms.chunks(slots_per_page).enumerate() {
            let bytes_for = |slots: usize| (slots.saturating_sub(1)) * stride + OBJECT_BYTES;
            let page = match self.direct_pages.get(index) {
                Some(page) if page.capacity >= objects.len() => page,
                _ => {
                    let capacity = objects.len().next_power_of_two().min(slots_per_page);
                    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("iletki-3d cap direct object page"),
                        size: bytes_for(capacity) as u64,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("iletki-3d cap direct object binding"),
                        layout: object_layout,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &buffer,
                                offset: 0,
                                size: NonZeroU64::new(OBJECT_BYTES as u64),
                            }),
                        }],
                    });
                    let page = DirectCapUniformPage {
                        capacity,
                        buffer,
                        bind_group,
                    };
                    if let Some(slot) = self.direct_pages.get_mut(index) {
                        *slot = page;
                        slot
                    } else {
                        self.direct_pages.push(page);
                        let Some(page) = self.direct_pages.last() else {
                            continue;
                        };
                        page
                    }
                }
            };
            let bytes = if let [only] = objects {
                bytemuck::bytes_of(only)
            } else {
                self.direct_staging.resize(bytes_for(objects.len()), 0);
                for (slot, obj) in self.direct_staging.chunks_mut(stride).zip(objects) {
                    if let Some(payload) = slot.get_mut(..OBJECT_BYTES) {
                        payload.copy_from_slice(bytemuck::bytes_of(obj));
                    }
                }
                &self.direct_staging
            };
            queue.write_buffer(&page.buffer, 0, bytes);
        }
    }

    pub(crate) fn bind_direct(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        direct_idx: usize,
        dynamic: bool,
    ) -> bool {
        let slots_per_page = self.direct_slots_per_page.max(1);
        let Some(page) = self.direct_pages.get(direct_idx / slots_per_page) else {
            return false;
        };
        let offset = (direct_idx % slots_per_page) * self.direct_stride;
        if dynamic {
            pass.set_bind_group(1, &page.bind_group, &[offset as u32]);
        } else {
            pass.set_bind_group(1, &page.bind_group, &[]);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iletki_core::Vec2;

    #[test]
    fn default_config_is_disabled_and_sanitization_guards_hostile_values() {
        let def = SectionCapConfig::default();
        assert!(!def.enabled);
        assert_eq!(def.pattern, SectionCapPattern::Solid);

        let hostile = SectionCapConfig {
            enabled: true,
            color: [f32::NAN, -2.0, 5.0, f32::INFINITY],
            pattern: SectionCapPattern::CrossHatch,
            hatch_color: [f32::NEG_INFINITY, 0.5, f32::NAN, -0.5],
            hatch_spacing: -1.0,
            hatch_width_px: f32::NAN,
            hatch_angle_deg: f32::INFINITY,
        };
        let clean = hostile.sanitized();
        assert!(clean.enabled);
        assert!(
            clean
                .color
                .iter()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
        );
        assert!(
            clean
                .hatch_color
                .iter()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
        );
        assert!(clean.hatch_spacing > 0.0 && clean.hatch_spacing.is_finite());
        assert!(clean.hatch_width_px >= 0.5 && clean.hatch_width_px <= 64.0);
        assert!(clean.hatch_angle_deg.is_finite());
    }

    #[test]
    fn plane_basis_is_orthonormal_and_faces_clipped_side() {
        let normals = [
            Vec3::X,
            -Vec3::X,
            Vec3::Y,
            -Vec3::Y,
            Vec3::Z,
            -Vec3::Z,
            Vec3::new(1.0, 2.0, 3.0).normalized().unwrap(),
            Vec3::new(-0.3, 0.2, -0.95).normalized().unwrap(),
        ];
        for n in normals {
            for deg in [0.0_f64, 30.0, 45.0, 90.0, 135.0, 270.0] {
                let (u, v) = plane_basis(n, deg.to_radians()).unwrap();
                assert!((u.length() - 1.0).abs() < 1e-12);
                assert!((v.length() - 1.0).abs() < 1e-12);
                assert!(u.dot(v).abs() < 1e-12);
                assert!(u.dot(n).abs() < 1e-12);
                assert!(v.dot(n).abs() < 1e-12);
                // u x v must equal -n so the cap quad faces the clipped side (viewer).
                let cross = u.cross(v);
                assert!((cross + n).length() < 1e-12);
            }
        }
        assert!(plane_basis(Vec3::ZERO, 0.0).is_none());
    }

    #[test]
    fn visibility_handles_perspective_ortho3d_and_ortho2d() {
        // Plane z <= 0 (kept is -Z, clipped is +Z).
        let plane = Plane::from_normal_point(-Vec3::Z, Vec3::ZERO).unwrap();

        let persp_clipped_side =
            Camera::perspective_3d(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y, 0.8, 0.05);
        assert!(is_cap_plane_visible(&persp_clipped_side, plane));

        let persp_kept_side =
            Camera::perspective_3d(Vec3::new(0.0, 0.0, -5.0), Vec3::ZERO, Vec3::Y, 0.8, 0.05);
        assert!(!is_cap_plane_visible(&persp_kept_side, plane));

        let ortho_looking_into_kept = Camera::ortho_3d(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::ZERO,
            Vec3::Y,
            4.0,
            -10.0,
            20.0,
        );
        assert!(is_cap_plane_visible(&ortho_looking_into_kept, plane));

        let ortho_looking_from_kept = Camera::ortho_3d(
            Vec3::new(0.0, 0.0, -5.0),
            Vec3::ZERO,
            Vec3::Y,
            4.0,
            -10.0,
            20.0,
        );
        assert!(!is_cap_plane_visible(&ortho_looking_from_kept, plane));

        let ortho_2d = Camera::ortho_2d(Vec2::ZERO, 1.0);
        assert!(!is_cap_plane_visible(&ortho_2d, plane));
    }

    #[test]
    fn utm_world_locked_hatch_phase_is_invariant_to_camera_motion() {
        let origin = Vec3::new(661_858.0, 4_491_159.0, 120.0);
        let half = Vec3::new(5.0, 5.0, 5.0);
        let bounds = Bounds3::from_corners(origin - half, origin + half);
        let plane = Plane::from_normal_point(-Vec3::Z, origin).unwrap();
        let cfg = SectionCapConfig::hatch([0.9, 0.2, 0.2, 1.0], [0.1, 0.1, 0.1, 1.0], 0.25);

        let mut cam_a = Camera::perspective_3d(
            origin + Vec3::new(0.0, -6.0, 8.0),
            origin,
            Vec3::Z,
            0.75,
            0.05,
        );
        cam_a.set_clip_planes(&[plane]);

        let mut cam_b = Camera::perspective_3d(
            origin + Vec3::new(3.5, 4.2, 9.1),
            origin,
            Vec3::Z,
            0.75,
            0.05,
        );
        cam_b.set_clip_planes(&[plane]);

        let caps_a = compute_visible_cap_planes(&cam_a, Some(bounds), false, 1.0, cfg);
        let caps_b = compute_visible_cap_planes(&cam_b, Some(bounds), false, 1.0, cfg);
        assert_eq!(caps_a.len(), 1);
        assert_eq!(caps_b.len(), 1);
        // Hatch phase depends solely on world anchor + plane, never camera position.
        assert_eq!(caps_a[0].1.hatch_phase, caps_b[0].1.hatch_phase);
        assert!(caps_a[0].1.hatch_phase[0] >= 0.0 && caps_a[0].1.hatch_phase[0] < 1.0);
        assert!(caps_a[0].1.hatch_phase[1] >= 0.0 && caps_a[0].1.hatch_phase[1] < 1.0);
        // Quad origin sits precisely on the camera-relative clip plane.
        let p = caps_a[0].0.plane;
        let o = caps_a[0].1.origin_extent;
        let residual = p[0] * o[0] + p[1] * o[1] + p[2] * o[2] + p[3];
        assert!(residual.abs() < 1e-4, "residual={residual}");
    }
}
