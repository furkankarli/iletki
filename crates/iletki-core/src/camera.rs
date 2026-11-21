//! 2D and 3D cameras in `f64` world space with split-precision 2D and Reversed-Z 3D uniforms.

use crate::math::{Bounds2, Bounds3, Frustum, Mat4, Plane, Quat, Ray, Vec2, Vec3, Vec4, split};

/// Minimum 2D zoom in logical points per world unit.
pub const MIN_ZOOM: f64 = 1e-9;
/// Maximum 2D zoom in logical points per world unit.
pub const MAX_ZOOM: f64 = 1e9;
/// Maximum number of active 3D section clip planes supported in the standard camera uniform block.
pub const MAX_CLIP_PLANES: usize = 6;

/// Projection model of a [`Camera`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Projection {
    /// 2D orthographic view (CAD / GIS / 2D canvas).
    Ortho2D {
        /// World point at the viewport's center.
        center: Vec2,
        /// Logical screen points per world unit.
        zoom: f64,
    },
    /// 3D orthographic view (CAD elevation/plan/axonometric).
    Ortho3D {
        /// World-space camera anchor / position.
        position: Vec3,
        /// Camera orientation (`+X` right, `+Y` up, `-Z` forward).
        rotation: Quat,
        /// Visible world height across the viewport.
        view_height: f64,
        /// Near clip distance (may be negative for orthographic volumes).
        near: f64,
        /// Far clip distance (`far > near`).
        far: f64,
    },
    /// 3D perspective view using infinite Reversed-Z (`Depth32Float`, near = `1.0`, far = `0.0`).
    Perspective3D {
        /// World-space eye position in `f64`.
        position: Vec3,
        /// Camera orientation (`+X` right, `+Y` up, `-Z` forward).
        rotation: Quat,
        /// Vertical field of view in radians.
        fov_y: f64,
        /// Positive near clip plane distance.
        near: f64,
    },
}

impl Default for Projection {
    fn default() -> Self {
        Self::Ortho2D {
            center: Vec2::ZERO,
            zoom: 1.0,
        }
    }
}

/// GPU-ready camera uniform payload matching the standard WGSL `CameraUniforms` layout (`@group(0)`).
///
/// Kept free of `bytemuck` in `iletki-core` so layer 0 stays dependency-free; `iletki-wgpu` wraps
/// or converts this directly into a `Pod` uniform buffer.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraUniformData {
    /// Camera-relative Reversed-Z view-projection matrix (`mat4x4<f32>`).
    pub view_proj_rel: [[f32; 4]; 4],
    /// Inverse of `view_proj_rel` (`mat4x4<f32>`).
    pub inv_view_proj_rel: [[f32; 4]; 4],
    /// High `f32` part of the 2D camera center.
    pub camera_hi: [f32; 2],
    /// Low `f32` remainder of the 2D camera center.
    pub camera_lo: [f32; 2],
    /// 2D world units to clip space on each axis.
    pub scale_2d: [f32; 2],
    /// Clip-space units per logical screen point on each axis.
    pub point_size: [f32; 2],
    /// Physical viewport size in pixels.
    pub viewport_px: [f32; 2],
    /// Logical-to-physical DPI scale factor.
    pub dpi_scale: f32,
    /// Bit 0: `is_perspective`, bit 1: `is_3d`, bit 2: `reversed_z`.
    pub flags: u32,
    /// Up to [`MAX_CLIP_PLANES`] camera-relative section clip planes `[nx, ny, nz, d_rel]`.
    pub clip_planes: [[f32; 4]; MAX_CLIP_PLANES],
    /// Active clip plane count (`0..=6`).
    pub clip_plane_count: u32,
    /// Padding to 16-byte WGSL uniform alignment.
    pub _pad: [u32; 3],
}

/// A 2D or 3D viewport camera with optional 3D section clip planes.
#[derive(Debug, Clone, PartialEq)]
pub struct Camera {
    /// Active projection parameters.
    pub projection: Projection,
    /// Active section clip planes in world space (capped at [`MAX_CLIP_PLANES`]).
    clip_planes: Vec<Plane>,
}

impl Default for Camera {
    fn default() -> Self {
        Self::ortho_2d(Vec2::ZERO, 1.0)
    }
}

impl Camera {
    /// Creates a 2D orthographic camera centered at `center` with `zoom` points per world unit.
    #[must_use]
    pub fn ortho_2d(center: Vec2, zoom: f64) -> Self {
        let center = if center.is_finite() {
            center
        } else {
            Vec2::ZERO
        };
        let zoom = if zoom.is_finite() {
            zoom.clamp(MIN_ZOOM, MAX_ZOOM)
        } else {
            1.0
        };
        Self {
            projection: Projection::Ortho2D { center, zoom },
            clip_planes: Vec::new(),
        }
    }

    /// Creates a 3D perspective camera looking from `eye` toward `target` with `up`.
    #[must_use]
    pub fn perspective_3d(eye: Vec3, target: Vec3, up: Vec3, fov_y_rad: f64, near: f64) -> Self {
        let position = if eye.is_finite() { eye } else { Vec3::Z };
        let forward = if target.is_finite() {
            target - position
        } else {
            -Vec3::Z
        };
        let rotation = Quat::look_direction(forward, up);
        let fov_y = if fov_y_rad.is_finite() {
            fov_y_rad.clamp(1e-3, std::f64::consts::PI - 1e-3)
        } else {
            std::f64::consts::FRAC_PI_3
        };
        let near = if near.is_finite() && near > 1e-6 {
            near
        } else {
            0.05
        };
        Self {
            projection: Projection::Perspective3D {
                position,
                rotation,
                fov_y,
                near,
            },
            clip_planes: Vec::new(),
        }
    }

    /// Creates a 3D orthographic camera looking from `eye` toward `target` with `up`.
    #[must_use]
    pub fn ortho_3d(
        eye: Vec3,
        target: Vec3,
        up: Vec3,
        view_height: f64,
        near: f64,
        far: f64,
    ) -> Self {
        let position = if eye.is_finite() { eye } else { Vec3::Z };
        let forward = if target.is_finite() {
            target - position
        } else {
            -Vec3::Z
        };
        let rotation = Quat::look_direction(forward, up);
        let view_height = if view_height.is_finite() && view_height > 1e-9 {
            view_height
        } else {
            10.0
        };
        let (near, far) = if near.is_finite() && far.is_finite() && far > near {
            (near, far)
        } else {
            (-1000.0, 1000.0)
        };
        Self {
            projection: Projection::Ortho3D {
                position,
                rotation,
                view_height,
                near,
                far,
            },
            clip_planes: Vec::new(),
        }
    }

    /// Returns the active section clip planes.
    #[must_use]
    pub fn clip_planes(&self) -> &[Plane] {
        &self.clip_planes
    }

    /// Replaces the active section clip planes (at most [`MAX_CLIP_PLANES`] are kept).
    pub fn set_clip_planes(&mut self, planes: &[Plane]) {
        self.clip_planes.clear();
        for &plane in planes.iter().take(MAX_CLIP_PLANES) {
            if plane.normal.is_finite() && plane.d.is_finite() {
                self.clip_planes.push(plane);
            }
        }
    }

    /// Clears all section clip planes.
    pub fn clear_clip_planes(&mut self) {
        self.clip_planes.clear();
    }

    /// World-space eye position (`(center.x, center.y, 0)` for 2D cameras).
    #[must_use]
    pub fn position_3d(&self) -> Vec3 {
        match self.projection {
            Projection::Ortho2D { center, .. } => Vec3::new(center.x, center.y, 0.0),
            Projection::Ortho3D { position, .. } | Projection::Perspective3D { position, .. } => {
                position
            }
        }
    }

    /// Camera orientation in 3D space (identity for 2D cameras).
    #[must_use]
    pub fn rotation_3d(&self) -> Quat {
        match self.projection {
            Projection::Ortho2D { .. } => Quat::IDENTITY,
            Projection::Ortho3D { rotation, .. } | Projection::Perspective3D { rotation, .. } => {
                rotation
            }
        }
    }

    /// Unit forward direction in world space (`-Z` in camera space).
    #[must_use]
    pub fn forward_3d(&self) -> Vec3 {
        self.rotation_3d().rotate(-Vec3::Z)
    }

    /// Unit right direction in world space (`+X` in camera space).
    #[must_use]
    pub fn right_3d(&self) -> Vec3 {
        self.rotation_3d().rotate(Vec3::X)
    }

    /// Unit up direction in world space (`+Y` in camera space).
    #[must_use]
    pub fn up_3d(&self) -> Vec3 {
        self.rotation_3d().rotate(Vec3::Y)
    }

    // -------------------------------------------------------------------------
    // 2D operations
    // -------------------------------------------------------------------------

    /// Fits a 2D orthographic camera to show all of `bounds` inside `viewport_size` points with
    /// `margin_pt` points of padding on each side.
    #[must_use]
    pub fn fit_2d(bounds: Bounds2, viewport_size: Vec2, margin_pt: f64) -> Self {
        let Some(center) = bounds.center() else {
            return Self::default();
        };
        let extent = bounds.size();
        let pad = if margin_pt.is_finite() && margin_pt >= 0.0 {
            2.0 * margin_pt
        } else {
            0.0
        };
        let usable = Vec2::new(
            (viewport_size.x - pad).max(1.0),
            (viewport_size.y - pad).max(1.0),
        );
        let zoom = [usable.x / extent.x, usable.y / extent.y]
            .into_iter()
            .filter(|z| z.is_finite() && *z > 0.0)
            .fold(f64::INFINITY, f64::min);
        let zoom = if zoom.is_finite() {
            zoom.clamp(MIN_ZOOM, MAX_ZOOM)
        } else {
            1.0
        };
        Self::ortho_2d(center, zoom)
    }

    /// Converts a viewport-local point (`(0,0)` top-left, `+Y` down) to a 2D world point.
    ///
    /// Returns `None` if this is not an [`Projection::Ortho2D`] camera or inputs are invalid.
    #[must_use]
    pub fn screen_to_world_2d(&self, screen_pos: Vec2, viewport_size: Vec2) -> Option<Vec2> {
        let Projection::Ortho2D { center, zoom } = self.projection else {
            return None;
        };
        if !screen_pos.is_finite() || !viewport_size.is_finite() || zoom <= 0.0 {
            return None;
        }
        let rel = screen_pos - viewport_size * 0.5;
        Some(center + Vec2::new(rel.x, -rel.y) / zoom)
    }

    /// Converts a 2D world point to viewport-local logical points (`(0,0)` top-left, `+Y` down).
    #[must_use]
    pub fn world_to_screen_2d(&self, world_pos: Vec2, viewport_size: Vec2) -> Option<Vec2> {
        let Projection::Ortho2D { center, zoom } = self.projection else {
            return None;
        };
        if !world_pos.is_finite() || !viewport_size.is_finite() {
            return None;
        }
        let d = (world_pos - center) * zoom;
        Some(viewport_size * 0.5 + Vec2::new(d.x, -d.y))
    }

    /// Returns the world-space rectangle visible in a 2D viewport of `viewport_size` points.
    #[must_use]
    pub fn visible_2d(&self, viewport_size: Vec2) -> Option<Bounds2> {
        let a = self.screen_to_world_2d(Vec2::ZERO, viewport_size)?;
        let b = self.screen_to_world_2d(viewport_size, viewport_size)?;
        Some(Bounds2::from_corners(a, b))
    }

    /// Pans a 2D camera by a screen-space drag `delta_pt` (logical points, `+Y` down).
    pub fn pan_2d(&mut self, delta_pt: Vec2) -> bool {
        if !delta_pt.is_finite() {
            return false;
        }
        if let Projection::Ortho2D { center, zoom } = &mut self.projection
            && *zoom > 0.0
        {
            *center -= Vec2::new(delta_pt.x, -delta_pt.y) / *zoom;
            return true;
        }
        false
    }

    /// Zooms a 2D camera by `factor`, keeping the world point under `anchor_pt` fixed.
    pub fn zoom_at_2d(&mut self, anchor_pt: Vec2, viewport_size: Vec2, factor: f64) -> bool {
        if !factor.is_finite() || factor <= 0.0 || !anchor_pt.is_finite() {
            return false;
        }
        let Some(before) = self.screen_to_world_2d(anchor_pt, viewport_size) else {
            return false;
        };
        if let Projection::Ortho2D { zoom, .. } = &mut self.projection {
            *zoom = (*zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
        }
        let Some(after) = self.screen_to_world_2d(anchor_pt, viewport_size) else {
            return false;
        };
        if let Projection::Ortho2D { center, .. } = &mut self.projection {
            *center += before - after;
            return true;
        }
        false
    }

    // -------------------------------------------------------------------------
    // 3D operations (Camera-Relative RTE + Reversed-Z)
    // -------------------------------------------------------------------------

    /// Camera-relative Reversed-Z view-projection matrix in `f64`.
    #[must_use]
    pub fn view_proj_relative(&self, viewport_size: Vec2) -> Mat4 {
        let aspect = if viewport_size.is_finite() && viewport_size.y > 0.0 && viewport_size.x > 0.0
        {
            viewport_size.x / viewport_size.y
        } else {
            1.0
        };
        match self.projection {
            Projection::Ortho2D { zoom, .. } => {
                let half_h = if viewport_size.y > 0.0 && zoom > 0.0 {
                    0.5 * viewport_size.y / zoom
                } else {
                    1.0
                };
                let half_w = half_h * aspect;
                Mat4::orthographic_reverse_rh(half_w, half_h, -1.0, 1.0)
            }
            Projection::Ortho3D {
                rotation,
                view_height,
                near,
                far,
                ..
            } => {
                let half_h = (0.5 * view_height).max(1e-9);
                let half_w = half_h * aspect;
                Mat4::orthographic_reverse_rh(half_w, half_h, near, far)
                    .mul_mat4(Mat4::view_relative(rotation))
            }
            Projection::Perspective3D {
                rotation,
                fov_y,
                near,
                ..
            } => Mat4::perspective_infinite_reverse_rh(fov_y, aspect, near)
                .mul_mat4(Mat4::view_relative(rotation)),
        }
    }

    /// Extracts the camera-relative view frustum for CPU culling against world bounds.
    #[must_use]
    pub fn frustum(&self, viewport_size: Vec2) -> Frustum {
        Frustum::from_view_proj_reverse_z(self.view_proj_relative(viewport_size))
    }

    /// Casts a world-space picking ray through `screen_pos` (viewport-local logical points).
    #[must_use]
    pub fn screen_ray(&self, screen_pos: Vec2, viewport_size: Vec2) -> Option<Ray> {
        if !screen_pos.is_finite()
            || !viewport_size.is_finite()
            || viewport_size.x <= 0.0
            || viewport_size.y <= 0.0
        {
            return None;
        }
        let ndc_x = (2.0 * screen_pos.x / viewport_size.x) - 1.0;
        let ndc_y = 1.0 - (2.0 * screen_pos.y / viewport_size.y);
        let inv_vp = self.view_proj_relative(viewport_size).inverse()?;
        let cam_pos = self.position_3d();

        // Reversed-Z: near is NDC z = 1.0, deeper point is NDC z = 0.5.
        let p_near = inv_vp.mul_vec4(Vec4::new(ndc_x, ndc_y, 1.0, 1.0));
        let p_mid = inv_vp.mul_vec4(Vec4::new(ndc_x, ndc_y, 0.5, 1.0));
        if p_near.w.abs() <= 1e-12 || p_mid.w.abs() <= 1e-12 {
            return None;
        }
        let near_rel = p_near.xyz() / p_near.w;
        let mid_rel = p_mid.xyz() / p_mid.w;
        Ray::new(cam_pos + near_rel, mid_rel - near_rel)
    }

    /// Reconstructs the exact `f64` world-space position from a screen point and a Reversed-Z
    /// depth buffer sample (`depth ∈ (0.0, 1.0]`).
    ///
    /// Returns `None` for background (`depth <= 0.0`) or non-finite inputs.
    #[must_use]
    pub fn unproject_depth(
        &self,
        screen_pos: Vec2,
        depth_reverse_z: f32,
        viewport_size: Vec2,
    ) -> Option<Vec3> {
        let z = f64::from(depth_reverse_z);
        if !z.is_finite()
            || z <= 0.0
            || !screen_pos.is_finite()
            || !viewport_size.is_finite()
            || viewport_size.x <= 0.0
            || viewport_size.y <= 0.0
        {
            return None;
        }
        let ndc_x = (2.0 * screen_pos.x / viewport_size.x) - 1.0;
        let ndc_y = 1.0 - (2.0 * screen_pos.y / viewport_size.y);
        let inv_vp = self.view_proj_relative(viewport_size).inverse()?;
        let clip = inv_vp.mul_vec4(Vec4::new(ndc_x, ndc_y, z, 1.0));
        if !clip.w.is_finite() || clip.w.abs() <= 1e-12 {
            return None;
        }
        Some(self.position_3d() + clip.xyz() / clip.w)
    }

    /// Builds the GPU-ready [`CameraUniformData`] for a viewport of `logical_size` points and
    /// `dpi_scale` physical pixels per point.
    #[must_use]
    pub fn to_uniform_data(&self, logical_size: Vec2, dpi_scale: f32) -> CameraUniformData {
        let dpi = if dpi_scale.is_finite() && dpi_scale > 0.0 {
            dpi_scale
        } else {
            1.0
        };
        let vp = self.view_proj_relative(logical_size);
        let inv_vp = vp.inverse().unwrap_or(Mat4::IDENTITY);
        let (camera_hi, camera_lo, scale_2d, flags) = match self.projection {
            Projection::Ortho2D { center, zoom } => {
                let (hx, lx) = split(center.x);
                let (hy, ly) = split(center.y);
                let sx = if logical_size.x > 0.0 {
                    (2.0 * zoom / logical_size.x) as f32
                } else {
                    0.0
                };
                let sy = if logical_size.y > 0.0 {
                    (2.0 * zoom / logical_size.y) as f32
                } else {
                    0.0
                };
                ([hx, hy], [lx, ly], [sx, sy], 0u32)
            }
            Projection::Ortho3D { .. } => ([0.0, 0.0], [0.0, 0.0], [0.0, 0.0], (1 << 1) | (1 << 2)),
            Projection::Perspective3D { .. } => (
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                (1 << 0) | (1 << 1) | (1 << 2),
            ),
        };
        let point_size = [
            if logical_size.x > 0.0 {
                (2.0 / logical_size.x) as f32
            } else {
                0.0
            },
            if logical_size.y > 0.0 {
                (2.0 / logical_size.y) as f32
            } else {
                0.0
            },
        ];
        let cam_pos = self.position_3d();
        let mut clip_planes = [[0.0f32; 4]; MAX_CLIP_PLANES];
        for (dst, plane) in clip_planes.iter_mut().zip(self.clip_planes.iter()) {
            *dst = plane.to_camera_relative_f32(cam_pos);
        }
        let clip_plane_count = self.clip_planes.len().min(MAX_CLIP_PLANES) as u32;

        CameraUniformData {
            view_proj_rel: vp.to_cols_array_f32(),
            inv_view_proj_rel: inv_vp.to_cols_array_f32(),
            camera_hi,
            camera_lo,
            scale_2d,
            point_size,
            viewport_px: [
                (logical_size.x.max(0.0) as f32) * dpi,
                (logical_size.y.max(0.0) as f32) * dpi,
            ],
            dpi_scale: dpi,
            flags,
            clip_planes,
            clip_plane_count,
            _pad: [0; 3],
        }
    }

    // -------------------------------------------------------------------------
    // 3D framing, standard CAD views & smooth transitions
    // -------------------------------------------------------------------------

    /// Fits an existing 3D camera (`Perspective3D` or `Ortho3D`) to frame all 8 corners of `bounds`
    /// inside `viewport_size` while keeping the current viewing direction.
    ///
    /// `padding` is a fractional margin multiplier (`>= 1.0`, e.g. `1.15` leaves ~15% breathing room
    /// around the object). Returns the new orbit pivot (`bounds.center()`), or `None` if `bounds`
    /// is empty or this camera is `Ortho2D`.
    pub fn fit_3d(&mut self, bounds: Bounds3, viewport_size: Vec2, padding: f64) -> Option<Vec3> {
        let center = bounds.center()?;
        let pad = if padding.is_finite() && padding >= 1.0 {
            padding
        } else {
            1.15
        };
        let aspect = if viewport_size.is_finite() && viewport_size.x > 0.0 && viewport_size.y > 0.0
        {
            viewport_size.x / viewport_size.y
        } else {
            1.0
        };

        let right = self.right_3d();
        let up = self.up_3d();
        let forward = self.forward_3d();

        let corners = [
            Vec3::new(bounds.min.x, bounds.min.y, bounds.min.z),
            Vec3::new(bounds.max.x, bounds.min.y, bounds.min.z),
            Vec3::new(bounds.min.x, bounds.max.y, bounds.min.z),
            Vec3::new(bounds.max.x, bounds.max.y, bounds.min.z),
            Vec3::new(bounds.min.x, bounds.min.y, bounds.max.z),
            Vec3::new(bounds.max.x, bounds.min.y, bounds.max.z),
            Vec3::new(bounds.min.x, bounds.max.y, bounds.max.z),
            Vec3::new(bounds.max.x, bounds.max.y, bounds.max.z),
        ];

        let mut half_x = 0.0f64;
        let mut half_y = 0.0f64;
        let mut half_z = 0.0f64;
        for c in corners {
            let d = c - center;
            half_x = half_x.max(d.dot(right).abs());
            half_y = half_y.max(d.dot(up).abs());
            half_z = half_z.max(d.dot(forward).abs());
        }
        let radius = bounds.size().length() * 0.5;
        let min_extent = radius.max(1e-3);

        match &mut self.projection {
            Projection::Perspective3D {
                position,
                fov_y,
                near,
                ..
            } => {
                let tan_y = (0.5 * *fov_y).tan().max(1e-4);
                let tan_x = (tan_y * aspect).max(1e-4);
                let dist_y = (half_y.max(min_extent * 0.1) * pad) / tan_y;
                let dist_x = (half_x.max(min_extent * 0.1) * pad) / tan_x;
                let dist = dist_y.max(dist_x) + half_z + *near * 2.0;
                *position = center - forward * dist.max(1e-3);
                Some(center)
            }
            Projection::Ortho3D {
                position,
                view_height,
                near,
                far,
                ..
            } => {
                let req_h = (2.0 * half_y).max(2.0 * half_x / aspect).max(1e-3) * pad;
                *view_height = req_h;
                let span = (half_z * 2.0 + req_h * 2.0).max(10.0);
                *near = 0.05;
                *far = span * 4.0;
                *position = center - forward * span;
                Some(center)
            }
            Projection::Ortho2D { .. } => None,
        }
    }

    /// Creates a standard CAD/BIM 3D view (`Top`, `Front`, `Right`, `Isometric`, etc.) framed onto
    /// `bounds` and returns `(camera, pivot)`.
    #[must_use]
    pub fn standard_view_3d(
        view: StandardView,
        bounds: Bounds3,
        viewport_size: Vec2,
        padding: f64,
        perspective: bool,
    ) -> (Self, Vec3) {
        let center = bounds.center().unwrap_or(Vec3::ZERO);
        let (dir_to_eye, up) = view.eye_direction_and_up();
        let eye = center + dir_to_eye * 10.0;
        let mut cam = if perspective {
            Self::perspective_3d(eye, center, up, std::f64::consts::FRAC_PI_4, 0.05)
        } else {
            Self::ortho_3d(eye, center, up, 10.0, 0.05, 1000.0)
        };
        let pivot = cam.fit_3d(bounds, viewport_size, padding).unwrap_or(center);
        (cam, pivot)
    }

    /// Smoothly interpolates between `self` and `other` (`t` in `0..=1`), using SLERP for 3D
    /// rotations. If projections have mismatched variants, snaps to `other` at `t >= 0.5`.
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f64) -> Self {
        let t = if t.is_finite() {
            t.clamp(0.0, 1.0)
        } else {
            1.0
        };
        let projection = match (self.projection, other.projection) {
            (
                Projection::Ortho2D {
                    center: c0,
                    zoom: z0,
                },
                Projection::Ortho2D {
                    center: c1,
                    zoom: z1,
                },
            ) => {
                // Log-space zoom interpolation feels constant-speed visually.
                let zoom = if z0 > 0.0 && z1 > 0.0 {
                    (z0.ln() + (z1.ln() - z0.ln()) * t).exp()
                } else {
                    z0 + (z1 - z0) * t
                };
                Projection::Ortho2D {
                    center: c0.lerp(c1, t),
                    zoom: zoom.clamp(MIN_ZOOM, MAX_ZOOM),
                }
            }
            (
                Projection::Perspective3D {
                    position: p0,
                    rotation: r0,
                    fov_y: f0,
                    near: n0,
                },
                Projection::Perspective3D {
                    position: p1,
                    rotation: r1,
                    fov_y: f1,
                    near: n1,
                },
            ) => Projection::Perspective3D {
                position: p0.lerp(p1, t),
                rotation: r0.slerp(r1, t),
                fov_y: f0 + (f1 - f0) * t,
                near: n0 + (n1 - n0) * t,
            },
            (
                Projection::Ortho3D {
                    position: p0,
                    rotation: r0,
                    view_height: h0,
                    near: n0,
                    far: f0,
                },
                Projection::Ortho3D {
                    position: p1,
                    rotation: r1,
                    view_height: h1,
                    near: n1,
                    far: f1,
                },
            ) => {
                let view_height = if h0 > 0.0 && h1 > 0.0 {
                    (h0.ln() + (h1.ln() - h0.ln()) * t).exp()
                } else {
                    h0 + (h1 - h0) * t
                };
                Projection::Ortho3D {
                    position: p0.lerp(p1, t),
                    rotation: r0.slerp(r1, t),
                    view_height,
                    near: n0 + (n1 - n0) * t,
                    far: f0 + (f1 - f0) * t,
                }
            }
            _ => {
                if t >= 0.5 {
                    other.projection
                } else {
                    self.projection
                }
            }
        };
        Self {
            projection,
            clip_planes: if t >= 0.5 {
                other.clip_planes.clone()
            } else {
                self.clip_planes.clone()
            },
        }
    }
}

/// Standard Z-up CAD/BIM/3D editor camera presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StandardView {
    /// Plan view looking down `-Z` (`+Y` up).
    Top,
    /// Looking up `+Z` (`-Y` up).
    Bottom,
    /// South elevation looking toward `+Y` (`+Z` up).
    Front,
    /// North elevation looking toward `-Y` (`+Z` up).
    Back,
    /// East elevation looking toward `-X` (`+Z` up).
    Right,
    /// West elevation looking toward `+X` (`+Z` up).
    Left,
    /// Standard southeast isometric 3/4 view (`+Z` up).
    Isometric,
}

impl StandardView {
    /// Unit vector from target toward eye and world up reference (Z-up convention).
    #[must_use]
    pub fn eye_direction_and_up(self) -> (Vec3, Vec3) {
        match self {
            Self::Top => (Vec3::Z, Vec3::Y),
            Self::Bottom => (-Vec3::Z, -Vec3::Y),
            Self::Front => (-Vec3::Y, Vec3::Z),
            Self::Back => (Vec3::Y, Vec3::Z),
            Self::Right => (Vec3::X, Vec3::Z),
            Self::Left => (-Vec3::X, Vec3::Z),
            Self::Isometric => (
                Vec3::new(1.0, -1.0, 0.85)
                    .normalized()
                    .unwrap_or(Vec3::new(0.6, -0.6, 0.5)),
                Vec3::Z,
            ),
        }
    }
}

/// Smooth ease-in-out animation between two [`Camera`] states (e.g. when clicking a standard view
/// preset or pressing "Fit to Scene").
#[derive(Debug, Clone, Default)]
pub struct CameraTransition {
    from: Camera,
    to: Camera,
    duration_secs: f64,
    elapsed_secs: f64,
    active: bool,
}

impl CameraTransition {
    /// Starts a smooth transition from `from` to `to` over `duration_secs` seconds.
    pub fn start(&mut self, from: Camera, to: Camera, duration_secs: f64) {
        self.from = from;
        self.to = to;
        self.duration_secs = if duration_secs.is_finite() && duration_secs > 1e-4 {
            duration_secs
        } else {
            0.0
        };
        self.elapsed_secs = 0.0;
        self.active = true;
    }

    /// Cancels any active transition (e.g. when the user starts manually orbiting/panning).
    pub fn cancel(&mut self) {
        self.active = false;
    }

    /// Returns `true` if a transition is currently in progress.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.active
    }

    /// Advances the transition by `dt_secs` and writes the interpolated camera into `camera`.
    ///
    /// Returns `true` while the camera changed (and another frame should be rendered).
    pub fn tick(&mut self, dt_secs: f64, camera: &mut Camera) -> bool {
        if !self.active {
            return false;
        }
        if self.duration_secs <= 0.0 {
            *camera = self.to.clone();
            self.active = false;
            return true;
        }
        let dt = if dt_secs.is_finite() && dt_secs >= 0.0 {
            dt_secs
        } else {
            0.0
        };
        self.elapsed_secs += dt;
        let raw_t = (self.elapsed_secs / self.duration_secs).clamp(0.0, 1.0);
        // Smoothstep cubic ease-in-out: 3t^2 - 2t^3
        let eased = raw_t * raw_t * (3.0 - 2.0 * raw_t);
        *camera = self.from.lerp(&self.to, eased);
        if raw_t >= 1.0 {
            self.active = false;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zoom_2d_keeps_anchor_fixed_at_national_grid_coordinates() {
        let size = Vec2::new(1280.0, 720.0);
        let mut cam = Camera::ortho_2d(Vec2::new(4.5e9, 5.0e8), 0.01);
        let anchor = Vec2::new(760.0, 320.0);
        let before = cam.screen_to_world_2d(anchor, size).unwrap();
        assert!(cam.zoom_at_2d(anchor, size, 1000.0));
        let after = cam.screen_to_world_2d(anchor, size).unwrap();
        assert!(after.approx_eq(before, 1e-3));
    }

    #[test]
    fn fit_2d_centers_bounds_and_handles_degenerate() {
        let b = Bounds2::from_corners(Vec2::new(0.0, 0.0), Vec2::new(1000.0, 500.0));
        let cam = Camera::fit_2d(b, Vec2::new(1040.0, 1040.0), 20.0);
        let Projection::Ortho2D { center, zoom } = cam.projection else {
            panic!("expected Ortho2D");
        };
        assert!(center.approx_eq(Vec2::new(500.0, 250.0), 1e-9));
        assert!((zoom - 1.0).abs() < 1e-9);

        let pt = Bounds2::from_corners(Vec2::new(5.0, 5.0), Vec2::new(5.0, 5.0));
        let deg = Camera::fit_2d(pt, Vec2::new(100.0, 100.0), 10.0);
        assert!(deg.position_3d().is_finite());
    }

    #[test]
    fn reverse_z_unproject_depth_recovers_world_point_at_50km() {
        let size = Vec2::new(1000.0, 1000.0);
        let eye = Vec3::new(661_858.0, 4_491_159.0, 120.0);
        let target_world = eye + Vec3::new(120.0, -45.0, -50_000.0);
        let cam = Camera::perspective_3d(
            eye,
            eye + Vec3::new(0.0, 0.0, -1.0),
            Vec3::Y,
            std::f64::consts::FRAC_PI_3,
            0.01,
        );
        let vp = cam.view_proj_relative(size);
        let rel = target_world - eye;
        let clip = vp.mul_vec4(Vec4::from_vec3(rel, 1.0));
        let ndc_x = clip.x / clip.w;
        let ndc_y = clip.y / clip.w;
        let ndc_z = (clip.z / clip.w) as f32;

        let screen = Vec2::new((ndc_x + 1.0) * 0.5 * size.x, (1.0 - ndc_y) * 0.5 * size.y);
        let recovered = cam.unproject_depth(screen, ndc_z, size).unwrap();
        assert!(
            recovered.approx_eq(target_world, 5e-2),
            "expected {target_world:?}, got {recovered:?}"
        );
    }

    #[test]
    fn clip_planes_are_capped_at_max() {
        let mut cam = Camera::default();
        let p = Plane::from_normal_point(Vec3::Z, Vec3::ZERO).unwrap();
        cam.set_clip_planes(&[p; 12]);
        assert_eq!(cam.clip_planes().len(), MAX_CLIP_PLANES);
        let u = cam.to_uniform_data(Vec2::new(800.0, 600.0), 2.0);
        assert_eq!(u.clip_plane_count, MAX_CLIP_PLANES as u32);
    }

    #[test]
    fn fit_3d_frames_all_eight_corners_for_perspective_and_ortho() {
        let size = Vec2::new(1280.0, 720.0);
        let bounds =
            Bounds3::from_corners(Vec3::new(100.0, 200.0, 10.0), Vec3::new(112.0, 208.0, 16.0));
        let corners = [
            Vec3::new(bounds.min.x, bounds.min.y, bounds.min.z),
            Vec3::new(bounds.max.x, bounds.min.y, bounds.min.z),
            Vec3::new(bounds.min.x, bounds.max.y, bounds.min.z),
            Vec3::new(bounds.max.x, bounds.max.y, bounds.min.z),
            Vec3::new(bounds.min.x, bounds.min.y, bounds.max.z),
            Vec3::new(bounds.max.x, bounds.min.y, bounds.max.z),
            Vec3::new(bounds.min.x, bounds.max.y, bounds.max.z),
            Vec3::new(bounds.max.x, bounds.max.y, bounds.max.z),
        ];

        for perspective in [true, false] {
            let (cam, pivot) =
                Camera::standard_view_3d(StandardView::Isometric, bounds, size, 1.15, perspective);
            assert!(pivot.approx_eq(bounds.center().unwrap(), 1e-9));
            let vp = cam.view_proj_relative(size);
            let eye = cam.position_3d();
            for c in corners {
                let clip = vp.mul_vec4(Vec4::from_vec3(c - eye, 1.0));
                assert!(clip.w > 0.0);
                let nx = clip.x / clip.w;
                let ny = clip.y / clip.w;
                assert!(
                    nx.abs() <= 1.0 && ny.abs() <= 1.0,
                    "perspective={perspective} corner {c:?} out of NDC: ({nx}, {ny})"
                );
            }
        }
    }

    #[test]
    fn camera_transition_interpolates_smoothly_and_finishes() {
        let bounds = Bounds3::from_corners(Vec3::new(-5.0, -5.0, 0.0), Vec3::new(5.0, 5.0, 4.0));
        let size = Vec2::new(800.0, 600.0);
        let (from, _) = Camera::standard_view_3d(StandardView::Front, bounds, size, 1.2, true);
        let (to, _) = Camera::standard_view_3d(StandardView::Isometric, bounds, size, 1.2, true);
        let mut cam = from.clone();
        let mut tr = CameraTransition::default();
        tr.start(from.clone(), to.clone(), 0.4);
        assert!(tr.is_active());
        assert!(tr.tick(0.2, &mut cam));
        // Midpoint should lie strictly between start and target positions.
        assert!(!cam.position_3d().approx_eq(from.position_3d(), 1e-3));
        assert!(!cam.position_3d().approx_eq(to.position_3d(), 1e-3));
        assert!(tr.tick(0.25, &mut cam));
        assert!(!tr.is_active());
        assert!(cam.position_3d().approx_eq(to.position_3d(), 1e-9));
    }
}
