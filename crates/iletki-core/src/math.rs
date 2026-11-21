//! Double-precision (`f64`) linear algebra, split-precision helpers and spatial primitives.

use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// A 2D vector or point in `f64` world coordinates.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Vec2 {
    /// Horizontal component.
    pub x: f64,
    /// Vertical component.
    pub y: f64,
}

impl Vec2 {
    /// The zero vector `(0, 0)`.
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };
    /// Unit X vector `(1, 0)`.
    pub const X: Self = Self { x: 1.0, y: 0.0 };
    /// Unit Y vector `(0, 1)`.
    pub const Y: Self = Self { x: 0.0, y: 1.0 };

    /// Creates a new 2D vector.
    #[must_use]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Returns `true` if both components are finite.
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }

    /// Dot product of `self` and `other`.
    #[must_use]
    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y
    }

    /// Squared Euclidean length.
    #[must_use]
    pub fn length_squared(self) -> f64 {
        self.dot(self)
    }

    /// Euclidean length.
    #[must_use]
    pub fn length(self) -> f64 {
        self.length_squared().sqrt()
    }

    /// Returns a unit vector in the same direction, or `None` if zero or non-finite.
    #[must_use]
    pub fn normalized(self) -> Option<Self> {
        let len = self.length();
        if len.is_finite() && len > 0.0 {
            Some(Self::new(self.x / len, self.y / len))
        } else {
            None
        }
    }

    /// Component-wise minimum.
    #[must_use]
    pub fn min(self, other: Self) -> Self {
        Self::new(self.x.min(other.x), self.y.min(other.y))
    }

    /// Component-wise maximum.
    #[must_use]
    pub fn max(self, other: Self) -> Self {
        Self::new(self.x.max(other.x), self.y.max(other.y))
    }

    /// Linearly interpolates between `self` and `other` by `t`.
    #[must_use]
    pub fn lerp(self, other: Self, t: f64) -> Self {
        self + (other - self) * t
    }

    /// Returns `true` if both components are within `tol` of `other`.
    #[must_use]
    pub fn approx_eq(self, other: Self, tol: f64) -> bool {
        (self.x - other.x).abs() <= tol && (self.y - other.y).abs() <= tol
    }

    /// Converts to a single-precision array.
    #[must_use]
    pub fn to_f32(self) -> [f32; 2] {
        [self.x as f32, self.y as f32]
    }
}

impl From<[f64; 2]> for Vec2 {
    fn from([x, y]: [f64; 2]) -> Self {
        Self { x, y }
    }
}

impl From<Vec2> for [f64; 2] {
    fn from(v: Vec2) -> Self {
        [v.x, v.y]
    }
}

impl Add for Vec2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl AddAssign for Vec2 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl Sub for Vec2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl SubAssign for Vec2 {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl Neg for Vec2 {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y)
    }
}

impl Mul<f64> for Vec2 {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self::new(self.x * rhs, self.y * rhs)
    }
}

impl MulAssign<f64> for Vec2 {
    fn mul_assign(&mut self, rhs: f64) {
        self.x *= rhs;
        self.y *= rhs;
    }
}

impl Div<f64> for Vec2 {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self::new(self.x / rhs, self.y / rhs)
    }
}

/// A 3D vector or point in `f64` world coordinates.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Vec3 {
    /// X component.
    pub x: f64,
    /// Y component.
    pub y: f64,
    /// Z component.
    pub z: f64,
}

impl Vec3 {
    /// The zero vector `(0, 0, 0)`.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    /// Unit X axis `(1, 0, 0)`.
    pub const X: Self = Self {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    };
    /// Unit Y axis `(0, 1, 0)`.
    pub const Y: Self = Self {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };
    /// Unit Z axis `(0, 0, 1)`.
    pub const Z: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };

    /// Creates a new 3D vector.
    #[must_use]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Returns `true` if all components are finite.
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// Dot product of `self` and `other`.
    #[must_use]
    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Cross product `self × other`.
    #[must_use]
    pub fn cross(self, other: Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    /// Squared Euclidean length.
    #[must_use]
    pub fn length_squared(self) -> f64 {
        self.dot(self)
    }

    /// Euclidean length.
    #[must_use]
    pub fn length(self) -> f64 {
        self.length_squared().sqrt()
    }

    /// Returns a unit vector in the same direction, or `None` if zero or non-finite.
    #[must_use]
    pub fn normalized(self) -> Option<Self> {
        let len = self.length();
        if len.is_finite() && len > 0.0 {
            Some(Self::new(self.x / len, self.y / len, self.z / len))
        } else {
            None
        }
    }

    /// Component-wise minimum.
    #[must_use]
    pub fn min(self, other: Self) -> Self {
        Self::new(
            self.x.min(other.x),
            self.y.min(other.y),
            self.z.min(other.z),
        )
    }

    /// Component-wise maximum.
    #[must_use]
    pub fn max(self, other: Self) -> Self {
        Self::new(
            self.x.max(other.x),
            self.y.max(other.y),
            self.z.max(other.z),
        )
    }

    /// Linearly interpolates between `self` and `other` by `t`.
    #[must_use]
    pub fn lerp(self, other: Self, t: f64) -> Self {
        self + (other - self) * t
    }

    /// Returns `true` if every component is within `tol` of `other`.
    #[must_use]
    pub fn approx_eq(self, other: Self, tol: f64) -> bool {
        (self.x - other.x).abs() <= tol
            && (self.y - other.y).abs() <= tol
            && (self.z - other.z).abs() <= tol
    }

    /// Converts to a single-precision array.
    #[must_use]
    pub fn to_f32(self) -> [f32; 3] {
        [self.x as f32, self.y as f32, self.z as f32]
    }
}

impl From<[f64; 3]> for Vec3 {
    fn from([x, y, z]: [f64; 3]) -> Self {
        Self { x, y, z }
    }
}

impl From<Vec3> for [f64; 3] {
    fn from(v: Vec3) -> Self {
        [v.x, v.y, v.z]
    }
}

impl Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl SubAssign for Vec3 {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

impl Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl MulAssign<f64> for Vec3 {
    fn mul_assign(&mut self, rhs: f64) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

impl Div<f64> for Vec3 {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

/// A 4D homogeneous vector in `f64`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Vec4 {
    /// X component.
    pub x: f64,
    /// Y component.
    pub y: f64,
    /// Z component.
    pub z: f64,
    /// W component.
    pub w: f64,
}

impl Vec4 {
    /// Creates a new 4D vector.
    #[must_use]
    pub const fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        Self { x, y, z, w }
    }

    /// Extends a 3D vector with `w`.
    #[must_use]
    pub const fn from_vec3(v: Vec3, w: f64) -> Self {
        Self::new(v.x, v.y, v.z, w)
    }

    /// Drops `w` to obtain the `(x, y, z)` part.
    #[must_use]
    pub const fn xyz(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }

    /// Converts to a single-precision array.
    #[must_use]
    pub fn to_f32(self) -> [f32; 4] {
        [self.x as f32, self.y as f32, self.z as f32, self.w as f32]
    }
}

/// A unit quaternion representing a 3D orientation in `f64`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quat {
    /// Vector X component.
    pub x: f64,
    /// Vector Y component.
    pub y: f64,
    /// Vector Z component.
    pub z: f64,
    /// Scalar W component.
    pub w: f64,
}

impl Default for Quat {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Quat {
    /// Identity rotation.
    pub const IDENTITY: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    };

    /// Creates a rotation of `angle` radians around `axis`.
    ///
    /// Returns [`Quat::IDENTITY`] if `axis` cannot be normalized or `angle` is non-finite.
    #[must_use]
    pub fn from_axis_angle(axis: Vec3, angle: f64) -> Self {
        let Some(n) = axis.normalized() else {
            return Self::IDENTITY;
        };
        if !angle.is_finite() {
            return Self::IDENTITY;
        }
        let half = 0.5 * angle;
        let (s, c) = half.sin_cos();
        Self {
            x: n.x * s,
            y: n.y * s,
            z: n.z * s,
            w: c,
        }
    }

    /// Normalizes the quaternion, falling back to [`Quat::IDENTITY`] if degenerate.
    #[must_use]
    pub fn normalized(self) -> Self {
        let len = (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt();
        if len.is_finite() && len > 0.0 {
            Self {
                x: self.x / len,
                y: self.y / len,
                z: self.z / len,
                w: self.w / len,
            }
        } else {
            Self::IDENTITY
        }
    }

    /// Conjugate (inverse for a unit quaternion).
    #[must_use]
    pub const fn conjugate(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: self.w,
        }
    }

    /// Composes `self` followed by `rhs` (`self * rhs`).
    #[must_use]
    pub fn mul_quat(self, rhs: Self) -> Self {
        Self {
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
        }
        .normalized()
    }

    /// Rotates vector `v` by this unit quaternion.
    #[must_use]
    pub fn rotate(self, v: Vec3) -> Vec3 {
        let qv = Vec3::new(self.x, self.y, self.z);
        let t = qv.cross(v) * 2.0;
        v + t * self.w + qv.cross(t)
    }

    /// Spherical linear interpolation between two unit quaternions along the shortest arc.
    #[must_use]
    pub fn slerp(self, other: Self, t: f64) -> Self {
        let t = if t.is_finite() {
            t.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let a = self.normalized();
        let mut b = other.normalized();
        let mut dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        if dot < 0.0 {
            dot = -dot;
            b = Self {
                x: -b.x,
                y: -b.y,
                z: -b.z,
                w: -b.w,
            };
        }
        if dot > 0.9995 {
            return Self {
                x: a.x + (b.x - a.x) * t,
                y: a.y + (b.y - a.y) * t,
                z: a.z + (b.z - a.z) * t,
                w: a.w + (b.w - a.w) * t,
            }
            .normalized();
        }
        let theta = dot.clamp(-1.0, 1.0).acos();
        let sin_theta = theta.sin();
        if !sin_theta.is_finite() || sin_theta.abs() < 1e-12 {
            return a;
        }
        let w0 = ((1.0 - t) * theta).sin() / sin_theta;
        let w1 = (t * theta).sin() / sin_theta;
        Self {
            x: a.x * w0 + b.x * w1,
            y: a.y * w0 + b.y * w1,
            z: a.z * w0 + b.z * w1,
            w: a.w * w0 + b.w * w1,
        }
        .normalized()
    }

    /// Builds a right-handed camera orientation looking along `forward` with approximate `up`.
    ///
    /// Camera local axes follow the standard right-handed convention:
    /// `+X` right, `+Y` up, `-Z` forward.
    #[must_use]
    pub fn look_direction(forward: Vec3, up: Vec3) -> Self {
        let Some(f) = forward.normalized() else {
            return Self::IDENTITY;
        };
        let fallback_up = if f.dot(Vec3::Z).abs() < 0.99 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        let up_ref = up.normalized().unwrap_or(fallback_up);
        let right = f
            .cross(up_ref)
            .normalized()
            .or_else(|| f.cross(fallback_up).normalized())
            .unwrap_or(Vec3::X);
        let true_up = right.cross(f);
        // Columns of rotation matrix M = [right, true_up, -f]
        let back = -f;
        Self::from_basis(right, true_up, back)
    }

    /// Constructs a quaternion from an orthonormal basis `(x_axis, y_axis, z_axis)`.
    #[must_use]
    pub fn from_basis(x_axis: Vec3, y_axis: Vec3, z_axis: Vec3) -> Self {
        let m00 = x_axis.x;
        let m10 = x_axis.y;
        let m20 = x_axis.z;
        let m01 = y_axis.x;
        let m11 = y_axis.y;
        let m21 = y_axis.z;
        let m02 = z_axis.x;
        let m12 = z_axis.y;
        let m22 = z_axis.z;

        let trace = m00 + m11 + m22;
        if trace > 0.0 {
            let s = (trace + 1.0).sqrt() * 2.0;
            Self {
                x: (m21 - m12) / s,
                y: (m02 - m20) / s,
                z: (m10 - m01) / s,
                w: 0.25 * s,
            }
            .normalized()
        } else if m00 > m11 && m00 > m22 {
            let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
            Self {
                x: 0.25 * s,
                y: (m01 + m10) / s,
                z: (m02 + m20) / s,
                w: (m21 - m12) / s,
            }
            .normalized()
        } else if m11 > m22 {
            let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
            Self {
                x: (m01 + m10) / s,
                y: 0.25 * s,
                z: (m12 + m21) / s,
                w: (m02 - m20) / s,
            }
            .normalized()
        } else {
            let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
            Self {
                x: (m02 + m20) / s,
                y: (m12 + m21) / s,
                z: 0.25 * s,
                w: (m10 - m01) / s,
            }
            .normalized()
        }
    }
}

/// Column-major 4×4 matrix in `f64`, compatible with WGSL column-major layout.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4 {
    /// Column 0.
    pub c0: Vec4,
    /// Column 1.
    pub c1: Vec4,
    /// Column 2.
    pub c2: Vec4,
    /// Column 3.
    pub c3: Vec4,
}

impl Default for Mat4 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Mat4 {
    /// Identity 4×4 matrix.
    pub const IDENTITY: Self = Self {
        c0: Vec4::new(1.0, 0.0, 0.0, 0.0),
        c1: Vec4::new(0.0, 1.0, 0.0, 0.0),
        c2: Vec4::new(0.0, 0.0, 1.0, 0.0),
        c3: Vec4::new(0.0, 0.0, 0.0, 1.0),
    };

    /// Constructs a matrix from four columns.
    #[must_use]
    pub const fn from_cols(c0: Vec4, c1: Vec4, c2: Vec4, c3: Vec4) -> Self {
        Self { c0, c1, c2, c3 }
    }

    /// Converts to column-major `[[f32; 4]; 4]` suitable for GPU uniform buffers.
    #[must_use]
    pub fn to_cols_array_f32(self) -> [[f32; 4]; 4] {
        [
            self.c0.to_f32(),
            self.c1.to_f32(),
            self.c2.to_f32(),
            self.c3.to_f32(),
        ]
    }

    /// Returns row `r` (0..4) as a [`Vec4`].
    #[must_use]
    pub const fn row(self, r: usize) -> Vec4 {
        match r {
            0 => Vec4::new(self.c0.x, self.c1.x, self.c2.x, self.c3.x),
            1 => Vec4::new(self.c0.y, self.c1.y, self.c2.y, self.c3.y),
            2 => Vec4::new(self.c0.z, self.c1.z, self.c2.z, self.c3.z),
            _ => Vec4::new(self.c0.w, self.c1.w, self.c2.w, self.c3.w),
        }
    }

    /// Multiplies `self` by a column vector `v`.
    #[must_use]
    pub fn mul_vec4(self, v: Vec4) -> Vec4 {
        Vec4::new(
            self.c0.x * v.x + self.c1.x * v.y + self.c2.x * v.z + self.c3.x * v.w,
            self.c0.y * v.x + self.c1.y * v.y + self.c2.y * v.z + self.c3.y * v.w,
            self.c0.z * v.x + self.c1.z * v.y + self.c2.z * v.z + self.c3.z * v.w,
            self.c0.w * v.x + self.c1.w * v.y + self.c2.w * v.z + self.c3.w * v.w,
        )
    }

    /// Matrix multiplication `self * rhs`.
    #[must_use]
    pub fn mul_mat4(self, rhs: Self) -> Self {
        Self::from_cols(
            self.mul_vec4(rhs.c0),
            self.mul_vec4(rhs.c1),
            self.mul_vec4(rhs.c2),
            self.mul_vec4(rhs.c3),
        )
    }

    /// Right-handed camera-relative view matrix from camera orientation `rotation`.
    ///
    /// Because world coordinates are shifted by `-camera_position` in `f64` before projection,
    /// the camera sits at the origin `(0, 0, 0)` and the view matrix is pure rotation (`R^T`).
    #[must_use]
    pub fn view_relative(rotation: Quat) -> Self {
        let right = rotation.rotate(Vec3::X);
        let up = rotation.rotate(Vec3::Y);
        let back = rotation.rotate(Vec3::Z);
        Self::from_cols(
            Vec4::new(right.x, up.x, back.x, 0.0),
            Vec4::new(right.y, up.y, back.y, 0.0),
            Vec4::new(right.z, up.z, back.z, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
        )
    }

    /// Right-handed infinite Reversed-Z perspective projection for `wgpu` NDC (`z ∈ [0, 1]`).
    ///
    /// Maps `z = -near` to NDC `1.0` and `z → -∞` to NDC `0.0`.
    /// Pair with `Depth32Float`, clear `0.0`, and `CompareFunction::GreaterEqual`.
    #[must_use]
    pub fn perspective_infinite_reverse_rh(fov_y_rad: f64, aspect: f64, near: f64) -> Self {
        let fov = fov_y_rad.clamp(1e-4, std::f64::consts::PI - 1e-4);
        let aspect = if aspect.is_finite() && aspect > 1e-6 {
            aspect
        } else {
            1.0
        };
        let near = if near.is_finite() && near > 1e-9 {
            near
        } else {
            0.01
        };
        let f = 1.0 / (0.5 * fov).tan();
        Self::from_cols(
            Vec4::new(f / aspect, 0.0, 0.0, 0.0),
            Vec4::new(0.0, f, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, -1.0),
            Vec4::new(0.0, 0.0, near, 0.0),
        )
    }

    /// Right-handed Reversed-Z orthographic projection for `wgpu` NDC (`z ∈ [0, 1]`).
    ///
    /// Maps `z = -near` to NDC `1.0` and `z = -far` to NDC `0.0`.
    #[must_use]
    pub fn orthographic_reverse_rh(half_width: f64, half_height: f64, near: f64, far: f64) -> Self {
        let hw = if half_width.is_finite() && half_width > 1e-12 {
            half_width
        } else {
            1.0
        };
        let hh = if half_height.is_finite() && half_height > 1e-12 {
            half_height
        } else {
            1.0
        };
        let span = far - near;
        let inv_span = if span.is_finite() && span.abs() > 1e-12 {
            1.0 / span
        } else {
            1.0
        };
        // z_ndc = (z_eye + far) / (far - near)
        // at z_eye = -near => (far - near)/(far - near) = 1.0
        // at z_eye = -far  => 0.0
        Self::from_cols(
            Vec4::new(1.0 / hw, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0 / hh, 0.0, 0.0),
            Vec4::new(0.0, 0.0, inv_span, 0.0),
            Vec4::new(0.0, 0.0, far * inv_span, 1.0),
        )
    }

    /// Computes the matrix inverse, or `None` if singular or non-finite.
    #[must_use]
    pub fn inverse(self) -> Option<Self> {
        let m00 = self.c0.x;
        let m01 = self.c1.x;
        let m02 = self.c2.x;
        let m03 = self.c3.x;
        let m10 = self.c0.y;
        let m11 = self.c1.y;
        let m12 = self.c2.y;
        let m13 = self.c3.y;
        let m20 = self.c0.z;
        let m21 = self.c1.z;
        let m22 = self.c2.z;
        let m23 = self.c3.z;
        let m30 = self.c0.w;
        let m31 = self.c1.w;
        let m32 = self.c2.w;
        let m33 = self.c3.w;

        let s0 = m00 * m11 - m10 * m01;
        let s1 = m00 * m12 - m10 * m02;
        let s2 = m00 * m13 - m10 * m03;
        let s3 = m01 * m12 - m11 * m02;
        let s4 = m01 * m13 - m11 * m03;
        let s5 = m02 * m13 - m12 * m03;

        let c5 = m22 * m33 - m32 * m23;
        let c4 = m21 * m33 - m31 * m23;
        let c3 = m21 * m32 - m31 * m22;
        let c2 = m20 * m33 - m30 * m23;
        let c1 = m20 * m32 - m30 * m22;
        let c0 = m20 * m31 - m30 * m21;

        let det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
        if !det.is_finite() || det.abs() <= 1e-24 {
            return None;
        }
        let inv = 1.0 / det;

        Some(Self::from_cols(
            Vec4::new(
                (m11 * c5 - m12 * c4 + m13 * c3) * inv,
                (-m10 * c5 + m12 * c2 - m13 * c1) * inv,
                (m10 * c4 - m11 * c2 + m13 * c0) * inv,
                (-m10 * c3 + m11 * c1 - m12 * c0) * inv,
            ),
            Vec4::new(
                (-m01 * c5 + m02 * c4 - m03 * c3) * inv,
                (m00 * c5 - m02 * c2 + m03 * c1) * inv,
                (-m00 * c4 + m01 * c2 - m03 * c0) * inv,
                (m00 * c3 - m01 * c1 + m02 * c0) * inv,
            ),
            Vec4::new(
                (m31 * s5 - m32 * s4 + m33 * s3) * inv,
                (-m30 * s5 + m32 * s2 - m33 * s1) * inv,
                (m30 * s4 - m31 * s2 + m33 * s0) * inv,
                (-m30 * s3 + m31 * s1 - m32 * s0) * inv,
            ),
            Vec4::new(
                (-m21 * s5 + m22 * s4 - m23 * s3) * inv,
                (m20 * s5 - m22 * s2 + m23 * s1) * inv,
                (-m20 * s4 + m21 * s2 - m23 * s0) * inv,
                (m20 * s3 - m21 * s1 + m22 * s0) * inv,
            ),
        ))
    }
}

/// Splits an `f64` coordinate into a high `f32` and the `f32` remainder.
///
/// Subtracting `(hi - cam_hi) + (lo - cam_lo)` in a WGSL vertex shader preserves ~48 bits of
/// precision near the camera, eliminating jitter at national grid coordinates (`10^6..10^9`).
#[must_use]
pub fn split(v: f64) -> (f32, f32) {
    if !v.is_finite() {
        return (0.0, 0.0);
    }
    let hi = v as f32;
    let lo = (v - f64::from(hi)) as f32;
    (hi, lo)
}

/// Splits a 2D `f64` point into `([hi_x, hi_y], [lo_x, lo_y])`.
#[must_use]
pub fn split2(p: Vec2) -> ([f32; 2], [f32; 2]) {
    let (hx, lx) = split(p.x);
    let (hy, ly) = split(p.y);
    ([hx, hy], [lx, ly])
}

/// An axis-aligned 2D bounding box in `f64`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds2 {
    /// Minimum corner.
    pub min: Vec2,
    /// Maximum corner.
    pub max: Vec2,
}

impl Default for Bounds2 {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl Bounds2 {
    /// An empty bounding box.
    pub const EMPTY: Self = Self {
        min: Vec2::new(f64::INFINITY, f64::INFINITY),
        max: Vec2::new(f64::NEG_INFINITY, f64::NEG_INFINITY),
    };

    /// Creates a box containing both corners in any order.
    #[must_use]
    pub fn from_corners(a: Vec2, b: Vec2) -> Self {
        if !a.is_finite() || !b.is_finite() {
            return Self::EMPTY;
        }
        Self {
            min: a.min(b),
            max: a.max(b),
        }
    }

    /// Returns `true` if the box contains no finite points.
    #[must_use]
    pub fn is_empty(self) -> bool {
        !self.min.is_finite()
            || !self.max.is_finite()
            || self.min.x > self.max.x
            || self.min.y > self.max.y
    }

    /// Expands the box to include `p`.
    pub fn include(&mut self, p: Vec2) {
        if !p.is_finite() {
            return;
        }
        if self.is_empty() {
            self.min = p;
            self.max = p;
        } else {
            self.min = self.min.min(p);
            self.max = self.max.max(p);
        }
    }

    /// Center of the box, or `None` if empty.
    #[must_use]
    pub fn center(self) -> Option<Vec2> {
        (!self.is_empty()).then(|| (self.min + self.max) * 0.5)
    }

    /// Extent of the box, or [`Vec2::ZERO`] if empty.
    #[must_use]
    pub fn size(self) -> Vec2 {
        if self.is_empty() {
            Vec2::ZERO
        } else {
            self.max - self.min
        }
    }

    /// Returns `true` if `self` and `other` overlap (touching edges count).
    #[must_use]
    pub fn intersects(self, other: Self) -> bool {
        !self.is_empty()
            && !other.is_empty()
            && self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
    }
}

/// An axis-aligned 3D bounding box in `f64`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds3 {
    /// Minimum corner.
    pub min: Vec3,
    /// Maximum corner.
    pub max: Vec3,
}

impl Default for Bounds3 {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl Bounds3 {
    /// An empty 3D bounding box.
    pub const EMPTY: Self = Self {
        min: Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
        max: Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
    };

    /// Creates a box containing both corners in any order.
    #[must_use]
    pub fn from_corners(a: Vec3, b: Vec3) -> Self {
        if !a.is_finite() || !b.is_finite() {
            return Self::EMPTY;
        }
        Self {
            min: a.min(b),
            max: a.max(b),
        }
    }

    /// Returns `true` if the box contains no finite points.
    #[must_use]
    pub fn is_empty(self) -> bool {
        !self.min.is_finite()
            || !self.max.is_finite()
            || self.min.x > self.max.x
            || self.min.y > self.max.y
            || self.min.z > self.max.z
    }

    /// Expands the box to include `p`.
    pub fn include(&mut self, p: Vec3) {
        if !p.is_finite() {
            return;
        }
        if self.is_empty() {
            self.min = p;
            self.max = p;
        } else {
            self.min = self.min.min(p);
            self.max = self.max.max(p);
        }
    }

    /// Center of the box, or `None` if empty.
    #[must_use]
    pub fn center(self) -> Option<Vec3> {
        (!self.is_empty()).then(|| (self.min + self.max) * 0.5)
    }

    /// Extent of the box, or [`Vec3::ZERO`] if empty.
    #[must_use]
    pub fn size(self) -> Vec3 {
        if self.is_empty() {
            Vec3::ZERO
        } else {
            self.max - self.min
        }
    }

    /// Bounding sphere radius around [`Bounds3::center`].
    #[must_use]
    pub fn radius(self) -> f64 {
        self.size().length() * 0.5
    }
}

/// A 3D ray in `f64` world space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray {
    /// Origin of the ray.
    pub origin: Vec3,
    /// Unit direction vector.
    pub direction: Vec3,
}

impl Ray {
    /// Creates a ray, normalizing `direction`. Returns `None` if invalid.
    #[must_use]
    pub fn new(origin: Vec3, direction: Vec3) -> Option<Self> {
        if !origin.is_finite() {
            return None;
        }
        let dir = direction.normalized()?;
        Some(Self {
            origin,
            direction: dir,
        })
    }

    /// Evaluates the point at distance `t` along the ray.
    #[must_use]
    pub fn at(self, t: f64) -> Vec3 {
        self.origin + self.direction * t
    }
}

/// An oriented plane in `f64` world space: `normal · p + d = 0`.
///
/// Points with `signed_distance(p) >= 0` lie on the kept (front) side.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plane {
    /// Unit normal pointing toward the kept half-space.
    pub normal: Vec3,
    /// Signed offset `d` such that `normal · p + d = 0`.
    pub d: f64,
}

impl Plane {
    /// Creates a plane from a normal and a point on the plane.
    #[must_use]
    pub fn from_normal_point(normal: Vec3, point: Vec3) -> Option<Self> {
        if !point.is_finite() {
            return None;
        }
        let n = normal.normalized()?;
        Some(Self {
            normal: n,
            d: -n.dot(point),
        })
    }

    /// Creates a normalized plane from raw coefficients `(a, b, c, d)`.
    #[must_use]
    pub fn from_vec4(v: Vec4) -> Option<Self> {
        let len = v.xyz().length();
        if !len.is_finite() || len <= 0.0 || !v.w.is_finite() {
            return None;
        }
        Some(Self {
            normal: v.xyz() / len,
            d: v.w / len,
        })
    }

    /// Signed distance from `point` to the plane (`>= 0` is inside/front).
    #[must_use]
    pub fn signed_distance(self, point: Vec3) -> f64 {
        self.normal.dot(point) + self.d
    }

    /// Converts the plane to camera-relative `[nx, ny, nz, d_rel]` for GPU clip planes.
    ///
    /// For any camera-relative point `p_rel = p - camera_pos`:
    /// `n · p_rel + d_rel = n · p + d`.
    #[must_use]
    pub fn to_camera_relative_f32(self, camera_pos: Vec3) -> [f32; 4] {
        let d_rel = self.d + self.normal.dot(camera_pos);
        [
            self.normal.x as f32,
            self.normal.y as f32,
            self.normal.z as f32,
            d_rel as f32,
        ]
    }
}

/// Six-plane view frustum in camera-relative space (for fast CPU culling).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frustum {
    /// Left, right, bottom, top, near and (optional) far planes in camera-relative space.
    ///
    /// For an infinite Reversed-Z frustum the far plane is degenerate and represented as `None`.
    pub planes: [Option<Plane>; 6],
}

impl Frustum {
    /// Extracts frustum planes from a `wgpu` NDC (`z ∈ [0, 1]`) Reversed-Z view-projection matrix.
    ///
    /// The returned planes are in the same coordinate space as the matrix input (camera-relative
    /// when built from [`Mat4::view_relative`]).
    #[must_use]
    pub fn from_view_proj_reverse_z(vp: Mat4) -> Self {
        let r0 = vp.row(0);
        let r1 = vp.row(1);
        let r2 = vp.row(2);
        let r3 = vp.row(3);

        let add = |a: Vec4, b: Vec4| Vec4::new(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
        let sub = |a: Vec4, b: Vec4| Vec4::new(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);

        // wgpu NDC: -w <= x <= w, -w <= y <= w, 0 <= z <= w (Reversed-Z: z=w is near, z=0 is far).
        let left = Plane::from_vec4(add(r3, r0));
        let right = Plane::from_vec4(sub(r3, r0));
        let bottom = Plane::from_vec4(add(r3, r1));
        let top = Plane::from_vec4(sub(r3, r1));
        let near = Plane::from_vec4(sub(r3, r2));
        let far = Plane::from_vec4(r2);

        Self {
            planes: [left, right, bottom, top, near, far],
        }
    }

    /// Returns `true` if a world-space [`Bounds3`] may intersect this camera-relative frustum.
    #[must_use]
    pub fn intersects_bounds_world(&self, bounds: Bounds3, camera_pos: Vec3) -> bool {
        if bounds.is_empty() {
            return false;
        }
        let rel_min = bounds.min - camera_pos;
        let rel_max = bounds.max - camera_pos;
        for plane in self.planes.into_iter().flatten() {
            // Positive vertex (farthest along plane normal).
            let px = if plane.normal.x >= 0.0 {
                rel_max.x
            } else {
                rel_min.x
            };
            let py = if plane.normal.y >= 0.0 {
                rel_max.y
            } else {
                rel_min.y
            };
            let pz = if plane.normal.z >= 0.0 {
                rel_max.z
            } else {
                rel_min.z
            };
            if plane.signed_distance(Vec3::new(px, py, pz)) < 0.0 {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_preserves_precision_far_from_origin() {
        let x = 4_512_345_678.125_f64;
        let (hi, lo) = split(x);
        assert!((f64::from(hi) + f64::from(lo) - x).abs() < 1e-3);
        assert!((f64::from(x as f32) - x).abs() > 1.0);
    }

    #[test]
    fn split_handles_non_finite_safely() {
        assert_eq!(split(f64::NAN), (0.0, 0.0));
        assert_eq!(split(f64::INFINITY), (0.0, 0.0));
    }

    #[test]
    fn mat4_inverse_round_trips_perspective_reverse_z() {
        let p = Mat4::perspective_infinite_reverse_rh(1.0, 16.0 / 9.0, 0.05);
        let inv = p.inverse().expect("perspective matrix is invertible");
        let prod = p.mul_mat4(inv);
        for r in 0..4 {
            let row = prod.row(r);
            let expected = match r {
                0 => Vec4::new(1.0, 0.0, 0.0, 0.0),
                1 => Vec4::new(0.0, 1.0, 0.0, 0.0),
                2 => Vec4::new(0.0, 0.0, 1.0, 0.0),
                _ => Vec4::new(0.0, 0.0, 0.0, 1.0),
            };
            assert!((row.x - expected.x).abs() < 1e-12);
            assert!((row.y - expected.y).abs() < 1e-12);
            assert!((row.z - expected.z).abs() < 1e-12);
            assert!((row.w - expected.w).abs() < 1e-12);
        }
    }

    #[test]
    fn reverse_z_maps_near_to_one_and_far_to_zero() {
        let near = 0.01;
        let p = Mat4::perspective_infinite_reverse_rh(std::f64::consts::FRAC_PI_3, 1.0, near);
        let clip_near = p.mul_vec4(Vec4::new(0.0, 0.0, -near, 1.0));
        assert!((clip_near.z / clip_near.w - 1.0).abs() < 1e-12);

        let clip_far = p.mul_vec4(Vec4::new(0.0, 0.0, -50_000.0, 1.0));
        let z_far = clip_far.z / clip_far.w;
        assert!(z_far > 0.0 && z_far < 1e-5);
    }

    #[test]
    fn frustum_culls_behind_and_keeps_in_front() {
        let rot = Quat::IDENTITY; // looking along -Z
        let vp = Mat4::perspective_infinite_reverse_rh(std::f64::consts::FRAC_PI_2, 1.0, 0.1)
            .mul_mat4(Mat4::view_relative(rot));
        let frustum = Frustum::from_view_proj_reverse_z(vp);
        let cam = Vec3::new(1_000_000.0, 2_000_000.0, 500.0);

        let in_front = Bounds3::from_corners(
            cam + Vec3::new(-1.0, -1.0, -12.0),
            cam + Vec3::new(1.0, 1.0, -8.0),
        );
        let behind = Bounds3::from_corners(
            cam + Vec3::new(-1.0, -1.0, 5.0),
            cam + Vec3::new(1.0, 1.0, 10.0),
        );
        let far_right = Bounds3::from_corners(
            cam + Vec3::new(50.0, -1.0, -12.0),
            cam + Vec3::new(60.0, 1.0, -8.0),
        );

        assert!(frustum.intersects_bounds_world(in_front, cam));
        assert!(!frustum.intersects_bounds_world(behind, cam));
        assert!(!frustum.intersects_bounds_world(far_right, cam));
    }
}
