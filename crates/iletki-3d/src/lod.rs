//! Conservative projection of a mesh's declared geometric error.

use iletki_core::{Bounds3, Mat4, Vec3};

#[derive(Clone, Copy)]
struct Row {
    normal: Vec3,
    constant: f64,
    absolute: Vec3,
    norm: f64,
}

impl Row {
    fn new([x, y, z, w]: [f32; 4]) -> Self {
        let normal = Vec3::new(f64::from(x), f64::from(y), f64::from(z));
        Self {
            normal,
            constant: f64::from(w),
            absolute: Vec3::new(normal.x.abs(), normal.y.abs(), normal.z.abs()),
            norm: normal.length(),
        }
    }

    fn envelope(self, center: Vec3, half_size: Vec3, extent_from_eye: Vec3) -> (f64, f64, f64) {
        let middle = self.normal.dot(center) + self.constant;
        let extent = self.absolute.dot(half_size);
        let rounding = (self.absolute.dot(extent_from_eye) + self.constant.abs())
            * (16.0 * f64::from(f32::EPSILON));
        (middle, extent + rounding, rounding)
    }
}

pub(crate) struct LodProjection {
    x: Row,
    y: Row,
    w: Row,
    pixels: [f64; 2],
    max_error: f64,
}

impl LodProjection {
    pub(crate) fn new(matrix: Mat4, size: [u32; 2], max_error: f64) -> Self {
        Self {
            x: Row::new(matrix.row(0).to_f32()),
            y: Row::new(matrix.row(1).to_f32()),
            w: Row::new(matrix.row(3).to_f32()),
            pixels: size.map(f64::from),
            max_error,
        }
    }

    pub(crate) fn allows(&self, bounds: Bounds3, offset: Vec3, world_error: f64) -> bool {
        let Some(local_center) = bounds.center() else {
            return false;
        };
        let center = offset + local_center;
        let half_size = bounds.size() * 0.5;
        let extent_from_eye = Vec3::new(center.x.abs(), center.y.abs(), center.z.abs()) + half_size;
        let position_bound = extent_from_eye.x + extent_from_eye.y + extent_from_eye.z;
        let world_error = world_error + position_bound * (2.0 * f64::from(f32::EPSILON));
        let (middle_w, extent_w, rounding_w) = self.w.envelope(center, half_size, extent_from_eye);
        let min_w = middle_w - extent_w;
        let dw = self.w.norm * world_error + 2.0 * rounding_w;
        let denominator = min_w - dw;
        if !denominator.is_finite() || denominator <= 0.0 {
            return false;
        }

        // Bound the difference of the two perspective quotients over the whole box.
        let axis_error = |row: Row, pixels: f64| {
            let (middle, extent, rounding) = row.envelope(center, half_size, extent_from_eye);
            let max_ndc = (middle.abs() + extent) / min_w;
            (row.norm * world_error + 2.0 * rounding + max_ndc * dw) / denominator * pixels * 0.5
        };
        let x = axis_error(self.x, self.pixels[0]) / self.max_error;
        let y = axis_error(self.y, self.pixels[1]) / self.max_error;
        x * x + y * y <= 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iletki_core::{Camera, Vec2};

    #[test]
    fn perspective_error_preserves_near_geometry_and_grows_with_resolution() {
        let camera = Camera::perspective_3d(-Vec3::Y, Vec3::ZERO, Vec3::Z, 1.0, 0.1);
        let matrix = camera.view_proj_relative(Vec2::new(960.0, 1080.0));
        let small = LodProjection::new(matrix, [960, 1080], 0.75);
        let large = LodProjection::new(matrix, [1920, 2160], 0.75);
        let half_size = Vec3::new(0.5, 0.5, 0.5);
        let bounds = Bounds3::from_corners(-half_size, half_size);
        assert!(!small.allows(bounds, Vec3::Y, 0.04));
        assert!(!small.allows(bounds, -Vec3::Y, 0.04));
        assert!(small.allows(bounds, Vec3::Y * 100.0, 0.04));
        assert!(!large.allows(bounds, Vec3::Y * 100.0, 0.04));
    }
}
