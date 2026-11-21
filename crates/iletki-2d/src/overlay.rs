//! Reusable 2D CAD overlay helpers: adaptive grid, crosshair and window/crossing selection box.

use crate::layer::Mesh2D;
use crate::vertex::DrawStyle2D;
use iletki_core::{Bounds2, PickId, Vec2};

/// Maximum grid lines emitted along one axis (guards against hostile or extreme zoom values).
const MAX_GRID_LINES_PER_AXIS: usize = 512;

/// Visual styling for [`push_adaptive_grid`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridStyle {
    /// Minimum desired spacing in logical points between minor grid lines (default `16.0`).
    pub min_spacing_pt: f64,
    /// Number of minor divisions per major line (default `5` or `10`).
    pub major_every: u32,
    /// Minor grid line RGBA color.
    pub minor_color: [u8; 4],
    /// Major grid line RGBA color.
    pub major_color: [u8; 4],
    /// World X-axis (`y = 0`) RGBA color.
    pub x_axis_color: [u8; 4],
    /// World Y-axis (`x = 0`) RGBA color.
    pub y_axis_color: [u8; 4],
}

impl Default for GridStyle {
    fn default() -> Self {
        Self {
            min_spacing_pt: 18.0,
            major_every: 5,
            minor_color: [255, 255, 255, 16],
            major_color: [255, 255, 255, 34],
            x_axis_color: [220, 72, 72, 160],
            y_axis_color: [72, 200, 96, 160],
        }
    }
}

/// Computes a power-of-10 adaptive grid spacing in world units for a given 2D `zoom` (points per
/// world unit) and `min_spacing_pt`.
#[must_use]
pub fn adaptive_grid_step(zoom: f64, min_spacing_pt: f64) -> Option<f64> {
    if !zoom.is_finite() || zoom <= 0.0 || !min_spacing_pt.is_finite() || min_spacing_pt <= 0.0 {
        return None;
    }
    let raw = min_spacing_pt / zoom;
    if !raw.is_finite() || raw <= 0.0 {
        return None;
    }
    let exp = raw.log10().floor();
    let base = 10.0_f64.powf(exp);
    let step = if raw <= base {
        base
    } else if raw <= 2.0 * base {
        2.0 * base
    } else if raw <= 5.0 * base {
        5.0 * base
    } else {
        10.0 * base
    };
    step.is_finite().then_some(step)
}

/// Appends an AutoCAD-style adaptive 2D grid and origin axes covering `visible` into `mesh`.
pub fn push_adaptive_grid(
    mesh: &mut Mesh2D,
    visible: Bounds2,
    zoom: f64,
    style: GridStyle,
    linear: bool,
) {
    if visible.is_empty() {
        return;
    }
    let Some(step) = adaptive_grid_step(zoom, style.min_spacing_pt) else {
        return;
    };
    let major_every = i64::from(style.major_every.max(1));
    let min_ix = (visible.min.x / step).floor() as i64;
    let max_ix = (visible.max.x / step).ceil() as i64;
    let min_iy = (visible.min.y / step).floor() as i64;
    let max_iy = (visible.max.y / step).ceil() as i64;

    if max_ix.saturating_sub(min_ix) <= MAX_GRID_LINES_PER_AXIS as i64 {
        for ix in min_ix..=max_ix {
            let x = (ix as f64) * step;
            let (color, depth) = if ix == 0 {
                (style.y_axis_color, 0.03)
            } else if ix.rem_euclid(major_every) == 0 {
                (style.major_color, 0.02)
            } else {
                (style.minor_color, 0.01)
            };
            mesh.push_line(
                Vec2::new(x, visible.min.y),
                Vec2::new(x, visible.max.y),
                DrawStyle2D::new(color, linear, depth, PickId::NONE),
            );
        }
    }

    if max_iy.saturating_sub(min_iy) <= MAX_GRID_LINES_PER_AXIS as i64 {
        for iy in min_iy..=max_iy {
            let y = (iy as f64) * step;
            let (color, depth) = if iy == 0 {
                (style.x_axis_color, 0.03)
            } else if iy.rem_euclid(major_every) == 0 {
                (style.major_color, 0.02)
            } else {
                (style.minor_color, 0.01)
            };
            mesh.push_line(
                Vec2::new(visible.min.x, y),
                Vec2::new(visible.max.x, y),
                DrawStyle2D::new(color, linear, depth, PickId::NONE),
            );
        }
    }
}

/// Appends a full-viewport drafting crosshair centered at `cursor_world`.
pub fn push_crosshair(
    mesh: &mut Mesh2D,
    visible: Bounds2,
    cursor_world: Vec2,
    color: [u8; 4],
    linear: bool,
) {
    if visible.is_empty() || !cursor_world.is_finite() {
        return;
    }
    let style = DrawStyle2D::new(color, linear, 0.995, PickId::NONE);
    mesh.push_line(
        Vec2::new(visible.min.x, cursor_world.y),
        Vec2::new(visible.max.x, cursor_world.y),
        style,
    );
    mesh.push_line(
        Vec2::new(cursor_world.x, visible.min.y),
        Vec2::new(cursor_world.x, visible.max.y),
        style,
    );
}

/// Appends an AutoCAD-style selection rectangle (`window`: blue fill/outline if `crossing == false`,
/// green if `crossing == true`).
pub fn push_selection_box(
    mesh: &mut Mesh2D,
    corner_a: Vec2,
    corner_b: Vec2,
    crossing: bool,
    linear: bool,
) {
    let bounds = Bounds2::from_corners(corner_a, corner_b);
    if bounds.is_empty() {
        return;
    }
    let (fill, border) = if crossing {
        ([64, 210, 110, 48], [80, 235, 130, 220])
    } else {
        ([64, 150, 245, 48], [96, 175, 255, 220])
    };
    mesh.push_rect_fill(
        bounds.min,
        bounds.max,
        DrawStyle2D::new(fill, linear, 0.992, PickId::NONE),
    );
    mesh.push_rect_outline(
        bounds.min,
        bounds.max,
        DrawStyle2D::new(border, linear, 0.994, PickId::NONE),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_grid_produces_bounded_lines() {
        let mut mesh = Mesh2D::default();
        let visible = Bounds2::from_corners(Vec2::new(-50.0, -50.0), Vec2::new(50.0, 50.0));
        push_adaptive_grid(&mut mesh, visible, 10.0, GridStyle::default(), false);
        assert!(!mesh.lines.is_empty());
        assert!(mesh.lines.len() < 400);
    }
}
