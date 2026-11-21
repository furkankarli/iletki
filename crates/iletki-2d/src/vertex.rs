//! Split-precision 2D vertex types, drawing-order Reversed-Z mapping and sRGB helpers.

use bytemuck::{Pod, Zeroable};
use iletki_core::{PickId, Vec2, split2};

/// A compact 24-byte split-precision 2D vertex for 1 px lines and filled triangles.
///
/// Picking identifiers live in an optional parallel `[u32; 2]` stream so color passes pay zero
/// picking bandwidth overhead.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable, PartialEq)]
pub struct Vertex2D {
    /// High `f32` part of world position.
    pub hi: [f32; 2],
    /// Low `f32` remainder of world position.
    pub lo: [f32; 2],
    /// RGBA color bytes (pre-linearized if targeting an `*Srgb` attachment).
    pub color: [u8; 4],
    /// Reversed-Z depth in `(0.0, 1.0]`: higher values are nearer and stay on top.
    pub depth: f32,
}

impl Vertex2D {
    /// Constructs a split-precision 2D vertex from `f64` world coordinates.
    #[must_use]
    pub fn new(pos: Vec2, color: [u8; 4], linear: bool, depth: f32) -> Self {
        let (hi, lo) = split2(pos);
        Self {
            hi,
            lo,
            color: if linear { to_linear(color) } else { color },
            depth,
        }
    }
}

/// One corner of a screen-space wide line segment quad (6 vertices per segment, 52 bytes each).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable, PartialEq)]
pub struct WideVertex2D {
    /// High `f32` part of segment start `a`.
    pub a_hi: [f32; 2],
    /// Low `f32` remainder of segment start `a`.
    pub a_lo: [f32; 2],
    /// High `f32` part of segment end `b`.
    pub b_hi: [f32; 2],
    /// Low `f32` remainder of segment end `b`.
    pub b_lo: [f32; 2],
    /// Corner selector: `x ∈ {0, 1}` selects end `a`/`b`, `y ∈ {-1, 1}` selects side.
    pub corner: [f32; 2],
    /// Stroke width in logical screen points.
    pub width: f32,
    /// RGBA color bytes.
    pub color: [u8; 4],
    /// Reversed-Z depth in `(0.0, 1.0]`.
    pub depth: f32,
}

/// Styling attributes shared by 2D mesh builders.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrawStyle2D {
    /// RGBA color bytes.
    pub color: [u8; 4],
    /// Convert sRGB color bytes to linear (set `true` when targeting `*Srgb` attachments).
    pub linear: bool,
    /// Reversed-Z depth in `(0.0, 1.0]` (use [`order_to_depth`]; higher draws on top).
    pub depth: f32,
    /// 64-bit picking ID (use [`PickId::NONE`] for non-pickable helpers/overlays).
    pub pick_id: PickId,
}

impl DrawStyle2D {
    /// Creates a new 2D draw style.
    #[must_use]
    pub const fn new(color: [u8; 4], linear: bool, depth: f32, pick_id: PickId) -> Self {
        Self {
            color,
            linear,
            depth,
            pick_id,
        }
    }
}

/// Appends the 6 vertices of a screen-space wide line segment from `a` to `b`.
///
/// Returns `true` if the segment was valid and 6 vertices were pushed.
pub fn push_wide_segment(
    out: &mut Vec<WideVertex2D>,
    a: Vec2,
    b: Vec2,
    width_pt: f32,
    style: DrawStyle2D,
) -> bool {
    if !a.is_finite() || !b.is_finite() || a == b || !width_pt.is_finite() || width_pt <= 0.0 {
        return false;
    }
    let (a_hi, a_lo) = split2(a);
    let (b_hi, b_lo) = split2(b);
    let color = if style.linear {
        to_linear(style.color)
    } else {
        style.color
    };
    for corner in [
        [0.0, -1.0],
        [1.0, -1.0],
        [1.0, 1.0],
        [0.0, -1.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ] {
        out.push(WideVertex2D {
            a_hi,
            a_lo,
            b_hi,
            b_lo,
            corner,
            width: width_pt,
            color,
            depth: style.depth,
        });
    }
    true
}

/// Maps a 2D drawing order `order ∈ [0..=max_order]` into Reversed-Z depth `(0.0, 0.99]`.
///
/// Later orders produce larger depth values so `CompareFunction::GreaterEqual` keeps them on top,
/// leaving `0.0` for the cleared background and `(0.99, 1.0]` for highlights and overlays.
#[must_use]
pub fn order_to_depth(order: u32, max_order: u32) -> f32 {
    let denom = f64::from(max_order) + 2.0;
    (((f64::from(order) + 1.0) / denom) * 0.99) as f32
}

/// Converts sRGB `[r, g, b, a]` bytes into linear bytes for `*Srgb` render targets.
#[must_use]
pub fn to_linear([r, g, b, a]: [u8; 4]) -> [u8; 4] {
    let f = |c: u8| {
        let c = f32::from(c) / 255.0;
        let l = if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        };
        (l * 255.0).round().clamp(0.0, 255.0) as u8
    };
    [f(r), f(g), f(b), a]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_sizes_are_compact_and_aligned() {
        assert_eq!(std::mem::size_of::<Vertex2D>(), 24);
        assert_eq!(std::mem::size_of::<WideVertex2D>(), 52);
    }

    #[test]
    fn higher_order_maps_to_nearer_reverse_z_depth() {
        let d0 = order_to_depth(0, 100);
        let d50 = order_to_depth(50, 100);
        let d100 = order_to_depth(100, 100);
        assert!(d0 > 0.0);
        assert!(d50 > d0);
        assert!(d100 > d50);
        assert!(d100 <= 0.99);
    }
}
