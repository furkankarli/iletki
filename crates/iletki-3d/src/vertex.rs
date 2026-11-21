//! Compact 28-byte 3D mesh vertices and geometric builders (boxes, arrows, triangles).

use bytemuck::{Pod, Zeroable};
use iletki_core::{Bounds3, PickId, Vec3};

/// Compact 28-byte camera-relative / chunk-local 3D mesh vertex.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable, PartialEq)]
pub struct Vertex3D {
    /// Local position relative to the owning chunk's `f64` world origin.
    pub position: [f32; 3],
    /// Unit surface normal.
    pub normal: [f32; 3],
    /// Linear or sRGB RGBA color bytes.
    pub color: [u8; 4],
}

impl Vertex3D {
    /// Constructs a 3D vertex from local position, normal and color.
    #[must_use]
    pub const fn new(position: [f32; 3], normal: [f32; 3], color: [u8; 4]) -> Self {
        Self {
            position,
            normal,
            color,
        }
    }
}

/// CPU-side 3D triangle mesh anchored at an `f64` world origin for jitter-free large worlds.
#[derive(Debug, Clone, Default)]
pub struct Mesh3D {
    /// World-space `f64` anchor of this mesh; vertices are stored relative to `origin`.
    pub origin: Vec3,
    /// Triangle list (`3` vertices per triangle).
    pub vertices: Vec<Vertex3D>,
    /// Optional 3D wireframe/silhouette edge list (`2` vertices per line segment) rendered with a
    /// Reversed-Z depth offset so edges stay crisp over solid faces (*Shaded with Edges* CAD mode).
    pub edges: Vec<Vertex3D>,
    /// Optional parallel 64-bit pick IDs (`[u32; 2]` per vertex, lazily allocated).
    pub pick_ids: Vec<[u32; 2]>,
    /// Local bounding box relative to [`Mesh3D::origin`].
    pub local_bounds: Bounds3,
}

impl Mesh3D {
    /// Creates an empty 3D mesh anchored at `origin`.
    #[must_use]
    pub fn at_origin(origin: Vec3) -> Self {
        Self {
            origin: if origin.is_finite() {
                origin
            } else {
                Vec3::ZERO
            },
            vertices: Vec::new(),
            edges: Vec::new(),
            pick_ids: Vec::new(),
            local_bounds: Bounds3::EMPTY,
        }
    }

    /// World-space bounding box of this mesh.
    #[must_use]
    pub fn world_bounds(&self) -> Bounds3 {
        if self.local_bounds.is_empty() {
            Bounds3::EMPTY
        } else {
            Bounds3::from_corners(
                self.origin + self.local_bounds.min,
                self.origin + self.local_bounds.max,
            )
        }
    }

    /// Returns `true` if neither triangles nor edges have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty() && self.edges.is_empty()
    }

    /// Pushes a 3D edge line segment `(a, b)` in chunk-local coordinates (drawn with a Reversed-Z
    /// depth offset so it never z-fights against coplanar solid faces).
    pub fn push_edge_line(&mut self, a: Vec3, b: Vec3, color: [u8; 4]) {
        if !a.is_finite() || !b.is_finite() {
            return;
        }
        self.edges
            .push(Vertex3D::new(a.to_f32(), [0.0, 0.0, 1.0], color));
        self.edges
            .push(Vertex3D::new(b.to_f32(), [0.0, 0.0, 1.0], color));
        self.local_bounds.include(a);
        self.local_bounds.include(b);
    }

    fn record_pick(&mut self, prev: usize, count: usize, pick_id: PickId) {
        if pick_id.is_none() && self.pick_ids.is_empty() {
            return;
        }
        if self.pick_ids.len() < prev {
            self.pick_ids.resize(prev, [0, 0]);
        }
        let packed = pick_id.to_rg32u();
        for _ in 0..count {
            self.pick_ids.push(packed);
        }
    }

    /// Pushes a flat-shaded triangle in chunk-local coordinates.
    pub fn push_triangle(&mut self, a: Vec3, b: Vec3, c: Vec3, color: [u8; 4], pick_id: PickId) {
        if !a.is_finite() || !b.is_finite() || !c.is_finite() {
            return;
        }
        let normal = (b - a).cross(c - a).normalized().unwrap_or(Vec3::Z);
        let n = normal.to_f32();
        let prev = self.vertices.len();
        self.vertices.push(Vertex3D::new(a.to_f32(), n, color));
        self.vertices.push(Vertex3D::new(b.to_f32(), n, color));
        self.vertices.push(Vertex3D::new(c.to_f32(), n, color));
        self.local_bounds.include(a);
        self.local_bounds.include(b);
        self.local_bounds.include(c);
        self.record_pick(prev, 3, pick_id);
    }

    /// Pushes a counter-clockwise quad `(a, b, c, d)` split into two triangles.
    pub fn push_quad(
        &mut self,
        a: Vec3,
        b: Vec3,
        c: Vec3,
        d: Vec3,
        color: [u8; 4],
        pick_id: PickId,
    ) {
        self.push_triangle(a, b, c, color, pick_id);
        self.push_triangle(a, c, d, color, pick_id);
    }

    /// Pushes an axis-aligned solid box spanning `min..=max` in chunk-local coordinates.
    ///
    /// Each face can optionally carry a distinct `sub` ID (`0..=5`) when `sub_face_ids` is `true`
    /// so GPU picking distinguishes individual faces of the solid.
    pub fn push_box(
        &mut self,
        min: Vec3,
        max: Vec3,
        color: [u8; 4],
        base_pick_id: PickId,
        sub_face_ids: bool,
    ) {
        let b = Bounds3::from_corners(min, max);
        if b.is_empty() {
            return;
        }
        let p000 = Vec3::new(b.min.x, b.min.y, b.min.z);
        let p100 = Vec3::new(b.max.x, b.min.y, b.min.z);
        let p110 = Vec3::new(b.max.x, b.max.y, b.min.z);
        let p010 = Vec3::new(b.min.x, b.max.y, b.min.z);
        let p001 = Vec3::new(b.min.x, b.min.y, b.max.z);
        let p101 = Vec3::new(b.max.x, b.min.y, b.max.z);
        let p111 = Vec3::new(b.max.x, b.max.y, b.max.z);
        let p011 = Vec3::new(b.min.x, b.max.y, b.max.z);

        let face_id = |sub: u16| {
            if sub_face_ids && !base_pick_id.is_none() {
                PickId::new(base_pick_id.layer, base_pick_id.object, sub)
            } else {
                base_pick_id
            }
        };

        // +Z (top), -Z (bottom), +X (right), -X (left), +Y (back), -Y (front)
        self.push_quad(p001, p101, p111, p011, color, face_id(0));
        self.push_quad(p010, p110, p100, p000, color, face_id(1));
        self.push_quad(p100, p110, p111, p101, color, face_id(2));
        self.push_quad(p010, p000, p001, p011, color, face_id(3));
        self.push_quad(p110, p010, p011, p111, color, face_id(4));
        self.push_quad(p000, p100, p101, p001, color, face_id(5));
    }

    /// Pushes an axis-aligned solid box together with its 12 crisp CAD wireframe edges
    /// (*Shaded with Edges* style).
    pub fn push_box_with_edges(
        &mut self,
        min: Vec3,
        max: Vec3,
        face_color: [u8; 4],
        edge_color: [u8; 4],
        base_pick_id: PickId,
        sub_face_ids: bool,
    ) {
        self.push_box(min, max, face_color, base_pick_id, sub_face_ids);
        let b = Bounds3::from_corners(min, max);
        if b.is_empty() {
            return;
        }
        let p000 = Vec3::new(b.min.x, b.min.y, b.min.z);
        let p100 = Vec3::new(b.max.x, b.min.y, b.min.z);
        let p110 = Vec3::new(b.max.x, b.max.y, b.min.z);
        let p010 = Vec3::new(b.min.x, b.max.y, b.min.z);
        let p001 = Vec3::new(b.min.x, b.min.y, b.max.z);
        let p101 = Vec3::new(b.max.x, b.min.y, b.max.z);
        let p111 = Vec3::new(b.max.x, b.max.y, b.max.z);
        let p011 = Vec3::new(b.min.x, b.max.y, b.max.z);

        // Bottom 4 edges
        self.push_edge_line(p000, p100, edge_color);
        self.push_edge_line(p100, p110, edge_color);
        self.push_edge_line(p110, p010, edge_color);
        self.push_edge_line(p010, p000, edge_color);
        // Top 4 edges
        self.push_edge_line(p001, p101, edge_color);
        self.push_edge_line(p101, p111, edge_color);
        self.push_edge_line(p111, p011, edge_color);
        self.push_edge_line(p011, p001, edge_color);
        // Vertical 4 edges
        self.push_edge_line(p000, p001, edge_color);
        self.push_edge_line(p100, p101, edge_color);
        self.push_edge_line(p110, p111, edge_color);
        self.push_edge_line(p010, p011, edge_color);
    }

    /// Pushes an arrow along `axis` (unit vector) from `0` to `length` for 3D transform gizmos.
    pub fn push_axis_arrow(
        &mut self,
        axis: Vec3,
        length: f64,
        shaft_radius: f64,
        head_radius: f64,
        color: [u8; 4],
        pick_id: PickId,
    ) {
        let Some(dir) = axis.normalized() else {
            return;
        };
        let ref_v = if dir.dot(Vec3::Z).abs() < 0.9 {
            Vec3::Z
        } else {
            Vec3::X
        };
        let Some(u) = dir.cross(ref_v).normalized() else {
            return;
        };
        let v = dir.cross(u);

        let segments = 12usize;
        let shaft_len = length * 0.78;
        let tip = dir * length;

        for i in 0..segments {
            let a0 = (i as f64) * std::f64::consts::TAU / (segments as f64);
            let a1 = ((i + 1) as f64) * std::f64::consts::TAU / (segments as f64);
            let r0 = u * a0.cos() + v * a0.sin();
            let r1 = u * a1.cos() + v * a1.sin();

            // Shaft quad
            let s00 = r0 * shaft_radius;
            let s10 = r1 * shaft_radius;
            let s11 = dir * shaft_len + r1 * shaft_radius;
            let s01 = dir * shaft_len + r0 * shaft_radius;
            self.push_quad(s00, s10, s11, s01, color, pick_id);

            // Cone head triangle
            let h0 = dir * shaft_len + r0 * head_radius;
            let h1 = dir * shaft_len + r1 * head_radius;
            self.push_triangle(h0, h1, tip, color, pick_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_3d_is_compact_28_bytes() {
        assert_eq!(std::mem::size_of::<Vertex3D>(), 28);
    }

    #[test]
    fn box_builder_emits_12_triangles_with_face_sub_ids() {
        let mut m = Mesh3D::at_origin(Vec3::new(100.0, 200.0, 300.0));
        m.push_box(
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, 1.0),
            [200, 200, 200, 255],
            PickId::new(2, 55, 0),
            true,
        );
        assert_eq!(m.vertices.len(), 36);
        assert_eq!(m.pick_ids.len(), 36);
        let wb = m.world_bounds();
        assert!(wb.min.approx_eq(Vec3::new(99.0, 199.0, 299.0), 1e-12));
        assert!(wb.max.approx_eq(Vec3::new(101.0, 201.0, 301.0), 1e-12));
    }
}
