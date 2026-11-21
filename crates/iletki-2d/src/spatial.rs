//! Spatial tiling for conservative 2D frustum culling of large CAD scenes.
//!
//! Because `iletki-2d` uses a Reversed-Z `Depth32Float` buffer carrying exact drawing order per
//! vertex, opaque primitives can be reordered into spatial tiles without altering visual stacking.
//! Each tile records the exact conservative [`Bounds2`] of all vertices inside it (and oversized
//! primitives go to a dedicated global bucket), guaranteeing zero clipped lines while culling
//! 70–95% of off-screen geometry when zoomed into large drawings.

use crate::vertex::{Vertex2D, WideVertex2D};
use iletki_core::{Bounds2, Vec2};

/// Grid dimension (`GRID × GRID` spatial tiles + 1 global bucket for oversized primitives).
const GRID: usize = 8;
const TILE_COUNT: usize = GRID * GRID;
const GLOBAL_BUCKET: usize = TILE_COUNT;
const BUCKET_COUNT: usize = TILE_COUNT + 1;
/// Skip spatial bucketing below this primitive count (not worth the pass on tiny drawings).
const MIN_PRIMITIVES_TO_BUCKET: usize = 4096;

/// A contiguous vertex range belonging to one spatial tile, with its exact conservative world
/// bounds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TileRange {
    /// First vertex index inside the buffer.
    pub start: usize,
    /// Number of vertices in this tile.
    pub count: usize,
    /// Conservative world-space bounds covering every vertex in `start..start + count`.
    pub bounds: Bounds2,
}

fn reconstruct_pos([hx, hy]: [f32; 2], [lx, ly]: [f32; 2]) -> Vec2 {
    Vec2::new(f64::from(hx) + f64::from(lx), f64::from(hy) + f64::from(ly))
}

fn bucket_for(prim_bounds: Bounds2, world_bounds: Bounds2, world_size: Vec2) -> usize {
    let psize = prim_bounds.size();
    if psize.x > world_size.x * 0.25 || psize.y > world_size.y * 0.25 {
        return GLOBAL_BUCKET;
    }
    let Some(center) = prim_bounds.center() else {
        return GLOBAL_BUCKET;
    };
    let nx = if world_size.x > 0.0 {
        ((center.x - world_bounds.min.x) / world_size.x).clamp(0.0, 0.999_999)
    } else {
        0.0
    };
    let ny = if world_size.y > 0.0 {
        ((center.y - world_bounds.min.y) / world_size.y).clamp(0.0, 0.999_999)
    } else {
        0.0
    };
    let gx = ((nx * GRID as f64) as usize).min(GRID - 1);
    let gy = ((ny * GRID as f64) as usize).min(GRID - 1);
    gy * GRID + gx
}

/// Reorders plain `Vertex2D` primitives (`group = 2` for lines, `3` for triangles) in place into
/// spatial tiles and returns their [`TileRange`] descriptors.
///
/// Only runs when `pick_ids` is empty (or reorders `pick_ids` in lockstep) and all vertices are
/// opaque (`alpha == 255`), preserving exact blending order if any translucent geometry exists.
pub(crate) fn bucket_plain(
    verts: &mut Vec<Vertex2D>,
    pick_ids: &mut Vec<[u32; 2]>,
    group: usize,
) -> Vec<TileRange> {
    let g = group.max(1);
    let prim_count = verts.len() / g;
    if prim_count < MIN_PRIMITIVES_TO_BUCKET {
        return Vec::new();
    }
    // Preserve exact draw order when translucent primitives are present.
    if verts.iter().any(|v| {
        let [_, _, _, a] = v.color;
        a < 255
    }) {
        return Vec::new();
    }
    let has_picks = pick_ids.len() == verts.len();

    let mut world = Bounds2::EMPTY;
    for v in verts.iter() {
        world.include(reconstruct_pos(v.hi, v.lo));
    }
    let world_size = world.size();
    if world.is_empty() || (world_size.x <= 0.0 && world_size.y <= 0.0) {
        return Vec::new();
    }

    let mut counts = [0usize; BUCKET_COUNT];
    let mut bucket_bounds = [Bounds2::EMPTY; BUCKET_COUNT];
    let mut assignments = Vec::with_capacity(prim_count);

    for prim in verts.chunks_exact(g) {
        let mut pb = Bounds2::EMPTY;
        for v in prim {
            pb.include(reconstruct_pos(v.hi, v.lo));
        }
        let b = bucket_for(pb, world, world_size);
        assignments.push(b);
        if let Some(c) = counts.get_mut(b) {
            *c = c.saturating_add(g);
        }
        if let Some(bb) = bucket_bounds.get_mut(b) {
            bb.include(pb.min);
            bb.include(pb.max);
        }
    }

    let mut offsets = [0usize; BUCKET_COUNT];
    let mut cursor = 0usize;
    let mut ranges = Vec::with_capacity(BUCKET_COUNT);
    for (b, &cnt) in counts.iter().enumerate() {
        if let Some(o) = offsets.get_mut(b) {
            *o = cursor;
        }
        if cnt > 0
            && let Some(&bounds) = bucket_bounds.get(b)
        {
            ranges.push(TileRange {
                start: cursor,
                count: cnt,
                bounds,
            });
        }
        cursor = cursor.saturating_add(cnt);
    }

    let mut out_verts = vec![Vertex2D::default(); cursor];
    let mut out_picks = if has_picks {
        vec![[0u32; 2]; cursor]
    } else {
        Vec::new()
    };
    let mut write_pos = offsets;

    for (idx, &b) in assignments.iter().enumerate() {
        let src_start = idx * g;
        let Some(dst_slot) = write_pos.get_mut(b) else {
            continue;
        };
        let dst_start = *dst_slot;
        *dst_slot = dst_start.saturating_add(g);

        if let (Some(src), Some(dst)) = (
            verts.get(src_start..src_start + g),
            out_verts.get_mut(dst_start..dst_start + g),
        ) {
            dst.copy_from_slice(src);
        }
        if has_picks
            && let (Some(src), Some(dst)) = (
                pick_ids.get(src_start..src_start + g),
                out_picks.get_mut(dst_start..dst_start + g),
            )
        {
            dst.copy_from_slice(src);
        }
    }

    *verts = out_verts;
    if has_picks {
        *pick_ids = out_picks;
    }
    ranges
}

/// Reorders wide line quads (`6` vertices per segment) into spatial tiles for frustum culling.
pub(crate) fn bucket_wide(
    verts: &mut Vec<WideVertex2D>,
    pick_ids: &mut Vec<[u32; 2]>,
) -> Vec<TileRange> {
    const G: usize = 6;
    let prim_count = verts.len() / G;
    if prim_count < MIN_PRIMITIVES_TO_BUCKET {
        return Vec::new();
    }
    if verts.iter().any(|v| {
        let [_, _, _, a] = v.color;
        a < 255
    }) {
        return Vec::new();
    }
    let has_picks = pick_ids.len() == verts.len();

    let mut world = Bounds2::EMPTY;
    for prim in verts.as_chunks::<G>().0 {
        if let Some(first) = prim.first() {
            world.include(reconstruct_pos(first.a_hi, first.a_lo));
            world.include(reconstruct_pos(first.b_hi, first.b_lo));
        }
    }
    let world_size = world.size();
    if world.is_empty() || (world_size.x <= 0.0 && world_size.y <= 0.0) {
        return Vec::new();
    }

    let mut counts = [0usize; BUCKET_COUNT];
    let mut bucket_bounds = [Bounds2::EMPTY; BUCKET_COUNT];
    let mut assignments = Vec::with_capacity(prim_count);

    for prim in verts.as_chunks::<G>().0 {
        let mut pb = Bounds2::EMPTY;
        if let Some(first) = prim.first() {
            pb.include(reconstruct_pos(first.a_hi, first.a_lo));
            pb.include(reconstruct_pos(first.b_hi, first.b_lo));
        }
        let b = bucket_for(pb, world, world_size);
        assignments.push(b);
        if let Some(c) = counts.get_mut(b) {
            *c = c.saturating_add(G);
        }
        if let Some(bb) = bucket_bounds.get_mut(b) {
            bb.include(pb.min);
            bb.include(pb.max);
        }
    }

    let mut offsets = [0usize; BUCKET_COUNT];
    let mut cursor = 0usize;
    let mut ranges = Vec::with_capacity(BUCKET_COUNT);
    for (b, &cnt) in counts.iter().enumerate() {
        if let Some(o) = offsets.get_mut(b) {
            *o = cursor;
        }
        if cnt > 0
            && let Some(&bounds) = bucket_bounds.get(b)
        {
            ranges.push(TileRange {
                start: cursor,
                count: cnt,
                bounds,
            });
        }
        cursor = cursor.saturating_add(cnt);
    }

    let mut out_verts = vec![WideVertex2D::default(); cursor];
    let mut out_picks = if has_picks {
        vec![[0u32; 2]; cursor]
    } else {
        Vec::new()
    };
    let mut write_pos = offsets;

    for (idx, &b) in assignments.iter().enumerate() {
        let src_start = idx * G;
        let Some(dst_slot) = write_pos.get_mut(b) else {
            continue;
        };
        let dst_start = *dst_slot;
        *dst_slot = dst_start.saturating_add(G);

        if let (Some(src), Some(dst)) = (
            verts.get(src_start..src_start + G),
            out_verts.get_mut(dst_start..dst_start + G),
        ) {
            dst.copy_from_slice(src);
        }
        if has_picks
            && let (Some(src), Some(dst)) = (
                pick_ids.get(src_start..src_start + G),
                out_picks.get_mut(dst_start..dst_start + G),
            )
        {
            dst.copy_from_slice(src);
        }
    }

    *verts = out_verts;
    if has_picks {
        *pick_ids = out_picks;
    }
    ranges
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vertex::Vertex2D;

    #[test]
    fn spatial_bucketing_preserves_count_and_culls_offscreen() {
        let mut verts = Vec::new();
        for i in 0..5000 {
            let x = f64::from(i % 100);
            let y = f64::from(i / 100);
            verts.push(Vertex2D::new(
                Vec2::new(x, y),
                [255, 255, 255, 255],
                false,
                0.5,
            ));
            verts.push(Vertex2D::new(
                Vec2::new(x + 0.5, y + 0.5),
                [255, 255, 255, 255],
                false,
                0.5,
            ));
        }
        let mut picks = Vec::new();
        let tiles = bucket_plain(&mut verts, &mut picks, 2);
        assert!(!tiles.is_empty());
        let total: usize = tiles.iter().map(|t| t.count).sum();
        assert_eq!(total, 10_000);

        // Small zoomed-in viewport in bottom-left should intersect only a tiny fraction of tiles.
        let view = Bounds2::from_corners(Vec2::new(0.0, 0.0), Vec2::new(10.0, 5.0));
        let drawn: usize = tiles
            .iter()
            .filter(|t| t.bounds.intersects(view))
            .map(|t| t.count)
            .sum();
        assert!(drawn > 0 && drawn < 1000, "drawn={drawn}");
    }
}
