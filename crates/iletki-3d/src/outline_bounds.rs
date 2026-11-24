//! Conservative per-chunk outline bounds for a prepared viewport.

use iletki_core::{Bounds3, Camera, Mat4, Vec2, Vec3};
use iletki_wgpu::{LayerRevision, PrepareCtx};

#[derive(PartialEq)]
struct ViewKey {
    camera: Camera,
    size: [u32; 2],
    logical: Vec2,
    geometry: LayerRevision,
}

type Corners = [u32; 4];

#[derive(Default)]
pub(crate) struct OutlineBounds {
    view: Option<ViewKey>,
    revision: Option<LayerRevision>,
    view_gen: u32,
    // `(gen, cached)`: `gen != view_gen` is uncomputed; `cached == None` requires a full-view mask.
    chunks: Vec<(u32, Option<Corners>)>,
    rect: Option<[u32; 4]>,
}

pub(crate) enum OutlineUpdate {
    Unchanged,
    Selection,
    View,
}

impl OutlineBounds {
    pub(crate) fn rect(&self) -> Option<[u32; 4]> {
        self.rect
    }

    /// Updates the bounds and identifies which render data needs rebuilding.
    pub(crate) fn update<W>(
        &mut self,
        ctx: &PrepareCtx<'_, W>,
        revision: LayerRevision,
        chunk_count: usize,
        selected: impl IntoIterator<Item = (usize, Bounds3, Vec3)>,
    ) -> OutlineUpdate {
        let geometry = revision.static_only();
        let same_view = self.view.as_ref().is_some_and(|key| {
            key.camera == *ctx.camera
                && key.size == ctx.active_px
                && key.logical == ctx.logical_size
                && key.geometry == geometry
        });
        if !same_view {
            self.view = Some(ViewKey {
                camera: ctx.camera.clone(),
                size: ctx.active_px,
                logical: ctx.logical_size,
                geometry,
            });
            self.revision = None;
            self.view_gen = self.view_gen.wrapping_add(1);
            if self.view_gen == 0 {
                self.chunks.fill((0, None));
                self.view_gen = 1;
            }
        }
        if self.revision == Some(revision) {
            return OutlineUpdate::Unchanged;
        }
        self.revision = Some(revision);
        self.rect = self.combine(ctx, chunk_count, selected);
        if same_view {
            OutlineUpdate::Selection
        } else {
            OutlineUpdate::View
        }
    }

    /// Updates the outline rectangle from a single conservative `(local_bounds, origin)` envelope
    /// shared across viewports for large selections, avoiding per-chunk projection and cache scans.
    pub(crate) fn update_envelope<W>(
        &mut self,
        ctx: &PrepareCtx<'_, W>,
        revision: LayerRevision,
        envelope: Option<(Bounds3, Vec3)>,
    ) -> OutlineUpdate {
        let geometry = revision.static_only();
        let same_view = self.view.as_ref().is_some_and(|key| {
            key.camera == *ctx.camera
                && key.size == ctx.active_px
                && key.logical == ctx.logical_size
                && key.geometry == geometry
        });
        if !same_view {
            self.view = Some(ViewKey {
                camera: ctx.camera.clone(),
                size: ctx.active_px,
                logical: ctx.logical_size,
                geometry,
            });
            self.revision = None;
            self.view_gen = self.view_gen.wrapping_add(1);
            if self.view_gen == 0 {
                self.chunks.fill((0, None));
                self.view_gen = 1;
            }
        }
        if self.revision == Some(revision) {
            return OutlineUpdate::Unchanged;
        }
        self.revision = Some(revision);
        self.rect = envelope.and_then(|(bounds, origin)| {
            let projector = BoundsProjector::new(ctx.camera, ctx.logical_size, ctx.active_px);
            let [x0, y0, x1, y1] = projector.project(bounds, origin)?;
            Some([x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0)])
        });
        if same_view {
            OutlineUpdate::Selection
        } else {
            OutlineUpdate::View
        }
    }

    fn combine<W>(
        &mut self,
        ctx: &PrepareCtx<'_, W>,
        chunk_count: usize,
        selected: impl IntoIterator<Item = (usize, Bounds3, Vec3)>,
    ) -> Option<[u32; 4]> {
        let mut selected = selected.into_iter().peekable();
        let [width, height] = ctx.active_px;
        if selected.peek().is_some() && self.chunks.len() < chunk_count {
            self.chunks.resize(chunk_count, (0, None));
        }
        let projector = BoundsProjector::new(ctx.camera, ctx.logical_size, ctx.active_px);
        let [mut left, mut top, mut right, mut bottom] = [width, height, 0, 0];
        let view_gen = self.view_gen;
        for (index, bounds, origin) in selected {
            let slot = self.chunks.get_mut(index)?;
            if slot.0 != view_gen {
                slot.0 = view_gen;
                slot.1 = projector.project(bounds, origin);
            }
            let [x0, y0, x1, y1] = slot.1?;
            left = left.min(x0);
            top = top.min(y0);
            right = right.max(x1);
            bottom = bottom.max(y1);
            if left == 0 && top == 0 && right >= width && bottom >= height {
                break;
            }
        }
        Some([
            left,
            top,
            right.saturating_sub(left),
            bottom.saturating_sub(top),
        ])
    }
}

struct BoundsProjector {
    projection: Mat4,
    camera_pos: Vec3,
    width: f64,
    height: f64,
    abs_error_rows: [[f64; 4]; 3],
}

impl BoundsProjector {
    fn new(camera: &Camera, logical: Vec2, size: [u32; 2]) -> Self {
        let projection = camera.view_proj_relative(logical);
        let camera_pos = camera.position_3d();
        let [width, height] = size.map(f64::from);
        let abs_error_rows = [
            [
                projection.c0.x.abs(),
                projection.c1.x.abs(),
                projection.c2.x.abs(),
                projection.c3.x.abs(),
            ],
            [
                projection.c0.y.abs(),
                projection.c1.y.abs(),
                projection.c2.y.abs(),
                projection.c3.y.abs(),
            ],
            [
                projection.c0.w.abs(),
                projection.c1.w.abs(),
                projection.c2.w.abs(),
                projection.c3.w.abs(),
            ],
        ];
        Self {
            projection,
            camera_pos,
            width,
            height,
            abs_error_rows,
        }
    }

    fn project(&self, bounds: Bounds3, origin: Vec3) -> Option<Corners> {
        let Self {
            projection,
            camera_pos,
            width,
            height,
            abs_error_rows,
        } = *self;
        if bounds.is_empty() {
            return None;
        }
        // Use the same rounded local coordinates and origin as the vertex shader.
        let [ox, oy, oz] = (origin - camera_pos).to_f32();
        let bx0 = f64::from(bounds.min.x as f32 + ox);
        let bx1 = f64::from(bounds.max.x as f32 + ox);
        let by0 = f64::from(bounds.min.y as f32 + oy);
        let by1 = f64::from(bounds.max.y as f32 + oy);
        let bz0 = f64::from(bounds.min.z as f32 + oz);
        let bz1 = f64::from(bounds.max.z as f32 + oz);
        let mx = bx0.abs().max(bx1.abs());
        let my = by0.abs().max(by1.abs());
        let mz = bz0.abs().max(bz1.abs());
        // Conservative upper bound on f32 uniform and dot-product rounding across all 8 corners.
        let eps = 16.0 * f64::from(f32::EPSILON);
        let min_pos = f64::from(f32::MIN_POSITIVE);
        let [x_error, y_error, w_error] =
            abs_error_rows.map(|[a, b, c, d]| (a * mx + b * my + c * mz + d) * eps + min_pos);
        if !x_error.is_finite()
            || !y_error.is_finite()
            || !w_error.is_finite()
            || x_error.max(y_error) > 1.0
        {
            return None;
        }

        let c0 = projection.c0;
        let c1 = projection.c1;
        let c2 = projection.c2;
        let c3 = projection.c3;
        let scale = |c: iletki_core::Vec4, s: f64| [c.x * s, c.y * s, c.z * s, c.w * s];
        let x0 = scale(c0, bx0);
        let x1 = scale(c0, bx1);
        let y0 = scale(c1, by0);
        let y1 = scale(c1, by1);
        let z0 = [
            c2.x * bz0 + c3.x,
            c2.y * bz0 + c3.y,
            c2.z * bz0 + c3.z,
            c2.w * bz0 + c3.w,
        ];
        let z1 = [
            c2.x * bz1 + c3.x,
            c2.y * bz1 + c3.y,
            c2.z * bz1 + c3.z,
            c2.w * bz1 + c3.w,
        ];

        let [_, _, _, x0w] = x0;
        let [_, _, _, x1w] = x1;
        let [_, _, _, y0w] = y0;
        let [_, _, _, y1w] = y1;
        let [_, _, _, z0w] = z0;
        let [_, _, _, z1w] = z1;
        let w_min = x0w.min(x1w) + y0w.min(y1w) + z0w.min(z1w);
        let w_max = x0w.max(x1w) + y0w.max(y1w) + z0w.max(z1w);
        if !w_min.is_finite()
            || !w_max.is_finite()
            || w_min <= w_error
            || w_max >= f64::from(f32::MAX) * 0.5
        {
            return None;
        }

        let mut min_nx = f64::INFINITY;
        let mut max_nx = f64::NEG_INFINITY;
        let mut min_ny = f64::INFINITY;
        let mut max_ny = f64::NEG_INFINITY;
        for [xx, xy, xz, xw] in [x0, x1] {
            for [yx, yy, yz, yw] in [y0, y1] {
                let xy_x = xx + yx;
                let xy_y = xy + yy;
                let xy_z = xz + yz;
                let xy_w = xw + yw;
                for [zx, zy, zz, zw] in [z0, z1] {
                    let cx = xy_x + zx;
                    let cy = xy_y + zy;
                    let cz = xy_z + zz;
                    let cw = xy_w + zw;
                    if !cx.is_finite() || !cy.is_finite() || !cz.is_finite() {
                        return None;
                    }
                    let inv_w = 1.0 / cw;
                    let nx = cx * inv_w;
                    let ny = cy * inv_w;
                    let inv_w_err = 1.0 / (cw - w_error);
                    let dx = (x_error + nx.abs() * w_error) * inv_w_err;
                    let dy = (y_error + ny.abs() * w_error) * inv_w_err;
                    min_nx = min_nx.min(nx - dx);
                    max_nx = max_nx.max(nx + dx);
                    min_ny = min_ny.min(ny - dy);
                    max_ny = max_ny.max(ny + dy);
                }
            }
        }
        let left = ((min_nx + 1.0) * width * 0.5).floor();
        let right = ((max_nx + 1.0) * width * 0.5).ceil();
        let top = ((1.0 - max_ny) * height * 0.5).floor();
        let bottom = ((1.0 - min_ny) * height * 0.5).ceil();
        Some([
            left.clamp(0.0, width) as u32,
            top.clamp(0.0, height) as u32,
            right.clamp(0.0, width) as u32,
            bottom.clamp(0.0, height) as u32,
        ])
    }
}
