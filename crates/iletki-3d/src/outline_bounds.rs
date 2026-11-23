//! Conservative per-chunk outline bounds for a prepared viewport.

use iletki_core::{Bounds3, Camera, Mat4, Vec2, Vec3, Vec4};
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
    // Empty entries are uncomputed. A computed None requires a full-view mask.
    chunks: Vec<Option<Option<Corners>>>,
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
            self.chunks.fill(None);
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

    fn combine<W>(
        &mut self,
        ctx: &PrepareCtx<'_, W>,
        chunk_count: usize,
        selected: impl IntoIterator<Item = (usize, Bounds3, Vec3)>,
    ) -> Option<[u32; 4]> {
        let mut selected = selected.into_iter().peekable();
        let [width, height] = ctx.active_px;
        if selected.peek().is_some() && self.chunks.len() < chunk_count {
            self.chunks.resize(chunk_count, None);
        }
        let projector = BoundsProjector::new(ctx.camera, ctx.logical_size, ctx.active_px);
        let [mut left, mut top, mut right, mut bottom] = [width, height, 0, 0];
        for (index, bounds, origin) in selected {
            let slot = self.chunks.get_mut(index)?;
            let [x0, y0, x1, y1] =
                (*slot.get_or_insert_with(|| projector.project(bounds, origin)))?;
            left = left.min(x0);
            top = top.min(y0);
            right = right.max(x1);
            bottom = bottom.max(y1);
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
    error_rows: [[f64; 4]; 3],
}

impl BoundsProjector {
    fn new(camera: &Camera, logical: Vec2, size: [u32; 2]) -> Self {
        let projection = camera.view_proj_relative(logical);
        let camera_pos = camera.position_3d();
        let [width, height] = size.map(f64::from);
        let error_rows = [
            [
                projection.c0.x,
                projection.c1.x,
                projection.c2.x,
                projection.c3.x,
            ],
            [
                projection.c0.y,
                projection.c1.y,
                projection.c2.y,
                projection.c3.y,
            ],
            [
                projection.c0.w,
                projection.c1.w,
                projection.c2.w,
                projection.c3.w,
            ],
        ];
        Self {
            projection,
            camera_pos,
            width,
            height,
            error_rows,
        }
    }

    fn project(&self, bounds: Bounds3, origin: Vec3) -> Option<Corners> {
        let Self {
            projection,
            camera_pos,
            width,
            height,
            error_rows,
        } = *self;
        let mut min = [width, height];
        let mut max = [0.0_f64; 2];
        if bounds.is_empty() {
            return None;
        }
        let [ox, oy, oz] = (origin - camera_pos).to_f32();
        for x in [bounds.min.x, bounds.max.x] {
            for y in [bounds.min.y, bounds.max.y] {
                for z in [bounds.min.z, bounds.max.z] {
                    // Use the same rounded local coordinates and origin as the vertex shader.
                    let point = Vec4::new(
                        f64::from(x as f32 + ox),
                        f64::from(y as f32 + oy),
                        f64::from(z as f32 + oz),
                        1.0,
                    );
                    let clip = projection.mul_vec4(point);
                    // Bound f32 uniform and dot-product rounding before the perspective divide.
                    let [x_error, y_error, w_error] = error_rows.map(|[a, b, c, d]| {
                        let magnitude = (a * point.x).abs()
                            + (b * point.y).abs()
                            + (c * point.z).abs()
                            + d.abs();
                        magnitude * (16.0 * f64::from(f32::EPSILON)) + f64::from(f32::MIN_POSITIVE)
                    });
                    if ![clip.x, clip.y, clip.z, clip.w]
                        .into_iter()
                        .all(f64::is_finite)
                        || clip.w <= w_error
                        || clip.w >= f64::from(f32::MAX) * 0.5
                        || x_error.max(y_error) > 1.0
                    {
                        return None;
                    }
                    let nx = clip.x / clip.w;
                    let ny = clip.y / clip.w;
                    let dx = (x_error + nx.abs() * w_error) / (clip.w - w_error);
                    let dy = (y_error + ny.abs() * w_error) / (clip.w - w_error);
                    let [left, top] = [
                        (nx - dx + 1.0) * width * 0.5,
                        (1.0 - ny - dy) * height * 0.5,
                    ];
                    let [right, bottom] = [
                        (nx + dx + 1.0) * width * 0.5,
                        (1.0 - ny + dy) * height * 0.5,
                    ];
                    let [l, t] = min;
                    let [r, b] = max;
                    min = [l.min(left), t.min(top)];
                    max = [r.max(right), b.max(bottom)];
                }
            }
        }
        let [left, top] = min.map(f64::floor);
        let [right, bottom] = max.map(f64::ceil);
        Some([
            left.clamp(0.0, width) as u32,
            top.clamp(0.0, height) as u32,
            right.clamp(0.0, width) as u32,
            bottom.clamp(0.0, height) as u32,
        ])
    }
}
