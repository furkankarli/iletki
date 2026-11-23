//! Layer trait and render contexts connecting application scenes to [`crate::ViewportHost`].

use iletki_core::{Camera, DrawPhase, DrawPhaseMask, UploadBudget, Vec2, ViewportId};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

static NEXT_LAYER_REVISION: AtomicU64 = AtomicU64::new(1);

/// Revisions used by each viewport to invalidate its cached layer output independently.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LayerRevision {
    pub(crate) static_scene: u64,
    pub(crate) dynamic: u64,
}

impl LayerRevision {
    /// Starts a tracked revision with a distinct identity for a newly created layer.
    #[must_use]
    pub fn new() -> Self {
        let revision = NEXT_LAYER_REVISION.fetch_add(1, Ordering::Relaxed);
        Self {
            static_scene: revision,
            dynamic: revision,
        }
    }

    /// Invalidates opaque and transparent scene content.
    pub fn mark_static(&mut self) {
        self.static_scene = NEXT_LAYER_REVISION.fetch_add(1, Ordering::Relaxed);
    }

    /// Invalidates outline, gizmo and overlay content.
    pub fn mark_dynamic(&mut self) {
        self.dynamic = NEXT_LAYER_REVISION.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns the revision without its dynamic component.
    /// Compare this only for work known to ignore dynamic scene content.
    #[must_use]
    pub const fn static_only(self) -> Self {
        Self {
            static_scene: self.static_scene,
            dynamic: 0,
        }
    }
}

/// Weak reference to a viewport's lifetime, for caches stored inside shared layers.
#[derive(Debug, Clone)]
pub struct ViewportLifetime(pub(crate) Weak<()>);

impl ViewportLifetime {
    /// Returns whether the viewport still exists.
    #[must_use]
    pub fn is_alive(&self) -> bool {
        self.0.strong_count() > 0
    }
}

/// Shared byte and CPU time allowance for a preparation pass.
#[derive(Debug)]
pub struct UploadTracker {
    budget: UploadBudget,
    uploaded: usize,
    upload_ms: f64,
    started_upload: Option<f64>,
    clock: Option<(fn() -> f64, f64)>,
    #[cfg(not(target_arch = "wasm32"))]
    started: std::time::Instant,
}

impl UploadTracker {
    /// Starts a budget using a native monotonic clock when available.
    #[must_use]
    pub fn new(budget: UploadBudget) -> Self {
        Self::with_clock(budget, None)
    }

    pub(crate) fn with_clock(budget: UploadBudget, clock: Option<fn() -> f64>) -> Self {
        let clock = clock.and_then(|clock| {
            let started = clock();
            started.is_finite().then_some((clock, started))
        });
        Self {
            budget,
            uploaded: 0,
            upload_ms: 0.0,
            started_upload: None,
            clock,
            #[cfg(not(target_arch = "wasm32"))]
            started: std::time::Instant::now(),
        }
    }

    /// Reserves bytes for the next indivisible upload. One upload can exceed the allowance to
    /// guarantee progress. Subsequent uploads must fit both the remaining bytes and time.
    pub fn reserve(&mut self, bytes: usize) -> bool {
        self.finish_upload();
        if bytes == 0 {
            return true;
        }
        let total = self.uploaded.saturating_add(bytes);
        if self.uploaded > 0
            && (total > self.budget.max_bytes_per_frame
                || self.upload_ms >= f64::from(self.budget.max_time_ms))
        {
            return false;
        }
        self.uploaded = total;
        self.started_upload = Some(self.elapsed_ms());
        true
    }

    /// Accounts for CPU time spent staging the reserved upload. Call immediately after staging so
    /// unrelated work, such as pipeline creation, does not consume the upload time allowance.
    pub fn finish_upload(&mut self) {
        if let Some(started) = self.started_upload.take() {
            self.upload_ms += (self.elapsed_ms() - started).max(0.0);
        }
    }

    fn elapsed_ms(&self) -> f64 {
        if let Some((clock, start)) = self.clock {
            return (clock() - start).max(0.0);
        }
        #[cfg(not(target_arch = "wasm32"))]
        return self.started.elapsed().as_secs_f64() * 1000.0;
        #[cfg(target_arch = "wasm32")]
        0.0
    }
}

/// Context handed to [`ViewportLayer::prepare`] before drawing a frame or pick pass.
pub struct PrepareCtx<'a, W = ()> {
    pub(crate) viewport_lifetime: &'a Arc<()>,
    pub(crate) section_antialiasing: bool,
    pub(crate) picking_rect: Option<[u32; 4]>,
    /// Unique identifier of the viewport being prepared (use as key for per-viewport GPU buffers
    /// when one scene is shared across multiple split viewports).
    pub viewport_id: ViewportId,
    /// Shared GPU device.
    pub device: &'a wgpu::Device,
    /// Shared GPU command queue.
    pub queue: &'a wgpu::Queue,
    /// Current viewport camera.
    pub camera: &'a Camera,
    /// Shared `@group(0)` bind group layout for [`crate::CAMERA_WGSL`].
    pub camera_bind_group_layout: &'a wgpu::BindGroupLayout,
    /// Shared application world/scene reference.
    pub world: &'a W,
    /// Active viewport size in physical pixels.
    pub active_px: [u32; 2],
    /// Active viewport size in logical points.
    pub logical_size: Vec2,
    /// Logical-to-physical DPI scale.
    pub dpi_scale: f32,
    /// Per-frame GPU upload budget for progressive streaming.
    pub budget: UploadBudget,
    /// Shared allowance for scene uploads. Reserve bytes here before staging geometry.
    pub uploads: UploadTracker,
    /// Target color texture format.
    pub color_format: wgpu::TextureFormat,
    /// Target depth texture format (`Depth32Float`).
    pub depth_format: wgpu::TextureFormat,
    /// MSAA sample count for main scene/gizmo/overlay passes (`1` or `4`).
    pub msaa_samples: u32,
}

impl<W> PrepareCtx<'_, W> {
    /// Returns the clipped physical-pixel picking rectangle, or `None` when preparing a frame.
    ///
    /// Layers may use this hint to discard objects that cannot contribute to the pick pass.
    #[must_use]
    pub const fn picking_rect_px(&self) -> Option<[u32; 4]> {
        self.picking_rect
    }

    /// Returns whether this viewport can use antialiased 3D section clipping at 4× MSAA.
    #[must_use]
    pub const fn supports_section_antialiasing(&self) -> bool {
        self.section_antialiasing
    }

    /// Returns a weak reference that lets a layer retire viewport-local GPU state after close.
    #[must_use]
    pub fn viewport_lifetime(&self) -> ViewportLifetime {
        ViewportLifetime(Arc::downgrade(self.viewport_lifetime))
    }
}

impl<W> std::fmt::Debug for PrepareCtx<'_, W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrepareCtx")
            .field("active_px", &self.active_px)
            .field("dpi_scale", &self.dpi_scale)
            .field("color_format", &self.color_format)
            .field("msaa_samples", &self.msaa_samples)
            .finish()
    }
}

/// Context handed to [`ViewportLayer::paint`] during an active `wgpu::RenderPass`.
pub struct PaintCtx<'a, W = ()> {
    /// Unique identifier of the viewport being drawn.
    pub viewport_id: ViewportId,
    /// Current viewport camera.
    pub camera: &'a Camera,
    /// Shared `@group(0)` bind group for [`crate::CAMERA_WGSL`].
    pub camera_bind_group: &'a wgpu::BindGroup,
    /// Shared application world/scene reference.
    pub world: &'a W,
    /// Active viewport size in physical pixels.
    pub active_px: [u32; 2],
    /// Active viewport size in logical points.
    pub logical_size: Vec2,
    /// Logical-to-physical DPI scale.
    pub dpi_scale: f32,
}

impl<W> std::fmt::Debug for PaintCtx<'_, W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PaintCtx")
            .field("active_px", &self.active_px)
            .field("dpi_scale", &self.dpi_scale)
            .finish()
    }
}

/// A pluggable rendering layer drawn inside a [`crate::ViewportHost`].
///
/// Applications implement this trait for 2D CAD geometry, 3D meshes, gizmos, grids, or screen-space
/// overlays. Multiple viewports can share the same `W` world reference with zero duplicated GPU
/// geometry.
pub trait ViewportLayer<W = ()>: wgpu::WasmNotSendSync {
    /// Unique identifier for this layer (embedded in 64-bit [`iletki_core::PickId::layer`]).
    fn id(&self) -> u16;

    /// Declares which [`DrawPhase`]s this layer participates in. Unused phases are skipped cleanly.
    fn phases(&self) -> DrawPhaseMask;

    /// Returns the revision of the layer's cached output after preparation.
    ///
    /// Layers that return the default revision can use [`crate::ViewportHost::mark_dirty`] for
    /// content changes. The host always tracks layer order, membership and enabled phases.
    fn revision(&self) -> LayerRevision {
        LayerRevision::default()
    }

    /// Updates GPU resources (pipelines, vertex buffers, instances) within `ctx.uploads`.
    ///
    /// Return `true` if more streaming uploads remain for subsequent frames.
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_, W>) -> bool;

    /// Returns whether this layer prepared a 4× MSAA pipeline for its outline mask.
    ///
    /// The host uses a single-sample mask when any participating layer uses the default.
    fn supports_multisampled_outline_mask(&self) -> bool {
        false
    }

    /// Records the outline mask with a 4× MSAA `R8Unorm` color attachment and no depth attachment.
    ///
    /// Layers that report support must implement this method. The camera bind group is bound.
    fn paint_multisampled_outline_mask(
        &self,
        _pass: &mut wgpu::RenderPass<'_>,
        _ctx: &PaintCtx<'_, W>,
    ) {
    }

    /// Returns a conservative physical pixel rectangle `[x, y, width, height]` for mask geometry.
    ///
    /// Return `None` to use the full viewport, or an empty rectangle for an invisible mask.
    /// The host adds space for the outline and its filter. Mask drawing must stay within this
    /// rectangle before filtering. Update the layer revision when its outline geometry changes.
    fn outline_mask_rect(&self, _ctx: &PaintCtx<'_, W>) -> Option<[u32; 4]> {
        None
    }

    /// Records draw commands for `phase` into `pass`.
    ///
    /// `@group(0)` is already bound to `ctx.camera_bind_group` and the viewport/scissor is already
    /// constrained to the active sub-rectangle.
    fn paint(&self, phase: DrawPhase, pass: &mut wgpu::RenderPass<'_>, ctx: &PaintCtx<'_, W>);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upload_tracker_enforces_shared_bytes_and_elapsed_time() {
        static TIME_MS: AtomicU64 = AtomicU64::new(0);
        fn clock() -> f64 {
            TIME_MS.load(Ordering::Relaxed) as f64
        }

        let budget = UploadBudget {
            max_bytes_per_frame: 100,
            max_time_ms: 4.0,
        };
        let mut tracker = UploadTracker::with_clock(budget, Some(clock));
        assert!(tracker.reserve(60));
        assert!(!tracker.reserve(41));
        assert!(tracker.reserve(39));
        TIME_MS.store(4, Ordering::Relaxed);
        assert!(!tracker.reserve(1));
    }
}
