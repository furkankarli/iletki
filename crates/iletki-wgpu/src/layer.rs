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
    pub(crate) selection_id_attachment: bool,
    pub(crate) gpu_culling: bool,
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
    /// Target depth texture format (`Depth32Float`, or `Depth32FloatStencil8` when a layer
    /// requests stencil and the device supports it).
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

    /// Returns whether the upcoming color pass attaches a second [`crate::SELECTION_ID_FORMAT`]
    /// target (`@location(1)`) for zero-extra-pass selection previews.
    #[must_use]
    pub const fn selection_id_attachment(&self) -> bool {
        self.selection_id_attachment
    }

    /// Returns whether the host viewport supports two-phase GPU culling and Hi-Z occlusion
    /// culling on this frame.
    #[must_use]
    pub const fn supports_gpu_culling(&self) -> bool {
        self.gpu_culling
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

/// Context handed to two-phase GPU culling hooks before and between scene passes.
pub struct GpuCullCtx<'a, W = ()> {
    /// Unique identifier of the viewport being rendered.
    pub viewport_id: ViewportId,
    /// Shared GPU device.
    pub device: &'a wgpu::Device,
    /// Shared GPU command queue.
    pub queue: &'a wgpu::Queue,
    /// Current viewport camera.
    pub camera: &'a Camera,
    /// Shared application world/scene reference.
    pub world: &'a W,
    /// Active viewport size in physical pixels.
    pub active_px: [u32; 2],
    /// Active viewport size in logical points.
    pub logical_size: Vec2,
}

impl<W> std::fmt::Debug for GpuCullCtx<'_, W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuCullCtx")
            .field("viewport_id", &self.viewport_id)
            .field("active_px", &self.active_px)
            .finish()
    }
}

/// Optional render-graph pass hook stage executed inside [`crate::ViewportHost`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PassHookStage {
    /// Runs before the primary scene color pass (e.g. shadow maps or custom compute preparation).
    PreScene = 0,
    /// Runs immediately after all opaque geometry has been drawn into the scene color/depth
    /// attachments (and before transparent draws or MSAA resolve), with full access to the scene
    /// depth/stencil target (e.g. GPU stencil section caps).
    PostOpaque = 1,
    /// Runs after scene color has been resolved to a single-sample texture while the scene depth
    /// buffer is still preserved (e.g. half-resolution GTAO or custom screen-space post passes).
    PostSceneResolved = 2,
}

impl PassHookStage {
    /// Returns the single-bit [`PassHookMask`] corresponding to this stage.
    #[must_use]
    pub const fn mask(self) -> PassHookMask {
        PassHookMask(1 << (self as u8))
    }
}

/// Bitmask of optional [`PassHookStage`]s requested by a [`ViewportLayer`] on a frame.
///
/// Unused stages are pruned from the viewport pass graph with zero GPU or CPU overhead.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct PassHookMask(u8);

impl PassHookMask {
    /// Requests no optional pass hooks (zero overhead).
    pub const NONE: Self = Self(0);
    /// Requests [`PassHookStage::PreScene`].
    pub const PRE_SCENE: Self = PassHookStage::PreScene.mask();
    /// Requests [`PassHookStage::PostOpaque`].
    pub const POST_OPAQUE: Self = PassHookStage::PostOpaque.mask();
    /// Requests [`PassHookStage::PostSceneResolved`].
    pub const POST_SCENE_RESOLVED: Self = PassHookStage::PostSceneResolved.mask();
    /// Requests every optional pass hook stage.
    pub const ALL: Self = Self(0x07);

    /// Returns `true` if no hook bits are set.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Combines two hook masks.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Returns `true` if `stage` is enabled in this mask.
    #[must_use]
    pub const fn contains(self, stage: PassHookStage) -> bool {
        (self.0 & stage.mask().0) != 0
    }
}

/// Typed context handed to [`ViewportLayer::encode_pass_hook`] during optional frame stages.
pub struct PassHookCtx<'a, W = ()> {
    /// Unique identifier of the viewport being rendered.
    pub viewport_id: ViewportId,
    /// Shared GPU device.
    pub device: &'a wgpu::Device,
    /// Shared GPU command queue.
    pub queue: &'a wgpu::Queue,
    /// Current viewport camera.
    pub camera: &'a Camera,
    /// Shared `@group(0)` bind group for [`crate::CAMERA_WGSL`].
    pub camera_bind_group: &'a wgpu::BindGroup,
    /// Shared application world/scene reference.
    pub world: &'a W,
    /// Active viewport size in physical pixels.
    pub active_px: [u32; 2],
    /// Allocated block-aligned backing target size in physical pixels.
    pub allocated_px: [u32; 2],
    /// Active viewport size in logical points.
    pub logical_size: Vec2,
    /// Logical-to-physical DPI scale.
    pub dpi_scale: f32,
    /// Target color texture format.
    pub color_format: wgpu::TextureFormat,
    /// Target depth (or depth-stencil) texture format.
    pub depth_format: wgpu::TextureFormat,
    /// MSAA sample count of the primary scene attachments (`1` or `4`).
    pub msaa_samples: u32,
    /// Monotonic generation of the viewport's backing targets (use to invalidate cached bind groups).
    pub target_generation: u64,
    /// Scene color attachment view (MSAA when `msaa_samples > 1`, otherwise 1×).
    pub color_attachment_view: &'a wgpu::TextureView,
    /// Resolved 1× scene color view when available (always `Some` in [`PassHookStage::PostSceneResolved`]).
    pub resolved_color_view: Option<&'a wgpu::TextureView>,
    /// Scene depth (or depth-stencil) attachment view (`TextureAspect::All`).
    pub depth_stencil_view: &'a wgpu::TextureView,
    /// Depth-only sampleable view (`TextureAspect::DepthOnly`), or `None` when depth sampling is
    /// unavailable on the active backend (e.g. MSAA depth on OpenGL).
    pub depth_sample_view: Option<&'a wgpu::TextureView>,
    /// Active MRT selection slot-ID attachment view when color-frame selection is enabled.
    pub selection_id_view: Option<&'a wgpu::TextureView>,
}

impl<W> std::fmt::Debug for PassHookCtx<'_, W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PassHookCtx")
            .field("viewport_id", &self.viewport_id)
            .field("active_px", &self.active_px)
            .field("color_format", &self.color_format)
            .field("depth_format", &self.depth_format)
            .field("msaa_samples", &self.msaa_samples)
            .finish()
    }
}

/// Context handed to [`ViewportLayer::paint`] during an active `wgpu::RenderPass`.
pub struct PaintCtx<'a, W = ()> {
    pub(crate) selection_id_attachment: bool,
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

impl<W> PaintCtx<'_, W> {
    /// Returns whether the active color render pass includes a second
    /// [`crate::SELECTION_ID_FORMAT`] (`Rg16Uint`) color attachment at `@location(1)`.
    #[must_use]
    pub const fn selection_id_attachment(&self) -> bool {
        self.selection_id_attachment
    }
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

    /// Returns whether this layer can paint into a color pass that attaches a second
    /// [`crate::SELECTION_ID_FORMAT`] (`@location(1)`) slot ID target.
    ///
    /// Layers that return `true` must use pipelines whose fragment targets include
    /// [`crate::SELECTION_ID_FORMAT`] at slot 1 whenever [`PaintCtx::selection_id_attachment`]
    /// is `true` (writing either 1-based slot IDs or masking writes with `ColorWrites::empty()`).
    fn supports_color_frame_selection(&self) -> bool {
        false
    }

    /// Returns the exclusive upper bound on 0-based slot IDs written by this layer during
    /// color-frame selection (`0` if this layer does not write slot IDs).
    fn selection_slot_count(&self) -> u32 {
        0
    }

    /// Returns whether this layer prepared two-phase GPU culling and Hi-Z occlusion culling for
    /// `viewport_id` on the current frame.
    fn uses_gpu_culling(&self, _viewport_id: ViewportId) -> bool {
        false
    }

    /// Records phase-1 GPU compute culling work before the primary opaque render pass.
    fn encode_gpu_cull_phase1(
        &mut self,
        _encoder: &mut wgpu::CommandEncoder,
        _ctx: &GpuCullCtx<'_, W>,
    ) {
    }

    /// Records phase-2 GPU Hi-Z occlusion culling work against the phase-1 Reversed-Z min-depth
    /// pyramid (`hiz_view`, power-of-two mip 0 extent `hiz_size`, `hiz_mip_count` levels and
    /// monotonic `hiz_generation` for bind-group caching).
    fn encode_gpu_cull_phase2(
        &mut self,
        _encoder: &mut wgpu::CommandEncoder,
        _hiz_view: &wgpu::TextureView,
        _hiz_size: [u32; 2],
        _hiz_mip_count: u32,
        _hiz_generation: u64,
        _ctx: &GpuCullCtx<'_, W>,
    ) {
    }

    /// Records phase-2 draw commands (newly visible opaque, translucent and edge geometry) into
    /// the second scene render pass after Hi-Z occlusion culling.
    fn paint_phase2(
        &self,
        _phase: DrawPhase,
        _pass: &mut wgpu::RenderPass<'_>,
        _ctx: &PaintCtx<'_, W>,
    ) {
    }

    /// Returns whether this layer requires a stencil aspect (`Depth32FloatStencil8`) on the scene
    /// depth attachment for `viewport_id` (for example, GPU stencil section caps).
    ///
    /// When no participating layer returns `true`, the viewport keeps plain `Depth32Float` with
    /// zero extra memory overhead.
    fn requires_stencil(&self, _viewport_id: ViewportId) -> bool {
        false
    }

    /// Declares which optional [`PassHookStage`]s this layer wants executed on `viewport_id` for
    /// the current frame. Unrequested stages are pruned from the pass graph before encoding.
    fn pass_hooks(&self, _viewport_id: ViewportId) -> PassHookMask {
        PassHookMask::NONE
    }

    /// Records custom compute or render work for an opt-in [`PassHookStage`].
    fn encode_pass_hook(
        &mut self,
        _stage: PassHookStage,
        _encoder: &mut wgpu::CommandEncoder,
        _ctx: &PassHookCtx<'_, W>,
    ) {
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
