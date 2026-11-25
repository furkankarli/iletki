//! [`ViewportHost`]: dual-mode (`OnDemand` / `Continuous`), dual-path (`OwnedTarget` /
//! `BorrowedPass`) GPU viewport orchestrator with 64-bit ID+depth picking and headless readback.

use crate::blit::{BlitResources, DepthCopyPipeline};
use crate::error::ViewportError;
use crate::hiz::{HiZPyramid, supports_hiz};
use crate::layer::{
    GpuCullCtx, LayerRevision, PaintCtx, PassHookCtx, PassHookMask, PassHookStage, PrepareCtx,
    UploadTracker, ViewportLayer,
};
use crate::outline::OutlineEngine;
use crate::pass_graph::{FramePlanInput, PassGraph, PassId};
use crate::selection_preview::{
    PendingSlotPreview, SelectionPreviewReducer, SelectionPreviewSnapshot, SelectionPreviewStats,
    SlotStagingCache, StagedSlotPreview, supports_compute_preview,
};
use crate::target::{
    DEPTH_FORMAT, DEPTH_STENCIL_FORMAT, PickTargets, SelectionPreviewTargets, TargetSet,
    ViewportConfig, color_attachment_features, validate_depth_attachment, validate_samples,
};
use crate::uniforms::CameraGpu;
use iletki_core::{
    Camera, CameraRig, DirtyFlags, DrawPhase, DrawPhaseMask, OutlineConfig, PickHit, PickId,
    RigOutcome, TargetSize, Vec2, ViewportCommand, ViewportId, ViewportInput, ViewportMode,
};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

static NEXT_VIEWPORT_ID: AtomicU64 = AtomicU64::new(1);

/// Maximum readback allocation cap (256 MiB) to guard against hostile sizes.
const MAX_READBACK_BYTES: usize = 256 * 1024 * 1024;
const MAX_CACHED_PICK_BYTES: u64 = 512 * 512 * 12;

/// Descriptor of an external host `wgpu::RenderPass` for [`ViewportHost::prepare_for_pass`] and
/// [`ViewportHost::paint_into_pass`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PassTargetDesc {
    /// Color attachment format of the host pass.
    pub format: wgpu::TextureFormat,
    /// MSAA sample count of the host pass (`1` or `4`).
    pub sample_count: u32,
    /// Optional depth-stencil attachment format of the host pass.
    pub depth_format: Option<wgpu::TextureFormat>,
    /// If `true`, composites with premultiplied alpha over whatever the host already drew behind
    /// the viewport (e.g. a background color or UI grid).
    pub blend_alpha: bool,
}

/// Per-frame telemetry returned inside [`FrameOutput`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrameStats {
    /// `true` if the offscreen static scene pass ran on this frame.
    pub static_rebuilt: bool,
    /// `true` if any dynamic post-static pass (gizmo/overlay) ran on this frame.
    pub dynamic_drawn: bool,
    /// `true` if the backing GPU target textures were reallocated on this frame.
    pub target_reallocated: bool,
    /// `true` if any layer reported remaining progressive upload work.
    pub streaming_pending: bool,
    /// Number of layer `paint` calls executed on this frame.
    pub layer_draws: u32,
}

/// Zero-copy GPU output of [`ViewportHost::render_to_texture`].
pub struct FrameOutput<'a> {
    /// Resolved 1x texture view ready to be displayed by any UI framework.
    pub texture_view: &'a wgpu::TextureView,
    /// Normalized UV ceiling `(active_px / allocated_px)` for cropping the 256 px block texture.
    pub uv_max: [f32; 2],
    /// Active viewport size in physical pixels.
    pub active_px: [u32; 2],
    /// Per-frame execution statistics.
    pub stats: FrameStats,
}

impl std::fmt::Debug for FrameOutput<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameOutput")
            .field("uv_max", &self.uv_max)
            .field("active_px", &self.active_px)
            .field("stats", &self.stats)
            .finish()
    }
}

/// Tightly packed RGBA8 image read back from the GPU (suitable for headless tests and PNG export).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbaImage {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Row-major `RGBA8` bytes (`width * height * 4`).
    pub pixels: Vec<u8>,
}

impl RgbaImage {
    /// Returns the `[r, g, b, a]` pixel at `(x, y)`, or `None` if out of bounds.
    #[must_use]
    pub fn pixel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let idx = (y as usize)
            .checked_mul(self.width as usize)?
            .checked_add(x as usize)?
            .checked_mul(4)?;
        let slice = self.pixels.get(idx..idx.checked_add(4)?)?;
        let [r, g, b, a] = slice else {
            return None;
        };
        Some([*r, *g, *b, *a])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActiveOutput {
    StaticCache,
    DynamicOutput,
}

struct LayerState {
    id: u16,
    phases: DrawPhaseMask,
    revision: LayerRevision,
}

/// Independent GPU viewport instance sharing a host `wgpu::Device` and `wgpu::Queue`.
///
/// Multiple [`ViewportHost`] instances can render the same application `World` from independent
/// cameras (e.g. CAD quad views or 2D plan + 3D perspective) with zero duplicated scene buffers.
pub struct ViewportHost {
    id: ViewportId,
    lifetime: Arc<()>,
    config: ViewportConfig,
    section_antialiasing: bool,
    camera: Camera,
    dpi_scale: f32,
    dirty: DirtyFlags,
    camera_gpu: CameraGpu,
    blit: BlitResources,
    targets: TargetSet,
    target_generation: u64,
    outline: Option<OutlineEngine>,
    pick_targets: Option<PickTargets>,
    depth_copy: Option<DepthCopyPipeline>,
    pick_readback: PickBufferCache,
    selection_preview_rect: Option<[u32; 4]>,
    selection_targets: Option<SelectionPreviewTargets>,
    selection_reducer: Option<SelectionPreviewReducer>,
    selection_staging: SlotStagingCache,
    selection_id_valid: bool,
    staged_selection_preview: Option<StagedSlotPreview>,
    selection_preview_stats: Arc<Mutex<SelectionPreviewStats>>,
    hiz_pyramid: Option<HiZPyramid>,
    active_output: ActiveOutput,
    static_valid: bool,
    pending_reallocated: bool,
    streaming_pending: bool,
    upload_clock: Option<fn() -> f64>,
    cached_layers: Vec<LayerState>,
    prepared_layers: Vec<LayerState>,
}

impl std::fmt::Debug for ViewportHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ViewportHost")
            .field("id", &self.id)
            .field("config", &self.config)
            .field("camera", &self.camera)
            .field("dpi_scale", &self.dpi_scale)
            .field("dirty", &self.dirty)
            .field("targets", &self.targets)
            .field("active_output", &self.active_output)
            .field("static_valid", &self.static_valid)
            .finish()
    }
}

impl ViewportHost {
    /// Creates a new viewport host on an existing `wgpu::Device`.
    /// Returns an error for unsupported color, depth or multisample attachments.
    pub fn new(device: &wgpu::Device, config: ViewportConfig) -> Result<Self, ViewportError> {
        Self::create(device, config, false)
    }

    /// Creates a viewport host and checks the adapter for antialiased section clipping support.
    ///
    /// Pass the adapter that created `device`. Built-in 3D layers use the capability for
    /// 4× MSAA section cuts. Enable [`wgpu::Features::CLIP_DISTANCES`] at device creation for
    /// hardware clipping when supported. Other non-OpenGL devices need multisampled shading.
    /// OpenGL uses pixel-rate clipping. Returns an error for unsupported color, depth or
    /// multisample attachments.
    pub fn new_with_adapter(
        device: &wgpu::Device,
        adapter: &wgpu::Adapter,
        config: ViewportConfig,
    ) -> Result<Self, ViewportError> {
        let info = adapter.get_info();
        // OpenGL's sample interpolation qualifier can require a higher GLSL version than
        // the one needed for the adapter's MULTISAMPLED_SHADING flag.
        // The OpenGL sample qualifier can require a higher GLSL version than the
        // adapter's MULTISAMPLED_SHADING flag guarantees.
        let section_antialiasing = config.msaa_samples == 4
            && info.backend != wgpu::Backend::Gl
            && info == device.adapter_info()
            && (device.features().contains(wgpu::Features::CLIP_DISTANCES)
                || adapter
                    .get_downlevel_capabilities()
                    .flags
                    .contains(wgpu::DownlevelFlags::MULTISAMPLED_SHADING));
        Self::create(device, config, section_antialiasing)
    }

    fn create(
        device: &wgpu::Device,
        config: ViewportConfig,
        section_antialiasing: bool,
    ) -> Result<Self, ViewportError> {
        config.validate(device)?;
        let id = ViewportId(NEXT_VIEWPORT_ID.fetch_add(1, Ordering::Relaxed));
        let msaa = config.msaa_samples;
        let dpi_scale = if config.dpi_scale.is_finite() && config.dpi_scale > 0.0 {
            config.dpi_scale
        } else {
            1.0
        };
        let camera_gpu = CameraGpu::new(device);
        let mut blit = BlitResources::new(device);
        // Pre-create the internal blit pipeline used when copying static_cache into output target.
        blit.ensure_pipeline(device, config.color_format, msaa, Some(DEPTH_FORMAT), false);
        let targets = TargetSet::new(device, &blit, config.initial_px, config.color_format, msaa);
        Ok(Self {
            id,
            lifetime: Arc::new(()),
            config,
            section_antialiasing,
            camera: Camera::default(),
            dpi_scale,
            dirty: DirtyFlags::ALL,
            camera_gpu,
            blit,
            targets,
            target_generation: 1,
            outline: None,
            pick_targets: None,
            depth_copy: None,
            pick_readback: Arc::default(),
            selection_preview_rect: None,
            selection_targets: None,
            selection_reducer: None,
            selection_staging: Arc::default(),
            selection_id_valid: false,
            staged_selection_preview: None,
            selection_preview_stats: Arc::default(),
            hiz_pyramid: None,
            active_output: ActiveOutput::StaticCache,
            static_valid: false,
            pending_reallocated: false,
            streaming_pending: false,
            upload_clock: None,
            cached_layers: Vec::new(),
            prepared_layers: Vec::new(),
        })
    }

    /// Returns this viewport's unique [`ViewportId`].
    #[must_use]
    pub const fn viewport_id(&self) -> ViewportId {
        self.id
    }

    /// Updates the Jump-Flooding selection outline style and marks [`DirtyFlags::SELECTION`].
    pub fn set_outline(&mut self, outline: OutlineConfig) {
        self.config.outline = outline;
        self.dirty.insert(DirtyFlags::SELECTION);
    }

    /// Returns the viewport's configuration.
    #[must_use]
    pub fn config(&self) -> ViewportConfig {
        self.config
    }

    /// Sets a monotonic, millisecond clock supplied by the host. Web callers can supply their
    /// browser clock to enforce the soft upload time limit alongside the byte limit.
    pub fn set_upload_clock(&mut self, clock: Option<fn() -> f64>) {
        self.upload_clock = clock;
    }

    /// Returns the active [`ViewportMode`].
    #[must_use]
    pub fn mode(&self) -> ViewportMode {
        self.config.mode
    }

    /// Returns the unique identifier of this viewport instance (used by shared layers to maintain
    /// per-viewport GPU uniform buffers in multi-viewport setups).
    #[must_use]
    pub const fn id(&self) -> ViewportId {
        self.id
    }

    /// Switches between [`ViewportMode::OnDemand`] and [`ViewportMode::Continuous`].
    pub fn set_mode(&mut self, mode: ViewportMode) {
        if self.config.mode != mode {
            self.config.mode = mode;
            self.dirty.insert(DirtyFlags::ALL);
            self.static_valid = false;
            self.selection_id_valid = false;
        }
    }

    /// Returns whether `device` and this viewport's target configuration support MRT color-frame
    /// selection previews with compute bitset reduction.
    #[must_use]
    pub fn supports_color_frame_selection(&self, device: &wgpu::Device) -> bool {
        supports_compute_preview(device, self.targets.format(), self.targets.msaa_samples())
    }

    /// Returns the active scene depth attachment format (`Depth32Float`, or
    /// `Depth32FloatStencil8` when a layer requested stencil support).
    #[must_use]
    pub fn depth_format(&self) -> wgpu::TextureFormat {
        self.targets.depth_format()
    }

    /// Returns whether `device` supports upgrading this viewport's depth target to
    /// [`crate::DEPTH_STENCIL_FORMAT`] (`Depth32FloatStencil8`) at the active MSAA sample count.
    #[must_use]
    pub fn supports_stencil(&self, device: &wgpu::Device) -> bool {
        validate_depth_attachment(device, DEPTH_STENCIL_FORMAT, self.targets.msaa_samples()).is_ok()
    }

    /// Returns the monotonic generation counter of this viewport's backing render targets.
    #[must_use]
    pub const fn target_generation(&self) -> u64 {
        self.target_generation
    }

    /// Enables or disables color-frame selection preview capture for the next frame.
    ///
    /// Pass `Some([x, y, width, height])` in physical pixels during an active marquee drag so the
    /// normal color pass writes compact slot IDs to a second attachment and reduces the rectangle
    /// via compute in the same command encoder. Pass `None` when the gesture ends to release the
    /// auxiliary target immediately.
    pub fn set_color_frame_selection_rect(&mut self, rect_px: Option<[u32; 4]>) {
        self.selection_preview_rect = rect_px;
        if rect_px.is_none() {
            self.selection_targets = None;
            self.selection_id_valid = false;
            if self.staged_selection_preview.take().is_some()
                && let Ok(mut stats) = self.selection_preview_stats.lock()
            {
                stats.superseded = stats.superseded.saturating_add(1);
            }
        }
    }

    /// Takes the staged color-frame selection preview produced by the most recent
    /// [`Self::render_to_texture`] or [`Self::prepare_for_pass`] call and starts mapping its
    /// compact bitset staging buffer.
    ///
    /// Call immediately **after** submitting the frame's `CommandEncoder` to `queue`.
    pub fn take_color_frame_selection_preview(
        &mut self,
        device: &wgpu::Device,
    ) -> Option<PendingSlotPreview> {
        let staged = self.staged_selection_preview.take()?;
        Some(PendingSlotPreview::from_staged(device, staged))
    }

    /// Returns cumulative lifecycle counters for color-frame selection previews on this viewport.
    #[must_use]
    pub fn selection_preview_stats(&self) -> SelectionPreviewStats {
        self.selection_preview_stats
            .lock()
            .map(|stats| *stats)
            .unwrap_or_default()
    }

    /// Records that a completed color-frame selection preview was applied and displayed.
    pub fn record_selection_preview_displayed(&self) {
        if let Ok(mut stats) = self.selection_preview_stats.lock() {
            stats.displayed = stats.displayed.saturating_add(1);
        }
    }

    /// Records that a color-frame selection preview was discarded as stale before display.
    pub fn record_selection_preview_stale(&self) {
        if let Ok(mut stats) = self.selection_preview_stats.lock() {
            stats.stale = stats.stale.saturating_add(1);
        }
    }

    /// Records that a color-frame selection preview was superseded by a newer request.
    pub fn record_selection_preview_superseded(&self) {
        if let Ok(mut stats) = self.selection_preview_stats.lock() {
            stats.superseded = stats.superseded.saturating_add(1);
        }
    }

    /// Updates the background clear color and invalidates the static scene cache.
    pub fn set_clear_color(&mut self, rgba: [f64; 4]) {
        self.config.clear_color = rgba;
        self.dirty.insert(DirtyFlags::STATIC_SCENE);
    }

    /// Returns the shared `@group(0)` bind group layout for [`crate::CAMERA_WGSL`].
    #[must_use]
    pub fn camera_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        self.camera_gpu.layout()
    }

    /// Returns an immutable reference to the camera.
    #[must_use]
    pub fn camera(&self) -> &Camera {
        &self.camera
    }

    /// Returns a mutable reference to the camera and marks [`DirtyFlags::CAMERA`].
    pub fn camera_mut(&mut self) -> &mut Camera {
        self.dirty.insert(DirtyFlags::CAMERA);
        &mut self.camera
    }

    /// Marks the given dirty flags so the next render updates the affected passes.
    pub fn mark_dirty(&mut self, flags: DirtyFlags) {
        self.dirty.insert(flags);
    }

    /// Returns currently pending dirty flags.
    #[must_use]
    pub fn dirty(&self) -> DirtyFlags {
        self.dirty
    }

    /// Returns the current active and allocated target dimensions.
    #[must_use]
    pub fn size(&self) -> TargetSize {
        self.targets.size()
    }

    /// Active logical size in points (`active_px / dpi_scale`).
    #[must_use]
    pub fn logical_size(&self) -> Vec2 {
        let [w, h] = self.targets.size().active_px;
        let inv = 1.0 / f64::from(self.dpi_scale);
        Vec2::new(f64::from(w) * inv, f64::from(h) * inv)
    }

    /// Updates physical pixel dimensions and DPI scale. The physical size is capped at the
    /// device's maximum texture dimension, which is reflected in [`Self::size`].
    ///
    /// Reallocates backing GPU textures only when crossing a 256 px block boundary or after the
    /// shrink hysteresis window.
    pub fn resize(&mut self, device: &wgpu::Device, physical_px: [u32; 2], dpi_scale: f32) {
        let prev = self.targets.size().active_px;
        let dpi = if dpi_scale.is_finite() && dpi_scale > 0.0 {
            dpi_scale
        } else {
            self.dpi_scale
        };
        let reallocated = self.targets.resize(device, &self.blit, physical_px);
        let next = self.targets.size().active_px;
        if reallocated || next != prev || (dpi - self.dpi_scale).abs() > 1e-4 {
            self.dpi_scale = dpi;
            self.dirty.insert(DirtyFlags::SIZE);
            if reallocated {
                self.target_generation = self.target_generation.wrapping_add(1);
                self.static_valid = false;
                self.selection_id_valid = false;
                self.pending_reallocated = true;
            }
        }
    }

    /// Routes input to the camera rig and returns its requested host commands.
    /// Use [`Self::handle_input`] to also receive whether the rig consumed the event.
    pub fn on_input(
        &mut self,
        input: &ViewportInput,
        rig: &mut dyn CameraRig,
    ) -> Vec<ViewportCommand> {
        self.handle_input(input, rig).commands
    }

    /// Routes input to the rig, preserving its decision about further event dispatch.
    pub fn handle_input(&mut self, input: &ViewportInput, rig: &mut dyn CameraRig) -> RigOutcome {
        let logical = self.logical_size();
        let outcome = rig.handle_input(input, &mut self.camera, logical);
        if outcome.camera_changed {
            self.dirty.insert(DirtyFlags::CAMERA);
        }
        outcome
    }

    /// Advances time-based rig animation (e.g. fly movement or smooth damping).
    pub fn tick(&mut self, dt_secs: f64, rig: &mut dyn CameraRig) -> bool {
        if rig.tick(dt_secs, &mut self.camera) {
            self.dirty.insert(DirtyFlags::CAMERA);
            true
        } else {
            false
        }
    }

    /// Path A (`OwnedTarget`): renders into the viewport's managed offscreen targets and returns a
    /// zero-copy [`FrameOutput`] texture view ready for compositing in any UI framework.
    pub fn render_to_texture<W>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        world: &W,
        layers: &mut [&mut dyn ViewportLayer<W>],
    ) -> FrameOutput<'_> {
        let stats = self.execute_frame(device, queue, encoder, world, layers);
        let size = self.targets.size();
        let texture_view = match self.active_output {
            ActiveOutput::StaticCache => self.targets.static_view(),
            ActiveOutput::DynamicOutput => self.targets.output_view(),
        };
        FrameOutput {
            texture_view,
            uv_max: size.uv_max,
            active_px: size.active_px,
            stats,
        }
    }

    /// Path B (`BorrowedPass`, step 1): prepares offscreen targets and ensures a blit pipeline
    /// compatible with the host's [`PassTargetDesc`] exists. Unsupported descriptors return an
    /// error before the viewport records a frame.
    pub fn prepare_for_pass<W>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        world: &W,
        layers: &mut [&mut dyn ViewportLayer<W>],
        pass_target: PassTargetDesc,
    ) -> Result<FrameStats, ViewportError> {
        let (format, samples, depth, blend) = (
            pass_target.format,
            pass_target.sample_count,
            pass_target.depth_format,
            pass_target.blend_alpha,
        );
        if self
            .blit
            .get_pipeline(format, samples, depth, blend)
            .is_none()
        {
            let features = color_attachment_features(device, format, blend)?;
            validate_samples(format, features.flags, samples)?;
            if let Some(depth_format) = depth {
                validate_depth_attachment(device, depth_format, samples)?;
            }
            self.blit
                .ensure_pipeline(device, format, samples, depth, blend);
        }
        let stats = self.execute_frame(device, queue, encoder, world, layers);
        self.blit
            .write_crop_uv_max(queue, self.targets.size().uv_max);
        Ok(stats)
    }

    /// Path B (`BorrowedPass`, step 2): composites the resolved offscreen color into the host's
    /// active render pass. The host sets the viewport and scissor. Scene depth stays offscreen;
    /// an external depth attachment only affects pipeline compatibility.
    pub fn paint_into_pass(&self, pass: &mut wgpu::RenderPass<'_>, pass_target: PassTargetDesc) {
        let Some(pipeline) = self.blit.get_pipeline(
            pass_target.format,
            pass_target.sample_count,
            pass_target.depth_format,
            pass_target.blend_alpha,
        ) else {
            return;
        };
        let bg = match self.active_output {
            ActiveOutput::StaticCache => self.targets.static_crop_bg(),
            ActiveOutput::DynamicOutput => self.targets.output_crop_bg(),
        };
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.draw(0..3, 0..1);
    }

    fn execute_frame<W>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        world: &W,
        layers: &mut [&mut dyn ViewportLayer<W>],
    ) -> FrameStats {
        let _profile = crate::trace::scope("frame.encode");
        let mut stats = FrameStats {
            target_reallocated: self.pending_reallocated,
            ..FrameStats::default()
        };
        self.pending_reallocated = false;

        let size = self.targets.size();
        let [aw, ah] = size.active_px;
        let logical = self.logical_size();

        // Lazily upgrade or downgrade the depth attachment between Depth32Float and
        // Depth32FloatStencil8 according to active layer stencil requirements (zero-tax when off).
        let want_stencil = layers.iter().any(|l| l.requires_stencil(self.id));
        let desired_depth_format = if want_stencil && self.supports_stencil(device) {
            DEPTH_STENCIL_FORMAT
        } else {
            DEPTH_FORMAT
        };
        if self.targets.set_depth_format(device, desired_depth_format) {
            self.blit.ensure_pipeline(
                device,
                self.targets.format(),
                self.targets.msaa_samples(),
                Some(desired_depth_format),
                false,
            );
            self.target_generation = self.target_generation.wrapping_add(1);
            self.static_valid = false;
            self.selection_id_valid = false;
            self.dirty.insert(DirtyFlags::STATIC_SCENE);
            stats.target_reallocated = true;
        }

        // Update shared camera uniforms (set bit 3 when targeting an sRGB color attachment).
        let mut cam_data = self.camera.to_uniform_data(logical, self.dpi_scale);
        if self.targets.format().is_srgb() {
            cam_data.flags |= 8;
        }
        self.camera_gpu.write(queue, cam_data);

        // Determine whether this frame can run MRT color-frame selection + compute bitset reduction.
        let clipped_preview_rect = self.selection_preview_rect.and_then(|[rx, ry, rw, rh]| {
            let x0 = rx.min(aw);
            let y0 = ry.min(ah);
            let x1 = rx.saturating_add(rw).min(aw);
            let y1 = ry.saturating_add(rh).min(ah);
            let w = x1.saturating_sub(x0);
            let h = y1.saturating_sub(y0);
            (w > 0 && h > 0).then_some([x0, y0, w, h])
        });
        let static_phases = DrawPhaseMask::OPAQUE.union(DrawPhaseMask::TRANSPARENT);
        let layers_support_preview = clipped_preview_rect.is_some()
            && self.supports_color_frame_selection(device)
            && layers.iter().all(|l| {
                l.phases().intersection(static_phases) == DrawPhaseMask::NONE
                    || l.supports_color_frame_selection()
            });

        let (pending, untracked_pending) =
            self.prepare_layers(device, queue, world, layers, None, layers_support_preview);
        stats.streaming_pending = pending;
        let was_streaming = std::mem::replace(&mut self.streaming_pending, untracked_pending);
        if self.config.mode == ViewportMode::OnDemand {
            self.update_layer_cache(layers);
            if was_streaming || untracked_pending {
                self.dirty.insert(DirtyFlags::STATIC_SCENE);
            }
        }

        let preview_slot_count = if layers_support_preview {
            layers
                .iter()
                .map(|l| l.selection_slot_count())
                .max()
                .unwrap_or(0)
        } else {
            0
        };
        let preview_rect = clipped_preview_rect.filter(|_| preview_slot_count > 0);
        if preview_rect.is_none()
            && self.staged_selection_preview.take().is_some()
            && let Ok(mut stats) = self.selection_preview_stats.lock()
        {
            stats.superseded = stats.superseded.saturating_add(1);
        }

        let depth_format = self.targets.depth_format();
        let use_gpu_cull = self.targets.depth_sampleable()
            && supports_hiz(device, depth_format, self.targets.msaa_samples())
            && layers.iter().any(|l| l.uses_gpu_culling(self.id));
        let want_hiz = use_gpu_cull
            && (self.config.mode == ViewportMode::Continuous
                || !self.static_valid
                || (preview_rect.is_some() && !self.selection_id_valid)
                || self.dirty.needs_static_rebuild());
        if want_hiz {
            HiZPyramid::ensure(
                &mut self.hiz_pyramid,
                device,
                self.targets.depth_sample_view(),
                size.active_px,
                self.targets.msaa_samples(),
                self.target_generation,
            );
        }

        let paint_ctx = PaintCtx {
            selection_id_attachment: false,
            viewport_id: self.id,
            camera: &self.camera,
            camera_bind_group: self.camera_gpu.bind_group(),
            world,
            active_px: size.active_px,
            logical_size: logical,
            dpi_scale: self.dpi_scale,
        };
        let mrt_paint_ctx = PaintCtx {
            selection_id_attachment: true,
            viewport_id: self.id,
            camera: &self.camera,
            camera_bind_group: self.camera_gpu.bind_group(),
            world,
            active_px: size.active_px,
            logical_size: logical,
            dpi_scale: self.dpi_scale,
        };

        let outline_rect = Self::outline_rect(layers, &paint_ctx);
        let has_gizmo_or_overlay = layers.iter().any(|l| {
            let m = l.phases();
            m.contains(DrawPhase::Gizmo) || m.contains(DrawPhase::Overlay)
        });
        let has_dynamic_layers = outline_rect.is_some() || has_gizmo_or_overlay;
        let has_transparent_layers = layers
            .iter()
            .any(|l| l.phases().contains(DrawPhase::Transparent));
        let mut combined_hooks = PassHookMask::NONE;
        for layer in layers.iter() {
            combined_hooks = combined_hooks.union(layer.pass_hooks(self.id));
        }

        if outline_rect.is_some() {
            let multisampled_mask = self.targets.msaa_samples() == 4
                && layers
                    .iter()
                    .filter(|layer| layer.phases().contains(DrawPhase::OutlineMask))
                    .all(|layer| layer.supports_multisampled_outline_mask());
            OutlineEngine::ensure(
                &mut self.outline,
                device,
                size.allocated_px,
                self.targets.format(),
                self.targets.msaa_samples(),
                depth_format,
                multisampled_mask,
            );
        }

        match self.config.mode {
            ViewportMode::Continuous => {
                let use_mrt = preview_rect.is_some();
                let plan = PassGraph::compile(FramePlanInput {
                    rebuild_scene: true,
                    use_gpu_cull,
                    use_mrt_selection: use_mrt,
                    run_selection_reduce: use_mrt,
                    has_stencil: depth_format.has_stencil_aspect(),
                    has_pre_scene_hook: combined_hooks.contains(PassHookStage::PreScene),
                    has_post_opaque_hook: combined_hooks.contains(PassHookStage::PostOpaque),
                    has_transparent_layers,
                    has_post_scene_resolved_hook: combined_hooks
                        .contains(PassHookStage::PostSceneResolved),
                    has_outline: outline_rect.is_some(),
                    need_dynamic: has_dynamic_layers,
                    msaa_samples: self.targets.msaa_samples(),
                    continuous_second_pass: has_gizmo_or_overlay,
                });
                if plan.is_live(PassId::OutlineMask)
                    && let Some(rect) = outline_rect
                {
                    self.run_outline_mask_and_jfa(
                        queue, encoder, rect, layers, &paint_ctx, &mut stats,
                    );
                }
                if use_mrt {
                    SelectionPreviewTargets::ensure(
                        &mut self.selection_targets,
                        device,
                        size.allocated_px,
                        self.targets.msaa_samples(),
                    );
                }
                let sel_view = use_mrt
                    .then(|| {
                        self.selection_targets
                            .as_ref()
                            .map(SelectionPreviewTargets::id_view)
                    })
                    .flatten();
                let scene_ctx = if sel_view.is_some() {
                    &mrt_paint_ctx
                } else {
                    &paint_ctx
                };
                self.render_continuous(
                    device,
                    queue,
                    encoder,
                    aw,
                    ah,
                    outline_rect,
                    sel_view,
                    use_gpu_cull,
                    &plan,
                    layers,
                    scene_ctx,
                    &paint_ctx,
                    &mut stats,
                );
                if plan.is_live(PassId::SelectionReduce)
                    && let Some(rect) = preview_rect
                {
                    self.encode_selection_preview_reduction(
                        device,
                        queue,
                        encoder,
                        rect,
                        preview_slot_count,
                        layers,
                    );
                }
                self.active_output = ActiveOutput::DynamicOutput;
                self.static_valid = false;
                self.selection_id_valid = false;
                self.dirty = DirtyFlags::NONE;
            }
            ViewportMode::OnDemand => {
                let use_mrt = preview_rect.is_some();
                let need_static = !self.static_valid
                    || (use_mrt && !self.selection_id_valid)
                    || self.dirty.needs_static_rebuild();
                let need_dynamic =
                    has_dynamic_layers && (need_static || self.dirty.needs_dynamic_pass());
                let plan = PassGraph::compile(FramePlanInput {
                    rebuild_scene: need_static,
                    use_gpu_cull,
                    use_mrt_selection: use_mrt && need_static,
                    run_selection_reduce: preview_rect.is_some()
                        && (need_static || self.selection_id_valid),
                    has_stencil: depth_format.has_stencil_aspect(),
                    has_pre_scene_hook: combined_hooks.contains(PassHookStage::PreScene),
                    has_post_opaque_hook: combined_hooks.contains(PassHookStage::PostOpaque),
                    has_transparent_layers,
                    has_post_scene_resolved_hook: combined_hooks
                        .contains(PassHookStage::PostSceneResolved),
                    has_outline: outline_rect.is_some(),
                    need_dynamic,
                    msaa_samples: self.targets.msaa_samples(),
                    continuous_second_pass: false,
                });

                if plan.is_live(PassId::ScenePhase1) {
                    if use_mrt {
                        SelectionPreviewTargets::ensure(
                            &mut self.selection_targets,
                            device,
                            size.allocated_px,
                            self.targets.msaa_samples(),
                        );
                    }
                    let sel_view = use_mrt
                        .then(|| {
                            self.selection_targets
                                .as_ref()
                                .map(SelectionPreviewTargets::id_view)
                        })
                        .flatten();
                    let scene_ctx = if sel_view.is_some() {
                        &mrt_paint_ctx
                    } else {
                        &paint_ctx
                    };
                    self.render_static_cache(
                        device,
                        queue,
                        encoder,
                        aw,
                        ah,
                        sel_view,
                        use_gpu_cull,
                        &plan,
                        layers,
                        scene_ctx,
                        &mut stats,
                    );
                    self.static_valid = true;
                    self.selection_id_valid = sel_view.is_some();
                    stats.static_rebuilt = true;
                }

                if plan.is_live(PassId::DynamicComposite) {
                    if plan.is_live(PassId::OutlineMask)
                        && let Some(rect) = outline_rect
                    {
                        self.run_outline_mask_and_jfa(
                            queue, encoder, rect, layers, &paint_ctx, &mut stats,
                        );
                    }
                    self.render_dynamic_over_cache(
                        encoder,
                        aw,
                        ah,
                        outline_rect,
                        layers,
                        &paint_ctx,
                        &mut stats,
                    );
                    self.active_output = ActiveOutput::DynamicOutput;
                    stats.dynamic_drawn = true;
                } else if !has_dynamic_layers {
                    self.active_output = ActiveOutput::StaticCache;
                }
                if plan.is_live(PassId::SelectionReduce)
                    && let Some(rect) = preview_rect
                    && self.selection_id_valid
                {
                    self.encode_selection_preview_reduction(
                        device,
                        queue,
                        encoder,
                        rect,
                        preview_slot_count,
                        layers,
                    );
                }
                self.dirty = DirtyFlags::NONE;
            }
        }

        stats
    }

    fn encode_selection_preview_reduction<W>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        rect: [u32; 4],
        slot_count: u32,
        layers: &[&mut dyn ViewportLayer<W>],
    ) {
        let Some(targets) = &self.selection_targets else {
            return;
        };
        let reducer = SelectionPreviewReducer::ensure(
            &mut self.selection_reducer,
            device,
            self.targets.msaa_samples(),
        );
        let Some((staging, word_count)) = reducer.encode(
            device,
            queue,
            encoder,
            targets.id_view(),
            rect,
            slot_count,
            &self.selection_staging,
        ) else {
            return;
        };
        let scene_revisions = layers
            .iter()
            .map(|layer| (layer.id(), layer.revision().static_only()))
            .collect();
        let snapshot = SelectionPreviewSnapshot {
            viewport_id: self.id,
            target_generation: self.target_generation,
            camera: self.camera.clone(),
            rect_px: rect,
            scene_revisions,
        };
        let superseded = self
            .staged_selection_preview
            .replace(StagedSlotPreview {
                staging,
                word_count,
                slot_count,
                cache: Arc::clone(&self.selection_staging),
                snapshot,
                stats_ready: Arc::clone(&self.selection_preview_stats),
            })
            .is_some();
        if let Ok(mut stats) = self.selection_preview_stats.lock() {
            stats.submitted = stats.submitted.saturating_add(1);
            if superseded {
                stats.superseded = stats.superseded.saturating_add(1);
            }
        }
    }

    fn prepare_layers<W>(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        world: &W,
        layers: &mut [&mut dyn ViewportLayer<W>],
        picking_rect: Option<[u32; 4]>,
        selection_id_attachment: bool,
    ) -> (bool, bool) {
        let _profile = crate::trace::scope("layers.prepare");
        let size = self.targets.size();
        let logical = self.logical_size();
        let mut prep_ctx = PrepareCtx {
            viewport_lifetime: &self.lifetime,
            section_antialiasing: self.section_antialiasing,
            selection_id_attachment,
            gpu_culling: picking_rect.is_none()
                && self.targets.depth_sampleable()
                && supports_hiz(
                    device,
                    self.targets.depth_format(),
                    self.targets.msaa_samples(),
                ),
            picking_rect,
            viewport_id: self.id,
            device,
            queue,
            camera: &self.camera,
            camera_bind_group_layout: self.camera_gpu.layout(),
            world,
            active_px: size.active_px,
            logical_size: logical,
            dpi_scale: self.dpi_scale,
            budget: self.config.upload_budget,
            uploads: UploadTracker::with_clock(self.config.upload_budget, self.upload_clock),
            color_format: self.targets.format(),
            depth_format: self.targets.depth_format(),
            msaa_samples: self.targets.msaa_samples(),
        };
        let mut pending = false;
        let mut untracked_pending = false;
        for layer in layers {
            if layer.prepare(&mut prep_ctx) {
                pending = true;
                untracked_pending |= layer.revision() == LayerRevision::default();
            }
        }
        (pending, untracked_pending)
    }

    fn update_layer_cache<W>(&mut self, layers: &[&mut dyn ViewportLayer<W>]) {
        self.prepared_layers.clear();
        self.prepared_layers
            .extend(layers.iter().map(|layer| LayerState {
                id: layer.id(),
                phases: layer.phases(),
                revision: layer.revision(),
            }));

        let static_phases = DrawPhaseMask::OPAQUE.union(DrawPhaseMask::TRANSPARENT);
        let dynamic_phases = DrawPhaseMask::OUTLINE_MASK
            .union(DrawPhaseMask::GIZMO)
            .union(DrawPhaseMask::OVERLAY);
        let static_layer = |state: &LayerState| {
            let phases = state.phases.intersection(static_phases);
            (phases != DrawPhaseMask::NONE).then_some((
                state.id,
                phases,
                state.revision.static_scene,
            ))
        };
        let dynamic_layer = |state: &LayerState| {
            let phases = state.phases.intersection(dynamic_phases);
            (phases != DrawPhaseMask::NONE).then_some((state.id, phases, state.revision.dynamic))
        };
        if self
            .cached_layers
            .iter()
            .filter_map(static_layer)
            .ne(self.prepared_layers.iter().filter_map(static_layer))
        {
            self.dirty.insert(DirtyFlags::STATIC_SCENE);
        }
        if self
            .cached_layers
            .iter()
            .filter_map(dynamic_layer)
            .ne(self.prepared_layers.iter().filter_map(dynamic_layer))
        {
            self.dirty.insert(DirtyFlags::OVERLAY);
        }
        std::mem::swap(&mut self.cached_layers, &mut self.prepared_layers);
    }

    fn outline_rect<W>(
        layers: &[&mut dyn ViewportLayer<W>],
        ctx: &PaintCtx<'_, W>,
    ) -> Option<[u32; 4]> {
        let _profile = crate::trace::scope("outline.bounds");
        let [aw, ah] = ctx.active_px;
        let mut min_x = aw;
        let mut min_y = ah;
        let mut max_x = 0;
        let mut max_y = 0;
        for layer in layers
            .iter()
            .filter(|layer| layer.phases().contains(DrawPhase::OutlineMask))
        {
            let Some([x, y, width, height]) = layer.outline_mask_rect(ctx) else {
                return Some([0, 0, aw, ah]);
            };
            let right = x.saturating_add(width).min(aw);
            let bottom = y.saturating_add(height).min(ah);
            if right > x && bottom > y {
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(right);
                max_y = max_y.max(bottom);
            }
        }
        if min_x >= max_x || min_y >= max_y {
            return None;
        }
        // Cover the largest supported outline, its feather, and the seed filter's neighbors.
        const PADDING: u32 = 17;
        let x = min_x.saturating_sub(PADDING);
        let y = min_y.saturating_sub(PADDING);
        let right = max_x.saturating_add(PADDING).min(aw);
        let bottom = max_y.saturating_add(PADDING).min(ah);
        Some([x, y, right - x, bottom - y])
    }

    #[allow(clippy::too_many_arguments)]
    fn run_outline_mask_and_jfa<W>(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        rect: [u32; 4],
        layers: &mut [&mut dyn ViewportLayer<W>],
        paint_ctx: &PaintCtx<'_, W>,
        stats: &mut FrameStats,
    ) {
        let Some(engine) = &self.outline else {
            return;
        };
        let [aw, ah] = paint_ctx.active_px;
        let [x, y, w, h] = rect;
        {
            let _profile = crate::trace::scope("outline.mask_encode");
            let (view, resolve_target) = engine.mask_color_views();
            let gpu_timer = crate::trace::pass("iletki outline mask");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("iletki outline mask"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: resolved_color_store(resolve_target),
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: gpu_timer.as_ref().and_then(|timer| timer.writes()),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_viewport(0.0, 0.0, aw as f32, ah as f32, 0.0, 1.0);
            pass.set_scissor_rect(x, y, w, h);
            pass.set_bind_group(0, self.camera_gpu.bind_group(), &[]);
            for layer in layers.iter() {
                if layer.phases().contains(DrawPhase::OutlineMask) {
                    if resolve_target.is_some() {
                        layer.paint_multisampled_outline_mask(&mut pass, paint_ctx);
                    } else {
                        layer.paint(DrawPhase::OutlineMask, &mut pass, paint_ctx);
                    }
                    stats.layer_draws = stats.layer_draws.saturating_add(1);
                }
            }
        }
        let _profile = crate::trace::scope("outline.jump_flood_encode");
        engine.run_jump_flood(
            queue,
            encoder,
            paint_ctx.active_px,
            rect,
            self.config.outline,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_pass_hooks<W>(
        &self,
        stage: PassHookStage,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        color_attachment_view: &wgpu::TextureView,
        resolved_color_view: Option<&wgpu::TextureView>,
        selection_id_view: Option<&wgpu::TextureView>,
        layers: &mut [&mut dyn ViewportLayer<W>],
        paint_ctx: &PaintCtx<'_, W>,
    ) {
        let size = self.targets.size();
        let hook_ctx = PassHookCtx {
            viewport_id: self.id,
            device,
            queue,
            camera: &self.camera,
            camera_bind_group: self.camera_gpu.bind_group(),
            world: paint_ctx.world,
            active_px: size.active_px,
            allocated_px: size.allocated_px,
            logical_size: paint_ctx.logical_size,
            dpi_scale: self.dpi_scale,
            color_format: self.targets.format(),
            depth_format: self.targets.depth_format(),
            msaa_samples: self.targets.msaa_samples(),
            target_generation: self.target_generation,
            color_attachment_view,
            resolved_color_view,
            depth_stencil_view: self.targets.depth_view(),
            depth_sample_view: self
                .targets
                .depth_sampleable()
                .then(|| self.targets.depth_sample_view()),
            selection_id_view,
        };
        for layer in layers.iter_mut() {
            if layer.pass_hooks(self.id).contains(stage) {
                layer.encode_pass_hook(stage, encoder, &hook_ctx);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_post_opaque_and_transparent<W>(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        aw: u32,
        ah: u32,
        color_view: &wgpu::TextureView,
        final_resolve_target: Option<&wgpu::TextureView>,
        final_store: wgpu::StoreOp,
        selection_id_view: Option<&wgpu::TextureView>,
        plan: &PassGraph,
        layers: &mut [&mut dyn ViewportLayer<W>],
        paint_ctx: &PaintCtx<'_, W>,
        stats: &mut FrameStats,
    ) {
        let depth_format = self.targets.depth_format();
        if plan.is_live(PassId::PostOpaqueHooks) {
            self.encode_pass_hooks(
                PassHookStage::PostOpaque,
                device,
                queue,
                encoder,
                color_view,
                None,
                selection_id_view,
                layers,
                paint_ctx,
            );
        }
        let keep_depth = plan.is_live(PassId::PostSceneResolvedHooks);
        if plan.is_live(PassId::SceneTransparent) {
            let gpu_timer = crate::trace::pass("iletki scene transparent");
            let color_attachment = Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                depth_slice: None,
                resolve_target: final_resolve_target,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: final_store,
                },
            });
            let single = [color_attachment.clone()];
            let mrt = selection_id_view.map(|id_view| {
                [
                    color_attachment,
                    Some(wgpu::RenderPassColorAttachment {
                        view: id_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ]
            });
            let attachments: &[Option<wgpu::RenderPassColorAttachment<'_>>] = match &mrt {
                Some(pair) => pair,
                None => &single,
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("iletki scene transparent"),
                color_attachments: attachments,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: self.targets.depth_view(),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: if keep_depth {
                            wgpu::StoreOp::Store
                        } else {
                            wgpu::StoreOp::Discard
                        },
                    }),
                    stencil_ops: stencil_ops(
                        depth_format,
                        wgpu::LoadOp::Load,
                        wgpu::StoreOp::Discard,
                    ),
                }),
                timestamp_writes: gpu_timer.as_ref().and_then(|timer| timer.writes()),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_viewport(0.0, 0.0, aw as f32, ah as f32, 0.0, 1.0);
            pass.set_scissor_rect(0, 0, aw, ah);
            pass.set_bind_group(0, self.camera_gpu.bind_group(), &[]);
            for layer in layers.iter() {
                if layer.phases().contains(DrawPhase::Transparent) {
                    layer.paint(DrawPhase::Transparent, &mut pass, paint_ctx);
                    stats.layer_draws = stats.layer_draws.saturating_add(1);
                }
            }
        } else if plan.is_live(PassId::SceneResolve) && final_resolve_target.is_some() {
            // Resolve MSAA color into the 1x target after post-opaque hooks when no transparent
            // pass runs. Keep `Store` so software drivers resolve cleanly.
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("iletki msaa resolve"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    depth_slice: None,
                    resolve_target: final_resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn render_two_phase_scene<W>(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        aw: u32,
        ah: u32,
        color_view: &wgpu::TextureView,
        final_resolve_target: Option<&wgpu::TextureView>,
        final_store: wgpu::StoreOp,
        selection_id_view: Option<&wgpu::TextureView>,
        label_p1: &'static str,
        label_p2: &'static str,
        hiz: &HiZPyramid,
        plan: &PassGraph,
        layers: &mut [&mut dyn ViewportLayer<W>],
        paint_ctx: &PaintCtx<'_, W>,
        stats: &mut FrameStats,
    ) {
        let depth_format = self.targets.depth_format();
        let keep_stencil_after_p1 = plan.is_live(PassId::PostOpaqueHooks);
        let cull_ctx = GpuCullCtx {
            viewport_id: self.id,
            device,
            queue,
            camera: &self.camera,
            world: paint_ctx.world,
            active_px: [aw, ah],
            logical_size: paint_ctx.logical_size,
        };
        for layer in layers.iter_mut() {
            if layer.uses_gpu_culling(self.id) {
                layer.encode_gpu_cull_phase1(encoder, &cull_ctx);
            }
        }
        {
            let gpu_timer = crate::trace::pass(label_p1);
            let color_attachment = Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(self.config.wgpu_clear_color()),
                    store: wgpu::StoreOp::Store,
                },
            });
            let single = [color_attachment.clone()];
            let mrt = selection_id_view.map(|id_view| {
                [
                    color_attachment,
                    Some(wgpu::RenderPassColorAttachment {
                        view: id_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ]
            });
            let attachments: &[Option<wgpu::RenderPassColorAttachment<'_>>] = match &mrt {
                Some(pair) => pair,
                None => &single,
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label_p1),
                color_attachments: attachments,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: self.targets.depth_view(),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: stencil_ops(
                        depth_format,
                        wgpu::LoadOp::Clear(0),
                        if keep_stencil_after_p1 {
                            wgpu::StoreOp::Store
                        } else {
                            wgpu::StoreOp::Discard
                        },
                    ),
                }),
                timestamp_writes: gpu_timer.as_ref().and_then(|timer| timer.writes()),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if let Some(timer) = &gpu_timer {
                timer.begin_stats(&mut pass);
            }
            pass.set_viewport(0.0, 0.0, aw as f32, ah as f32, 0.0, 1.0);
            pass.set_scissor_rect(0, 0, aw, ah);
            pass.set_bind_group(0, self.camera_gpu.bind_group(), &[]);
            for layer in layers.iter() {
                if layer.phases().contains(DrawPhase::Opaque) {
                    layer.paint(DrawPhase::Opaque, &mut pass, paint_ctx);
                    stats.layer_draws = stats.layer_draws.saturating_add(1);
                }
            }
            if let Some(timer) = &gpu_timer {
                timer.end_stats(&mut pass);
            }
        }

        hiz.encode(encoder);

        for layer in layers.iter_mut() {
            if layer.uses_gpu_culling(self.id) {
                layer.encode_gpu_cull_phase2(
                    encoder,
                    hiz.full_view(),
                    hiz.size(),
                    hiz.mip_count(),
                    hiz.generation(),
                    &cull_ctx,
                );
            }
        }

        {
            let phase2_resolve = if plan.keep_msaa_after_phase2() {
                None
            } else {
                final_resolve_target
            };
            let phase2_store = if plan.keep_msaa_after_phase2() {
                wgpu::StoreOp::Store
            } else {
                final_store
            };
            let gpu_timer = crate::trace::pass(label_p2);
            let color_attachment = Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                depth_slice: None,
                resolve_target: phase2_resolve,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: phase2_store,
                },
            });
            let single = [color_attachment.clone()];
            let mrt = selection_id_view.map(|id_view| {
                [
                    color_attachment,
                    Some(wgpu::RenderPassColorAttachment {
                        view: id_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ]
            });
            let attachments: &[Option<wgpu::RenderPassColorAttachment<'_>>] = match &mrt {
                Some(pair) => pair,
                None => &single,
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label_p2),
                color_attachments: attachments,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: self.targets.depth_view(),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: if plan.keep_depth_after_phase2() {
                            wgpu::StoreOp::Store
                        } else {
                            wgpu::StoreOp::Discard
                        },
                    }),
                    stencil_ops: stencil_ops(
                        depth_format,
                        wgpu::LoadOp::Load,
                        if keep_stencil_after_p1 {
                            wgpu::StoreOp::Store
                        } else {
                            wgpu::StoreOp::Discard
                        },
                    ),
                }),
                timestamp_writes: gpu_timer.as_ref().and_then(|timer| timer.writes()),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if let Some(timer) = &gpu_timer {
                timer.begin_stats(&mut pass);
            }
            pass.set_viewport(0.0, 0.0, aw as f32, ah as f32, 0.0, 1.0);
            pass.set_scissor_rect(0, 0, aw, ah);
            pass.set_bind_group(0, self.camera_gpu.bind_group(), &[]);
            for layer in layers.iter() {
                if layer.uses_gpu_culling(self.id) && layer.phases().contains(DrawPhase::Opaque) {
                    layer.paint_phase2(DrawPhase::Opaque, &mut pass, paint_ctx);
                    stats.layer_draws = stats.layer_draws.saturating_add(1);
                }
            }
            if !plan.separate_transparent_pass() {
                for layer in layers.iter() {
                    if layer.phases().contains(DrawPhase::Transparent) {
                        layer.paint(DrawPhase::Transparent, &mut pass, paint_ctx);
                        stats.layer_draws = stats.layer_draws.saturating_add(1);
                    }
                }
            }
            if let Some(timer) = &gpu_timer {
                timer.end_stats(&mut pass);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn render_static_cache<W>(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        aw: u32,
        ah: u32,
        selection_id_view: Option<&wgpu::TextureView>,
        use_gpu_cull: bool,
        plan: &PassGraph,
        layers: &mut [&mut dyn ViewportLayer<W>],
        paint_ctx: &PaintCtx<'_, W>,
        stats: &mut FrameStats,
    ) {
        let (color_view, resolve_target) = self.targets.static_color_views();
        let depth_format = self.targets.depth_format();
        if plan.is_live(PassId::PreSceneHooks) {
            self.encode_pass_hooks(
                PassHookStage::PreScene,
                device,
                queue,
                encoder,
                color_view,
                None,
                selection_id_view,
                layers,
                paint_ctx,
            );
        }
        if let Some(hiz) = use_gpu_cull.then_some(self.hiz_pyramid.as_ref()).flatten() {
            self.render_two_phase_scene(
                device,
                queue,
                encoder,
                aw,
                ah,
                color_view,
                resolve_target,
                resolved_color_store(resolve_target),
                selection_id_view,
                "iletki static scene",
                "iletki static scene phase2",
                hiz,
                plan,
                layers,
                paint_ctx,
                stats,
            );
        } else {
            let primary_resolve = if plan.keep_msaa_after_primary() {
                None
            } else {
                resolve_target
            };
            let primary_store = if plan.keep_msaa_after_primary() {
                wgpu::StoreOp::Store
            } else {
                resolved_color_store(resolve_target)
            };
            let gpu_timer = crate::trace::pass("iletki static scene");
            let color_attachment = Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                depth_slice: None,
                resolve_target: primary_resolve,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(self.config.wgpu_clear_color()),
                    store: primary_store,
                },
            });
            let single = [color_attachment.clone()];
            let mrt = selection_id_view.map(|id_view| {
                [
                    color_attachment,
                    Some(wgpu::RenderPassColorAttachment {
                        view: id_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ]
            });
            let attachments: &[Option<wgpu::RenderPassColorAttachment<'_>>] = match &mrt {
                Some(pair) => pair,
                None => &single,
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("iletki static scene"),
                color_attachments: attachments,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: self.targets.depth_view(),
                    depth_ops: Some(wgpu::Operations {
                        // Reversed-Z clear: 0.0 is far plane.
                        load: wgpu::LoadOp::Clear(0.0),
                        store: if plan.keep_depth_after_primary() {
                            wgpu::StoreOp::Store
                        } else {
                            wgpu::StoreOp::Discard
                        },
                    }),
                    stencil_ops: stencil_ops(
                        depth_format,
                        wgpu::LoadOp::Clear(0),
                        if plan.is_live(PassId::PostOpaqueHooks) {
                            wgpu::StoreOp::Store
                        } else {
                            wgpu::StoreOp::Discard
                        },
                    ),
                }),
                timestamp_writes: gpu_timer.as_ref().and_then(|timer| timer.writes()),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if let Some(timer) = &gpu_timer {
                timer.begin_stats(&mut pass);
            }
            pass.set_viewport(0.0, 0.0, aw as f32, ah as f32, 0.0, 1.0);
            pass.set_scissor_rect(0, 0, aw, ah);
            pass.set_bind_group(0, self.camera_gpu.bind_group(), &[]);

            for layer in layers.iter() {
                if layer.phases().contains(DrawPhase::Opaque) {
                    layer.paint(DrawPhase::Opaque, &mut pass, paint_ctx);
                    stats.layer_draws = stats.layer_draws.saturating_add(1);
                }
            }
            if !plan.separate_transparent_pass() {
                for layer in layers.iter() {
                    if layer.phases().contains(DrawPhase::Transparent) {
                        layer.paint(DrawPhase::Transparent, &mut pass, paint_ctx);
                        stats.layer_draws = stats.layer_draws.saturating_add(1);
                    }
                }
            }
            if let Some(timer) = &gpu_timer {
                timer.end_stats(&mut pass);
            }
        }

        if plan.separate_transparent_pass() {
            self.encode_post_opaque_and_transparent(
                device,
                queue,
                encoder,
                aw,
                ah,
                color_view,
                resolve_target,
                resolved_color_store(resolve_target),
                selection_id_view,
                plan,
                layers,
                paint_ctx,
                stats,
            );
        }

        if plan.is_live(PassId::PostSceneResolvedHooks) {
            self.encode_pass_hooks(
                PassHookStage::PostSceneResolved,
                device,
                queue,
                encoder,
                color_view,
                Some(self.targets.static_view()),
                selection_id_view,
                layers,
                paint_ctx,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn render_dynamic_over_cache<W>(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        aw: u32,
        ah: u32,
        outline_rect: Option<[u32; 4]>,
        layers: &mut [&mut dyn ViewportLayer<W>],
        paint_ctx: &PaintCtx<'_, W>,
        stats: &mut FrameStats,
    ) {
        let has_gizmo_or_overlay = layers.iter().any(|l| {
            let m = l.phases();
            m.contains(DrawPhase::Gizmo) || m.contains(DrawPhase::Overlay)
        });
        if !has_gizmo_or_overlay
            && let (Some(rect), Some(outline)) = (outline_rect, &self.outline)
            && self.targets.can_copy_output()
        {
            // Fast 1x outline-only path: copy the resolved static cache directly into the output
            // texture and alpha-blend the Jump-Flood outline halo in 1x without clearing, blitting
            // or resolving the 4x MSAA color/depth targets.
            encoder.copy_texture_to_texture(
                self.targets.static_texture().as_image_copy(),
                self.targets.output_texture().as_image_copy(),
                wgpu::Extent3d {
                    width: aw,
                    height: ah,
                    depth_or_array_layers: 1,
                },
            );
            let gpu_timer = crate::trace::pass("iletki outline composite 1x");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("iletki outline composite 1x"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: self.targets.output_view(),
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: gpu_timer.as_ref().and_then(|timer| timer.writes()),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            outline.composite_into_pass_1x(&mut pass, rect);
            return;
        }

        let depth_format = self.targets.depth_format();
        let (color_view, resolve_target) = self.targets.output_color_views();
        let gpu_timer = crate::trace::pass("iletki dynamic overlay");
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("iletki dynamic overlay"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                depth_slice: None,
                resolve_target,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(self.config.wgpu_clear_color()),
                    store: resolved_color_store(resolve_target),
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: self.targets.depth_view(),
                depth_ops: Some(wgpu::Operations {
                    // Fresh Reversed-Z depth so gizmos/overlays depth-test among themselves on top.
                    load: wgpu::LoadOp::Clear(0.0),
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: stencil_ops(
                    depth_format,
                    wgpu::LoadOp::Clear(0),
                    wgpu::StoreOp::Discard,
                ),
            }),
            timestamp_writes: gpu_timer.as_ref().and_then(|timer| timer.writes()),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        let [alloc_w, alloc_h] = self.targets.size().allocated_px;
        // Full allocated viewport for 1:1 texel copy from static_cache, clipped to active rect.
        pass.set_viewport(0.0, 0.0, alloc_w as f32, alloc_h as f32, 0.0, 1.0);
        pass.set_scissor_rect(0, 0, aw, ah);
        if let Some(blit_pipe) = self.blit.get_pipeline(
            self.targets.format(),
            self.targets.msaa_samples(),
            Some(depth_format),
            false,
        ) {
            pass.set_pipeline(blit_pipe);
            pass.set_bind_group(0, self.targets.static_identity_bg(), &[]);
            pass.draw(0..3, 0..1);
        }

        if let (Some(rect), Some(outline)) = (outline_rect, &self.outline) {
            outline.composite_into_pass(&mut pass, rect);
        }

        // Now switch viewport to active rect for gizmo and screen-space overlay layers.
        pass.set_viewport(0.0, 0.0, aw as f32, ah as f32, 0.0, 1.0);
        pass.set_scissor_rect(0, 0, aw, ah);
        pass.set_bind_group(0, self.camera_gpu.bind_group(), &[]);
        for phase in [DrawPhase::Gizmo, DrawPhase::Overlay] {
            for layer in layers.iter() {
                if layer.phases().contains(phase) {
                    layer.paint(phase, &mut pass, paint_ctx);
                    stats.layer_draws = stats.layer_draws.saturating_add(1);
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn render_continuous<W>(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        aw: u32,
        ah: u32,
        outline_rect: Option<[u32; 4]>,
        selection_id_view: Option<&wgpu::TextureView>,
        use_gpu_cull: bool,
        plan: &PassGraph,
        layers: &mut [&mut dyn ViewportLayer<W>],
        scene_paint_ctx: &PaintCtx<'_, W>,
        overlay_paint_ctx: &PaintCtx<'_, W>,
        stats: &mut FrameStats,
    ) {
        let depth_format = self.targets.depth_format();
        let has_gizmo_or_overlay = layers.iter().any(|l| {
            let m = l.phases();
            m.contains(DrawPhase::Gizmo) || m.contains(DrawPhase::Overlay)
        });
        let has_post_resolved = plan.is_live(PassId::PostSceneResolvedHooks);
        // When both a 1x post-scene hook (e.g. GTAO) and MSAA gizmo/overlay pass run in Continuous
        // mode, resolve scene into static_view, run the 1x hook there, then composite overlays.
        if has_gizmo_or_overlay && has_post_resolved {
            self.render_static_cache(
                device,
                queue,
                encoder,
                aw,
                ah,
                selection_id_view,
                use_gpu_cull,
                plan,
                layers,
                scene_paint_ctx,
                stats,
            );
            stats.static_rebuilt = true;
            self.render_dynamic_over_cache(
                encoder,
                aw,
                ah,
                outline_rect,
                layers,
                overlay_paint_ctx,
                stats,
            );
            stats.dynamic_drawn = true;
            return;
        }

        let (color_view, resolve_target) = self.targets.output_color_views();
        if plan.is_live(PassId::PreSceneHooks) {
            self.encode_pass_hooks(
                PassHookStage::PreScene,
                device,
                queue,
                encoder,
                color_view,
                None,
                selection_id_view,
                layers,
                scene_paint_ctx,
            );
        }
        let fast_outline_1x = !has_gizmo_or_overlay && outline_rect.is_some();
        let has_second_pass = has_gizmo_or_overlay;
        let final_resolve = if has_second_pass {
            None
        } else {
            resolve_target
        };
        let final_store = if has_second_pass {
            wgpu::StoreOp::Store
        } else {
            resolved_color_store(resolve_target)
        };
        if let Some(hiz) = use_gpu_cull.then_some(self.hiz_pyramid.as_ref()).flatten() {
            self.render_two_phase_scene(
                device,
                queue,
                encoder,
                aw,
                ah,
                color_view,
                final_resolve,
                final_store,
                selection_id_view,
                "iletki continuous scene",
                "iletki continuous scene phase2",
                hiz,
                plan,
                layers,
                scene_paint_ctx,
                stats,
            );
            stats.static_rebuilt = true;
        } else {
            let primary_resolve = if plan.keep_msaa_after_primary() {
                None
            } else {
                final_resolve
            };
            let primary_store = if plan.keep_msaa_after_primary() {
                wgpu::StoreOp::Store
            } else {
                final_store
            };
            let gpu_timer = crate::trace::pass("iletki continuous scene");
            let color_attachment = Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                depth_slice: None,
                resolve_target: primary_resolve,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(self.config.wgpu_clear_color()),
                    store: primary_store,
                },
            });
            let single = [color_attachment.clone()];
            let mrt = selection_id_view.map(|id_view| {
                [
                    color_attachment,
                    Some(wgpu::RenderPassColorAttachment {
                        view: id_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ]
            });
            let attachments: &[Option<wgpu::RenderPassColorAttachment<'_>>] = match &mrt {
                Some(pair) => pair,
                None => &single,
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("iletki continuous scene"),
                color_attachments: attachments,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: self.targets.depth_view(),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: if plan.keep_depth_after_primary() {
                            wgpu::StoreOp::Store
                        } else {
                            wgpu::StoreOp::Discard
                        },
                    }),
                    stencil_ops: stencil_ops(
                        depth_format,
                        wgpu::LoadOp::Clear(0),
                        if plan.is_live(PassId::PostOpaqueHooks) {
                            wgpu::StoreOp::Store
                        } else {
                            wgpu::StoreOp::Discard
                        },
                    ),
                }),
                timestamp_writes: gpu_timer.as_ref().and_then(|timer| timer.writes()),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if let Some(timer) = &gpu_timer {
                timer.begin_stats(&mut pass);
            }
            pass.set_viewport(0.0, 0.0, aw as f32, ah as f32, 0.0, 1.0);
            pass.set_scissor_rect(0, 0, aw, ah);
            pass.set_bind_group(0, self.camera_gpu.bind_group(), &[]);
            for layer in layers.iter() {
                if layer.phases().contains(DrawPhase::Opaque) {
                    layer.paint(DrawPhase::Opaque, &mut pass, scene_paint_ctx);
                    stats.layer_draws = stats.layer_draws.saturating_add(1);
                }
            }
            if !plan.separate_transparent_pass() {
                for layer in layers.iter() {
                    if layer.phases().contains(DrawPhase::Transparent) {
                        layer.paint(DrawPhase::Transparent, &mut pass, scene_paint_ctx);
                        stats.layer_draws = stats.layer_draws.saturating_add(1);
                    }
                }
            }
            if let Some(timer) = &gpu_timer {
                timer.end_stats(&mut pass);
            }
            stats.static_rebuilt = true;
        }

        if plan.separate_transparent_pass() {
            self.encode_post_opaque_and_transparent(
                device,
                queue,
                encoder,
                aw,
                ah,
                color_view,
                final_resolve,
                final_store,
                selection_id_view,
                plan,
                layers,
                scene_paint_ctx,
                stats,
            );
        }

        if has_post_resolved {
            self.encode_pass_hooks(
                PassHookStage::PostSceneResolved,
                device,
                queue,
                encoder,
                color_view,
                Some(self.targets.output_view()),
                selection_id_view,
                layers,
                scene_paint_ctx,
            );
        }

        if fast_outline_1x && let (Some(rect), Some(outline)) = (outline_rect, &self.outline) {
            // Fast 1x outline-only path: Pass 1 already resolved into `output_view`, so alpha-blend
            // the Jump-Flood halo directly in 1x without storing/reloading the 4x MSAA color buffer.
            let gpu_timer = crate::trace::pass("iletki outline composite 1x");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("iletki outline composite 1x"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: self.targets.output_view(),
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: gpu_timer.as_ref().and_then(|timer| timer.writes()),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            outline.composite_into_pass_1x(&mut pass, rect);
            stats.dynamic_drawn = true;
        }

        if has_second_pass {
            let gpu_timer = crate::trace::pass("iletki continuous gizmo/overlay");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("iletki continuous gizmo/overlay"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    depth_slice: None,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        // Keep `Store` when loading an existing MSAA attachment with a resolve
                        // target and partial outline scissor so software drivers resolve cleanly.
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: self.targets.depth_view(),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: stencil_ops(
                        depth_format,
                        wgpu::LoadOp::Clear(0),
                        wgpu::StoreOp::Discard,
                    ),
                }),
                timestamp_writes: gpu_timer.as_ref().and_then(|timer| timer.writes()),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if let (Some(rect), Some(outline)) = (outline_rect, &self.outline) {
                outline.composite_into_pass(&mut pass, rect);
            }
            pass.set_viewport(0.0, 0.0, aw as f32, ah as f32, 0.0, 1.0);
            pass.set_scissor_rect(0, 0, aw, ah);
            pass.set_bind_group(0, self.camera_gpu.bind_group(), &[]);
            for phase in [DrawPhase::Gizmo, DrawPhase::Overlay] {
                for layer in layers.iter() {
                    if layer.phases().contains(phase) {
                        layer.paint(phase, &mut pass, overlay_paint_ctx);
                        stats.layer_draws = stats.layer_draws.saturating_add(1);
                    }
                }
            }
            stats.dynamic_drawn = true;
        }
    }

    /// Reads back the active viewport output into a tightly packed [`RgbaImage`] (native only).
    ///
    /// Intended for headless visual regression tests, CLI rendering and AI agent visual inspection.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn readback_rgba_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<RgbaImage, ViewportError> {
        let is_bgra = match self.targets.format() {
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => false,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => true,
            _ => return Err(ViewportError::UnsupportedReadbackFormat),
        };
        let [width, height] = self.targets.size().active_px;
        if width == 0 || height == 0 {
            return Err(ViewportError::ZeroSize);
        }
        let tex = match self.active_output {
            ActiveOutput::StaticCache => self.targets.static_texture(),
            ActiveOutput::DynamicOutput => self.targets.output_texture(),
        };
        let raw = readback_texture_2d_blocking(device, queue, tex, [0, 0, width, height], 4)?;
        let mut pixels = raw;
        if is_bgra {
            for px in pixels.as_chunks_mut::<4>().0 {
                px.swap(0, 2);
            }
        }
        Ok(RgbaImage {
            width,
            height,
            pixels,
        })
    }

    /// Prepares the current layers and submits a 64-bit pick over `[x, y, width, height]`
    /// in physical pixels. The returned request owns its camera snapshot and GPU readback.
    ///
    /// Submit any previously encoded frames for this host before calling. On native, drive the
    /// request with [`PendingPick::try_read`] in the host loop or poll the device while awaiting it.
    /// WebGPU drives mapping callbacks automatically.
    pub fn begin_pick_rect<W>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        world: &W,
        layers: &mut [&mut dyn ViewportLayer<W>],
        rect_px: [u32; 4],
    ) -> Result<PendingPick, ViewportError> {
        let staged = self.submit_pick_rect(device, queue, world, layers, rect_px)?;
        Ok(PendingPick {
            #[cfg(not(target_arch = "wasm32"))]
            device: device.clone(),
            camera: self.camera.clone(),
            dpi_scale: self.dpi_scale,
            id_map: request_staging_map(&staged.buffers.ids),
            depth_map: request_staging_map(&staged.buffers.depth),
            staged: Some(staged),
        })
    }

    /// Awaits the same request as [`Self::begin_pick_rect`], returning hits nearest-first.
    ///
    /// Native callers must keep driving `wgpu` polling while awaiting. For an event loop that
    /// sleeps between frames, use [`Self::begin_pick_rect`] and schedule [`PendingPick::try_read`].
    pub async fn pick_rect_async<W>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        world: &W,
        layers: &mut [&mut dyn ViewportLayer<W>],
        rect_px: [u32; 4],
    ) -> Result<Vec<PickHit>, ViewportError> {
        self.begin_pick_rect(device, queue, world, layers, rect_px)?
            .await
    }

    /// Executes an on-demand 64-bit ID + Reversed-Z depth picking pass over `rect_px`
    /// (`[x, y, width, height]` in physical pixels) and returns unique [`PickHit`] results sorted
    /// nearest-first (native blocking wrapper).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn pick_rect_blocking<W>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        world: &W,
        layers: &mut [&mut dyn ViewportLayer<W>],
        rect_px: [u32; 4],
    ) -> Result<Vec<PickHit>, ViewportError> {
        let staged = self.submit_pick_rect(device, queue, world, layers, rect_px)?;
        let (id_bytes, depth_bytes) = {
            let _profile = crate::trace::scope("pick.readback");
            (
                map_staging_blocking(device, &staged.buffers.ids, staged.id_layout)?,
                map_staging_blocking(device, &staged.buffers.depth, staged.depth_layout)?,
            )
        };
        Ok(staged.decode(&self.camera, self.dpi_scale, &id_bytes, &depth_bytes))
    }

    fn submit_pick_rect<W>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        world: &W,
        layers: &mut [&mut dyn ViewportLayer<W>],
        rect_px: [u32; 4],
    ) -> Result<StagedPick, ViewportError> {
        let _profile = crate::trace::scope("pick.submit");
        let size = self.targets.size();
        let [aw, ah] = size.active_px;
        let [rx, ry, rw, rh] = rect_px;
        let x0 = rx.min(aw);
        let y0 = ry.min(ah);
        let x1 = rx.saturating_add(rw).min(aw);
        let y1 = ry.saturating_add(rh).min(ah);
        let w = x1.saturating_sub(x0);
        let h = y1.saturating_sub(y0);
        if w == 0 || h == 0 {
            return Err(ViewportError::ZeroSize);
        }

        let max_buffer_size = device.limits().max_buffer_size;
        let id_layout = StagingLayout::compute(w, h, 8, max_buffer_size)?;
        let depth_layout = StagingLayout::compute(w, h, 4, max_buffer_size)?;

        self.streaming_pending |= self
            .prepare_layers(device, queue, world, layers, Some([x0, y0, w, h]), false)
            .1;

        let logical = self.logical_size();
        self.camera_gpu
            .write(queue, self.camera.to_uniform_data(logical, self.dpi_scale));

        let pick = PickTargets::ensure(&mut self.pick_targets, device, size.allocated_px);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("iletki pick encoder"),
        });
        {
            let _profile = crate::trace::scope("pick.draw_encode");
            let paint_ctx = PaintCtx {
                selection_id_attachment: false,
                viewport_id: self.id,
                camera: &self.camera,
                camera_bind_group: self.camera_gpu.bind_group(),
                world,
                active_px: size.active_px,
                logical_size: logical,
                dpi_scale: self.dpi_scale,
            };
            let gpu_timer = crate::trace::pass("iletki pick pass");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("iletki pick pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: pick.id_view(),
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: pick.depth_view(),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: gpu_timer.as_ref().and_then(|timer| timer.writes()),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if let Some(timer) = &gpu_timer {
                timer.begin_stats(&mut pass);
            }
            pass.set_viewport(0.0, 0.0, aw as f32, ah as f32, 0.0, 1.0);
            pass.set_scissor_rect(x0, y0, w, h);
            pass.set_bind_group(0, self.camera_gpu.bind_group(), &[]);
            for layer in layers.iter() {
                if layer.phases().contains(DrawPhase::Picking) {
                    layer.paint(DrawPhase::Picking, &mut pass, &paint_ctx);
                }
            }
            if let Some(timer) = &gpu_timer {
                timer.end_stats(&mut pass);
            }
        }
        let depth_r32f = {
            let _profile = crate::trace::scope("pick.depth_copy_encode");
            let depth_copy = self
                .depth_copy
                .get_or_insert_with(|| DepthCopyPipeline::new(device));
            depth_copy.copy_subrect(
                device,
                queue,
                &mut encoder,
                pick.depth_view(),
                [x0, y0],
                [w, h],
            )
        };

        // Copy both ID and depth sub-rects into staging buffers in the SAME command submission.
        let buffers = self
            .pick_readback
            .lock()
            .ok()
            .and_then(|mut available| {
                if available.as_ref().is_some_and(|buffers| {
                    buffers.ids.size() >= id_layout.total_bytes as u64
                        && buffers.depth.size() >= depth_layout.total_bytes as u64
                }) {
                    available.take()
                } else {
                    None
                }
            })
            .unwrap_or_else(|| PickBuffers {
                ids: readback_buffer(device, id_layout),
                depth: readback_buffer(device, depth_layout),
            });
        encode_texture_readback(
            &mut encoder,
            pick.id_texture(),
            [x0, y0, w, h],
            id_layout,
            &buffers.ids,
        );
        encode_texture_readback(
            &mut encoder,
            &depth_r32f,
            [0, 0, w, h],
            depth_layout,
            &buffers.depth,
        );
        queue.submit(std::iter::once(encoder.finish()));

        Ok(StagedPick {
            buffers,
            cache: Arc::clone(&self.pick_readback),
            id_layout,
            depth_layout,
            rect: [x0, y0, w, h],
            logical,
        })
    }
}

#[derive(Clone, Copy)]
struct StagingLayout {
    unpadded_row: u32,
    padded_row: u32,
    height: u32,
    total_bytes: usize,
}

impl StagingLayout {
    fn compute(
        width: u32,
        height: u32,
        bytes_per_pixel: u32,
        max_buffer_size: u64,
    ) -> Result<Self, ViewportError> {
        if width == 0 || height == 0 {
            return Err(ViewportError::ZeroSize);
        }
        let unpadded_row = width
            .checked_mul(bytes_per_pixel)
            .ok_or(ViewportError::RegionTooLarge)?;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_row = unpadded_row
            .checked_add(align - 1)
            .map(|v| (v / align) * align)
            .ok_or(ViewportError::RegionTooLarge)?;
        let total_bytes = (padded_row as usize)
            .checked_mul(height as usize)
            .ok_or(ViewportError::RegionTooLarge)?;
        if total_bytes > MAX_READBACK_BYTES || total_bytes as u64 > max_buffer_size {
            return Err(ViewportError::RegionTooLarge);
        }
        Ok(Self {
            unpadded_row,
            padded_row,
            height,
            total_bytes,
        })
    }
}

struct PickBuffers {
    ids: wgpu::Buffer,
    depth: wgpu::Buffer,
}

type PickBufferCache = Arc<Mutex<Option<PickBuffers>>>;

struct StagedPick {
    buffers: PickBuffers,
    cache: PickBufferCache,
    id_layout: StagingLayout,
    depth_layout: StagingLayout,
    rect: [u32; 4],
    logical: Vec2,
}

impl StagedPick {
    // Both buffers have been read and unmapped by the caller. A dropped or failed request never
    // returns its buffers to the cache.
    fn decode(self, camera: &Camera, dpi_scale: f32, ids: &[u8], depths: &[u8]) -> Vec<PickHit> {
        let size = self
            .buffers
            .ids
            .size()
            .saturating_add(self.buffers.depth.size());
        if size <= MAX_CACHED_PICK_BYTES
            && let Ok(mut available) = self.cache.lock()
            && available.as_ref().is_none_or(|buffers| {
                self.buffers.ids.size() >= buffers.ids.size()
                    && self.buffers.depth.size() >= buffers.depth.size()
            })
        {
            *available = Some(self.buffers);
        }
        decode_pick_hits(camera, self.logical, dpi_scale, self.rect, ids, depths)
    }
}

// Only the single-sample output is used after the final color pass.
fn resolved_color_store(resolve_target: Option<&wgpu::TextureView>) -> wgpu::StoreOp {
    if resolve_target.is_some() {
        wgpu::StoreOp::Discard
    } else {
        wgpu::StoreOp::Store
    }
}

fn stencil_ops(
    depth_format: wgpu::TextureFormat,
    load: wgpu::LoadOp<u32>,
    store: wgpu::StoreOp,
) -> Option<wgpu::Operations<u32>> {
    depth_format
        .has_stencil_aspect()
        .then_some(wgpu::Operations { load, store })
}

fn readback_buffer(device: &wgpu::Device, layout: StagingLayout) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("iletki readback staging"),
        size: layout.total_bytes as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

fn encode_texture_readback(
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    rect: [u32; 4],
    layout: StagingLayout,
    staging: &wgpu::Buffer,
) {
    let [x, y, width, height] = rect;
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x, y, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(layout.padded_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}

fn extract_mapped_rows(
    staging: &wgpu::Buffer,
    layout: StagingLayout,
) -> Result<Vec<u8>, ViewportError> {
    let _profile = crate::trace::scope("readback.copy_rows");
    let slice = staging.slice(..);
    let mapped = slice
        .get_mapped_range()
        .map_err(|_| ViewportError::BufferMapFailed)?;
    let row_bytes = layout.unpadded_row as usize;
    let stride = layout.padded_row as usize;
    let h = layout.height as usize;
    let mut out = Vec::with_capacity(row_bytes.saturating_mul(h));
    for r in 0..h {
        let start = r.saturating_mul(stride);
        let end = start.saturating_add(row_bytes);
        let row = mapped
            .get(start..end)
            .ok_or(ViewportError::BufferMapFailed)?;
        out.extend_from_slice(row);
    }
    drop(mapped);
    staging.unmap();
    Ok(out)
}

struct MapState {
    result: Option<Result<(), wgpu::BufferAsyncError>>,
    waker: Option<Waker>,
}

/// An in-flight GPU pick. The viewport and its layers can be reused while this request completes.
pub struct PendingPick {
    #[cfg(not(target_arch = "wasm32"))]
    device: wgpu::Device,
    camera: Camera,
    dpi_scale: f32,
    id_map: Arc<Mutex<MapState>>,
    depth_map: Arc<Mutex<MapState>>,
    staged: Option<StagedPick>,
}

impl std::fmt::Debug for PendingPick {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingPick")
            .field("region", &self.staged.as_ref().map(|staged| staged.rect))
            .finish()
    }
}

impl PendingPick {
    /// Checks for a completed pick, driving one nonblocking GPU poll on native devices.
    ///
    /// While this returns `Poll::Pending`, the host can schedule another check without holding
    /// a borrow of the viewport. A ready result is consumed by this call.
    pub fn try_read(&mut self) -> Poll<Result<Vec<PickHit>, ViewportError>> {
        self.poll_completion(None)
    }

    fn poll_completion(
        &mut self,
        wake: Option<&Waker>,
    ) -> Poll<Result<Vec<PickHit>, ViewportError>> {
        let result = self.poll_staged(wake);
        if result.is_ready() {
            self.staged = None;
        }
        result
    }

    fn poll_staged(&mut self, wake: Option<&Waker>) -> Poll<Result<Vec<PickHit>, ViewportError>> {
        if self.staged.is_none() {
            return Poll::Ready(Err(ViewportError::PickAlreadyRead));
        }

        {
            let _profile = crate::trace::scope("pick.poll_maps");
            #[cfg(not(target_arch = "wasm32"))]
            self.device
                .poll(wgpu::PollType::Poll)
                .map_err(|_| ViewportError::BufferMapFailed)?;

            let id_ready = map_is_ready(&self.id_map, wake)?;
            let depth_ready = map_is_ready(&self.depth_map, wake)?;
            if !id_ready || !depth_ready {
                return Poll::Pending;
            }
        }

        let staged = self.staged.take().ok_or(ViewportError::PickAlreadyRead)?;
        let id_bytes = extract_mapped_rows(&staged.buffers.ids, staged.id_layout)?;
        let depth_bytes = extract_mapped_rows(&staged.buffers.depth, staged.depth_layout)?;
        Poll::Ready(Ok(staged.decode(
            &self.camera,
            self.dpi_scale,
            &id_bytes,
            &depth_bytes,
        )))
    }
}

impl Future for PendingPick {
    type Output = Result<Vec<PickHit>, ViewportError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.poll_completion(Some(cx.waker()))
    }
}

fn map_is_ready(state: &Mutex<MapState>, wake: Option<&Waker>) -> Result<bool, ViewportError> {
    let mut guard = state.lock().map_err(|_| ViewportError::BufferMapFailed)?;
    match &guard.result {
        Some(Ok(())) => Ok(true),
        Some(Err(_)) => Err(ViewportError::BufferMapFailed),
        None => {
            if let Some(waker) = wake {
                guard.waker = Some(waker.clone());
            }
            Ok(false)
        }
    }
}

fn request_staging_map(staging: &wgpu::Buffer) -> Arc<Mutex<MapState>> {
    let state = Arc::new(Mutex::new(MapState {
        result: None,
        waker: None,
    }));
    let cb_state = Arc::clone(&state);
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let wake = cb_state.lock().ok().and_then(|mut guard| {
                guard.result = Some(result);
                guard.waker.take()
            });
            if let Some(waker) = wake {
                waker.wake();
            }
        });
    state
}

#[cfg(not(target_arch = "wasm32"))]
fn map_staging_blocking(
    device: &wgpu::Device,
    staging: &wgpu::Buffer,
    layout: StagingLayout,
) -> Result<Vec<u8>, ViewportError> {
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|_| ViewportError::BufferMapFailed)?;
    rx.recv()
        .map_err(|_| ViewportError::BufferMapFailed)?
        .map_err(|_| ViewportError::BufferMapFailed)?;
    extract_mapped_rows(staging, layout)
}

fn decode_pick_hits(
    camera: &Camera,
    logical: Vec2,
    dpi_scale: f32,
    rect: [u32; 4],
    id_bytes: &[u8],
    depth_bytes: &[u8],
) -> Vec<PickHit> {
    let _profile = crate::trace::scope("pick.decode");
    let [x0, y0, w, h] = rect;
    let width = w as usize;
    let mut indices = HashMap::new();
    let mut winners = Vec::<(u64, f32, usize)>::new();
    let inv_dpi = 1.0 / f64::from(dpi_scale);
    let (ids, _) = id_bytes.as_chunks::<8>();
    let (depths, _) = depth_bytes.as_chunks::<4>();

    for (pixel, (id, depth)) in ids
        .iter()
        .zip(depths)
        .take(width.saturating_mul(h as usize))
        .enumerate()
    {
        let id = u64::from_le_bytes(*id);
        if id == 0 {
            continue;
        }
        let depth = f32::from_le_bytes(*depth);
        let index = *indices.entry(id).or_insert_with(|| {
            let index = winners.len();
            winners.push((id, depth, pixel));
            index
        });
        // Keep the first pixel at equal depths and retain encounter order across IDs.
        if let Some(winner) = winners.get_mut(index)
            && depth > winner.1
        {
            *winner = (id, depth, pixel);
        }
    }
    // Sort nearest-first (highest Reversed-Z depth first).
    winners.sort_by(|a, b| b.1.total_cmp(&a.1));
    if winners.is_empty() {
        return Vec::new();
    }
    let unproject = camera.depth_unprojector(logical);
    winners
        .into_iter()
        .map(|(id, depth, pixel)| {
            let col = (pixel % width) as u32;
            let row = (pixel / width) as u32;
            let screen_pt = Vec2::new(
                (f64::from(x0.saturating_add(col)) + 0.5) * inv_dpi,
                (f64::from(y0.saturating_add(row)) + 0.5) * inv_dpi,
            );
            PickHit {
                id: PickId::from_u64(id),
                depth,
                world_pos: unproject(screen_pt, depth),
            }
        })
        .collect()
}

#[cfg(not(target_arch = "wasm32"))]
fn readback_texture_2d_blocking(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    rect: [u32; 4],
    bytes_per_pixel: u32,
) -> Result<Vec<u8>, ViewportError> {
    let [_, _, width, height] = rect;
    let layout = StagingLayout::compute(
        width,
        height,
        bytes_per_pixel,
        device.limits().max_buffer_size,
    )?;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("iletki readback copy"),
    });
    let staging = readback_buffer(device, layout);
    encode_texture_readback(&mut encoder, texture, rect, layout, &staging);
    queue.submit(std::iter::once(encoder.finish()));
    map_staging_blocking(device, &staging, layout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use iletki_core::Vec3;

    #[test]
    fn pick_decoding_keeps_nearest_pixels_and_preserves_ties() {
        let a = PickId::new(1, 10, 20);
        let b = PickId::new(1, 20, 30);
        let ids: Vec<_> = [a, b, a, a]
            .into_iter()
            .flat_map(|id| id.to_u64().to_le_bytes())
            .collect();
        let depths: Vec<_> = [0.25_f32, 0.5, 0.5, 0.5]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let camera =
            Camera::perspective_3d(Vec3::new(0.0, -4.0, 0.0), Vec3::ZERO, Vec3::Z, 1.0, 0.1);
        let logical = Vec2::new(16.0, 8.0);
        let hits = decode_pick_hits(&camera, logical, 2.0, [2, 4, 4, 1], &ids, &depths);
        assert_eq!(hits.iter().map(|hit| hit.id).collect::<Vec<_>>(), [a, b]);
        let first = hits.first().unwrap();
        assert_eq!(first.depth.to_bits(), 0.5_f32.to_bits());
        assert_eq!(
            first.world_pos,
            camera.unproject_depth(Vec2::new(2.25, 2.25), 0.5, logical)
        );
    }
}
