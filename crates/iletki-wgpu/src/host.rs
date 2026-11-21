//! [`ViewportHost`]: dual-mode (`OnDemand` / `Continuous`), dual-path (`OwnedTarget` /
//! `BorrowedPass`) GPU viewport orchestrator with 64-bit ID+depth picking and headless readback.

use crate::blit::{BlitResources, DepthCopyPipeline};
use crate::error::ViewportError;
use crate::layer::{LayerRevision, PaintCtx, PrepareCtx, UploadTracker, ViewportLayer};
use crate::outline::OutlineEngine;
use crate::target::{
    DEPTH_FORMAT, PickTargets, TargetSet, ViewportConfig, color_attachment_features,
    validate_depth_attachment, validate_samples,
};
use crate::uniforms::CameraGpu;
use iletki_core::{
    Camera, CameraRig, DirtyFlags, DrawPhase, DrawPhaseMask, OutlineConfig, PickHit, PickId,
    RigOutcome, TargetSize, Vec2, ViewportCommand, ViewportId, ViewportInput, ViewportMode,
};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

static NEXT_VIEWPORT_ID: AtomicU64 = AtomicU64::new(1);

/// Maximum readback allocation cap (256 MiB) to guard against hostile sizes.
const MAX_READBACK_BYTES: usize = 256 * 1024 * 1024;

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
    camera: Camera,
    dpi_scale: f32,
    dirty: DirtyFlags,
    camera_gpu: CameraGpu,
    blit: BlitResources,
    targets: TargetSet,
    outline: Option<OutlineEngine>,
    pick_targets: Option<PickTargets>,
    depth_copy: Option<DepthCopyPipeline>,
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
            camera: Camera::default(),
            dpi_scale,
            dirty: DirtyFlags::ALL,
            camera_gpu,
            blit,
            targets,
            outline: None,
            pick_targets: None,
            depth_copy: None,
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
                self.static_valid = false;
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
        let mut stats = FrameStats {
            target_reallocated: self.pending_reallocated,
            ..FrameStats::default()
        };
        self.pending_reallocated = false;

        let size = self.targets.size();
        let [aw, ah] = size.active_px;
        let logical = self.logical_size();

        // Update shared camera uniforms (set bit 3 when targeting an sRGB color attachment).
        let mut cam_data = self.camera.to_uniform_data(logical, self.dpi_scale);
        if self.targets.format().is_srgb() {
            cam_data.flags |= 8;
        }
        self.camera_gpu.write(queue, cam_data);

        let (pending, untracked_pending) = self.prepare_layers(device, queue, world, layers);
        stats.streaming_pending = pending;
        let was_streaming = std::mem::replace(&mut self.streaming_pending, untracked_pending);
        if self.config.mode == ViewportMode::OnDemand {
            self.update_layer_cache(layers);
            if was_streaming || untracked_pending {
                self.dirty.insert(DirtyFlags::STATIC_SCENE);
            }
        }

        let has_outline_layers = layers
            .iter()
            .any(|l| l.phases().contains(DrawPhase::OutlineMask));
        let has_dynamic_layers = has_outline_layers
            || layers.iter().any(|l| {
                let m = l.phases();
                m.contains(DrawPhase::Gizmo) || m.contains(DrawPhase::Overlay)
            });

        if has_outline_layers {
            OutlineEngine::ensure(
                &mut self.outline,
                device,
                size.allocated_px,
                self.targets.format(),
                self.targets.msaa_samples(),
                DEPTH_FORMAT,
            );
        }

        let paint_ctx = PaintCtx {
            viewport_id: self.id,
            camera: &self.camera,
            camera_bind_group: self.camera_gpu.bind_group(),
            world,
            active_px: size.active_px,
            logical_size: logical,
            dpi_scale: self.dpi_scale,
        };

        match self.config.mode {
            ViewportMode::Continuous => {
                if has_outline_layers {
                    self.run_outline_mask_and_jfa(
                        queue,
                        encoder,
                        [aw, ah],
                        layers,
                        &paint_ctx,
                        &mut stats,
                    );
                }
                self.render_continuous(
                    encoder,
                    aw,
                    ah,
                    has_outline_layers,
                    layers,
                    &paint_ctx,
                    &mut stats,
                );
                self.active_output = ActiveOutput::DynamicOutput;
                self.static_valid = false;
                self.dirty = DirtyFlags::NONE;
            }
            ViewportMode::OnDemand => {
                let need_static = !self.static_valid || self.dirty.needs_static_rebuild();
                let need_dynamic =
                    has_dynamic_layers && (need_static || self.dirty.needs_dynamic_pass());

                if need_static {
                    self.render_static_cache(encoder, aw, ah, layers, &paint_ctx, &mut stats);
                    self.static_valid = true;
                    stats.static_rebuilt = true;
                }

                if need_dynamic {
                    if has_outline_layers {
                        self.run_outline_mask_and_jfa(
                            queue,
                            encoder,
                            [aw, ah],
                            layers,
                            &paint_ctx,
                            &mut stats,
                        );
                    }
                    self.render_dynamic_over_cache(
                        encoder,
                        aw,
                        ah,
                        has_outline_layers,
                        layers,
                        &paint_ctx,
                        &mut stats,
                    );
                    self.active_output = ActiveOutput::DynamicOutput;
                    stats.dynamic_drawn = true;
                } else if !has_dynamic_layers {
                    self.active_output = ActiveOutput::StaticCache;
                }
                self.dirty = DirtyFlags::NONE;
            }
        }

        stats
    }

    fn prepare_layers<W>(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        world: &W,
        layers: &mut [&mut dyn ViewportLayer<W>],
    ) -> (bool, bool) {
        let size = self.targets.size();
        let logical = self.logical_size();
        let mut prep_ctx = PrepareCtx {
            viewport_lifetime: &self.lifetime,
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
            depth_format: DEPTH_FORMAT,
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

    #[allow(clippy::too_many_arguments)]
    fn run_outline_mask_and_jfa<W>(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        active_px: [u32; 2],
        layers: &mut [&mut dyn ViewportLayer<W>],
        paint_ctx: &PaintCtx<'_, W>,
        stats: &mut FrameStats,
    ) {
        let Some(engine) = &self.outline else {
            return;
        };
        let [aw, ah] = active_px;
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("iletki outline mask"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: engine.mask_view(),
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_viewport(0.0, 0.0, aw as f32, ah as f32, 0.0, 1.0);
            pass.set_scissor_rect(0, 0, aw, ah);
            pass.set_bind_group(0, self.camera_gpu.bind_group(), &[]);
            for layer in layers.iter() {
                if layer.phases().contains(DrawPhase::OutlineMask) {
                    layer.paint(DrawPhase::OutlineMask, &mut pass, paint_ctx);
                    stats.layer_draws = stats.layer_draws.saturating_add(1);
                }
            }
        }
        engine.run_jump_flood(queue, encoder, active_px, self.config.outline);
    }

    fn render_static_cache<W>(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        aw: u32,
        ah: u32,
        layers: &mut [&mut dyn ViewportLayer<W>],
        paint_ctx: &PaintCtx<'_, W>,
        stats: &mut FrameStats,
    ) {
        let (color_view, resolve_target) = self.targets.static_color_views();
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("iletki static scene"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                depth_slice: None,
                resolve_target,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(self.config.wgpu_clear_color()),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: self.targets.depth_view(),
                depth_ops: Some(wgpu::Operations {
                    // Reversed-Z clear: 0.0 is far plane.
                    load: wgpu::LoadOp::Clear(0.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_viewport(0.0, 0.0, aw as f32, ah as f32, 0.0, 1.0);
        pass.set_scissor_rect(0, 0, aw, ah);
        pass.set_bind_group(0, self.camera_gpu.bind_group(), &[]);

        for phase in [DrawPhase::Opaque, DrawPhase::Transparent] {
            for layer in layers.iter() {
                if layer.phases().contains(phase) {
                    layer.paint(phase, &mut pass, paint_ctx);
                    stats.layer_draws = stats.layer_draws.saturating_add(1);
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn render_dynamic_over_cache<W>(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        aw: u32,
        ah: u32,
        composite_outline: bool,
        layers: &mut [&mut dyn ViewportLayer<W>],
        paint_ctx: &PaintCtx<'_, W>,
        stats: &mut FrameStats,
    ) {
        let (color_view, resolve_target) = self.targets.output_color_views();
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("iletki dynamic overlay"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                depth_slice: None,
                resolve_target,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(self.config.wgpu_clear_color()),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: self.targets.depth_view(),
                depth_ops: Some(wgpu::Operations {
                    // Fresh Reversed-Z depth so gizmos/overlays depth-test among themselves on top.
                    load: wgpu::LoadOp::Clear(0.0),
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
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
            Some(DEPTH_FORMAT),
            false,
        ) {
            pass.set_pipeline(blit_pipe);
            pass.set_bind_group(0, self.targets.static_identity_bg(), &[]);
            pass.draw(0..3, 0..1);
        }

        if composite_outline && let Some(outline) = &self.outline {
            outline.composite_into_pass(&mut pass, [aw, ah]);
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
        encoder: &mut wgpu::CommandEncoder,
        aw: u32,
        ah: u32,
        composite_outline: bool,
        layers: &mut [&mut dyn ViewportLayer<W>],
        paint_ctx: &PaintCtx<'_, W>,
        stats: &mut FrameStats,
    ) {
        let (color_view, resolve_target) = self.targets.output_color_views();
        let has_second_pass = composite_outline
            || layers.iter().any(|l| {
                let m = l.phases();
                m.contains(DrawPhase::Gizmo) || m.contains(DrawPhase::Overlay)
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("iletki continuous scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    depth_slice: None,
                    resolve_target: if has_second_pass {
                        None
                    } else {
                        resolve_target
                    },
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.config.wgpu_clear_color()),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: self.targets.depth_view(),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_viewport(0.0, 0.0, aw as f32, ah as f32, 0.0, 1.0);
            pass.set_scissor_rect(0, 0, aw, ah);
            pass.set_bind_group(0, self.camera_gpu.bind_group(), &[]);
            for phase in [DrawPhase::Opaque, DrawPhase::Transparent] {
                for layer in layers.iter() {
                    if layer.phases().contains(phase) {
                        layer.paint(phase, &mut pass, paint_ctx);
                        stats.layer_draws = stats.layer_draws.saturating_add(1);
                    }
                }
            }
            stats.static_rebuilt = true;
        }

        if has_second_pass {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("iletki continuous gizmo/overlay"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    depth_slice: None,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: self.targets.depth_view(),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if composite_outline && let Some(outline) = &self.outline {
                outline.composite_into_pass(&mut pass, [aw, ah]);
            }
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
            id_map: request_staging_map(&staged.id_buf),
            depth_map: request_staging_map(&staged.depth_buf),
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
        let id_bytes = map_staging_blocking(device, &staged.id_buf, staged.id_layout)?;
        let depth_bytes = map_staging_blocking(device, &staged.depth_buf, staged.depth_layout)?;
        Ok(decode_pick_hits(
            &self.camera,
            staged.logical,
            self.dpi_scale,
            staged.rect,
            &id_bytes,
            &depth_bytes,
        ))
    }

    fn submit_pick_rect<W>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        world: &W,
        layers: &mut [&mut dyn ViewportLayer<W>],
        rect_px: [u32; 4],
    ) -> Result<StagedPick, ViewportError> {
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

        self.streaming_pending |= self.prepare_layers(device, queue, world, layers).1;

        let logical = self.logical_size();
        self.camera_gpu
            .write(queue, self.camera.to_uniform_data(logical, self.dpi_scale));

        let pick = PickTargets::ensure(&mut self.pick_targets, device, size.allocated_px);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("iletki pick encoder"),
        });
        {
            let paint_ctx = PaintCtx {
                viewport_id: self.id,
                camera: &self.camera,
                camera_bind_group: self.camera_gpu.bind_group(),
                world,
                active_px: size.active_px,
                logical_size: logical,
                dpi_scale: self.dpi_scale,
            };
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
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_viewport(0.0, 0.0, aw as f32, ah as f32, 0.0, 1.0);
            pass.set_scissor_rect(x0, y0, w, h);
            pass.set_bind_group(0, self.camera_gpu.bind_group(), &[]);
            for layer in layers.iter() {
                if layer.phases().contains(DrawPhase::Picking) {
                    layer.paint(DrawPhase::Picking, &mut pass, &paint_ctx);
                }
            }
        }
        let depth_copy = self
            .depth_copy
            .get_or_insert_with(|| DepthCopyPipeline::new(device));
        let depth_r32f = depth_copy.copy_subrect(
            device,
            queue,
            &mut encoder,
            pick.depth_view(),
            [x0, y0],
            [w, h],
        );

        // Copy both ID and depth sub-rects into staging buffers in the SAME command submission.
        let id_buf = encode_texture_to_staging(
            device,
            &mut encoder,
            pick.id_texture(),
            [x0, y0, w, h],
            id_layout,
        );
        let depth_buf = encode_texture_to_staging(
            device,
            &mut encoder,
            &depth_r32f,
            [0, 0, w, h],
            depth_layout,
        );
        queue.submit(std::iter::once(encoder.finish()));

        Ok(StagedPick {
            id_buf,
            id_layout,
            depth_buf,
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

struct StagedPick {
    id_buf: wgpu::Buffer,
    id_layout: StagingLayout,
    depth_buf: wgpu::Buffer,
    depth_layout: StagingLayout,
    rect: [u32; 4],
    logical: Vec2,
}

fn encode_texture_to_staging(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    rect: [u32; 4],
    layout: StagingLayout,
) -> wgpu::Buffer {
    let [x, y, width, height] = rect;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("iletki readback staging"),
        size: layout.total_bytes as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x, y, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
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
    staging
}

fn extract_mapped_rows(
    staging: &wgpu::Buffer,
    layout: StagingLayout,
) -> Result<Vec<u8>, ViewportError> {
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

        #[cfg(not(target_arch = "wasm32"))]
        self.device
            .poll(wgpu::PollType::Poll)
            .map_err(|_| ViewportError::BufferMapFailed)?;

        let id_ready = map_is_ready(&self.id_map, wake)?;
        let depth_ready = map_is_ready(&self.depth_map, wake)?;
        if !id_ready || !depth_ready {
            return Poll::Pending;
        }

        let staged = self.staged.take().ok_or(ViewportError::PickAlreadyRead)?;
        let id_bytes = extract_mapped_rows(&staged.id_buf, staged.id_layout)?;
        let depth_bytes = extract_mapped_rows(&staged.depth_buf, staged.depth_layout)?;
        Poll::Ready(Ok(decode_pick_hits(
            &self.camera,
            staged.logical,
            self.dpi_scale,
            staged.rect,
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
    let [x0, y0, w, h] = rect;
    let mut hits: Vec<PickHit> = Vec::new();
    let inv_dpi = 1.0 / f64::from(dpi_scale);

    for row in 0..h {
        for col in 0..w {
            let px_index = (row as usize)
                .saturating_mul(w as usize)
                .saturating_add(col as usize);
            let id_off = px_index.saturating_mul(8);
            let depth_off = px_index.saturating_mul(4);
            let (Some(id_slice), Some(d_slice)) = (
                id_bytes.get(id_off..id_off.saturating_add(8)),
                depth_bytes.get(depth_off..depth_off.saturating_add(4)),
            ) else {
                continue;
            };
            let (Ok(lo_arr), Ok(hi_arr), Ok(d_arr)) = (
                <[u8; 4]>::try_from(id_slice.get(0..4).unwrap_or_default()),
                <[u8; 4]>::try_from(id_slice.get(4..8).unwrap_or_default()),
                <[u8; 4]>::try_from(d_slice),
            ) else {
                continue;
            };
            let id = PickId::from_rg32u([u32::from_le_bytes(lo_arr), u32::from_le_bytes(hi_arr)]);
            if id.is_none() {
                continue;
            }
            let depth = f32::from_le_bytes(d_arr);
            let screen_pt = Vec2::new(
                (f64::from(x0 + col) + 0.5) * inv_dpi,
                (f64::from(y0 + row) + 0.5) * inv_dpi,
            );
            let world_pos = camera.unproject_depth(screen_pt, depth, logical);

            if let Some(existing) = hits.iter_mut().find(|h| h.id == id) {
                // Reversed-Z: larger depth is closer to the camera.
                if depth > existing.depth {
                    existing.depth = depth;
                    existing.world_pos = world_pos;
                }
            } else {
                hits.push(PickHit {
                    id,
                    depth,
                    world_pos,
                });
            }
        }
    }
    // Sort nearest-first (highest Reversed-Z depth first).
    hits.sort_by(|a, b| b.depth.total_cmp(&a.depth));
    hits
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
    let staging = encode_texture_to_staging(device, &mut encoder, texture, rect, layout);
    queue.submit(std::iter::once(encoder.finish()));
    map_staging_blocking(device, &staging, layout)
}
