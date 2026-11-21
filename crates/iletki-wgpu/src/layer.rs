//! Layer trait and render contexts connecting application scenes to [`crate::ViewportHost`].

use iletki_core::{Camera, DrawPhase, DrawPhaseMask, UploadBudget, Vec2, ViewportId};

/// Context handed to [`ViewportLayer::prepare`] once per frame before any render passes open.
pub struct PrepareCtx<'a, W = ()> {
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
    /// Target color texture format.
    pub color_format: wgpu::TextureFormat,
    /// Target depth texture format (`Depth32Float`).
    pub depth_format: wgpu::TextureFormat,
    /// MSAA sample count for main scene/gizmo/overlay passes (`1` or `4`).
    pub msaa_samples: u32,
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

    /// Updates GPU resources (pipelines, vertex buffers, instances) within `ctx.budget`.
    ///
    /// Return `true` if more streaming uploads remain for subsequent frames.
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_, W>) -> bool;

    /// Records draw commands for `phase` into `pass`.
    ///
    /// `@group(0)` is already bound to `ctx.camera_bind_group` and the viewport/scissor is already
    /// constrained to the active sub-rectangle.
    fn paint(&self, phase: DrawPhase, pass: &mut wgpu::RenderPass<'_>, ctx: &PaintCtx<'_, W>);
}
