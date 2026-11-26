//! Framework-agnostic `wgpu` viewport runtime for `iletki`.
//!
//! `iletki-wgpu` plugs into any host that provides a `wgpu::Device` and `wgpu::Queue` (`egui`,
//! `winit`, `iced`, `slint`, custom hybrid UI frameworks, or headless CLI/CI runners):
//!
//! - **Dual rendering paths:**
//!   - [`ViewportHost::render_to_texture`] produces a zero-copy `wgpu::TextureView` cropped by
//!     `uv_max`.
//!   - [`ViewportHost::prepare_for_pass`] + [`ViewportHost::paint_into_pass`] blits directly into
//!     a host-owned `wgpu::RenderPass`.
//! - **Dual scheduling modes:** [`iletki_core::ViewportMode::OnDemand`] caches the static scene in
//!   an offscreen MSAA target so pointer moves incur zero static draw calls;
//!   [`iletki_core::ViewportMode::Continuous`] streams directly at full frame rate.
//! - **256 px block-aligned targets:** stutter-free resizing with shrink hysteresis.
//! - **64-bit ID + Reversed-Z depth picking** and **headless RGBA readback**.

mod blit;
pub mod error;
mod hiz;
pub mod host;
pub mod layer;
mod outline;
mod pass_graph;
#[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
pub mod profiling;
mod selection_preview;
pub mod target;
mod trace;
pub mod uniforms;

pub use error::ViewportError;
pub use host::{FrameOutput, FrameStats, PassTargetDesc, PendingPick, RgbaImage, ViewportHost};
pub use layer::{
    GpuCullCtx, LayerRevision, PaintCtx, PassHookCtx, PassHookMask, PassHookStage, PrepareCtx,
    UploadTracker, ViewportLayer, ViewportLifetime,
};
pub use outline::OUTLINE_MASK_FORMAT;
pub use selection_preview::{PendingSlotPreview, SelectionPreviewSnapshot, SelectionPreviewStats};
pub use target::{
    DEPTH_FORMAT, DEPTH_STENCIL_FORMAT, PICK_ID_FORMAT, SELECTION_ID_FORMAT, ViewportConfig,
};
pub use uniforms::{CAMERA_WGSL, CameraUniformsGpu};
pub use wgpu;
