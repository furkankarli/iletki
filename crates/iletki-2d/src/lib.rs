//! High-precision 2D CAD and vector rendering kit for `iletki`.
//!
//! - Split `hi/lo` `f32` vertex precision ([`vertex`]) for jitter-free coordinates up to `10^9`.
//! - Paged incremental GPU vertex buffers ([`paged_buffer`]) with `UploadBudget` throttling.
//! - Ready-to-use [`Scene2DLayer`] with cached static geometry, zero-static-redraw dynamic overlays
//!   and 64-bit GPU picking ([`layer`]).
//! - Adaptive grid, crosshair and window/crossing selection box helpers ([`overlay`]).

pub mod layer;
pub mod overlay;
pub mod paged_buffer;
pub mod spatial;
pub mod vertex;

pub use layer::{Chunk2DHandle, Mesh2D, Scene2DLayer};
pub use overlay::{
    GridStyle, adaptive_grid_step, push_adaptive_grid, push_crosshair, push_selection_box,
};
pub use paged_buffer::{PagedBuffer, SlotHandle};
pub use spatial::TileRange;
pub use vertex::{
    DrawStyle2D, Vertex2D, WideVertex2D, order_to_depth, push_wide_segment, to_linear,
};
