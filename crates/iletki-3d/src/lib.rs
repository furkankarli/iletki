//! Camera-relative Reversed-Z 3D scene, infinite ground grid, section clip planes and transform
//! gizmos for `iletki`.
//!
//! - [`vertex`]: compact 28-byte [`Vertex3D`] and `f64`-anchored [`Mesh3D`] builders.
//! - [`scene`]: [`Scene3DLayer`] with per-chunk `f64` camera-relative transforms, 3D frustum
//!   culling, 6 section clip planes and 64-bit sub-object GPU picking.
//! - [`grid`]: [`InfiniteGrid3DLayer`] analytic anti-aliased 3D ground grid (XY Z-up or XZ Y-up)
//!   with exact Reversed-Z depth output.
//! - [`gizmo`]: [`TransformGizmoLayer`] screen-size-invariant X/Y/Z axis arrows with per-axis
//!   64-bit GPU picking.
//! - [`instanced`]: [`InstancedMesh3DLayer`] GPU-instanced 3D batching (100k+ instances in 1 draw
//!   call) with per-instance 64-bit GPU picking and multi-viewport support.

mod face_batch;
pub mod gizmo;
mod gpu_cull;
pub mod grid;
pub mod instanced;
mod objects;
mod outline_bounds;
mod provoking;
pub mod scene;
mod section;
mod storage_batch;
pub mod vertex;

pub use gizmo::{GizmoAxis, TransformGizmoLayer};
pub use grid::{Grid3DConfig, GridPlane, InfiniteGrid3DLayer};
pub use instanced::{Instance3D, InstancedMesh3DLayer};
pub use scene::{Chunk3DId, Scene3DLayer};
pub use vertex::{Mesh3D, Vertex3D};
