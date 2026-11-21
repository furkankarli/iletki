//! `iletki` — a fast, framework-agnostic `wgpu` viewport for 2D/3D CAD, game editors, GIS and
//! visualization.
//!
//! This top-level crate re-exports the workspace layers behind feature flags.

pub use iletki_core as core;
pub use iletki_core::*;

#[cfg(feature = "wgpu")]
pub use iletki_wgpu as gpu;
#[cfg(feature = "wgpu")]
pub use iletki_wgpu::{
    CAMERA_WGSL, CameraUniformsGpu, DEPTH_FORMAT, FrameOutput, FrameStats, OUTLINE_MASK_FORMAT,
    PICK_ID_FORMAT, PaintCtx, PassTargetDesc, PrepareCtx, RgbaImage, ViewportConfig, ViewportError,
    ViewportHost, ViewportLayer,
};

#[cfg(feature = "cad2d")]
pub use iletki_2d as cad2d;
#[cfg(feature = "cad2d")]
pub use iletki_2d::{
    Chunk2DHandle, DrawStyle2D, GridStyle, Mesh2D, PagedBuffer, Scene2DLayer, SlotHandle,
    TileRange, Vertex2D, WideVertex2D, order_to_depth, push_wide_segment, to_linear,
};

#[cfg(feature = "cad3d")]
pub use iletki_3d as cad3d;
#[cfg(feature = "cad3d")]
pub use iletki_3d::{
    Chunk3DId, GizmoAxis, Grid3DConfig, GridPlane, InfiniteGrid3DLayer, Instance3D,
    InstancedMesh3DLayer, Mesh3D, Scene3DLayer, TransformGizmoLayer, Vertex3D,
};
