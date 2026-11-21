//! Framework-agnostic core types for the `iletki` GPU viewport.
//!
//! `iletki-core` holds all GPU-independent viewport contracts:
//! - Double-precision (`f64`) 2D/3D linear algebra, split `hi/lo` coordinates and Reversed-Z
//!   projection matrices ([`math`]).
//! - 2D orthographic, 3D orthographic and 3D perspective cameras with section clip planes and
//!   camera-relative GPU uniform packing ([`camera`]).
//! - Execution modes, dirty flags, draw phases, upload budgets and 64-bit sub-object pick IDs
//!   ([`mode`]).
//! - UI-agnostic pointer/keyboard/touch/pen input and host commands ([`input`]).
//! - Pluggable camera controllers for 2D CAD, 3D orbit and 3D fly navigation ([`rig`]).
//! - 256-pixel block-aligned offscreen target sizing with shrink hysteresis ([`target`]).

pub mod camera;
pub mod input;
pub mod math;
pub mod mode;
pub mod rig;
pub mod target;

pub use camera::{
    Camera, CameraTransition, CameraUniformData, MAX_CLIP_PLANES, MAX_ZOOM, MIN_ZOOM, Projection,
    StandardView,
};
pub use input::{
    CursorKind, KeyCode, Modifiers, PointerButton, ScrollUnit, ViewportCommand, ViewportInput,
};
pub use math::{
    Bounds2, Bounds3, Frustum, Mat4, Plane, Quat, Ray, Vec2, Vec3, Vec4, split, split2,
};
pub use mode::{
    DirtyFlags, DrawPhase, DrawPhaseMask, OutlineConfig, PickHit, PickId, UploadBudget, ViewportId,
    ViewportMode,
};
pub use rig::{Cad2DRig, CameraRig, FlyRig, OrbitMode, OrbitRig, RigOutcome};
pub use target::{BLOCK_ALIGN_PX, BlockSizing, MAX_TARGET_DIM, TargetSize};
