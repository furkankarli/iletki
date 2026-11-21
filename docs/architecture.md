# Architecture

iletki is a Cargo workspace of small, strictly layered crates. A crate may depend only on crates
below it, and core layers never pull in a UI or windowing framework.

| Layer | Crate | Status | Role |
|---|---|---|---|
| 0 | `iletki-core` | ready | Zero-dependency `f64` math, 2D/3D cameras (split `hi/lo` + camera-relative Reversed-Z), frustums, clip planes, camera rigs (`Cad2DRig`, `OrbitRig`, `FlyRig`), input/command contracts, `DrawPhase`, `OutlineConfig`, `ViewportId`, `UploadBudget`, 64-bit `PickId` and 256 px block-aligned target sizing |
| 1 | `iletki-wgpu` | ready | `ViewportHost` with dual integration paths (`OwnedTarget` zero-copy `TextureView` + `BorrowedPass`), `OnDemand` static-scene cache vs `Continuous` mode, `DrawPhase` render graph, Jump-Flood Algorithm (JFA) screen-space selection outlines, 64-bit ID + Reversed-Z depth picking, headless RGBA readback and shared WGSL camera uniforms |
| 2 | `iletki-2d` | ready | High-precision 2D CAD/vector layer kit: split `hi/lo` compact 24-byte lines & fills, screen-space wide lines, 8×8 spatial tile frustum culling, paged incremental vertex pools (`PagedBuffer`), lazy parallel 64-bit picking, adaptive grids and overlays |
| 2 | `iletki-3d` | ready | 3D editor helper kit: camera-relative Reversed-Z meshes, *Shaded with Edges* depth-offset wireframes, GPU instancing (`InstancedMesh3DLayer`, 100k+ objects in 1 draw call), 6 section clip planes, 3D frustum culling, per-`ViewportId` multi-viewport uniform buffers, infinite anti-aliased ground grid and screen-size-invariant transform gizmos with 64-bit sub-object picking |
| 3 | `iletki-egui` | ready | Optional thin `egui` / `eframe` adapter (`ViewportPaintCallback`, input & command translation) |
| 3 | `iletki-slint` | ready | Optional Slint adapter using an application-supplied image importer, with viewport sizing and picking in logical coordinates |
| 3 | `iletki` | ready | Top-level facade re-exporting the workspace crates behind features (`wgpu`, `cad2d`, `cad3d`) plus headless visual/benchmark showcase (`examples/showcase.rs`) |
| app | `iletki-demo` | ready | Standalone interactive desktop test app (`winit` + `wgpu`, unpublished) exercising 3D Arch, 4-Way Quad-View, 2D 500k UTM CAD, 100k 3D Instancing, Section Clips and 64-bit GPU Picking |

## Principles

- **Shared world, many viewports.** One scene/world holds GPU geometry; N `ViewportHost` instances
  render it from independent cameras (quad views, 2D + 3D side-by-side) with zero duplicated mesh
  memory.
- **Two execution modes.** `OnDemand` caches the static scene in an offscreen MSAA texture so
  pointer movement redraws only lightweight overlays; `Continuous` streams directly at full frame
  rate for games and animation.
- **Large-world precision.** World coordinates stay `f64`. 2D shaders subtract split `hi/lo` `f32`
  pairs; 3D shaders render camera-relative with Reversed-Z (`Depth32Float`, `GreaterEqual`,
  infinite far plane).
- **Defensive input handling.** The core clamps target sizes and filters non-finite camera inputs.
  GPU readback checks device buffer limits. The libraries forbid `unsafe` and deny panicking
  primitives.

The application owns the device, event loop and presentation. `OwnedTarget` exposes resolved
offscreen color and its UV bounds. `BorrowedPass` prepares the same offscreen rendering, then
composites color into the application's pass using its viewport and scissor. The application's depth
attachment affects pipeline compatibility. Viewport scene depth is stored in the offscreen target.

`OnDemand` refreshes its cached passes when shared layers report new revisions or their membership
changes. Hosts can schedule another frame while `FrameStats::streaming_pending` is set. Native
hosts using `begin_pick_rect` keep checking `PendingPick::try_read` while the request is pending.
WebGPU drives its mapping callbacks automatically.
