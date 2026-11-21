# iletki

A fast, UI-framework-agnostic `wgpu` viewport for 2D/3D CAD, BIM, GIS, game editors and technical visualization, written in pure Rust.

## Why iletki?

- **Framework-independent integration.** The core crates have no UI or windowing dependencies. Use the resolved offscreen `TextureView` (`OwnedTarget`) or composite its color into your own `RenderPass` (`BorrowedPass`). The [`iletki-egui`](crates/iletki-egui) adapter is optional.
- **Large-world `f64` precision.** 2D geometry uses split `hi/lo` `f32` vertex subtraction. 3D scenes render camera-relative with infinite Reversed-Z (`Depth32Float`). These paths retain local detail near national grid coordinates such as `(661_858 m, 4_491_159 m)`.
- **Dual execution modes.** `OnDemand` caches the static MSAA scene offscreen (0 static draw calls on pointer hover / overlay updates); `Continuous` streams at full frame rate for games and live simulation.
- **Multi-viewport by design.** Share one scene across N independent `ViewportHost` instances (e.g. Top / Front / Right / Perspective CAD quad views) inside a single `wgpu::CommandEncoder` via per-`ViewportId` uniform buffers.
- **Built-in editor primitives.** 64-bit GPU picking (`Rg32Uint` + Reversed-Z depth reconstruction, sync & cross-platform async for native + WebGPU), Jump-Flood (JFA) screen-space selection outlines, 6 world-space section clip planes, *Shaded with Edges* depth-offset wireframes, 100k+ GPU instancing, infinite 3D ground grid, screen-size-invariant 3D transform gizmo, `Camera::fit_3d`, standard CAD views and smooth `CameraTransition`s.
- **Defensive input handling and permissive licensing.** The libraries forbid `unsafe` and deny panicking primitives and unchecked indexing. They build for native and `wasm32-unknown-unknown` and use `MIT OR Apache-2.0`.

## Benchmarks

The headless showcase measures a 500,000-segment 2D CAD scene at 1024×768, 100,000 instanced boxes at 800×600 (1.2 million triangles), and a 120-frame drag-resize workload. See the [benchmark guide](docs/benchmarks.md) for commands, timing boundaries and output files.

## Quick start

```rust
use iletki::{
    Camera, Mesh3D, PickId, Scene3DLayer, StandardView, Vec3,
    ViewportConfig, ViewportHost, ViewportLayer, Bounds3,
};

// 1. Create a viewport sharing your existing wgpu::Device
let mut host = ViewportHost::new(&device, ViewportConfig::default())?;

// 2. Build a 3D scene anchored at f64 world coordinates
let mut scene = Scene3DLayer::new(1);
let mut mesh = Mesh3D::at_origin(Vec3::ZERO);
mesh.push_box_with_edges(
    Vec3::splat(-1.0),
    Vec3::splat(1.0),
    [70, 185, 235, 255],
    [12, 16, 24, 110],
    PickId::new(1, 42, 0),
    true,
);
let id = scene.insert(mesh);
scene.set_selected(id, true); // Enables Jump-Flood selection outline

// 3. Frame a standard CAD isometric view
let bounds = Bounds3::from_corners(Vec3::splat(-1.0), Vec3::splat(1.0));
let (cam, _pivot) = Camera::standard_view_3d(
    StandardView::Isometric,
    bounds,
    host.logical_size(),
    1.2,
    true,
);
*host.camera_mut() = cam;

// 4. Render to a zero-copy TextureView (or use prepare_for_pass / paint_into_pass)
let mut layers: [&mut dyn ViewportLayer<()>; 1] = [&mut scene];
let frame = host.render_to_texture(&device, &queue, &mut encoder, &(), &mut layers);
```

For native asynchronous picking in an event-driven UI, submit the frame, call `begin_pick_rect`, and schedule checks with `PendingPick::try_read` until the result is ready. WebGPU drives its mapping callbacks asynchronously. See the [`ViewportHost` API](crates/iletki-wgpu/src/host.rs).

## Workspace layout

See [docs/architecture.md](docs/architecture.md) for the crate layering rules.

- `crates/iletki-core` — Zero-dependency `f64` math, 2D/3D cameras, `fit_2d`/`fit_3d`, `StandardView`, `CameraTransition`, clip planes, frustums, camera rigs, input/command contracts, `PickId`, `ViewportId`, 256 px block sizing.
- `crates/iletki-wgpu` — `ViewportHost`, `OnDemand` / `Continuous`, `OwnedTarget` / `BorrowedPass`, Jump-Flood selection outlines, 64-bit sync + async GPU picking, headless RGBA readback.
- `crates/iletki-2d` — Split `hi/lo` 2D lines, fills, wide lines, 8×8 spatial culling, 64 MiB `PagedBuffer`, adaptive grids and overlays.
- `crates/iletki-3d` — Camera-relative Reversed-Z 3D meshes, *Shaded with Edges*, `InstancedMesh3DLayer`, infinite 3D grid, 92 px transform gizmo.
- `crates/iletki-egui` — Optional thin `egui` / `eframe` adapter (`ViewportPaintCallback`, input & command translation).
- `crates/iletki` — Top-level facade re-exporting workspace crates behind feature flags (`wgpu`, `cad2d`, `cad3d`).
- `apps/iletki-demo` — Standalone interactive native demo (`winit` + `wgpu`, unpublished).

## Running demos & CI

```sh
# Interactive desktop demo (3D Arch, 4-Way Quad-View, 2D 500k UTM CAD, 100k 3D Instances, GPU Pick)
cargo run --release -p iletki-demo

# Headless visual showcase (writes PNGs and timing CSVs to target/showcase)
cargo run --release -p iletki --example showcase

# Full quality gate (fmt, clippy, tests, docs, cargo-deny, wasm32)
rustup target add wasm32-unknown-unknown
cargo install --locked cargo-deny
cargo xtask ci
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
