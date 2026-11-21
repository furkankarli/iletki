# iletki

A fast, UI-framework-agnostic `wgpu` viewport for 2D/3D CAD, BIM, GIS, game editors and technical visualization, written in pure Rust.

## Why iletki?

- **Zero UI lock-in.** Core crates (`iletki-core`, `iletki-wgpu`, `iletki-2d`, `iletki-3d`) depend only on `wgpu` and `bytemuck` — never on `egui`, `winit`, `iced` or `slint`. Render either to a zero-copy `TextureView` (`OwnedTarget`) or directly into an external `RenderPass` (`BorrowedPass`). An optional thin [`iletki-egui`](crates/iletki-egui) adapter is provided separately.
- **Large-world `f64` precision.** 2D geometry uses split `hi/lo` `f32` vertex subtraction; 3D scenes render camera-relative with infinite Reversed-Z (`Depth32Float`). Stay sub-millimetre accurate at national grid coordinates (`661_858 m, 4_491_159 m`) where single-precision `f32` renderers snap in 50 cm jumps.
- **Dual execution modes.** `OnDemand` caches the static MSAA scene offscreen (0 static draw calls on pointer hover / overlay updates); `Continuous` streams at full frame rate for games and live simulation.
- **Multi-viewport by design.** Share one scene across N independent `ViewportHost` instances (e.g. Top / Front / Right / Perspective CAD quad views) inside a single `wgpu::CommandEncoder` via per-`ViewportId` uniform buffers.
- **Built-in editor primitives.** 64-bit GPU picking (`Rg32Uint` + Reversed-Z depth reconstruction, sync & cross-platform async for native + WebGPU), Jump-Flood (JFA) screen-space selection outlines, 6 world-space section clip planes, *Shaded with Edges* depth-offset wireframes, 100k+ GPU instancing, infinite 3D ground grid, screen-size-invariant 3D transform gizmo, `Camera::fit_3d`, standard CAD views and smooth `CameraTransition`s.
- **Never panics, zero `unsafe`, permissive license.** `unsafe_code = "forbid"`, panicking primitives and unchecked indexing denied in library code, `MIT OR Apache-2.0`, builds for native and `wasm32-unknown-unknown`.

## Benchmarks (NVIDIA RTX 4050 Laptop, Vulkan, 800×600, `--release`)

| Workload | Metric |
|---|---|
| **2D National-Grid CAD** (500,000 segments at UTM `661 km / 4,491 km`, 4× MSAA, 8×8 spatial culling) | Pan+Zoom: **0.33–0.64 ms** (~1,550–3,050 fps) · Idle crosshair (`OnDemand`): **0.11 ms** (0 static draws) |
| **3D 100,000 Instanced Boxes** (3.6M triangles, 1 draw call) | 1× MSAA: **1.22 ms (823 fps)** · 4× MSAA: **1.47 ms (681 fps)** (p99 1.58 ms) |
| **3D Architectural Scene** (4× MSAA + Shaded Edges + JFA Outline + Infinite Grid + 92 px Gizmo) | Orbit: **0.43 ms (2,329 fps)** · Idle (`OnDemand`): **0.063 ms** |
| **120 Continuous Drag-Resize Frames** (600 px ↔ 1200 px) | **3** GPU texture reallocations total (256 px block hysteresis) |

## Quick start

```rust
use iletki::{
    Camera, Mesh3D, PickId, Scene3DLayer, StandardView, Vec3,
    ViewportConfig, ViewportHost, ViewportLayer, Bounds3,
};

// 1. Create a viewport sharing your existing wgpu::Device
let mut host = ViewportHost::new(&device, ViewportConfig::default());

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

# Headless visual showcase & stress benchmarks (outputs PNGs to target/showcase)
cargo run --release -p iletki --example showcase

# Full quality gate (fmt, clippy, tests, docs, cargo-deny, wasm32)
rustup target add wasm32-unknown-unknown
cargo install --locked cargo-deny
cargo xtask ci
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
