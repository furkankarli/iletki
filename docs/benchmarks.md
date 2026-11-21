# Headless benchmark guide

## Published measurements

The [README results](../README.md#benchmarks) come from three release runs of
`3094b93745859378b939209a4c196dab9b2c5b19` on Linux x86-64 with Rust 1.100.0,
an NVIDIA GeForce RTX 4050 Laptop GPU, and NVIDIA Vulkan driver 620.70.06. Each workload
uses 4× MSAA. The p50 and p95 values use the nearest rank across the pooled samples
for that workload. The crosshair pass recorded zero static scene rebuilds in all three runs.

The CSVs include the run number, frame index within the run, and elapsed milliseconds.

- [500,000-segment pan and zoom](benchmarks/rtx-4050-vulkan/cad2d-pan-zoom.csv)
- [Crosshair updates](benchmarks/rtx-4050-vulkan/cad2d-crosshair.csv)
- [100,000-instance orbit](benchmarks/rtx-4050-vulkan/instanced-orbit.csv)

## Reproduce the workloads

```sh
cargo run --release --locked -p iletki --example showcase -- target/showcase
```

The example prints the selected adapter, backend and driver. Set `WGPU_BACKEND` to choose a backend.
On Linux with Vulkan, set `VK_DRIVER_FILES` when you need to select a specific driver. Confirm the
printed adapter on each run, especially on machines with integrated and discrete GPUs.

| Workload | Configuration | Timing output |
| --- | --- | --- |
| 2D pan and zoom | 500,000 segments, 1024×768, 4× MSAA, spatial culling | `cad2d-pan-zoom.csv`, 60 frames |
| 2D crosshair updates | Same scene, `OnDemand`, cached static geometry | `cad2d-crosshair.csv`, 30 frames and a static rebuild count in the terminal |
| 3D quad view | Four 512×512 viewports, 4× MSAA, one shared encoder | One first-frame wall time in the terminal |
| 3D instancing | 100,000 boxes, 1.2 million triangles, 800×600, 4× MSAA | `instanced-orbit.csv`, 60 frames |
| Drag resize | 120 frames from 600 to 1200 px wide and back | Total wall time and target-set reallocations after initialization |

The per-frame timers include command encoding, submission and waiting for GPU completion. They
exclude presentation, image readback, PNG output and CPU work performed before the timers start,
such as camera updates and crosshair mesh generation. These are synchronized offscreen latencies.
The resize workload waits at the end of the batch. Each target set can contain multiple textures.

Keep the CSVs in frame order and run several repetitions. When comparing renderers, fix the GPU,
driver, visible geometry, camera, lighting, MSAA and output format. Measure initial setup,
steady-state rendering, cache hits and presentation as separate workloads. Use enough samples for
any tail percentiles you publish. Large-world precision requires a measured coordinate or
projection error over a camera path.
