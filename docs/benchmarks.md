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

## Independent 3D parts

The assembly workload stores each part as a separate mesh chunk with a pick ID. It orbits
1,000, 4,000 and 20,000 parts in near, wide and dual views at 1024×768 with 4× MSAA. The
4,000-part near view also measures a blocking 5×5 pick, a blocking 256×256 marquee, and
a moving asynchronous pick with one request in flight. For this near view the 60 Hz
targets are 12 ms orbit, 8 ms blocking click, and 16.67 ms marquee at p95. The
20,000-part wide-view targets are 12 ms at p95 and 16.67 ms at p99.

```sh
cargo build --release --locked -p iletki --features profiling --example assembly_bench
for run in 1 2 3; do
  ILETKI_PROFILE=1 target/release/examples/assembly_bench "out/assembly/$run" candidate "$run"
done
ILETKI_PRIMARY_ONLY=1 ILETKI_PROFILE=1 \
  target/release/examples/assembly_bench out/assembly/visibility-only visibility-only 1
ILETKI_ASSEMBLY_CASE=20000/wide ILETKI_PROFILE=1 \
  target/release/examples/assembly_bench out/assembly/geometry-direct geometry-direct 1
```

`frames.csv` includes the first submitted frame, time until full mesh upload, and
steady-state samples. For orbit,
`cpu_ms` covers encoding through command-buffer finalization and `wall_ms` also includes
submission and waiting for GPU completion. For asynchronous picking, `cpu_ms` is the
time through submission and `wall_ms` includes polling and decode. Orbit records a
reference image hash per case and run. Picking records the ordered ID and depth hash
of each request. Compare signatures on the same adapter and backend.

`ILETKI_PROFILE=1` records eight additional frames per acceptance or model workload in `profile.csv`,
outside the latency samples. CPU scopes and GPU pass timestamps are in nanoseconds,
along with counters for candidates, draws and uniform writes. GPU timestamps require
the adapter to support the feature. The `visibility-only` variant keeps visibility
filtering and uses individual uniform writes for the orbit control. The
`geometry-direct` variant uses one draw per chunk with batched camera-relative uniforms.
`ILETKI_ASSEMBLY_CASE=20000/wide` isolates one size and view. For revision A/B testing,
build each checkout in a separate Cargo target directory and alternate their run order.

The offscreen workloads exclude window acquisition, compositing and presentation. The
asynchronous pick workload polls continuously to measure request readiness independently
of the application's event loop or display refresh rate.

### Tessellated IFC and STEP models

Use [the fixture converter](benchmarks/repro/convert-model.py) with NumPy and
IfcOpenShell 0.8.5 for IFC. Use OCCT DRAW 7.9 for STEP. The benchmark records loading
and scene insertion separately from upload and steady-state rendering. Record the
external conversion durations separately. The IFC converter preserves object placement
and filters space and opening solids. Openings in rendered objects are tessellated by
IfcOpenShell. The STEP converter reads triangle primitives and their node transforms
from OCCT's glTF export.

For the IFC workload, use the design model of Schependomlaan by the original project
owners, distributed under CC BY 4.0. For STEP, use `nist_ctc_02_asme1_ap242-e2.stp` and
`nist_ctc_04_asme1_ap242-e1.stp` from NIST's STEP archive. For each STEP file, load the
XDE document with `ReadStep`, take its shape with `XGetOneShape`, tessellate with
`incmesh shape 0.08 -angular 12 -parallel 1`, and use
`WriteGltf doc output.gltf -trsfFormat mat4 -mergeFaces 1`.
[Schependomlaan model](https://github.com/buildingsmart-community/Community-Sample-Test-Files/blob/main/IFC%202.3.0.1%20%28IFC%202x3%29/Schependomlaan/Design%20model%20IFC/IFC%20Schependomlaan.ifc)
[Model license](https://github.com/openBIMstandards/Archive-DataSetSchependomlaan/blob/master/LICENSE.MD)
[NIST CAD models](https://www.nist.gov/ctl/smart-connected-systems-division/smart-connected-manufacturing-systems-group/mbe-pmi-0)

```sh
python3 docs/benchmarks/repro/convert-model.py ifc out/schependomlaan.iltmesh out/schependomlaan.ifc
python3 docs/benchmarks/repro/convert-model.py step-gltf out/nist.iltmesh out/ctc02.gltf out/ctc04.gltf
ILETKI_ASSEMBLY_MESH=out/schependomlaan.iltmesh ILETKI_ASSEMBLY_SIZE=1920x1080 ILETKI_PROFILE=1 \
  target/release/examples/assembly_bench out/ifc/1 packed 1
```

Keep the CSVs in frame order and run several repetitions. When comparing renderers, fix the GPU,
driver, visible geometry, camera, lighting, MSAA and output format. Measure initial setup,
steady-state rendering, cache hits and presentation as separate workloads. Use enough samples for
any tail percentiles you publish. Large-world precision requires a measured coordinate or
projection error over a camera path.
