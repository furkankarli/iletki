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

Use `ILETKI_ASSEMBLY_STYLE=edges` for shaded edges and
`ILETKI_ASSEMBLY_STYLE=edges-section` for shaded edges and a rotating section plane.
The adapter chooses raster clipping when clip distances are available. Set
`ILETKI_ASSEMBLY_SAMPLED_SECTION=1` to exercise the sampled shader.
`ILETKI_ASSEMBLY_EDITS=1` moves every hundredth chunk on every sixth orbit tick.
Orbit timing includes those edits and camera updates. `ILETKI_ASSEMBLY_DUAL=1`
adds an orthographic view to a model run and records both reference image hashes.

The offscreen workloads exclude window acquisition, compositing and presentation. The
asynchronous pick workload polls continuously to measure request readiness independently
of the application's event loop or display refresh rate.

### Interactive window capture

```sh
cargo build --release --locked -p iletki-demo
target/release/iletki-demo --latency-bench out/interaction/wide \
  --scene wide --style edges-section --action orbit
python3 docs/benchmarks/repro/summarize-latency.py out/interaction/wide
```

The window workload uses the demo camera rig and picking API. It records 60 warm-up
events followed by 360 samples by default. Choose `--scene demo`, `--scene wide` or
`--scene path/to/model.iltmesh`. The model viewport requests 1920×1080 pixels and the
synthetic viewport requests 1024×768 pixels. `monitor.txt` records the actual window
size, output and active refresh rate. The replay runs at approximately 60 input
events per second. Actions are `orbit`, `pan`, `zoom`, `click`, `marquee` and `edit`.
The edit action orbits while moving every hundredth part once every six events.
`--action live` records OS input from dragging, scrolling and releasing a pick button.
Use Shift + left drag in the single 3D viewport to preview a visible-object selection
and release to commit it. Escape cancels the gesture and restores the committed selection.
`--dual` adds an orthographic view for camera and edit replays.

For the wide marquee gate, run the following command three times per adapter with separate
output directories. Use `--region boundary` and `--region empty` for separate controls.
The default `mixed` region alternates center and left rectangles for historical comparison.
Use `--rect-size 512` for large rectangles and `--motion orbit|section|edit` for invalidation
under camera moves, live section changes or 1% part edits. Section motion requires
`--style edges-section`. Replay requests run on a 60 Hz schedule and write a separate final
release event.

```sh
cargo run --release -p iletki-demo -- --latency-bench out/marquee/radeon/filled/1 \
  --scene wide --style edges-section --action marquee --region filled
python3 docs/benchmarks/repro/summarize-latency.py out/marquee
```

Use 16.67 ms p95, 33.3 ms p99 and at least 57 applied preview frames per scheduled second
as the fixed-view, filled 256×256 software gate. Check the highlighted object count to
distinguish dense, boundary and empty requests. Leave `ILETKI_PICK_PROFILE` unset for latency
gates. Build the demo with `--features profiling` and set `ILETKI_PICK_PROFILE=1` in a separate
run to log CPU phases, GPU passes and draw counts. Set `ILETKI_DIRECT_PICK=1` in that build
to A/B the per-chunk pick path while keeping the same preview and contour code.

For a controlled presentation comparison, select `--present fifo` or `--present mailbox`,
`--depth 1` or `--depth 2`, and optionally `--notify-present` for the Wayland frame
callback. Baseline runs omit these overrides. Each frame records surface acquisition,
encode, queue submit and present return timestamps. GPU completion uses callbacks
observed during nonblocking polling. The event CSV tracks the frame using each input
or pick result, the rectangle, raw hits and highlighted objects, and timestamps for
pick encode start, submission and observed completion. It distinguishes requests
coalesced before submission, results coalesced before a frame and stale results.
Compare the release digest between packed and direct runs using the same binary and adapter.
The digest includes the ordered hit IDs, depth bits and reconstructed world positions.
Surface failures retain their own status. The top-left marker encodes the low 16 Gray-code frame-ID bits in
left-to-right three-pixel cells. Match external optical captures to the frame ID and
use a synchronized physical input trigger to measure input-to-display latency.
The timing CSVs cover software observations and leave physical latency to that capture.

For physical acceptance, run in live mode on a fixed 60 Hz display. Trigger HID input and
an LED from a shared clock and capture the LED, frame marker and changed viewport region
together at 1,000 fps or faster, or use a calibrated equivalent. Vary trigger time across the
refresh interval. Measure the change at the object contour for selection so scanout at the
marker position does not substitute for scanout at the selected object. The rendering limits
are 12 ms p95 and 16.67 ms p99
for the 20,000-part workload with edges and a live section. The physical targets are
33.3/50 ms for camera p95/p99 and 50/66.7 ms from either a click or a marquee release to
the changed contour.

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
