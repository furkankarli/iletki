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

`ILETKI_PROFILE=1` records additional frames per acceptance or model workload in `profile.csv`,
outside the latency samples. Section orbit traces cover all 30 camera angles. Other workloads
record eight frames. CPU scopes and GPU pass timestamps are in nanoseconds,
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
Set `ILETKI_ASSEMBLY_NO_SECTION_CULL=1` for a raster control run that submits whole chunks
to the GPU even when they lie outside a section plane. The `scene.section_culled_chunks`
counter counts fully clipped chunks, and `scene.section_culled_triangles` counts their
triangles omitted by the CPU. Use `ILETKI_ASSEMBLY_NO_SECTION_BRIDGING=1` for the control
that runs individual tests on crossing groups and omits every fully clipped range.
The optimization activates when at least three quarters of object groups cross a section
plane inside the camera frustum.
`scene.section_deferred_chunks` counts small packed chunks tested only by the rasterizer.
The `scene.section_bridged_triangles` and
`scene.section_bridged_edge_vertices` counters track clipped geometry retained to merge draws.
Use `ILETKI_ASSEMBLY_NO_SECTION_GROUPS=1` to compare grouped bounds against
per-chunk culling. The `scene.section_chunk_tests` and `scene.frustum_chunk_tests` counters
track individual bounds tests, and `scene.opaque_triangles` counts triangles submitted
to the color pass. Use `ILETKI_ASSEMBLY_SHUFFLE=1` to permute the insertion order of the
synthetic assembly while preserving the geometry and input sequence.
`ILETKI_ASSEMBLY_EDITS=1` moves every hundredth chunk on every sixth orbit tick.
Orbit timing includes those edits and camera updates. `ILETKI_ASSEMBLY_DUAL=1`
adds a full-size orthographic view to a model run. Use `ILETKI_ASSEMBLY_SPLIT=1` to
divide the requested width between two views. For the synthetic grid, select
`20000/split`, `100000/split`, or `100000/wide-dual` through `ILETKI_ASSEMBLY_CASE`.
After 60 warm-up frames, each of these workloads records 360 orbit samples.
Use `ILETKI_ASSEMBLY_ORBIT_ONLY=1` to skip the separate picking workloads during
timing repetitions. `ILETKI_ASSEMBLY_SIGNATURES=1` records both views at each of the 30
orbit angles after timing and profiling. The angle in degrees is the orbit frame
number modulo 30, multiplied by 12.
`ILETKI_ASSEMBLY_VISIBILITY_EDITS=1` toggles visibility for the same subset.
With either edit control, `edit-frame` repeats the changing frames from the orbit
series for separate percentile analysis.

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
Scripted replay owns camera and selection input for the duration of a capture.
Use Shift + left drag in the single 3D viewport to preview a visible-object selection
and release to commit it. Escape cancels the gesture and restores the committed selection.
`--dual` adds an orthographic view for camera and edit replays.

For the wide marquee gate, run the following command three times per adapter with separate
output directories. Use `--region boundary` and `--region empty` for separate controls.
The default `mixed` region alternates center and left rectangles for historical comparison.
Use `--rect-size 512` for large rectangles and `--motion orbit|section|edit` for invalidation
under camera moves, live section changes or 1% part edits. Section motion requires
`--style edges-section`. Replay requests default to a 60 Hz schedule and write a separate
final release event. Set `--input-hz 240 --samples 1440` or
`--input-hz 500 --samples 3000` to replay six seconds of input faster than the display.
During a burst, the demo keeps the newest pending preview and schedules GPU work against
the window's nominal refresh interval. A ready preview gets a redraw opportunity before
the next preview is encoded. Release requests take priority.

For simultaneous orbit, live section, scene edits, and marquee previews, use
`--action marquee --style edges-section --motion combined`. Add `--dual` to split
the window between an orthographic view on the left and the perspective view on
the right. The replay picks in right-panel pixels and shows the selection in both
views. Use `--1080p` to run borderless on a 1920×1080 display and
`--scene shuffled-100k` for the 100,000-part permutation. Confirm the actual drawable
size in `monitor.txt`. The replay holds its final camera and scene
while the release request finishes. Compare the applied preview rate, the input
to present-return percentiles, and the release digest between builds.

Run the combined replay three times per adapter at 60 Hz. The software gate is
at least 57 applied preview frames per scheduled second, 16.67 ms p95 and
33.3 ms p99 from input to present return. Compare the final release digest with
the baseline for the same fixture, sequence, viewport layout, and GPU. Report
an absent preview percentile when no previews reach a frame.

```sh
cargo run --release -p iletki-demo -- --latency-bench out/marquee/radeon/filled/1 \
  --scene wide --style edges-section --action marquee --region filled
python3 docs/benchmarks/repro/summarize-latency.py out/marquee
```

Use 16.67 ms p95, 33.3 ms p99 and at least 57 applied preview frames per scheduled second
as the fixed-view, filled 256×256 software gate. Check the highlighted object count to
distinguish dense, boundary and empty requests. Leave `ILETKI_PICK_PROFILE` unset for latency
gates. Build the demo with `--features profiling` and set `ILETKI_PICK_PROFILE=1` in a separate
run to log CPU phases, GPU passes and draw counts, including the outline mask and bounds.
Compare the intervals between adjacent applied previews independently of input latency.
The software pacing targets for fixed-view previews are 18.33 ms p95 and 25 ms p99.
Set `ILETKI_DIRECT_PICK=1` in the profiling build
to A/B the per-chunk pick path while keeping the same preview and contour code.

For a controlled presentation comparison, select `--present fifo` or `--present mailbox`,
`--depth 1` or `--depth 2`, and optionally `--notify-present` for the Wayland frame
callback. Scripted baseline runs omit these overrides. The interactive demo and `--action live`
request the frame callback before presenting. The window title shows smoothed CPU encoding
time after surface acquisition and whole frame time through presentation return.
The first frame is labeled separately. The averages start fresh after a gap of at least 350 ms.
Each trace records surface acquisition,
encode, queue submit and present return timestamps. GPU completion uses callbacks
observed during nonblocking polling. The event CSV tracks the frame using each input
or pick result, the rectangle, raw hits and highlighted objects, and timestamps for
pick encode start, submission, observed completion, highlight application and redraw requests.
The frame start is recorded when the redraw handler begins. The frame trace includes viewport
encoding completion and whether the dynamic pass ran. It distinguishes requests
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

Export one CSV row per synchronized trigger. Use the demo event ID, the trigger and
first visible change times in nanoseconds on the optical capture clock, and the raw
16-bit Gray marker read alongside that change. Use blank change and marker fields for
a trigger with no observed change. Space trials so a frame carrying each change can
be identified before the next trial. The acceptance tool reports missed and unmatched
trials along with latency percentiles for each recorded action.

```csv
event,trigger_ns,changed_ns,gray16
60,1000000000,1035000000,0x002B
```

```sh
python3 docs/benchmarks/repro/summarize-optical.py out/interaction/live capture.csv
```

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

For an assembly-scale STEP run, use
`OpenArm Cell/Integration with OpenArm/OpenArm_Cell+OpenArm_2.0.STEP` from the
OpenArm Hardware 2.0.0 release archive, by Enactic, Inc., under CERN-OHL-S-2.0.
The archive SHA-256 is
`247ce0141a9be6ebbb5a6c206cf0557574929ea0b80e20dddbff7c0c0919db75`.
With the same meshing and export settings, OCCT 7.9.3 and the fixture converter
produced 538 chunks and 1,586,648 triangles. Record the generated fixture hash
with the results. [OpenArm release](https://github.com/enactic/openarm_hardware/releases/tag/2.0.0)

```sh
python3 docs/benchmarks/repro/convert-model.py ifc out/schependomlaan.iltmesh out/schependomlaan.ifc
python3 docs/benchmarks/repro/convert-model.py step-gltf out/nist.iltmesh out/ctc02.gltf out/ctc04.gltf
ILETKI_ASSEMBLY_MESH=out/schependomlaan.iltmesh ILETKI_ASSEMBLY_SIZE=1920x1080 ILETKI_PROFILE=1 \
  target/release/examples/assembly_bench out/ifc/1 packed 1
```

### Production-scale loading and edits

The West Riverside Hospital architecture and structural models form a coordinated
IFC4 federation. Attribution goes to Wawan Solihin and the University of Auckland
OpenIFC Model Repository. The files are distributed under CC BY 3.0. The converter
tessellates each file, preserves its placements, and assigns distinct pick IDs
across source files. Keep the downloaded and converted assets in a local directory.
[Model card and attribution](https://huggingface.co/datasets/sylvainHellin/ifc-bench/blob/ebacaf2de71cedd7226897497fcdf8b143c5b66f/projects/west_riverside_hospital/model_card.md)

```sh
mkdir -p out/hospital
hospital_url=https://huggingface.co/datasets/sylvainHellin/ifc-bench/resolve/ebacaf2de71cedd7226897497fcdf8b143c5b66f/projects/west_riverside_hospital
for file in arc_ifc4.ifc str_ifc4.ifc license.txt; do
  curl --fail --location "$hospital_url/$file" --output "out/hospital/$file"
done
python3 docs/benchmarks/repro/convert-model.py ifc out/hospital/hospital.iltmesh \
  out/hospital/arc_ifc4.ifc out/hospital/str_ifc4.ifc
cargo build --release --locked -p iletki --features profiling --example assembly_bench
ILETKI_ASSEMBLY_MESH=out/hospital/hospital.iltmesh ILETKI_ASSEMBLY_SIZE=1920x1080 \
  ILETKI_ASSEMBLY_STYLE=edges-section ILETKI_ASSEMBLY_EDITS=1 \
  ILETKI_ASSEMBLY_VISIBILITY_EDITS=1 ILETKI_PROFILE=1 ILETKI_ASSEMBLY_MEMORY=1 \
  /usr/bin/time -v -o out/hospital/process-memory.txt \
  target/release/examples/assembly_bench out/hospital/run current 1
```

`fixture-decode` and `scene-insert` separate binary fixture reading from insertion into
the viewport layer. `first-frame` and `streaming-complete` use GPU-completed frames.
On Linux, GNU time records peak process RSS. The optional `memory.csv` records live
allocations and reserved blocks reported by wgpu at open, first frame, completed
upload, steady state, picking and close. These are allocator snapshots. An unavailable
allocator report leaves the fields empty.

Set `ILETKI_ASSEMBLY_CASE=100000/wide` and unset `ILETKI_ASSEMBLY_MESH` to run the
100,000-chunk synthetic control. Its `scene-build` timing includes generating each
mesh and inserting it. Match flags, fixture checksums, resolution, adapter and
sample counts when comparing revisions.

Keep the CSVs in frame order and run several repetitions. When comparing renderers, fix the GPU,
driver, visible geometry, camera, lighting, MSAA and output format. Measure initial setup,
steady-state rendering, cache hits and presentation as separate workloads. Use enough samples for
any tail percentiles you publish. Large-world precision requires a measured coordinate or
projection error over a camera path.
