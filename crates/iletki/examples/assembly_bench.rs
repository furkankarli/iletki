//! Headless benchmark for rendering and picking thousands of independently stored 3D chunks.
//!
//! Run with `cargo run --release --locked -p iletki --features profiling --example assembly_bench
//! -- <output-dir> <variant> <run>`. Set `ILETKI_PRIMARY_ONLY=1` to run the acceptance fixture.
//! Set `ILETKI_ASSEMBLY_CASE=20000/wide` to isolate a particular size and view.
//! Set `ILETKI_ASSEMBLY_MESH` to load a tessellated model fixture instead of the synthetic grid.

#![allow(clippy::print_stderr)]

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    native::run()
}

#[cfg(not(target_arch = "wasm32"))]
mod assembly_fixture;

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use iletki::gpu::{RgbaImage, profiling, wgpu};
    use iletki::{
        Camera, Chunk3DId, Mesh3D, PickHit, PickId, Plane, Scene3DLayer, Vec3, ViewportConfig,
        ViewportHost,
    };
    use std::fs::{self, File};
    use std::future::Future;
    use std::io::{BufWriter, Write};
    use std::path::{Path, PathBuf};
    use std::pin::pin;
    use std::task::{Context, Poll, Waker};
    use std::time::Instant;

    type Error = Box<dyn std::error::Error>;
    const SIZE: [u32; 2] = [1024, 768];
    const ORBIT_PERIOD: u32 = 30;
    const WARMUP: u32 = 60;

    fn block_on<F: Future>(future: F) -> F::Output {
        let mut future = pin!(future);
        let mut context = Context::from_waker(Waker::noop());
        loop {
            match future.as_mut().poll(&mut context) {
                Poll::Ready(result) => return result,
                Poll::Pending => std::thread::yield_now(),
            }
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Style {
        Faces,
        Edges,
        Section,
    }

    impl Style {
        fn from_env() -> Result<Self, Error> {
            match std::env::var("ILETKI_ASSEMBLY_STYLE").as_deref() {
                Err(_) | Ok("faces") => Ok(Self::Faces),
                Ok("edges") => Ok(Self::Edges),
                Ok("edges-section") => Ok(Self::Section),
                _ => Err("expected faces, edges or edges-section".into()),
            }
        }
    }

    #[derive(Clone, Copy)]
    struct Case {
        chunks: u32,
        view: &'static str,
        model: Option<(Vec3, f64)>,
        style: Style,
    }

    impl Case {
        fn primary(self) -> bool {
            self.chunks == 4_000 && self.view == "near"
        }

        fn acceptance_wide(self) -> bool {
            matches!(self.chunks, 20_000 | 100_000)
                && matches!(self.view, "wide" | "split" | "wide-dual")
        }

        fn camera(self, frame: u32, secondary: bool) -> Camera {
            let mut camera = self.orbit_camera(frame, secondary);
            if self.style == Style::Section {
                let center = self.model.map_or(
                    Vec3::new(63.0, f64::from(self.chunks.div_ceil(64)), 0.0),
                    |model| model.0,
                );
                let angle = f64::from(frame % ORBIT_PERIOD) * std::f64::consts::TAU
                    / f64::from(ORBIT_PERIOD);
                if let Some(plane) =
                    Plane::from_normal_point(Vec3::new(angle.sin(), angle.cos(), 0.15), center)
                {
                    camera.set_clip_planes(&[plane]);
                }
            }
            camera
        }

        fn orbit_camera(self, frame: u32, secondary: bool) -> Camera {
            let angle =
                f64::from(frame % ORBIT_PERIOD) * std::f64::consts::TAU / f64::from(ORBIT_PERIOD);
            if let Some((focus, radius)) = self.model {
                if secondary {
                    return Camera::ortho_3d(
                        focus + Vec3::Z * radius,
                        focus,
                        Vec3::Y,
                        radius,
                        radius / 1500.0,
                        radius * 3.0,
                    );
                }
                let eye =
                    focus + Vec3::new(radius * angle.sin(), -radius * angle.cos(), radius * 0.65);
                return Camera::perspective_3d(
                    eye,
                    focus,
                    Vec3::Z,
                    55_f64.to_radians(),
                    radius / 1500.0,
                );
            }
            let focus = Vec3::new(30.0, 4.0, 0.8);
            if matches!(self.view, "wide" | "split" | "wide-dual") {
                let half_y = f64::from(self.chunks.div_ceil(64));
                let focus = Vec3::new(63.0, half_y, 0.8);
                let radius = half_y.max(64.0) * 2.0;
                if secondary {
                    return Camera::ortho_3d(
                        focus + Vec3::Z * radius,
                        focus,
                        Vec3::Y,
                        radius,
                        0.05,
                        radius * 3.0,
                    );
                }
                let eye = focus + Vec3::new(radius * angle.sin(), -radius * angle.cos(), radius);
                return Camera::perspective_3d(eye, focus, Vec3::Z, 55_f64.to_radians(), 0.05);
            }
            if secondary {
                Camera::ortho_3d(focus + Vec3::Z * 15.0, focus, Vec3::Y, 16.0, 0.05, 100.0)
            } else {
                let eye = focus + Vec3::new(7.0 * angle.sin(), -7.0 * angle.cos(), 5.0);
                Camera::perspective_3d(eye, focus, Vec3::Z, 55_f64.to_radians(), 0.05)
            }
        }

        fn make_scene(
            self,
            batch: bool,
            pack_geometry: bool,
        ) -> (Scene3DLayer, Vec<(Chunk3DId, Vec3)>) {
            let mut scene = Scene3DLayer::new(1);
            scene.set_profile_batch_uniforms(batch);
            scene.set_profile_batch_geometry(pack_geometry);
            let mut edits = Vec::new();
            let mut order: Vec<_> = (0..self.chunks).collect();
            if std::env::var_os("ILETKI_ASSEMBLY_SHUFFLE").is_some() {
                // An odd multiplier permutes the keys and distributes nearby parts across slots.
                order.sort_unstable_by_key(|i| i.wrapping_mul(0x9e37_79b9));
            }
            for i in order {
                let x = f64::from(i % 64) * 2.0;
                let y = f64::from(i / 64) * 2.0;
                let origin = Vec3::new(x, y, 0.0);
                let mut mesh = Mesh3D::at_origin(origin);
                let min = Vec3::new(-0.6, -0.6, 0.0);
                let max = Vec3::new(0.6, 0.6, 1.6);
                let id = PickId::new(1, i + 1, 0);
                let color = [120, 170, 200, 255];
                if self.style == Style::Faces {
                    mesh.push_box(min, max, color, id, false);
                } else {
                    mesh.push_box_with_edges(min, max, color, [24, 32, 40, 160], id, false);
                }
                let id = scene.insert(mesh);
                if i % 100 == 0 {
                    edits.push((id, origin));
                }
            }
            (scene, edits)
        }
    }

    struct Gpu {
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
    }

    impl Gpu {
        fn new() -> Result<Self, Error> {
            let instance = wgpu::Instance::new(
                wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
            );
            let adapter = block_on(instance.request_adapter(&Default::default()))?;
            let mut required_features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
            if std::env::var_os("ILETKI_ASSEMBLY_SAMPLED_SECTION").is_none() {
                required_features |= adapter.features() & wgpu::Features::CLIP_DISTANCES;
            }
            let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                required_features,
                ..Default::default()
            }))?;
            eprintln!(
                "{:?}; features: {:?}",
                adapter.get_info(),
                required_features
            );
            Ok(Self {
                adapter,
                device,
                queue,
            })
        }

        fn wait(&self, submission_index: wgpu::SubmissionIndex) -> Result<(), wgpu::PollError> {
            self.device.poll(wgpu::PollType::Wait {
                submission_index: Some(submission_index),
                timeout: None,
            })?;
            Ok(())
        }
    }

    fn hash_bytes(bytes: impl IntoIterator<Item = u8>) -> u64 {
        bytes.into_iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
        })
    }

    fn hit_hash(hits: &[PickHit]) -> u64 {
        hash_bytes(hits.iter().flat_map(|hit| {
            let position = hit.world_pos.map(|pos| [pos.x, pos.y, pos.z]);
            hit.id
                .to_u64()
                .to_le_bytes()
                .into_iter()
                .chain(hit.depth.to_le_bytes())
                .chain([u8::from(position.is_some())])
                .chain(position.into_iter().flatten().flat_map(f64::to_le_bytes))
        }))
    }

    fn record_capture(
        file: &mut impl Write,
        label: &str,
        frame: u32,
        capture: profiling::Capture,
    ) -> std::io::Result<()> {
        for (name, time) in capture.cpu {
            writeln!(file, "{label},{frame},cpu,{name},{}", time.as_nanos())?;
        }
        for (name, time) in capture.gpu {
            writeln!(file, "{label},{frame},gpu,{name},{}", time.as_nanos())?;
        }
        for (name, count) in capture.counters {
            writeln!(file, "{label},{frame},count,{name},{count}")?;
        }
        Ok(())
    }

    struct Benchmark<'a> {
        gpu: &'a Gpu,
        samples: BufWriter<File>,
        profile: BufWriter<File>,
        memory: Option<BufWriter<File>>,
        images: Option<PathBuf>,
        run: u32,
        variant: String,
        size: [u32; 2],
    }

    impl Benchmark<'_> {
        fn run_case(
            &mut self,
            case: Case,
            mut scene: Scene3DLayer,
            edits: &[(Chunk3DId, Vec3)],
        ) -> Result<(), Error> {
            if std::env::var_os("ILETKI_ASSEMBLY_NO_SECTION_CULL").is_some() {
                scene.set_profile_section_culling(false);
            }
            if std::env::var_os("ILETKI_ASSEMBLY_NO_SECTION_GROUPS").is_some() {
                scene.set_profile_section_groups(false);
            }
            if std::env::var_os("ILETKI_ASSEMBLY_NO_SECTION_BRIDGING").is_some() {
                scene.set_profile_section_bridging(false);
            }
            if std::env::var_os("ILETKI_ASSEMBLY_NO_REPEATED_PAGES").is_some() {
                scene.set_profile_repeated_pages(false);
            }
            self.record_memory(case, "before-targets")?;
            let host = |initial_px| {
                ViewportHost::new_with_adapter(
                    &self.gpu.device,
                    &self.gpu.adapter,
                    ViewportConfig {
                        initial_px,
                        ..Default::default()
                    },
                )
            };
            let [width, height] = self.size;
            let split = case.view == "split";
            if split && width < 2 {
                return Err("split benchmark requires at least two pixels of width".into());
            }
            let left = if split { width / 2 } else { width };
            let right = if split { width - left } else { width };
            let mut main = host([right, height])?;
            let mut second = matches!(case.view, "dual" | "split" | "wide-dual")
                .then(|| host([left, height]))
                .transpose()?;
            *main.camera_mut() = case.camera(0, false);
            if let Some(second) = second.as_mut() {
                *second.camera_mut() = case.camera(0, true);
            }
            self.orbit(case, &mut scene, &mut main, &mut second, edits)?;
            self.record_memory(case, "after-orbit")?;
            if (case.primary() || case.model.is_some() || case.chunks == 100_000)
                && self.variant != "visibility-only"
                && std::env::var_os("ILETKI_ASSEMBLY_ORBIT_ONLY").is_none()
            {
                self.picking(case, &mut scene, &mut main)?;
            }
            self.record_memory(case, "after-pick")?;
            drop(second);
            drop(main);
            drop(scene);
            if self.memory.is_some() {
                self.gpu.wait(self.gpu.queue.submit([]))?;
                self.record_memory(case, "closed")?;
            }
            self.samples.flush()?;
            self.profile.flush()?;
            if let Some(memory) = &mut self.memory {
                memory.flush()?;
            }
            Ok(())
        }

        fn record_memory(&mut self, case: Case, stage: &str) -> std::io::Result<()> {
            let Some(output) = &mut self.memory else {
                return Ok(());
            };
            let (allocated, reserved) = self.gpu.device.generate_allocator_report().map_or_else(
                || (String::new(), String::new()),
                |report| {
                    (
                        report.total_allocated_bytes.to_string(),
                        report.total_reserved_bytes.to_string(),
                    )
                },
            );
            writeln!(
                output,
                "{},{},{stage},{allocated},{reserved}",
                case.chunks, case.view
            )
        }

        fn orbit(
            &mut self,
            case: Case,
            scene: &mut Scene3DLayer,
            main: &mut ViewportHost,
            second: &mut Option<ViewportHost>,
            edits: &[(Chunk3DId, Vec3)],
        ) -> Result<(), Error> {
            let gpu = self.gpu;
            let stream_start = Instant::now();
            for frame in 0..=case.chunks {
                let start = Instant::now();
                let mut encoder = gpu.device.create_command_encoder(&Default::default());
                let mut streaming = main
                    .render_to_texture(&gpu.device, &gpu.queue, &mut encoder, &(), &mut [scene])
                    .stats
                    .streaming_pending;
                if let Some(second) = second.as_mut() {
                    streaming |= second
                        .render_to_texture(&gpu.device, &gpu.queue, &mut encoder, &(), &mut [scene])
                        .stats
                        .streaming_pending;
                }
                gpu.wait(gpu.queue.submit([encoder.finish()]))?;
                if frame == 0 {
                    self.record(case, "first-frame", frame, 0.0, elapsed_ms(start), "")?;
                    self.record_memory(case, "first-frame")?;
                }
                if !streaming {
                    self.record(
                        case,
                        "streaming-complete",
                        frame,
                        0.0,
                        elapsed_ms(stream_start),
                        "",
                    )?;
                    self.record_memory(case, "streaming-complete")?;
                    break;
                }
                if frame == case.chunks {
                    return Err("scene is still streaming after preparation".into());
                }
            }
            let long = case.primary() || case.acceptance_wide() || case.model.is_some();
            let samples = if long { 360 } else { 60 };
            let profile = std::env::var_os("ILETKI_PROFILE").is_some() && long;
            let traces = match (profile, case.style) {
                (true, Style::Section) => ORBIT_PERIOD,
                (true, _) => 8,
                (false, _) => 0,
            };
            let signatures =
                self.images.is_some() || std::env::var_os("ILETKI_ASSEMBLY_SIGNATURES").is_some();
            let edit_scene = std::env::var_os("ILETKI_ASSEMBLY_EDITS").is_some();
            let edit_visibility = std::env::var_os("ILETKI_ASSEMBLY_VISIBILITY_EDITS").is_some();
            let overlap = std::env::var_os("ILETKI_ASSEMBLY_OVERLAP").is_some();
            let trace_end = WARMUP + samples + traces;
            let mut model_seen = false;
            for frame in 0..trace_end + if signatures { ORBIT_PERIOD } else { 0 } {
                let start = Instant::now();
                let edited = (edit_scene || edit_visibility) && frame % 6 == 0;
                if edited {
                    let toggled = frame % 12 == 0;
                    let delta = if toggled { 0.2 } else { 0.0 };
                    for &(id, origin) in edits {
                        if edit_scene {
                            scene.set_origin(id, origin + Vec3::X * delta);
                        }
                        if edit_visibility {
                            scene.set_visible(id, !toggled);
                        }
                    }
                }
                *main.camera_mut() = case.camera(frame, false);
                if let Some(second) = second.as_mut() {
                    *second.camera_mut() = case.camera(frame, true);
                }
                let trace = (WARMUP + samples..trace_end).contains(&frame);
                if trace {
                    profiling::begin(&gpu.device);
                }
                let mut encoder = gpu.device.create_command_encoder(&Default::default());
                main.render_to_texture(&gpu.device, &gpu.queue, &mut encoder, &(), &mut [scene]);
                if let Some(second) = second.as_mut() {
                    if overlap {
                        gpu.queue.submit([encoder.finish()]);
                        encoder = gpu.device.create_command_encoder(&Default::default());
                    }
                    second.render_to_texture(
                        &gpu.device,
                        &gpu.queue,
                        &mut encoder,
                        &(),
                        &mut [scene],
                    );
                }
                let commands = encoder.finish();
                let encode = elapsed_ms(start);
                gpu.wait(gpu.queue.submit([commands]))?;
                let wall = elapsed_ms(start);
                if trace {
                    let capture = profiling::finish(&gpu.device, &gpu.queue)?;
                    let label = if case.acceptance_wide() {
                        "orbit-20000-wide"
                    } else {
                        "orbit"
                    };
                    record_capture(&mut self.profile, label, frame, capture)?;
                } else if frame >= WARMUP {
                    let reference_frame = if signatures {
                        frame >= trace_end
                    } else {
                        frame == WARMUP
                    };
                    let signature = if reference_frame {
                        let image = main.readback_rgba_blocking(&gpu.device, &gpu.queue)?;
                        if case.model.is_some() && !model_seen {
                            model_seen = image.pixels.get(..4).is_some_and(|background| {
                                image
                                    .pixels
                                    .as_chunks::<4>()
                                    .0
                                    .iter()
                                    .any(|pixel| pixel != background)
                            });
                        }
                        let phase = frame.saturating_sub(trace_end);
                        self.record_image(case, phase, "main", &image)?;
                        let hash = hash_bytes(image.pixels);
                        if let Some(second) = second.as_ref() {
                            let image = second.readback_rgba_blocking(&gpu.device, &gpu.queue)?;
                            self.record_image(case, phase, "second", &image)?;
                            format!("{hash:016x}/{:016x}", hash_bytes(image.pixels))
                        } else {
                            format!("{hash:016x}")
                        }
                    } else {
                        String::new()
                    };
                    if frame >= trace_end {
                        self.record(
                            case,
                            "image-signature",
                            frame - WARMUP,
                            0.0,
                            0.0,
                            &signature,
                        )?;
                    } else {
                        self.record(case, "orbit", frame - WARMUP, encode, wall, &signature)?;
                        if edited {
                            self.record(case, "edit-frame", frame - WARMUP, encode, wall, "")?;
                        }
                    }
                }
            }
            if case.model.is_some() && !model_seen {
                return Err("the model produced only uniform reference images".into());
            }
            Ok(())
        }

        fn record_image(
            &self,
            case: Case,
            phase: u32,
            view: &str,
            image: &RgbaImage,
        ) -> Result<(), Error> {
            if let Some(directory) = &self.images
                && phase.is_multiple_of(10)
            {
                let filename = format!(
                    "{}-{}-{phase:02}-{view}-{}x{}.rgba",
                    case.chunks, case.view, image.width, image.height,
                );
                fs::write(directory.join(filename), &image.pixels)?;
            }
            Ok(())
        }

        fn picking(
            &mut self,
            case: Case,
            scene: &mut Scene3DLayer,
            main: &mut ViewportHost,
        ) -> Result<(), Error> {
            let gpu = self.gpu;
            *main.camera_mut() = case.camera(35, false);
            let samples = 360;
            let profile = std::env::var_os("ILETKI_PROFILE").is_some();
            let [x, y] = main.size().active_px.map(|dimension| dimension / 2);
            let click = [x.saturating_sub(2), y.saturating_sub(2), 5, 5];
            let marquee = [x.saturating_sub(128), y.saturating_sub(128), 256, 256];
            for (label, rect) in [("click", click), ("marquee", marquee), ("hover", click)] {
                for frame in 0..WARMUP + samples + if profile { 8 } else { 0 } {
                    let trace = frame >= WARMUP + samples;
                    if trace {
                        profiling::begin(&gpu.device);
                    }
                    let start = Instant::now();
                    let (submit, hits) = if label == "hover" {
                        // One in flight, with deterministic cursor displacement. Polling here
                        // measures readiness under load without tying it to a GUI refresh tick.
                        let [x, y, w, h] = rect;
                        let mut pending = main.begin_pick_rect(
                            &gpu.device,
                            &gpu.queue,
                            &(),
                            &mut [scene],
                            [x.saturating_add(frame % 11), y, w, h],
                        )?;
                        let submit = elapsed_ms(start);
                        let hits = loop {
                            match pending.try_read() {
                                Poll::Ready(hits) => break hits?,
                                Poll::Pending => std::thread::yield_now(),
                            }
                        };
                        (submit, hits)
                    } else {
                        (
                            0.0,
                            main.pick_rect_blocking(
                                &gpu.device,
                                &gpu.queue,
                                &(),
                                &mut [scene],
                                rect,
                            )?,
                        )
                    };
                    let wall = elapsed_ms(start);
                    if trace {
                        let capture = profiling::finish(&gpu.device, &gpu.queue)?;
                        record_capture(&mut self.profile, label, frame, capture)?;
                    } else if frame >= WARMUP {
                        self.record(
                            case,
                            label,
                            frame - WARMUP,
                            submit,
                            wall,
                            &format!("{:016x}", hit_hash(&hits)),
                        )?;
                    }
                }
            }
            Ok(())
        }

        fn record(
            &mut self,
            case: Case,
            workload: &str,
            frame: u32,
            cpu: f64,
            wall: f64,
            signature: &str,
        ) -> std::io::Result<()> {
            writeln!(
                self.samples,
                "{},{},{},{},{},{},{cpu:.6},{wall:.6},{signature}",
                self.variant, self.run, case.chunks, case.view, workload, frame
            )
        }
    }

    fn elapsed_ms(start: Instant) -> f64 {
        start.elapsed().as_secs_f64() * 1000.0
    }

    fn viewport_size() -> Result<[u32; 2], Error> {
        let Ok(value) = std::env::var("ILETKI_ASSEMBLY_SIZE") else {
            return Ok(SIZE);
        };
        let (width, height) = value.split_once('x').ok_or("expected WIDTHxHEIGHT")?;
        let size = [width.parse()?, height.parse()?];
        if size.contains(&0) {
            return Err("benchmark dimensions must be positive".into());
        }
        Ok(size)
    }

    pub(super) fn run() -> Result<(), Error> {
        let mut args = std::env::args_os().skip(1);
        let output = args.next().ok_or("missing output directory")?;
        let output = Path::new(&output);
        let variant = args
            .next()
            .and_then(|arg| arg.into_string().ok())
            .ok_or("missing variant label")?;
        let run = args
            .next()
            .and_then(|arg| arg.into_string().ok())
            .ok_or("missing run number")?
            .parse()?;
        fs::create_dir_all(output)?;
        let gpu = Gpu::new()?;
        let memory = if std::env::var_os("ILETKI_ASSEMBLY_MEMORY").is_some() {
            let mut output = BufWriter::new(File::create(output.join("memory.csv"))?);
            writeln!(
                output,
                "chunks,view,stage,gpu_allocated_bytes,gpu_reserved_bytes"
            )?;
            Some(output)
        } else {
            None
        };
        let images = if std::env::var_os("ILETKI_ASSEMBLY_IMAGES").is_some() {
            let directory = output.join("images");
            fs::create_dir_all(&directory)?;
            Some(directory)
        } else {
            None
        };
        let mut bench = Benchmark {
            gpu: &gpu,
            samples: BufWriter::new(File::create(output.join("frames.csv"))?),
            profile: BufWriter::new(File::create(output.join("profile.csv"))?),
            memory,
            images,
            run,
            variant,
            size: viewport_size()?,
        };
        writeln!(
            bench.samples,
            "variant,run,chunks,view,workload,frame,cpu_ms,wall_ms,signature"
        )?;
        writeln!(bench.profile, "workload,frame,kind,label,value")?;
        let style = Style::from_env()?;
        let pack = !matches!(
            bench.variant.as_str(),
            "geometry-direct" | "visibility-only"
        );
        if let Some(path) = std::env::var_os("ILETKI_ASSEMBLY_MESH") {
            let started = Instant::now();
            let fixture = crate::assembly_fixture::load(Path::new(&path))?;
            let decoded_ms = elapsed_ms(started);
            let chunks = u32::try_from(fixture.meshes.len())?;
            let focus = fixture.bounds.center().ok_or("model has no bounds")?;
            let orbit_scale = if std::env::var_os("ILETKI_ASSEMBLY_MODEL_CLOSE").is_some() {
                0.85
            } else {
                2.6
            };
            let radius = fixture.bounds.radius() * orbit_scale;
            if !focus.is_finite() || !radius.is_finite() || radius <= 0.0 {
                return Err("invalid model camera bounds".into());
            }
            let case = Case {
                chunks,
                view: if std::env::var_os("ILETKI_ASSEMBLY_SPLIT").is_some() {
                    "split"
                } else if std::env::var_os("ILETKI_ASSEMBLY_DUAL").is_some() {
                    "dual"
                } else {
                    "model"
                },
                model: Some((focus, radius)),
                style,
            };
            let insert_started = Instant::now();
            let mut scene = Scene3DLayer::new(1);
            scene.set_profile_batch_geometry(pack);
            let mut edits = Vec::new();
            for (index, mesh) in fixture.meshes.into_iter().enumerate() {
                let origin = mesh.origin;
                let id = scene.insert(mesh);
                if index % 100 == 0 {
                    edits.push((id, origin));
                }
            }
            let insert_ms = elapsed_ms(insert_started);
            eprintln!(
                "model: {chunks} chunks, {} triangles, {:?} pixels",
                fixture.triangles, bench.size
            );
            bench.record(case, "scene-load", 0, 0.0, decoded_ms + insert_ms, "")?;
            bench.record(case, "fixture-decode", 0, 0.0, decoded_ms, "")?;
            bench.record(case, "scene-insert", 0, 0.0, insert_ms, "")?;
            return bench.run_case(case, scene, &edits);
        }
        let primary_only = std::env::var_os("ILETKI_PRIMARY_ONLY").is_some();
        let filter = std::env::var("ILETKI_ASSEMBLY_CASE").ok();
        let mut ran_case = false;
        for chunks in [1000, 4000, 20000, 100_000] {
            for view in ["near", "wide", "dual", "split", "wide-dual"] {
                let case = Case {
                    chunks,
                    view,
                    model: None,
                    style,
                };
                if (chunks == 100_000 && !case.acceptance_wide())
                    || (filter.is_none()
                        && (chunks == 100_000 || matches!(view, "split" | "wide-dual")))
                    || (primary_only && !case.primary())
                    || filter
                        .as_ref()
                        .is_some_and(|filter| filter != &format!("{chunks}/{view}"))
                {
                    continue;
                }
                ran_case = true;
                let started = Instant::now();
                let (scene, edits) = case.make_scene(bench.variant != "visibility-only", pack);
                bench.record(case, "scene-build", 0, 0.0, elapsed_ms(started), "")?;
                bench.run_case(case, scene, &edits)?;
            }
        }
        if !ran_case {
            return Err("no assembly workload matches the selected case".into());
        }
        Ok(())
    }
}
