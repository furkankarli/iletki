//! Headless benchmark for rendering and picking thousands of independently stored 3D chunks.
//!
//! Run with `cargo run --release --locked -p iletki --features profiling --example assembly_bench
//! -- <output-dir> <variant> <run>`. Set `ILETKI_PRIMARY_ONLY=1` to run the acceptance fixture.
//! Set `ILETKI_ASSEMBLY_CASE=20000/wide` to isolate a particular size and view.

#![allow(clippy::print_stderr)]

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    native::run()
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use iletki::gpu::{profiling, wgpu};
    use iletki::{
        Camera, Mesh3D, PickHit, PickId, Scene3DLayer, Vec3, ViewportConfig, ViewportHost,
    };
    use std::fs::{self, File};
    use std::future::Future;
    use std::io::{BufWriter, Write};
    use std::path::Path;
    use std::pin::pin;
    use std::task::{Context, Poll, Waker};
    use std::time::Instant;

    type Error = Box<dyn std::error::Error>;
    const SIZE: [u32; 2] = [1024, 768];
    const CLICK: [u32; 4] = [510, 382, 5, 5];
    const MARQUEE: [u32; 4] = [384, 256, 256, 256];

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

    #[derive(Clone, Copy)]
    struct Case {
        chunks: u32,
        view: &'static str,
    }

    impl Case {
        fn primary(self) -> bool {
            self.chunks == 4_000 && self.view == "near"
        }

        fn camera(self, frame: u32, secondary: bool) -> Camera {
            let angle = f64::from(frame % 30) * std::f64::consts::TAU / 30.0;
            let focus = Vec3::new(30.0, 4.0, 0.8);
            if self.view == "wide" {
                let half_y = f64::from(self.chunks.div_ceil(64));
                let focus = Vec3::new(63.0, half_y, 0.8);
                let radius = half_y.max(64.0) * 2.0;
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

        fn make_scene(self, batch: bool) -> Scene3DLayer {
            let mut scene = Scene3DLayer::new(1);
            scene.set_profile_batch_uniforms(batch);
            for i in 0..self.chunks {
                let x = f64::from(i % 64) * 2.0;
                let y = f64::from(i / 64) * 2.0;
                let mut mesh = Mesh3D::at_origin(Vec3::new(x, y, 0.0));
                mesh.push_box(
                    Vec3::new(-0.6, -0.6, 0.0),
                    Vec3::new(0.6, 0.6, 1.6),
                    [120, 170, 200, 255],
                    PickId::new(1, i + 1, 0),
                    false,
                );
                scene.insert(mesh);
            }
            scene
        }
    }

    struct Gpu {
        device: wgpu::Device,
        queue: wgpu::Queue,
    }

    impl Gpu {
        fn new() -> Result<Self, Error> {
            let instance = wgpu::Instance::new(
                wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
            );
            let adapter = block_on(instance.request_adapter(&Default::default()))?;
            let required_features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
            let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                required_features,
                ..Default::default()
            }))?;
            eprintln!(
                "{:?}; timestamp queries: {}",
                adapter.get_info(),
                !required_features.is_empty()
            );
            Ok(Self { device, queue })
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
            hit.id
                .to_u64()
                .to_le_bytes()
                .into_iter()
                .chain(hit.depth.to_le_bytes())
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
        run: u32,
        variant: String,
    }

    impl Benchmark<'_> {
        fn orbit(
            &mut self,
            case: Case,
            scene: &mut Scene3DLayer,
            main: &mut ViewportHost,
            second: &mut Option<ViewportHost>,
        ) -> Result<(), Error> {
            let gpu = self.gpu;
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
                }
                if !streaming {
                    break;
                }
                if frame == case.chunks {
                    return Err("scene is still streaming after preparation".into());
                }
            }
            let samples = if case.primary() { 360 } else { 60 };
            let profile = std::env::var_os("ILETKI_PROFILE").is_some() && case.primary();
            for frame in 0..12 + samples + if profile { 8 } else { 0 } {
                *main.camera_mut() = case.camera(frame, false);
                if let Some(second) = second.as_mut() {
                    *second.camera_mut() = case.camera(frame, true);
                }
                let trace = frame >= 12 + samples;
                if trace {
                    profiling::begin(&gpu.device);
                }
                let start = Instant::now();
                let mut encoder = gpu.device.create_command_encoder(&Default::default());
                main.render_to_texture(&gpu.device, &gpu.queue, &mut encoder, &(), &mut [scene]);
                if let Some(second) = second.as_mut() {
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
                    record_capture(&mut self.profile, "orbit", frame, capture)?;
                } else if frame >= 12 {
                    let signature = if frame == 12 {
                        let image = main.readback_rgba_blocking(&gpu.device, &gpu.queue)?;
                        format!("{:016x}", hash_bytes(image.pixels))
                    } else {
                        String::new()
                    };
                    self.record(case, "orbit", frame - 12, encode, wall, &signature)?;
                }
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
            let samples = if case.primary() { 360 } else { 30 };
            let profile = std::env::var_os("ILETKI_PROFILE").is_some() && case.primary();
            for (label, rect) in [("click", CLICK), ("marquee", MARQUEE), ("hover", CLICK)] {
                for frame in 0..12 + samples + if profile { 8 } else { 0 } {
                    let trace = frame >= 12 + samples;
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
                    } else if frame >= 12 {
                        self.record(
                            case,
                            label,
                            frame - 12,
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
        let mut bench = Benchmark {
            gpu: &gpu,
            samples: BufWriter::new(File::create(output.join("frames.csv"))?),
            profile: BufWriter::new(File::create(output.join("profile.csv"))?),
            run,
            variant,
        };
        writeln!(
            bench.samples,
            "variant,run,chunks,view,workload,frame,cpu_ms,wall_ms,signature"
        )?;
        writeln!(bench.profile, "workload,frame,kind,label,value")?;
        let primary_only = std::env::var_os("ILETKI_PRIMARY_ONLY").is_some();
        let filter = std::env::var("ILETKI_ASSEMBLY_CASE").ok();
        let mut ran_case = false;
        for chunks in [1000, 4000, 20000] {
            for view in ["near", "wide", "dual"] {
                let case = Case { chunks, view };
                if (primary_only && !case.primary())
                    || filter
                        .as_ref()
                        .is_some_and(|filter| filter != &format!("{chunks}/{view}"))
                {
                    continue;
                }
                ran_case = true;
                let mut scene = case.make_scene(bench.variant != "visibility-only");
                let host = || {
                    ViewportHost::new(
                        &gpu.device,
                        ViewportConfig {
                            initial_px: SIZE,
                            ..Default::default()
                        },
                    )
                };
                let mut main = host()?;
                let mut second = (view == "dual").then(host).transpose()?;
                *main.camera_mut() = case.camera(0, false);
                if let Some(second) = second.as_mut() {
                    *second.camera_mut() = case.camera(0, true);
                }
                bench.orbit(case, &mut scene, &mut main, &mut second)?;
                if case.primary() && bench.variant != "visibility-only" {
                    bench.picking(case, &mut scene, &mut main)?;
                }
                bench.samples.flush()?;
                bench.profile.flush()?;
            }
        }
        if !ran_case {
            return Err("no assembly workload matches the selected case".into());
        }
        Ok(())
    }
}
