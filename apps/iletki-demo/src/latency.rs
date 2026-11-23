//! Opt-in windowed input replay and timing capture for the demo.

use iletki::gpu::wgpu;
use iletki::{Mesh3D, PickId, Vec3};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

pub(crate) const PERIOD: Duration = Duration::from_nanos(16_666_667);
const WARMUP: u32 = 60;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Action {
    Orbit,
    Pan,
    Zoom,
    Click,
    Marquee,
    Edit,
    Live,
}

impl Action {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Orbit => "orbit",
            Self::Pan => "pan",
            Self::Zoom => "zoom",
            Self::Click => "click",
            Self::Marquee => "marquee",
            Self::Edit => "edit",
            Self::Live => "live",
        }
    }
}

pub(crate) enum Scene {
    Demo,
    Wide,
    Model(PathBuf),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Style {
    Faces,
    Edges,
    Section,
}

pub(crate) struct Options {
    pub(crate) directory: PathBuf,
    pub(crate) scene: Scene,
    pub(crate) action: Action,
    pub(crate) style: Style,
    pub(crate) dual: bool,
    pub(crate) notify: bool,
    pub(crate) present: Option<wgpu::PresentMode>,
    pub(crate) depth: Option<u32>,
    pub(crate) samples: u32,
}

impl Options {
    pub(crate) fn parse() -> Result<Option<Self>, String> {
        let mut args = std::env::args_os().skip(1);
        let Some(command) = args.next() else {
            return Ok(None);
        };
        if command != "--latency-bench" {
            return Err("expected --latency-bench <directory>".into());
        }
        let mut config = Self {
            directory: args.next().ok_or("missing capture directory")?.into(),
            scene: Scene::Demo,
            action: Action::Orbit,
            style: Style::Faces,
            dual: false,
            notify: false,
            present: None,
            depth: None,
            samples: 360,
        };
        while let Some(flag) = args.next() {
            match flag.to_str() {
                Some("--dual") => config.dual = true,
                Some("--notify-present") => config.notify = true,
                Some("--scene") => {
                    let value = args.next().ok_or("missing scene")?;
                    config.scene = match value.to_str() {
                        Some("demo") => Scene::Demo,
                        Some("wide") => Scene::Wide,
                        _ => Scene::Model(value.into()),
                    };
                }
                key => {
                    let value = args
                        .next()
                        .and_then(|arg| arg.into_string().ok())
                        .ok_or("missing or invalid option value")?;
                    match (key, value.as_str()) {
                        (Some("--action"), "orbit") => config.action = Action::Orbit,
                        (Some("--action"), "pan") => config.action = Action::Pan,
                        (Some("--action"), "zoom") => config.action = Action::Zoom,
                        (Some("--action"), "click") => config.action = Action::Click,
                        (Some("--action"), "marquee") => config.action = Action::Marquee,
                        (Some("--action"), "edit") => config.action = Action::Edit,
                        (Some("--action"), "live") => config.action = Action::Live,
                        (Some("--style"), "faces") => config.style = Style::Faces,
                        (Some("--style"), "edges") => config.style = Style::Edges,
                        (Some("--style"), "edges-section") => config.style = Style::Section,
                        (Some("--present"), "fifo") => {
                            config.present = Some(wgpu::PresentMode::Fifo)
                        }
                        (Some("--present"), "mailbox") => {
                            config.present = Some(wgpu::PresentMode::Mailbox)
                        }
                        (Some("--depth"), "1") => config.depth = Some(1),
                        (Some("--depth"), "2") => config.depth = Some(2),
                        (Some("--samples"), _) => {
                            config.samples = value.parse().map_err(|_| "invalid sample count")?;
                            if !(1..=10_000).contains(&config.samples) {
                                return Err("sample count must be between 1 and 10000".into());
                            }
                        }
                        _ => return Err(format!("invalid latency option {flag:?} {value}")),
                    }
                }
            }
        }
        if config.dual && matches!(config.action, Action::Click | Action::Marquee) {
            return Err("select a single viewport for the picking replay".into());
        }
        if config.action == Action::Edit && matches!(config.scene, Scene::Demo) {
            return Err("edit replay requires the wide or model scene".into());
        }
        Ok(Some(config))
    }

    pub(crate) fn dimensions(&self) -> [u32; 2] {
        match self.scene {
            Scene::Wide => [1024, 768],
            Scene::Model(_) => [1920, 1080],
            Scene::Demo => [1280, 800],
        }
    }
}

pub(crate) fn wide_mesh(index: u32, style: Style) -> Mesh3D {
    let mut mesh = Mesh3D::at_origin(Vec3::new(
        f64::from(index % 64) * 2.0,
        f64::from(index / 64) * 2.0,
        0.0,
    ));
    let min = Vec3::new(-0.6, -0.6, 0.0);
    let max = Vec3::new(0.6, 0.6, 1.6);
    let pick = PickId::new(1, index + 1, 0);
    let color = [120, 170, 200, 255];
    if style == Style::Faces {
        mesh.push_box(min, max, color, pick, false);
    } else {
        mesh.push_box_with_edges(min, max, color, [24, 32, 40, 160], pick, false);
    }
    mesh
}

pub(crate) struct Event {
    kind: &'static str,
    recorded: u128,
    scheduled: u128,
    redraw: Option<u128>,
    submitted: Option<u128>,
    ready: Option<u128>,
    frame: Option<u32>,
    hits: Option<usize>,
    hit: Option<u64>,
    status: &'static str,
    warmup: bool,
}

pub(crate) struct Frame {
    pub(crate) id: u32,
    pub(crate) started: u128,
    pub(crate) acquired: u128,
    pub(crate) encoded: u128,
    pub(crate) submitted: u128,
    pub(crate) presented: u128,
    pub(crate) status: &'static str,
    pub(crate) streaming: bool,
}

pub(crate) struct Trace {
    pub(crate) options: Options,
    pub(crate) next: Instant,
    pub(crate) ticks: u32,
    pub(crate) marker: wgpu::RenderPipeline,
    epoch: Instant,
    end: Option<Instant>,
    events: Vec<Event>,
    pending_frames: Vec<usize>,
    pending_selection_frame: Option<usize>,
    frames: Vec<Frame>,
    completed: Arc<Mutex<Vec<(u32, u128)>>>,
}

const MARKER_WGSL: &str = r#"
struct Vertex {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) bit: u32,
};

@vertex
fn vertex(@builtin(vertex_index) index: u32, @builtin(instance_index) frame: u32) -> Vertex {
    let x = f32((index << 1u) & 2u);
    let y = f32(index & 2u);
    return Vertex(vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0), frame ^ (frame >> 1u));
}

@fragment
fn fragment(input: Vertex) -> @location(0) vec4<f32> {
    let slot = min(u32(input.position.x) / 3u, 15u);
    let value = f32((input.bit >> slot) & 1u);
    return vec4<f32>(value, value, value, 1.0);
}
"#;

impl Trace {
    pub(crate) fn new(
        options: Options,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("latency frame marker"),
            source: wgpu::ShaderSource::Wgsl(MARKER_WGSL.into()),
        });
        let marker = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("latency frame marker"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment"),
                compilation_options: Default::default(),
                targets: &[Some(format.into())],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let epoch = Instant::now();
        Self {
            options,
            next: epoch,
            ticks: 0,
            marker,
            epoch,
            end: None,
            events: Vec::new(),
            pending_frames: Vec::new(),
            pending_selection_frame: None,
            frames: Vec::new(),
            completed: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub(crate) fn ns(&self, time: Instant) -> u128 {
        time.saturating_duration_since(self.epoch).as_nanos()
    }

    pub(crate) fn tick(&mut self, now: Instant) -> Option<(u32, usize)> {
        if self.options.action == Action::Live || self.end.is_some() || now < self.next {
            return None;
        }
        let scheduled = self.next;
        self.next = (self.next + PERIOD).max(now + Duration::from_millis(1));
        let tick = self.ticks;
        let id = self.event(self.options.action.name(), now, scheduled);
        Some((tick, id))
    }

    pub(crate) fn live(&mut self, kind: &'static str, now: Instant) -> Option<usize> {
        (self.options.action == Action::Live && self.end.is_none())
            .then(|| self.event(kind, now, now))
    }

    fn event(&mut self, kind: &'static str, now: Instant, scheduled: Instant) -> usize {
        let event = self.events.len();
        self.events.push(Event {
            kind,
            recorded: self.ns(now),
            scheduled: self.ns(scheduled),
            redraw: None,
            submitted: None,
            ready: None,
            frame: None,
            hits: None,
            hit: None,
            status: "pending",
            warmup: self.ticks < WARMUP,
        });
        self.ticks += 1;
        if self.ticks >= WARMUP + self.options.samples {
            self.end = Some(now + Duration::from_secs(1));
        }
        event
    }

    pub(crate) fn redraw(&mut self, id: usize) {
        let time = self.ns(Instant::now());
        if let Some(event) = self.events.get_mut(id) {
            event.redraw = Some(time);
            self.pending_frames.push(id);
        }
    }

    pub(crate) fn selection_status(
        &mut self,
        id: usize,
        status: &'static str,
        hits: Option<usize>,
    ) {
        let time = self.ns(Instant::now());
        if let Some(event) = self.events.get_mut(id) {
            event.status = status;
            match status {
                "submitted" => event.submitted = Some(time),
                "ready" => {
                    event.ready = Some(time);
                    event.hits = hits;
                }
                _ => {}
            }
        }
    }

    pub(crate) fn selection_redraw(&mut self, id: usize) {
        if let Some(previous) = self.pending_selection_frame.replace(id) {
            self.selection_status(previous, "coalesced-before-frame", None);
        }
        let time = self.ns(Instant::now());
        if let Some(event) = self.events.get_mut(id) {
            event.redraw = Some(time);
        }
    }

    pub(crate) fn frame_id(&self) -> u32 {
        self.frames.len().min(u32::MAX as usize - 1) as u32 + 1
    }

    pub(crate) fn hit(&mut self, id: usize, hit: Option<u64>) {
        if let Some(event) = self.events.get_mut(id) {
            event.hit = hit;
        }
    }

    pub(crate) fn track_gpu(&self, queue: &wgpu::Queue, id: u32) {
        let completed = self.completed.clone();
        let epoch = self.epoch;
        queue.on_submitted_work_done(move || {
            if let Ok(mut done) = completed.lock() {
                done.push((id, epoch.elapsed().as_nanos()));
            }
        });
    }

    pub(crate) fn frame(&mut self, frame: Frame) {
        if frame.status == "present-called" {
            for id in self
                .pending_frames
                .drain(..)
                .chain(self.pending_selection_frame.take())
            {
                if let Some(event) = self.events.get_mut(id) {
                    event.frame = Some(frame.id);
                    event.status = "applied";
                }
            }
        }
        self.frames.push(frame);
    }

    pub(crate) fn pending_gpu(&self) -> bool {
        self.completed.lock().is_ok_and(|done| {
            self.frames
                .iter()
                .rev()
                .find(|frame| frame.status == "present-called")
                .is_some_and(|frame| done.last().is_none_or(|(id, _)| *id < frame.id))
        })
    }

    pub(crate) fn deadline(&self, now: Instant) -> Option<Instant> {
        let next = self
            .end
            .or_else(|| (self.options.action != Action::Live).then_some(self.next));
        if self.pending_gpu() {
            let poll = now + Duration::from_millis(1);
            Some(next.map_or(poll, |next| next.min(poll)))
        } else {
            next
        }
    }

    pub(crate) fn finished(&self, now: Instant, has_pick: bool) -> bool {
        self.end.is_some_and(|end| {
            now >= end
                || (!has_pick
                    && self.pending_frames.is_empty()
                    && self.pending_selection_frame.is_none()
                    && !self.pending_gpu())
        })
    }

    pub(crate) fn write(
        &self,
        window: &winit::window::Window,
        camera: &iletki::Camera,
    ) -> std::io::Result<()> {
        fs::create_dir_all(&self.options.directory)?;
        let monitor = window.current_monitor();
        let mut info = File::create(self.options.directory.join("monitor.txt"))?;
        writeln!(info, "window={:?}", window.inner_size())?;
        writeln!(
            info,
            "monitor={:?}",
            monitor
                .as_ref()
                .and_then(winit::monitor::MonitorHandle::name)
        )?;
        writeln!(
            info,
            "refresh_millihertz={:?}",
            monitor.and_then(|monitor| monitor.refresh_rate_millihertz())
        )?;
        writeln!(info, "final_camera={camera:?}")?;
        let completed: std::collections::HashMap<_, _> = self
            .completed
            .lock()
            .map(|done| done.iter().copied().collect())
            .unwrap_or_default();
        let mut frames = BufWriter::new(File::create(self.options.directory.join("frames.csv"))?);
        writeln!(
            frames,
            "frame,start_ns,acquire_ns,encode_ns,submit_ns,gpu_observed_ns,present_return_ns,status,streaming"
        )?;
        for frame in &self.frames {
            let gpu = completed
                .get(&frame.id)
                .map(ToString::to_string)
                .unwrap_or_default();
            writeln!(
                frames,
                "{},{},{},{},{},{},{},{},{}",
                frame.id,
                frame.started,
                frame.acquired,
                frame.encoded,
                frame.submitted,
                gpu,
                frame.presented,
                frame.status,
                frame.streaming
            )?;
        }
        let mut events = BufWriter::new(File::create(self.options.directory.join("events.csv"))?);
        writeln!(
            events,
            "event,kind,warmup,scheduled_ns,input_ns,redraw_ns,pick_submit_ns,pick_ready_ns,frame,hits,first_hit,status"
        )?;
        for (id, event) in self.events.iter().enumerate() {
            fn field<T: ToString>(value: Option<T>) -> String {
                value.map(|value| value.to_string()).unwrap_or_default()
            }
            writeln!(
                events,
                "{id},{},{},{},{},{},{},{},{},{},{},{}",
                event.kind,
                event.warmup,
                event.scheduled,
                event.recorded,
                field(event.redraw),
                field(event.submitted),
                field(event.ready),
                field(event.frame),
                field(event.hits),
                field(event.hit),
                event.status
            )?;
        }
        events.flush()?;
        frames.flush()
    }
}
