//! Opt-in windowed input replay and timing capture for the demo.

use iletki::gpu::wgpu;
use iletki::{Mesh3D, PickHit, PickId, Vec3};
use std::{
    fs::{self, File},
    hash::{DefaultHasher, Hash, Hasher},
    io::{BufWriter, Write},
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

const DEFAULT_INPUT_HZ: u32 = 60;

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
    Shuffled,
    Model(PathBuf),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Style {
    Faces,
    Edges,
    Section,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Region {
    Mixed,
    Filled,
    Boundary,
    Empty,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Motion {
    Static,
    Orbit,
    Section,
    Edit,
    Combined,
}

pub(crate) struct Options {
    pub(crate) directory: PathBuf,
    pub(crate) scene: Scene,
    pub(crate) action: Action,
    pub(crate) style: Style,
    pub(crate) dual: bool,
    pub(crate) full_hd: bool,
    pub(crate) monitor: Option<String>,
    pub(crate) notify: bool,
    pub(crate) present: Option<wgpu::PresentMode>,
    pub(crate) depth: Option<u32>,
    pub(crate) samples: u32,
    pub(crate) input_hz: u32,
    pub(crate) region: Region,
    pub(crate) motion: Motion,
    pub(crate) rect_size: u32,
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
            full_hd: false,
            monitor: None,
            notify: false,
            present: None,
            depth: None,
            samples: 360,
            input_hz: DEFAULT_INPUT_HZ,
            region: Region::Mixed,
            motion: Motion::Static,
            rect_size: 256,
        };
        while let Some(flag) = args.next() {
            match flag.to_str() {
                Some("--dual") => config.dual = true,
                Some("--1080p") => config.full_hd = true,
                Some("--notify-present") => config.notify = true,
                Some("--scene") => {
                    let value = args.next().ok_or("missing scene")?;
                    config.scene = match value.to_str() {
                        Some("demo") => Scene::Demo,
                        Some("wide") => Scene::Wide,
                        Some("shuffled-100k") => Scene::Shuffled,
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
                        (Some("--monitor"), _) => config.monitor = Some(value),
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
                        (Some("--region"), "mixed") => config.region = Region::Mixed,
                        (Some("--region"), "filled") => config.region = Region::Filled,
                        (Some("--region"), "boundary") => config.region = Region::Boundary,
                        (Some("--region"), "empty") => config.region = Region::Empty,
                        (Some("--motion"), "static") => config.motion = Motion::Static,
                        (Some("--motion"), "orbit") => config.motion = Motion::Orbit,
                        (Some("--motion"), "section") => config.motion = Motion::Section,
                        (Some("--motion"), "edit") => config.motion = Motion::Edit,
                        (Some("--motion"), "combined") => config.motion = Motion::Combined,
                        (Some("--rect-size"), "256") => config.rect_size = 256,
                        (Some("--rect-size"), "512") => config.rect_size = 512,
                        (Some("--input-hz"), _) => {
                            config.input_hz = value.parse().map_err(|_| "invalid input rate")?;
                            if !(1..=1000).contains(&config.input_hz) {
                                return Err("input rate must be between 1 and 1000 Hz".into());
                            }
                        }
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
        if config.action == Action::Edit && matches!(config.scene, Scene::Demo) {
            return Err("edit replay requires the wide or model scene".into());
        }
        if config.monitor.is_some() && !config.full_hd {
            return Err("--monitor requires --1080p".into());
        }
        if config.motion != Motion::Static
            && (config.action != Action::Marquee || matches!(config.scene, Scene::Demo))
        {
            return Err("marquee motion requires the wide or model scene".into());
        }
        if matches!(config.motion, Motion::Section | Motion::Combined)
            && config.style != Style::Section
        {
            return Err("section motion requires --style edges-section".into());
        }
        if config.action == Action::Live && config.input_hz != DEFAULT_INPUT_HZ {
            return Err("input rate applies only to scripted replay".into());
        }
        Ok(Some(config))
    }

    pub(crate) fn dimensions(&self) -> [u32; 2] {
        if self.full_hd {
            return [1920, 1080];
        }
        match self.scene {
            Scene::Wide => [1024, 768],
            Scene::Model(_) | Scene::Shuffled => [1920, 1080],
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
    encoding: Option<u128>,
    submitted: Option<u128>,
    ready: Option<u128>,
    highlight_applied: Option<u128>,
    preview_view_changed: Option<bool>,
    frame: Option<u32>,
    hits: Option<usize>,
    rect: Option<[u32; 4]>,
    highlighted: Option<usize>,
    release_digest: Option<u64>,
    hit: Option<u64>,
    status: &'static str,
    warmup: bool,
}

pub(crate) struct Frame {
    pub(crate) id: u32,
    pub(crate) started: u128,
    pub(crate) acquired: u128,
    pub(crate) encoded: u128,
    pub(crate) viewport_encoded: u128,
    pub(crate) submitted: u128,
    pub(crate) presented: u128,
    pub(crate) status: &'static str,
    pub(crate) streaming: bool,
    pub(crate) dynamic_drawn: bool,
    pub(crate) preview_view_changed: bool,
}

pub(crate) struct Trace {
    pub(crate) options: Options,
    pub(crate) next: Instant,
    pub(crate) ticks: u32,
    pub(crate) marker: wgpu::RenderPipeline,
    epoch: Instant,
    period: Duration,
    warmup: u32,
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
        let input_hz = u64::from(options.input_hz);
        let period = Duration::from_nanos((1_000_000_000 + input_hz / 2) / input_hz);
        let warmup = options.input_hz.max(DEFAULT_INPUT_HZ);
        Self {
            options,
            next: epoch,
            ticks: 0,
            marker,
            epoch,
            period,
            warmup,
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
        self.next += self.period;
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
            encoding: None,
            submitted: None,
            ready: None,
            highlight_applied: None,
            preview_view_changed: None,
            frame: None,
            hits: None,
            rect: None,
            highlighted: None,
            release_digest: None,
            hit: None,
            status: "pending",
            warmup: self.ticks < self.warmup,
        });
        self.ticks += 1;
        if self.ticks >= self.warmup + self.options.samples {
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
                "encoding" => event.encoding = Some(time),
                "submitted" => event.submitted = Some(time),
                "ready" => {
                    event.ready = Some(time);
                    event.hits = hits;
                }
                _ => {}
            }
        }
    }

    pub(crate) fn selection_rect(&mut self, id: usize, rect: [u32; 4]) {
        if let Some(event) = self.events.get_mut(id) {
            event.rect = Some(rect);
        }
    }

    pub(crate) fn highlighted(&mut self, id: usize, count: usize) {
        let time = self.ns(Instant::now());
        if let Some(event) = self.events.get_mut(id) {
            event.highlighted = Some(count);
            event.highlight_applied = Some(time);
        }
    }

    pub(crate) fn release_digest(&mut self, id: usize, hits: &[PickHit]) {
        if let Some(event) = self.events.get_mut(id) {
            let mut hasher = DefaultHasher::new();
            hits.len().hash(&mut hasher);
            for hit in hits {
                hit.id.hash(&mut hasher);
                hit.depth.to_bits().hash(&mut hasher);
                hit.world_pos
                    .map(|p| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()])
                    .hash(&mut hasher);
            }
            event.release_digest = Some(hasher.finish());
        }
    }

    pub(crate) fn invalidate_selection_frame(&mut self, status: &'static str) {
        if let Some(id) = self.pending_selection_frame.take() {
            self.selection_status(id, status, None);
        }
    }

    pub(crate) fn replay_release(&mut self, now: Instant) -> Option<usize> {
        (self.ticks == self.warmup + self.options.samples)
            .then(|| self.event("marquee-release", now, now))
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
                    if event.kind == "marquee" {
                        event.preview_view_changed = Some(frame.preview_view_changed);
                    }
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
        if self.options.action != Action::Live {
            writeln!(info, "replay_input_hz={}", self.options.input_hz)?;
        }
        let completed: std::collections::HashMap<_, _> = self
            .completed
            .lock()
            .map(|done| done.iter().copied().collect())
            .unwrap_or_default();
        let mut frames = BufWriter::new(File::create(self.options.directory.join("frames.csv"))?);
        writeln!(
            frames,
            "frame,start_ns,acquire_ns,encode_ns,submit_ns,gpu_observed_ns,present_return_ns,status,streaming,viewport_encoded_ns,dynamic_drawn"
        )?;
        for frame in &self.frames {
            let gpu = completed
                .get(&frame.id)
                .map(ToString::to_string)
                .unwrap_or_default();
            writeln!(
                frames,
                "{},{},{},{},{},{},{},{},{},{},{}",
                frame.id,
                frame.started,
                frame.acquired,
                frame.encoded,
                frame.submitted,
                gpu,
                frame.presented,
                frame.status,
                frame.streaming,
                frame.viewport_encoded,
                frame.dynamic_drawn,
            )?;
        }
        let mut events = BufWriter::new(File::create(self.options.directory.join("events.csv"))?);
        writeln!(
            events,
            "event,kind,warmup,scheduled_ns,input_ns,redraw_ns,pick_submit_ns,pick_ready_ns,frame,hits,first_hit,status,pick_encode_ns,x,y,width,height,highlighted,release_digest,highlight_applied_ns,preview_view_changed"
        )?;
        for (id, event) in self.events.iter().enumerate() {
            fn field<T: ToString>(value: Option<T>) -> String {
                value.map(|value| value.to_string()).unwrap_or_default()
            }
            let [x, y, width, height] = event
                .rect
                .map(|rect| rect.map(|v| v.to_string()))
                .unwrap_or_default();
            writeln!(
                events,
                "{id},{},{},{},{},{},{},{},{},{},{},{},{},{x},{y},{width},{height},{},{},{},{}",
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
                event.status,
                field(event.encoding),
                field(event.highlighted),
                field(event.release_digest),
                field(event.highlight_applied),
                field(event.preview_view_changed),
            )?;
        }
        events.flush()?;
        frames.flush()
    }
}
