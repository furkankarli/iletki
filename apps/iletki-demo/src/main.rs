//! Interactive native desktop demo & stress tester for `iletki` (`winit` + `wgpu`).
//!
//! Run with:
//! ```text
//! cargo run --release -p iletki-demo
//! ```
//!
//! Controls:
//! - Middle/Right drag: Orbit (3D) or Pan (2D); hold Shift to Pan in 3D
//! - Scroll wheel: Dolly / Zoom
//! - Shift + Left drag: Select visible 3D objects within a rectangle
//! - Left click: 64-bit GPU Pick (highlights selected 3D chunk with Jump-Flood outline + moves gizmo)
//! - `1` : 3D Architectural Scene (Shaded with Edges + JFA Outline + Grid + Gizmo)
//! - `2` : 2D National-Grid CAD Stress (500,000 segments at UTM 661km / 4491km)
//! - `3` : 100,000-Instance 3D City Stress (1.2M triangles in 1 draw call)
//! - `Q` : Toggle 4-Way CAD Quad-View (Top / Persp / Front / Right in 1 CommandEncoder)
//! - `C` : Toggle 3D Section Clip Plane
//! - `M` : Toggle `OnDemand` (0 static draws on idle) vs `Continuous`
//! - `R` : Reset camera

#![allow(clippy::print_stdout, clippy::print_stderr)]

#[path = "../../../crates/iletki/examples/assembly_fixture/mod.rs"]
mod assembly_fixture;
mod latency;
mod preview;

use iletki::gpu::wgpu;
use iletki::{
    Bounds2, Bounds3, Cad2DRig, Camera, CameraTransition, Chunk3DId, DirtyFlags, DrawStyle2D,
    Grid3DConfig, GridStyle, InfiniteGrid3DLayer, Instance3D, InstancedMesh3DLayer, Mesh2D, Mesh3D,
    Modifiers, OrbitRig, OutlineConfig, PassTargetDesc, PendingPick, PickHit, PickId, Plane,
    PointerButton, Projection, Scene2DLayer, Scene3DLayer, ScrollUnit, StandardView,
    TransformGizmoLayer, Vec2, Vec3, ViewportConfig, ViewportHost, ViewportInput, ViewportLayer,
    ViewportMode, order_to_depth,
};
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Fullscreen, Window, WindowAttributes, WindowId};

const PICK_POLL_INTERVAL: Duration = Duration::from_micros(500);
const MAX_MOVING_PREVIEW_AGE: Duration = Duration::from_millis(50);
const TITLE_UPDATE_INTERVAL: Duration = Duration::from_millis(350);

const MARQUEE_SHADER: &str = r#"
@vertex
fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let x = f32((index << 1u) & 2u);
    let y = f32(index & 2u);
    return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}

@fragment
fn fragment() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.48, 0.09, 1.0);
}
"#;

fn marquee_pipeline(device: &wgpu::Device, format: wgpu::TextureFormat) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("marquee border"),
        source: wgpu::ShaderSource::Wgsl(MARQUEE_SHADER.into()),
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("marquee border"),
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
    })
}

fn marquee_border(
    pass: &mut wgpu::RenderPass<'_>,
    pipeline: &wgpu::RenderPipeline,
    rect: [u32; 4],
) {
    let [x, y, w, h] = rect;
    pass.set_pipeline(pipeline);
    for [x, y, w, h] in [
        [x, y, w, 1],
        [x, y + h - 1, w, 1],
        [x, y, 1, h],
        [x + w - 1, y, 1, h],
    ] {
        pass.set_scissor_rect(x, y, w, h);
        pass.draw(0..3, 0..1);
    }
}

fn block_on<F: Future>(fut: F) -> F::Output {
    let mut cx = Context::from_waker(Waker::noop());
    let mut boxed: Pin<Box<F>> = Box::pin(fut);
    loop {
        match boxed.as_mut().poll(&mut cx) {
            Poll::Ready(val) => return val,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DemoScene {
    Arch3D,
    Cad2D,
    Instanced100k,
}

struct ChunkMeta {
    pick_object: u32,
    chunk_id: Chunk3DId,
    gizmo_pos: Vec3,
    name: &'static str,
}

#[derive(Clone, PartialEq)]
struct PickView {
    scene: DemoScene,
    camera: Camera,
    size: [u32; 2],
    dpi: f64,
    revision: iletki::gpu::LayerRevision,
}

impl PickView {
    fn same_surface(&self, other: &Self) -> bool {
        self.scene == other.scene
            && self.size == other.size
            && self.dpi.to_bits() == other.dpi.to_bits()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PickKind {
    Click,
    Preview,
    Commit,
}

struct PickRequest {
    rect: [u32; 4],
    view: PickView,
    event: Option<usize>,
    kind: PickKind,
    epoch: u64,
    input_at: Instant,
}

struct SubmittedPick {
    pick: PendingPick,
    request: PickRequest,
    submitted_at: Instant,
}

struct GpuState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pass_desc: PassTargetDesc,

    // Viewports: main + 3 orthographic quad-view companions.
    vp_main: ViewportHost,
    vp_top: ViewportHost,
    vp_front: ViewportHost,
    vp_right: ViewportHost,

    // Camera rigs & smooth transition
    orbit_rig: OrbitRig,
    cad2d_rig: Cad2DRig,
    transition: CameraTransition,
    last_frame_time: Instant,

    // Shared scene layers
    scene_3d: Scene3DLayer,
    chunks_3d: Vec<ChunkMeta>,
    chunk_by_object: HashMap<u32, usize>,
    committed_selection: HashSet<Chunk3DId>,
    visible_selection: HashSet<Chunk3DId>,
    preview_view: Option<(PickView, Instant)>,
    grid_3d: InfiniteGrid3DLayer,
    gizmo_3d: TransformGizmoLayer,
    instanced_3d: InstancedMesh3DLayer,
    scene_2d: Scene2DLayer,
    origin_2d: Vec2,

    // Interactive state
    active_scene: DemoScene,
    quad_view: bool,
    clip_enabled: bool,
    modifiers: Modifiers,
    last_cursor: Vec2,
    press_pos: Option<Vec2>,
    marquee_origin: Option<Vec2>,
    marquee_rect: Option<[u32; 4]>,
    marquee_border: wgpu::RenderPipeline,
    pick_epoch: u64,
    pending_pick: Option<SubmittedPick>,
    queued_pick: Option<PickRequest>,
    preview_pacing: preview::PreviewPacing,
    frames_in_flight: Arc<AtomicUsize>,
    deferred_redraw: bool,
    latency: Option<latency::Trace>,
    #[cfg(feature = "profiling")]
    pick_profile: bool,
    bench_fit: Option<(Vec3, f64)>,
    bench_edits: Vec<(Chunk3DId, Vec3)>,
    bench_dual: bool,
    last_pick_text: String,
    timing_samples: u8,
    title_updated_at: Instant,
    smoothed_frame_ms: f64,
    smoothed_encode_ms: f64,
}

impl GpuState {
    fn new(window: Arc<Window>, bench: Option<latency::Options>) -> Result<Self, String> {
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance
            .create_surface(Arc::clone(&window))
            .map_err(|e| format!("create_surface failed: {e}"))?;

        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            ..wgpu::RequestAdapterOptions::default()
        }))
        .map_err(|e| format!("request_adapter failed: {e}"))?;

        let info = adapter.get_info();
        println!("iletki-demo GPU: {} ({:?})", info.name, info.backend);

        let features = wgpu::Features::CLIP_DISTANCES;
        #[cfg(feature = "profiling")]
        let pick_profile = std::env::var_os("ILETKI_PICK_PROFILE").is_some();
        #[cfg(feature = "profiling")]
        let features = if pick_profile {
            features | wgpu::Features::TIMESTAMP_QUERY
        } else {
            features
        };
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("iletki-demo device"),
            required_features: adapter.features() & features,
            ..wgpu::DeviceDescriptor::default()
        }))
        .map_err(|e| format!("request_device failed: {e}"))?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .or_else(|| caps.formats.first().copied())
            .ok_or_else(|| "surface reported no supported formats".to_string())?;
        let present_mode = if let Some(mode) = bench.as_ref().and_then(|options| options.present) {
            if !caps.present_modes.contains(&mode) {
                return Err(format!(
                    "present mode {mode:?} is unsupported by this surface"
                ));
            }
            mode
        } else if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
            wgpu::PresentMode::Mailbox
        } else {
            wgpu::PresentMode::Fifo
        };

        let mut surface_config = surface
            .get_default_config(&adapter, width, height)
            .ok_or_else(|| "surface reported no default configuration".to_string())?;
        surface_config.format = format;
        surface_config.present_mode = present_mode;
        if let Some(depth) = bench.as_ref().and_then(|options| options.depth) {
            surface_config.desired_maximum_frame_latency = depth;
        }
        if bench.is_some() {
            println!(
                "latency surface: {width}x{height}, {present_mode:?}, requested depth {}",
                surface_config.desired_maximum_frame_latency
            );
        }
        surface.configure(&device, &surface_config);

        let make_vp = |px: [u32; 2]| {
            ViewportHost::new_with_adapter(
                &device,
                &adapter,
                ViewportConfig {
                    color_format: format,
                    msaa_samples: 4,
                    clear_color: [0.09, 0.11, 0.15, 1.0],
                    mode: ViewportMode::OnDemand,
                    initial_px: px,
                    outline: OutlineConfig {
                        color: [1.0, 0.55, 0.05, 1.0],
                        width_px: 4.0,
                    },
                    ..ViewportConfig::default()
                },
            )
        };

        let vp_main = make_vp([width, height]).map_err(|e| e.to_string())?;
        let vp_top =
            make_vp([(width / 2).max(1), (height / 2).max(1)]).map_err(|e| e.to_string())?;
        let vp_front =
            make_vp([(width / 2).max(1), (height / 2).max(1)]).map_err(|e| e.to_string())?;
        let vp_right =
            make_vp([(width / 2).max(1), (height / 2).max(1)]).map_err(|e| e.to_string())?;

        let mut bench_fit = None;
        let mut bench_edits = Vec::new();
        let (scene_3d, chunks_3d) = match bench.as_ref().map(|b| &b.scene) {
            Some(latency::Scene::Wide) => {
                bench_fit = Some((Vec3::new(64.0, 312.0, 0.8), 626.0));
                let style = bench.as_ref().map_or(latency::Style::Faces, |b| b.style);
                bench_scene(
                    (0..20_000).map(|index| latency::wide_mesh(index, style)),
                    &mut bench_edits,
                )
            }
            Some(latency::Scene::Shuffled) => {
                let center = f64::from(100_000_u32.div_ceil(64));
                bench_fit = Some((Vec3::new(63.0, center, 0.8), center * 2.0));
                let style = bench.as_ref().map_or(latency::Style::Faces, |b| b.style);
                let mut order: Vec<u32> = (0..100_000).collect();
                order.sort_unstable_by_key(|index| index.wrapping_mul(0x9e37_79b9));
                bench_scene(
                    order
                        .into_iter()
                        .map(|index| latency::wide_mesh(index, style)),
                    &mut bench_edits,
                )
            }
            Some(latency::Scene::Model(path)) => {
                let fixture = assembly_fixture::load(path).map_err(|error| error.to_string())?;
                let center = fixture.bounds.center().ok_or("model bounds are empty")?;
                let radius = fixture.bounds.radius() * 2.6;
                if !center.is_finite() || !radius.is_finite() || radius <= 0.0 {
                    return Err("model camera bounds are invalid".into());
                }
                bench_fit = Some((center, radius));
                println!(
                    "latency model: {} chunks, {} triangles",
                    fixture.meshes.len(),
                    fixture.triangles
                );
                bench_scene(fixture.meshes, &mut bench_edits)
            }
            _ => build_arch_3d(),
        };
        let grid_3d = InfiniteGrid3DLayer::new(2, Grid3DConfig::default());
        let mut gizmo_3d = TransformGizmoLayer::new(3, 1);
        gizmo_3d.origin = Some(Vec3::new(-0.8, 0.4, 3.0));

        let instanced_3d = build_instanced_100k();

        let origin_2d = Vec2::new(661_858.0, 4_491_159.0);
        let mut scene_2d = Scene2DLayer::new(4);
        if bench.is_none() {
            let mut mesh_2d = build_cad_2d(origin_2d, format.is_srgb());
            mesh_2d.optimize_spatial();
            scene_2d.upload_static_mesh(&device, &queue, &mesh_2d);
        }
        let bench_dual = bench.as_ref().is_some_and(|options| options.dual);
        let clip_enabled = bench
            .as_ref()
            .is_some_and(|options| options.style == latency::Style::Section);
        let latency = bench.map(|options| latency::Trace::new(options, &device, format));

        let chunk_by_object = chunks_3d
            .iter()
            .enumerate()
            .map(|(index, chunk)| (chunk.pick_object, index))
            .collect();
        let committed_selection: HashSet<_> = chunks_3d
            .iter()
            .map(|chunk| chunk.chunk_id)
            .filter(|&id| scene_3d.is_selected(id))
            .collect();
        let visible_selection = committed_selection.clone();
        let marquee_border = marquee_pipeline(&device, format);
        let preview_pacing = preview::PreviewPacing::new(
            window
                .current_monitor()
                .and_then(|monitor| monitor.refresh_rate_millihertz()),
        );
        let mut state = Self {
            window,
            surface,
            surface_config,
            device,
            queue,
            pass_desc: PassTargetDesc {
                format,
                sample_count: 1,
                depth_format: None,
                blend_alpha: false,
            },
            vp_main,
            vp_top,
            vp_front,
            vp_right,
            orbit_rig: OrbitRig::new(Vec3::new(0.0, 0.0, 1.8), Vec3::Z),
            cad2d_rig: Cad2DRig::default(),
            transition: CameraTransition::default(),
            last_frame_time: Instant::now(),
            scene_3d,
            chunks_3d,
            chunk_by_object,
            committed_selection,
            visible_selection,
            preview_view: None,
            grid_3d,
            gizmo_3d,
            instanced_3d,
            scene_2d,
            origin_2d,
            active_scene: DemoScene::Arch3D,
            quad_view: false,
            clip_enabled,
            modifiers: Modifiers::default(),
            last_cursor: Vec2::ZERO,
            press_pos: None,
            marquee_origin: None,
            marquee_rect: None,
            marquee_border,
            pick_epoch: 0,
            pending_pick: None,
            queued_pick: None,
            preview_pacing,
            frames_in_flight: Arc::new(AtomicUsize::new(0)),
            deferred_redraw: false,
            latency,
            #[cfg(feature = "profiling")]
            pick_profile,
            bench_fit,
            bench_edits,
            bench_dual,
            last_pick_text: String::from("Click any 3D object to pick"),
            timing_samples: 0,
            title_updated_at: Instant::now(),
            smoothed_frame_ms: 0.0,
            smoothed_encode_ms: 0.0,
        };
        #[cfg(feature = "profiling")]
        if std::env::var_os("ILETKI_DIRECT_PICK").is_some() {
            state.scene_3d.set_profile_batch_picking(false);
        }
        state.reset_cameras();
        if let Some((center, radius)) = state.bench_fit {
            *state.vp_main.camera_mut() = Camera::perspective_3d(
                center + Vec3::new(0.0, -radius, radius * 0.65),
                center,
                Vec3::Z,
                55_f64.to_radians(),
                radius / 1500.0,
            );
            *state.vp_top.camera_mut() = Camera::ortho_3d(
                center + Vec3::Z * radius,
                center,
                Vec3::Y,
                radius,
                radius / 1500.0,
                radius * 3.0,
            );
            state.orbit_rig.pivot = center;
            state.gizmo_3d.origin = None;
            state.apply_clip_planes();
        }
        state.update_title();
        Ok(state)
    }

    fn reset_cameras(&mut self) {
        match self.active_scene {
            DemoScene::Arch3D => {
                *self.vp_main.camera_mut() = Camera::perspective_3d(
                    Vec3::new(9.5, -10.5, 7.5),
                    Vec3::new(0.0, 0.0, 1.8),
                    Vec3::Z,
                    std::f64::consts::FRAC_PI_4,
                    0.05,
                );
                self.orbit_rig.pivot = Vec3::new(0.0, 0.0, 1.8);
                *self.vp_top.camera_mut() = Camera::ortho_3d(
                    Vec3::new(0.0, 0.0, 20.0),
                    Vec3::new(0.0, 0.0, 1.5),
                    Vec3::Y,
                    14.0,
                    0.1,
                    100.0,
                );
                *self.vp_front.camera_mut() = Camera::ortho_3d(
                    Vec3::new(0.0, -20.0, 2.0),
                    Vec3::new(0.0, 0.0, 2.0),
                    Vec3::Z,
                    12.0,
                    0.1,
                    100.0,
                );
                *self.vp_right.camera_mut() = Camera::ortho_3d(
                    Vec3::new(20.0, 0.0, 2.0),
                    Vec3::new(0.0, 0.0, 2.0),
                    Vec3::Z,
                    12.0,
                    0.1,
                    100.0,
                );
            }
            DemoScene::Cad2D => {
                *self.vp_main.camera_mut() =
                    Camera::ortho_2d(self.origin_2d + Vec2::new(250.0, 250.0), 3.0);
                self.quad_view = false;
            }
            DemoScene::Instanced100k => {
                *self.vp_main.camera_mut() = Camera::perspective_3d(
                    Vec3::new(35.0, -45.0, 28.0),
                    Vec3::new(0.0, 0.0, 2.0),
                    Vec3::Z,
                    std::f64::consts::FRAC_PI_3,
                    0.1,
                );
                self.orbit_rig.pivot = Vec3::new(0.0, 0.0, 2.0);
                self.quad_view = false;
            }
        }
        self.apply_clip_planes();
        self.resize_viewports();
    }

    fn apply_clip_planes(&mut self) {
        let clip = if self.clip_enabled {
            Plane::from_normal_point(
                Vec3::Y,
                self.bench_fit
                    .map_or(Vec3::new(0.0, -0.35, 0.0), |fit| fit.0),
            )
        } else {
            None
        };
        for vp in [
            &mut self.vp_main,
            &mut self.vp_top,
            &mut self.vp_front,
            &mut self.vp_right,
        ] {
            if let Some(p) = clip {
                vp.camera_mut().set_clip_planes(&[p]);
            } else {
                vp.camera_mut().clear_clip_planes();
            }
            vp.mark_dirty(DirtyFlags::ALL);
        }
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.surface_config.width = new_size.width;
        self.surface_config.height = new_size.height;
        self.surface.configure(&self.device, &self.surface_config);
        self.resize_viewports();
    }

    fn resize_viewports(&mut self) {
        let w = self.surface_config.width.max(1);
        let h = self.surface_config.height.max(1);
        let dpi = self.window.scale_factor() as f32;
        if self.bench_dual {
            let left = (w / 2).max(1);
            self.vp_top.resize(&self.device, [left, h], dpi);
            self.vp_main
                .resize(&self.device, [(w - left).max(1), h], dpi);
        } else if self.quad_view && self.active_scene == DemoScene::Arch3D {
            let qw = (w / 2).max(1);
            let qh = (h / 2).max(1);
            self.vp_top.resize(&self.device, [qw, qh], dpi);
            self.vp_main.resize(&self.device, [w - qw, qh], dpi);
            self.vp_front.resize(&self.device, [qw, h - qh], dpi);
            self.vp_right.resize(&self.device, [w - qw, h - qh], dpi);
        } else {
            self.vp_main.resize(&self.device, [w, h], dpi);
        }
    }

    fn update_2d_overlay(&mut self, cursor_logical: Vec2) {
        let logical = self.vp_main.logical_size();
        let visible = self
            .vp_main
            .camera()
            .visible_2d(logical)
            .unwrap_or(Bounds2::EMPTY);
        let zoom = match self.vp_main.camera().projection {
            Projection::Ortho2D { zoom, .. } => zoom,
            _ => 1.0,
        };
        let linear = self.surface_config.format.is_srgb();
        let mut overlay = Mesh2D::default();
        iletki::cad2d::overlay::push_adaptive_grid(
            &mut overlay,
            visible,
            zoom,
            GridStyle::default(),
            linear,
        );
        if let Some(world) = self
            .vp_main
            .camera()
            .screen_to_world_2d(cursor_logical, logical)
        {
            iletki::cad2d::overlay::push_crosshair(
                &mut overlay,
                visible,
                world,
                [255, 255, 255, 190],
                linear,
            );
        }
        self.scene_2d.set_overlay(&overlay);
        self.vp_main.mark_dirty(DirtyFlags::OVERLAY);
    }

    fn pick_view(&self, kind: PickKind) -> PickView {
        let revision = match self.active_scene {
            DemoScene::Arch3D if kind != PickKind::Click => {
                ViewportLayer::<()>::revision(&self.scene_3d).static_only()
            }
            DemoScene::Arch3D => ViewportLayer::<()>::revision(&self.scene_3d),
            // The demo's 2D geometry is immutable. Cursor overlay writes are unrelated to picks.
            DemoScene::Cad2D => iletki::gpu::LayerRevision::default(),
            DemoScene::Instanced100k => ViewportLayer::<()>::revision(&self.instanced_3d),
        };
        PickView {
            scene: self.active_scene,
            camera: self.vp_main.camera().clone(),
            size: self.vp_main.size().active_px,
            dpi: self.window.scale_factor(),
            revision,
        }
    }

    fn handle_pick(&mut self, logical_pos: Vec2) {
        let now = Instant::now();
        let dpi = self.window.scale_factor();
        let px = (logical_pos.x * dpi).round().max(0.0) as u32;
        let px = if self.bench_dual {
            let Some(local) = px.checked_sub(self.surface_config.width / 2) else {
                return;
            };
            local
        } else {
            px
        };
        let py = (logical_pos.y * dpi).round().max(0.0) as u32;
        let event = self
            .latency
            .as_mut()
            .and_then(|trace| trace.live("click", now));
        self.request_pick(
            [px.saturating_sub(2), py.saturating_sub(2), 5, 5],
            PickKind::Click,
            event,
        );
    }

    fn pick_status(&mut self, event: Option<usize>, status: &'static str, hits: Option<usize>) {
        if let Some(trace) = &mut self.latency
            && let Some(id) = event
        {
            trace.selection_status(id, status, hits);
        }
    }

    fn marquee_at(&self, origin: Vec2, cursor: Vec2) -> Option<[u32; 4]> {
        if !origin.is_finite() || !cursor.is_finite() || origin == cursor {
            return None;
        }
        let dpi = self.window.scale_factor();
        let [w, h] = self.vp_main.size().active_px;
        let x = (origin.x.min(cursor.x) * dpi)
            .floor()
            .clamp(0.0, f64::from(w)) as u32;
        let y = (origin.y.min(cursor.y) * dpi)
            .floor()
            .clamp(0.0, f64::from(h)) as u32;
        let right = (origin.x.max(cursor.x) * dpi)
            .ceil()
            .clamp(0.0, f64::from(w)) as u32;
        let bottom = (origin.y.max(cursor.y) * dpi)
            .ceil()
            .clamp(0.0, f64::from(h)) as u32;
        (right > x && bottom > y).then_some([
            x,
            y,
            right.saturating_sub(x),
            bottom.saturating_sub(y),
        ])
    }

    fn update_marquee(&mut self, origin: Vec2, kind: PickKind, now: Instant) {
        let rect = self.marquee_at(origin, self.last_cursor);
        let redraw_border =
            kind != PickKind::Preview || self.marquee_rect.is_none() || rect.is_none();
        if kind == PickKind::Preview {
            if rect == self.marquee_rect {
                return;
            }
            self.marquee_rect = rect;
        } else {
            self.marquee_rect = None;
        }
        if let Some(rect) = rect {
            let label = if kind == PickKind::Commit {
                "marquee-release"
            } else {
                "marquee"
            };
            let event = self
                .latency
                .as_mut()
                .and_then(|trace| trace.live(label, now));
            self.request_pick(rect, kind, event);
        } else if kind == PickKind::Preview {
            self.invalidate_picks();
        } else {
            self.cancel_selection();
        }
        if redraw_border {
            self.window.request_redraw();
        }
    }

    fn cancel_selection(&mut self) {
        self.marquee_origin = None;
        self.marquee_rect = None;
        self.invalidate_picks();
        self.window.request_redraw();
    }

    fn invalidate_picks(&mut self) {
        self.preview_pacing = preview::PreviewPacing::new(
            self.window
                .current_monitor()
                .and_then(|monitor| monitor.refresh_rate_millihertz()),
        );
        self.pick_epoch = self.pick_epoch.wrapping_add(1);
        if let Some(queued) = self.queued_pick.take() {
            self.pick_status(queued.event, "cancelled", None);
        }
        self.clear_preview("cancelled");
    }

    fn clear_preview(&mut self, status: &'static str) {
        if self.preview_view.take().is_some() {
            self.preview_pacing.frame_started();
            if let Some(trace) = &mut self.latency {
                trace.invalidate_selection_frame(status);
            }
            self.sync_highlights(None);
        }
    }

    fn request_pick(&mut self, rect: [u32; 4], kind: PickKind, event: Option<usize>) {
        if self.quad_view {
            self.pick_status(event, "unsupported-view", None);
            return;
        }
        if let (Some(trace), Some(id)) = (&mut self.latency, event) {
            trace.selection_rect(id, rect);
        }
        let now = Instant::now();
        let request = PickRequest {
            rect,
            view: self.pick_view(kind),
            event,
            kind,
            epoch: self.pick_epoch,
            input_at: now,
        };
        if kind == PickKind::Preview {
            self.preview_pacing.input(now);
            if self
                .queued_pick
                .as_ref()
                .is_some_and(|queued| queued.kind == PickKind::Commit)
            {
                self.pick_status(event, "coalesced-before-submit", None);
                return;
            }
        }
        if let Some(old) = self.queued_pick.replace(request) {
            self.pick_status(old.event, "coalesced-before-submit", None);
        }
        self.submit_queued_pick(now);
    }

    fn submit_queued_pick(&mut self, now: Instant) {
        if self.pending_pick.is_some()
            || self
                .queued_pick_deadline()
                .is_some_and(|deadline| now < deadline)
        {
            return;
        }
        let Some(request) = self.queued_pick.take() else {
            return;
        };
        if request.epoch == self.pick_epoch
            && (request.kind == PickKind::Commit || request.view == self.pick_view(request.kind))
        {
            if request.kind == PickKind::Preview {
                self.preview_pacing.submitted(now);
            }
            self.begin_pick(request);
        } else {
            self.pick_status(request.event, "stale-before-submit", None);
        }
    }

    fn queued_pick_deadline(&self) -> Option<Instant> {
        self.queued_pick
            .as_ref()
            .filter(|request| request.kind == PickKind::Preview)?;
        self.preview_pacing.deadline()
    }

    fn begin_pick(&mut self, mut request: PickRequest) {
        let rect = request.rect;
        #[cfg(feature = "profiling")]
        if self.pick_profile
            && self
                .latency
                .as_ref()
                .is_some_and(|trace| trace.ticks >= 65 && trace.ticks.is_multiple_of(4))
            && !iletki::gpu::profiling::active()
        {
            iletki::gpu::profiling::begin(&self.device);
        }
        self.pick_status(request.event, "encoding", None);
        let pick = match self.active_scene {
            DemoScene::Arch3D if request.kind != PickKind::Click => self.vp_main.begin_pick_rect(
                &self.device,
                &self.queue,
                &(),
                &mut [&mut self.scene_3d],
                rect,
            ),
            DemoScene::Arch3D => {
                let mut layers: [&mut dyn ViewportLayer<()>; 3] =
                    [&mut self.scene_3d, &mut self.grid_3d, &mut self.gizmo_3d];
                self.vp_main
                    .begin_pick_rect(&self.device, &self.queue, &(), &mut layers, rect)
            }
            DemoScene::Instanced100k => {
                let mut layers: [&mut dyn ViewportLayer<()>; 1] = [&mut self.instanced_3d];
                self.vp_main
                    .begin_pick_rect(&self.device, &self.queue, &(), &mut layers, rect)
            }
            DemoScene::Cad2D => {
                let mut layers: [&mut dyn ViewportLayer<()>; 1] = [&mut self.scene_2d];
                self.vp_main
                    .begin_pick_rect(&self.device, &self.queue, &(), &mut layers, rect)
            }
        };
        if let Ok(pick) = pick {
            // Picking can itself finish progressive uploads during preparation.
            request.view = self.pick_view(request.kind);
            self.pick_status(request.event, "submitted", None);
            self.pending_pick = Some(SubmittedPick {
                pick,
                request,
                submitted_at: Instant::now(),
            });
        } else {
            self.pick_status(request.event, "error", None);
            if request.kind == PickKind::Commit {
                self.clear_preview("cancelled");
            }
        }
    }

    fn preview_deadline(&self) -> Option<Instant> {
        let (view, input_at) = self.preview_view.as_ref()?;
        (*view != self.pick_view(PickKind::Preview)).then(|| *input_at + MAX_MOVING_PREVIEW_AGE)
    }

    fn clear_stale_preview(&mut self, now: Instant) {
        if self
            .preview_deadline()
            .is_some_and(|deadline| now >= deadline)
        {
            self.clear_preview("stale-before-frame");
            self.window.request_redraw();
        }
    }

    fn poll_pick(&mut self) {
        let now = Instant::now();
        self.clear_stale_preview(now);
        let Some(pending) = &mut self.pending_pick else {
            return;
        };
        let Poll::Ready(result) = pending.pick.try_read() else {
            return;
        };
        let Some(SubmittedPick {
            mut request,
            submitted_at,
            ..
        }) = self.pending_pick.take()
        else {
            return;
        };
        if let Ok(hits) = &result {
            self.pick_status(request.event, "ready", Some(hits.len()));
        }
        let current = self.pick_view(request.kind);
        let preview = request.kind == PickKind::Preview;
        let commit_queued = self
            .queued_pick
            .as_ref()
            .is_some_and(|queued| queued.kind == PickKind::Commit);
        let recent_preview = preview
            && !commit_queued
            && request.view.same_surface(&current)
            && now.saturating_duration_since(request.input_at) < MAX_MOVING_PREVIEW_AGE;
        if request.epoch != self.pick_epoch || self.quad_view {
            self.pick_status(request.event, "cancelled", None);
            return;
        }
        if preview {
            self.preview_pacing.pick_ready(submitted_at, now);
        }
        if request.view != current && !recent_preview {
            if request.kind == PickKind::Commit {
                request.view = current;
                self.begin_pick(request);
                return;
            }
            self.pick_status(request.event, "stale", None);
        } else if let Ok(hits) = result {
            let event = request.event;
            if let Some(trace) = &mut self.latency
                && let Some(id) = event
            {
                trace.hit(id, hits.first().map(|hit| hit.id.to_u64()));
            }
            self.apply_pick(&hits, request);
            if let Some(trace) = &mut self.latency
                && let Some(id) = event
            {
                trace.selection_redraw(id);
            }
            if preview {
                self.preview_pacing.request_frame(Instant::now());
            }
            self.window.request_redraw();
        } else {
            self.pick_status(request.event, "error", None);
            if request.kind == PickKind::Commit {
                self.clear_preview("cancelled");
            }
        }
    }

    fn sync_highlights(&mut self, preview: Option<HashSet<Chunk3DId>>) {
        let desired = preview.as_ref().unwrap_or(&self.committed_selection);
        for &id in self.visible_selection.difference(desired) {
            self.scene_3d.set_selected(id, false);
        }
        for &id in desired.difference(&self.visible_selection) {
            self.scene_3d.set_selected(id, true);
        }
        if self.visible_selection != *desired {
            self.visible_selection = preview.unwrap_or_else(|| self.committed_selection.clone());
            self.vp_main.mark_dirty(DirtyFlags::SELECTION);
            if self.bench_dual {
                self.vp_top.mark_dirty(DirtyFlags::SELECTION);
            }
        }
    }

    fn apply_pick(&mut self, hits: &[PickHit], request: PickRequest) {
        if request.kind != PickKind::Click {
            let selected: HashSet<_> = hits
                .iter()
                .filter(|hit| hit.id.layer == 1)
                .filter_map(|hit| self.chunk_by_object.get(&hit.id.object))
                .filter_map(|&index| self.chunks_3d.get(index))
                .map(|chunk| chunk.chunk_id)
                .collect();
            if request.kind == PickKind::Commit {
                self.preview_view = None;
                self.committed_selection = selected;
                self.sync_highlights(None);
            } else {
                self.preview_view = Some((request.view, request.input_at));
                self.sync_highlights(Some(selected));
            }
            if request.kind == PickKind::Commit {
                self.gizmo_3d.origin = None;
                self.vp_main.mark_dirty(DirtyFlags::GIZMO);
            }
            self.last_pick_text = format!("{} selected", self.visible_selection.len());
            if let (Some(trace), Some(event)) = (&mut self.latency, request.event) {
                trace.highlighted(event, self.visible_selection.len());
                if request.kind == PickKind::Commit {
                    trace.release_digest(event, hits);
                }
            }
        } else if let Some(hit) = hits.first() {
            if self.active_scene == DemoScene::Arch3D && hit.id.layer == 1 {
                self.preview_view = None;
                self.committed_selection.clear();
                if let Some(meta) = self
                    .chunk_by_object
                    .get(&hit.id.object)
                    .and_then(|&index| self.chunks_3d.get(index))
                {
                    self.committed_selection.insert(meta.chunk_id);
                    self.gizmo_3d.origin = Some(meta.gizmo_pos);
                    self.last_pick_text = format!(
                        "Picked {} (obj={} face={})",
                        meta.name, hit.id.object, hit.id.sub
                    );
                }
                self.sync_highlights(None);
                self.vp_main.mark_dirty(DirtyFlags::GIZMO);
            } else {
                self.last_pick_text = format!(
                    "Picked layer={} obj={} sub={}",
                    hit.id.layer, hit.id.object, hit.id.sub
                );
            }
        } else {
            if self.active_scene == DemoScene::Arch3D {
                self.preview_view = None;
                self.committed_selection.clear();
                self.sync_highlights(None);
                self.gizmo_3d.origin = None;
                self.vp_main.mark_dirty(DirtyFlags::GIZMO);
            }
            self.last_pick_text = String::from("Empty background");
        }
    }

    fn update_title(&self) {
        let scene_name = match self.active_scene {
            DemoScene::Arch3D => {
                if self.quad_view {
                    "3D Quad-View"
                } else {
                    "3D Arch"
                }
            }
            DemoScene::Cad2D => "2D 500k CAD (UTM)",
            DemoScene::Instanced100k => "3D 100k Instances (1.2M tris)",
        };
        let mode = match self.vp_main.mode() {
            ViewportMode::OnDemand => "OnDemand",
            ViewportMode::Continuous => "Continuous",
        };
        let clip = if self.clip_enabled { " | CLIP" } else { "" };
        let frame_label = if self.timing_samples == 1 {
            "Startup frame"
        } else {
            "Frame"
        };
        self.window.set_title(&format!(
            "iletki-demo | {scene_name} | {mode}{clip} | CPU encode {:.2} ms | {frame_label} {:.2} ms | {} | [1/2/3 Q C M R]",
            self.smoothed_encode_ms, self.smoothed_frame_ms, self.last_pick_text
        ));
    }

    fn animate_to_view(&mut self, view: Option<StandardView>) {
        if self.active_scene != DemoScene::Arch3D {
            return;
        }
        let bounds = Bounds3::from_corners(Vec3::new(-4.5, -3.5, 0.0), Vec3::new(4.5, 3.5, 5.7));
        let logical = self.vp_main.logical_size();
        let from = self.vp_main.camera().clone();
        let (mut target_cam, pivot) = match view {
            Some(v) => Camera::standard_view_3d(v, bounds, logical, 1.25, true),
            None => {
                let mut fitted = from.clone();
                let p = fitted
                    .fit_3d(bounds, logical, 1.25)
                    .unwrap_or(Vec3::new(0.0, 0.0, 2.0));
                (fitted, p)
            }
        };
        if self.clip_enabled
            && let Some(clip) = Plane::from_normal_point(Vec3::Y, Vec3::new(0.0, -0.35, 0.0))
        {
            target_cam.set_clip_planes(&[clip]);
        }
        self.orbit_rig.pivot = pivot;
        self.transition.start(from, target_cam, 0.35);
        self.last_frame_time = Instant::now();
    }

    fn dispatch_input(&mut self, event: &ViewportInput) -> bool {
        let outcome = if self.active_scene == DemoScene::Cad2D {
            self.vp_main.on_input(event, &mut self.cad2d_rig)
        } else {
            self.vp_main.on_input(event, &mut self.orbit_rig)
        };
        outcome
            .iter()
            .any(|command| matches!(command, iletki::ViewportCommand::RequestRedraw))
    }

    fn track_live_input(&mut self, kind: &'static str, started: Instant) {
        if let Some(trace) = &mut self.latency
            && let Some(id) = trace.live(kind, started)
        {
            trace.redraw(id);
        }
    }

    fn replay_latency(&mut self, now: Instant) {
        let Some(trace) = &mut self.latency else {
            return;
        };
        let Some((tick, event)) = trace.tick(now) else {
            return;
        };
        let action = trace.options.action;
        let motion = trace.options.motion;
        let region = trace.options.region;
        let width = trace.options.rect_size;
        let release = if action == latency::Action::Marquee {
            trace.replay_release(now)
        } else {
            None
        };
        let [w, h] = self.vp_main.size().active_px;
        let center = self.vp_main.logical_size() * 0.5;
        let sign = if tick % 2 == 0 { 1.0 } else { -1.0 };
        let modifiers = Modifiers {
            shift: action == latency::Action::Pan,
            ..Modifiers::default()
        };
        match action {
            latency::Action::Click => {
                let x = if tick % 2 == 0 { w / 2 } else { w / 8 };
                self.request_pick(
                    [x.saturating_sub(2), (h / 2).saturating_sub(2), 5, 5],
                    PickKind::Click,
                    Some(event),
                );
                return;
            }
            latency::Action::Marquee => {
                if motion == latency::Motion::Orbit {
                    if tick == 0 {
                        self.dispatch_input(&ViewportInput::PointerDown {
                            pos: center,
                            button: PointerButton::Middle,
                            modifiers,
                        });
                    }
                    self.dispatch_input(&ViewportInput::PointerMove {
                        pos: center,
                        delta: Vec2::new(6.0 * sign, sign),
                        modifiers,
                    });
                }
                if motion == latency::Motion::Combined
                    && let Some((focus, radius)) = self.bench_fit
                {
                    let angle = f64::from(tick % 60) * std::f64::consts::TAU / 60.0;
                    let eye = focus
                        + Vec3::new(radius * angle.sin(), -radius * angle.cos(), radius * 0.65);
                    *self.vp_main.camera_mut() = Camera::perspective_3d(
                        eye,
                        focus,
                        Vec3::Z,
                        55_f64.to_radians(),
                        radius / 1500.0,
                    );
                }
                if matches!(motion, latency::Motion::Edit | latency::Motion::Combined)
                    && tick.is_multiple_of(6)
                {
                    let edited = tick.is_multiple_of(12);
                    let offset = if edited { 0.2 } else { 0.0 };
                    for &(id, origin) in &self.bench_edits {
                        self.scene_3d.set_origin(id, origin + Vec3::X * offset);
                        if motion == latency::Motion::Combined {
                            self.scene_3d.set_visible(id, !edited);
                        }
                    }
                }
                if matches!(motion, latency::Motion::Section | latency::Motion::Combined) {
                    self.rotate_bench_section(tick);
                }
                let (x, y) = match region {
                    latency::Region::Mixed => (if tick % 2 == 0 { w / 2 } else { w / 8 }, h / 2),
                    region => {
                        let x = match region {
                            latency::Region::Filled => w / 2,
                            latency::Region::Boundary => w / 3,
                            _ => 0,
                        };
                        let offset = (tick % 32) as i32 - 16;
                        (
                            x.saturating_add_signed(offset),
                            (h / 2).saturating_add_signed(offset / 2),
                        )
                    }
                };
                let [width, height] = [width.min(w), width.min(h)];
                let rect = [
                    x.saturating_sub(width / 2).min(w - width),
                    y.saturating_sub(height / 2).min(h - height),
                    width,
                    height,
                ];
                if motion != latency::Motion::Static {
                    self.window.request_redraw();
                }
                self.request_pick(rect, PickKind::Preview, Some(event));
                if let Some(release) = release {
                    self.request_pick(rect, PickKind::Commit, Some(release));
                }
                return;
            }
            latency::Action::Live => return,
            latency::Action::Zoom => {
                self.dispatch_input(&ViewportInput::Scroll {
                    pos: center,
                    delta: Vec2::new(0.0, sign),
                    unit: ScrollUnit::Line,
                    modifiers,
                });
            }
            latency::Action::Orbit | latency::Action::Pan | latency::Action::Edit => {
                if tick == 0 {
                    self.dispatch_input(&ViewportInput::PointerDown {
                        pos: center,
                        button: PointerButton::Middle,
                        modifiers,
                    });
                }
                if action == latency::Action::Edit && tick % 6 == 0 {
                    let offset = if tick % 12 == 0 { 0.2 } else { 0.0 };
                    for &(id, origin) in &self.bench_edits {
                        self.scene_3d.set_origin(id, origin + Vec3::X * offset);
                    }
                }
                self.dispatch_input(&ViewportInput::PointerMove {
                    pos: center,
                    delta: Vec2::new(6.0 * sign, sign),
                    modifiers,
                });
            }
        }
        if self.clip_enabled {
            self.rotate_bench_section(tick);
        }
        if let Some(trace) = &mut self.latency {
            trace.redraw(event);
        }
        self.window.request_redraw();
    }

    fn rotate_bench_section(&mut self, tick: u32) {
        if let Some((focus, _)) = self.bench_fit {
            let angle = f64::from(tick % 60) * std::f64::consts::TAU / 60.0;
            if let Some(plane) =
                Plane::from_normal_point(Vec3::new(angle.sin(), angle.cos(), 0.15), focus)
            {
                self.vp_main.camera_mut().set_clip_planes(&[plane]);
                if self.bench_dual {
                    self.vp_top.camera_mut().set_clip_planes(&[plane]);
                }
            }
        }
    }

    fn trace_surface_failure(&mut self, started: Instant, status: &'static str) {
        if let Some(trace) = &mut self.latency {
            trace.frame(latency::Frame {
                id: trace.frame_id(),
                started: trace.ns(started),
                acquired: trace.ns(Instant::now()),
                encoded: 0,
                viewport_encoded: 0,
                submitted: 0,
                presented: 0,
                status,
                streaming: false,
                dynamic_drawn: false,
                preview_view_changed: false,
            });
        }
    }

    fn pending_preview(&self) -> Option<&SubmittedPick> {
        self.pending_pick
            .as_ref()
            .filter(|pending| pending.request.kind == PickKind::Preview)
    }

    fn preview_active(&self) -> bool {
        self.marquee_origin.is_some()
            || self.preview_view.is_some()
            || self.pending_preview().is_some()
            || self
                .queued_pick
                .as_ref()
                .is_some_and(|request| request.kind == PickKind::Preview)
    }

    fn render_backlogged(&self, now: Instant) -> bool {
        self.frames_in_flight.load(Ordering::Relaxed) != 0
            && self.preview_active()
            && self
                .preview_pacing
                .gpu_backlogged(now, self.pending_preview().map(|pick| pick.submitted_at))
    }

    fn render(&mut self) -> Result<(), iletki::ViewportError> {
        let frame_start = Instant::now();
        if self.render_backlogged(frame_start) {
            self.deferred_redraw = true;
            return Ok(());
        }
        self.deferred_redraw = false;
        self.preview_pacing.frame_started();
        #[cfg(feature = "profiling")]
        if self.pick_profile
            && self.latency.as_ref().is_some_and(|trace| trace.ticks >= 65)
            && self.vp_main.dirty().contains(DirtyFlags::SELECTION)
            && !iletki::gpu::profiling::active()
        {
            iletki::gpu::profiling::begin(&self.device);
        }
        let frame_gap = frame_start.saturating_duration_since(self.last_frame_time);
        let dt = frame_gap.as_secs_f64().min(0.1);
        self.last_frame_time = frame_start;
        if self.transition.is_active() {
            let changed = self.transition.tick(dt, self.vp_main.camera_mut());
            if changed {
                self.vp_main.mark_dirty(DirtyFlags::CAMERA);
            }
        }
        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(tex)
            | wgpu::CurrentSurfaceTexture::Suboptimal(tex) => tex,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.surface_config);
                self.trace_surface_failure(frame_start, "reconfigured");
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Timeout => {
                self.trace_surface_failure(frame_start, "timeout");
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Occluded => {
                self.trace_surface_failure(frame_start, "occluded");
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                self.trace_surface_failure(frame_start, "validation");
                return Ok(());
            }
        };
        let acquired = Instant::now();
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("iletki-demo frame"),
            });

        let w = self.surface_config.width.max(1);
        let h = self.surface_config.height.max(1);

        self.clear_stale_preview(Instant::now());
        let mut streaming = false;
        let mut dynamic_drawn = false;
        if (self.quad_view || self.bench_dual) && self.active_scene == DemoScene::Arch3D {
            let qw = (w / 2) as f32;
            let qh = (h / 2) as f32;
            let rw = (w - w / 2) as f32;
            let rh = (h - h / 2) as f32;
            let mut views = [
                (
                    &mut self.vp_top,
                    [0.0, 0.0, qw, if self.bench_dual { h as f32 } else { qh }],
                ),
                (
                    &mut self.vp_main,
                    [qw, 0.0, rw, if self.bench_dual { h as f32 } else { qh }],
                ),
                (&mut self.vp_front, [0.0, qh, qw, rh]),
                (&mut self.vp_right, [qw, qh, rw, rh]),
            ];
            let count = if self.bench_dual { 2 } else { 4 };
            for (viewport, _) in views.iter_mut().take(count) {
                let stats = viewport.prepare_for_pass(
                    &self.device,
                    &self.queue,
                    &mut enc,
                    &(),
                    &mut [&mut self.scene_3d, &mut self.grid_3d, &mut self.gizmo_3d],
                    self.pass_desc,
                )?;
                streaming |= stats.streaming_pending;
                dynamic_drawn |= stats.dynamic_drawn;
            }
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("iletki-demo quad blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            for (viewport, [x, y, width, height]) in views.iter().take(count) {
                pass.set_viewport(*x, *y, *width, *height, 0.0, 1.0);
                viewport.paint_into_pass(&mut pass, self.pass_desc);
            }
        } else {
            let stats = match self.active_scene {
                DemoScene::Arch3D => {
                    let mut layers: [&mut dyn ViewportLayer<()>; 3] =
                        [&mut self.scene_3d, &mut self.grid_3d, &mut self.gizmo_3d];
                    self.vp_main.prepare_for_pass(
                        &self.device,
                        &self.queue,
                        &mut enc,
                        &(),
                        &mut layers,
                        self.pass_desc,
                    )?
                }
                DemoScene::Cad2D => {
                    let mut layers: [&mut dyn ViewportLayer<()>; 1] = [&mut self.scene_2d];
                    self.vp_main.prepare_for_pass(
                        &self.device,
                        &self.queue,
                        &mut enc,
                        &(),
                        &mut layers,
                        self.pass_desc,
                    )?
                }
                DemoScene::Instanced100k => {
                    let mut layers: [&mut dyn ViewportLayer<()>; 1] = [&mut self.instanced_3d];
                    self.vp_main.prepare_for_pass(
                        &self.device,
                        &self.queue,
                        &mut enc,
                        &(),
                        &mut layers,
                        self.pass_desc,
                    )?
                }
            };
            streaming |= stats.streaming_pending;
            dynamic_drawn |= stats.dynamic_drawn;
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("iletki-demo blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.vp_main.paint_into_pass(&mut pass, self.pass_desc);
            if let Some(rect) = self.marquee_rect {
                marquee_border(&mut pass, &self.marquee_border, rect);
            }
        }

        let viewport_encoded = self.latency.as_ref().map(|_| Instant::now());
        if let Some(trace) = &self.latency
            && w >= 48
            && h >= 8
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("latency optical marker"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_viewport(0.0, 0.0, 48.0, 8.0, 0.0, 1.0);
            pass.set_pipeline(&trace.marker);
            let id = trace.frame_id();
            pass.draw(0..3, id..id.saturating_add(1));
        }
        let commands = enc.finish();
        let encoded = Instant::now();
        self.queue.submit([commands]);
        if self.preview_active() {
            self.frames_in_flight.fetch_add(1, Ordering::Relaxed);
            let in_flight = self.frames_in_flight.clone();
            self.queue.on_submitted_work_done(move || {
                in_flight.fetch_sub(1, Ordering::Relaxed);
            });
        }
        let submitted = self.latency.as_ref().map(|_| Instant::now());
        if let Some(trace) = &self.latency {
            trace.track_gpu(&self.queue, trace.frame_id());
        }
        if self.latency.as_ref().is_none_or(|trace| {
            trace.options.notify || trace.options.action == latency::Action::Live
        }) {
            self.window.pre_present_notify();
        }
        self.queue.present(output);
        let presented = Instant::now();
        let preview_view_changed = self.latency.is_some() && self.preview_deadline().is_some();
        if let (Some(trace), Some(viewport_encoded), Some(submitted)) =
            (&mut self.latency, viewport_encoded, submitted)
        {
            trace.frame(latency::Frame {
                id: trace.frame_id(),
                started: trace.ns(frame_start),
                acquired: trace.ns(acquired),
                encoded: trace.ns(encoded),
                viewport_encoded: trace.ns(viewport_encoded),
                submitted: trace.ns(submitted),
                presented: trace.ns(presented),
                status: "present-called",
                streaming,
                dynamic_drawn,
                preview_view_changed,
            });
        }

        #[cfg(feature = "profiling")]
        if self.pick_profile && iletki::gpu::profiling::active() && self.pending_pick.is_none() {
            match iletki::gpu::profiling::finish(&self.device, &self.queue) {
                Ok(capture) => {
                    for (name, elapsed) in capture.cpu {
                        println!("profile,cpu,{name},{}", elapsed.as_nanos());
                    }
                    for (name, elapsed) in capture.gpu {
                        println!("profile,gpu,{name},{}", elapsed.as_nanos());
                    }
                    for (name, count) in capture.counters {
                        println!("profile,count,{name},{count}");
                    }
                }
                Err(error) => eprintln!("pick profile failed: {error}"),
            }
        }

        // Keep the startup frame and any previous burst of on-demand redraws out of the
        // timing shown for a fresh interaction.
        let reset_timing = self.timing_samples < 2 || frame_gap >= TITLE_UPDATE_INTERVAL;
        let smooth = |previous: f64, elapsed: Duration| {
            let ms = elapsed.as_secs_f64() * 1000.0;
            if reset_timing {
                ms
            } else {
                previous * 0.9 + ms * 0.1
            }
        };
        self.smoothed_frame_ms = smooth(
            self.smoothed_frame_ms,
            presented.duration_since(frame_start),
        );
        self.smoothed_encode_ms = smooth(self.smoothed_encode_ms, encoded.duration_since(acquired));
        self.timing_samples = self.timing_samples.saturating_add(1).min(2);
        if reset_timing || self.title_updated_at.elapsed() >= TITLE_UPDATE_INTERVAL {
            self.update_title();
            self.title_updated_at = Instant::now();
        }
        if streaming
            || self.vp_main.mode() == ViewportMode::Continuous
            || self.transition.is_active()
        {
            self.window.request_redraw();
        }
        Ok(())
    }
}

struct DemoApp {
    state: Option<GpuState>,
    bench: Option<latency::Options>,
}

impl ApplicationHandler for DemoApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let [width, height] = self
            .bench
            .as_ref()
            .map_or([1280, 800], latency::Options::dimensions);
        let mut attrs = WindowAttributes::default()
            .with_title("iletki-demo")
            .with_inner_size(PhysicalSize::new(width, height));
        if let Some(bench) = self.bench.as_ref().filter(|bench| bench.full_hd) {
            let monitor = if let Some(name) = &bench.monitor {
                let Some(monitor) = event_loop
                    .available_monitors()
                    .find(|monitor| monitor.name().as_ref() == Some(name))
                else {
                    eprintln!("latency benchmark monitor not found: {name}");
                    event_loop.exit();
                    return;
                };
                Some(monitor)
            } else {
                event_loop.primary_monitor()
            };
            attrs = attrs.with_fullscreen(Some(Fullscreen::Borderless(monitor)));
        }
        let Ok(window) = event_loop.create_window(attrs) else {
            event_loop.exit();
            return;
        };
        match GpuState::new(Arc::new(window), self.bench.take()) {
            Ok(state) => {
                state.window.request_redraw();
                self.state = Some(state);
            }
            Err(err) => {
                eprintln!("iletki-demo init failed: {err}");
                event_loop.exit();
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(state) = &mut self.state {
            state.poll_pick();
            let now = Instant::now();
            state.replay_latency(now);
            state.submit_queued_pick(Instant::now());
            let mut deadline = state
                .pending_pick
                .as_ref()
                .map(|_| Instant::now() + PICK_POLL_INTERVAL)
                .or_else(|| state.queued_pick_deadline());
            if state.deferred_redraw {
                let _ = state.device.poll(wgpu::PollType::Poll);
                if state.render_backlogged(Instant::now()) {
                    let poll = Instant::now() + PICK_POLL_INTERVAL;
                    deadline = Some(deadline.map_or(poll, |existing| existing.min(poll)));
                } else {
                    state.window.request_redraw();
                }
            }
            if let Some(expiry) = state.preview_deadline() {
                deadline = Some(deadline.map_or(expiry, |existing| existing.min(expiry)));
            }
            if let Some(trace) = &state.latency {
                if trace.pending_gpu() {
                    let _ = state.device.poll(wgpu::PollType::Poll);
                }
                if trace.finished(
                    now,
                    state.pending_pick.is_some() || state.queued_pick.is_some(),
                ) {
                    if let Err(error) = trace.write(&state.window, state.vp_main.camera()) {
                        eprintln!("latency trace write failed: {error}");
                    }
                    event_loop.exit();
                    return;
                }
                if let Some(next) = trace.deadline(now) {
                    deadline = Some(deadline.map_or(next, |existing| existing.min(next)));
                }
            }
            event_loop.set_control_flow(deadline.map_or(ControlFlow::Wait, ControlFlow::WaitUntil));
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(s) = &mut self.state else {
            return;
        };
        let replay = s
            .latency
            .as_ref()
            .is_some_and(|trace| trace.options.action != latency::Action::Live);
        match event {
            WindowEvent::CloseRequested => {
                if let Some(trace) = &s.latency
                    && let Err(error) = trace.write(&s.window, s.vp_main.camera())
                {
                    eprintln!("latency trace write failed: {error}");
                }
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if [size.width, size.height] != [s.surface_config.width, s.surface_config.height] {
                    s.cancel_selection();
                }
                s.resize(size);
                s.window.request_redraw();
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                s.cancel_selection();
                s.resize_viewports();
            }
            WindowEvent::Focused(false) if !replay => {
                s.press_pos = None;
                s.cancel_selection();
                s.dispatch_input(&ViewportInput::FocusLost);
            }
            WindowEvent::ModifiersChanged(m) if !replay => {
                let st = m.state();
                s.modifiers = Modifiers {
                    shift: st.shift_key(),
                    ctrl: st.control_key(),
                    alt: st.alt_key(),
                    meta: st.super_key(),
                };
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                if replay {
                    if event.physical_key == PhysicalKey::Code(KeyCode::Escape) {
                        event_loop.exit();
                    }
                    return;
                }
                if let PhysicalKey::Code(code) = event.physical_key {
                    match code {
                        KeyCode::Escape
                            if s.marquee_origin.is_some()
                                || s.preview_view.is_some()
                                || s.queued_pick
                                    .as_ref()
                                    .is_some_and(|request| request.kind == PickKind::Commit)
                                || s.pending_pick.as_ref().is_some_and(|pending| {
                                    pending.request.kind == PickKind::Commit
                                }) =>
                        {
                            s.cancel_selection()
                        }
                        KeyCode::Escape => event_loop.exit(),
                        KeyCode::Digit1 => {
                            s.cancel_selection();
                            s.active_scene = DemoScene::Arch3D;
                            s.reset_cameras();
                        }
                        KeyCode::Digit2 => {
                            s.cancel_selection();
                            s.active_scene = DemoScene::Cad2D;
                            s.reset_cameras();
                        }
                        KeyCode::Digit3 => {
                            s.cancel_selection();
                            s.active_scene = DemoScene::Instanced100k;
                            s.reset_cameras();
                        }
                        KeyCode::KeyQ => {
                            if s.active_scene == DemoScene::Arch3D {
                                s.cancel_selection();
                                s.quad_view = !s.quad_view;
                                s.resize_viewports();
                            }
                        }
                        KeyCode::KeyC => {
                            s.clip_enabled = !s.clip_enabled;
                            s.apply_clip_planes();
                        }
                        KeyCode::KeyM => {
                            let next = match s.vp_main.mode() {
                                ViewportMode::OnDemand => ViewportMode::Continuous,
                                ViewportMode::Continuous => ViewportMode::OnDemand,
                            };
                            s.vp_main.set_mode(next);
                        }
                        KeyCode::KeyR => {
                            s.transition.cancel();
                            s.reset_cameras();
                        }
                        KeyCode::KeyF => s.animate_to_view(None),
                        KeyCode::KeyT => s.animate_to_view(Some(StandardView::Top)),
                        KeyCode::KeyV => s.animate_to_view(Some(StandardView::Isometric)),
                        _ => {}
                    }
                    s.update_title();
                    s.window.request_redraw();
                }
            }
            WindowEvent::MouseInput { state, button, .. } if !replay => {
                let mapped = match button {
                    MouseButton::Middle | MouseButton::Right => Some(PointerButton::Middle),
                    MouseButton::Left => Some(PointerButton::Primary),
                    _ => None,
                };
                if let Some(btn) = mapped {
                    if btn == PointerButton::Middle && state == ElementState::Pressed {
                        s.transition.cancel();
                    }
                    if btn == PointerButton::Primary {
                        if state == ElementState::Pressed
                            && s.modifiers.shift
                            && s.active_scene == DemoScene::Arch3D
                            && !s.quad_view
                            && !s.bench_dual
                        {
                            s.cancel_selection();
                            s.transition.cancel();
                            s.press_pos = None;
                            s.marquee_origin = Some(s.last_cursor);
                            return;
                        }
                        if state == ElementState::Released
                            && let Some(origin) = s.marquee_origin.take()
                        {
                            s.update_marquee(origin, PickKind::Commit, Instant::now());
                            return;
                        }
                        match state {
                            ElementState::Pressed => s.press_pos = Some(s.last_cursor),
                            ElementState::Released => {
                                if let Some(down) = s.press_pos.take()
                                    && (s.last_cursor - down).length() < 5.0
                                {
                                    s.handle_pick(s.last_cursor);
                                }
                            }
                        }
                    }
                    let ev = match state {
                        ElementState::Pressed => ViewportInput::PointerDown {
                            pos: s.last_cursor,
                            button: btn,
                            modifiers: s.modifiers,
                        },
                        ElementState::Released => ViewportInput::PointerUp {
                            pos: s.last_cursor,
                            button: btn,
                            modifiers: s.modifiers,
                        },
                    };
                    s.dispatch_input(&ev);
                    s.window.request_redraw();
                }
            }
            WindowEvent::CursorMoved { position, .. } if !replay => {
                let started = Instant::now();
                let dpi = s.window.scale_factor();
                let pos = Vec2::new(position.x / dpi, position.y / dpi);
                let delta = pos - s.last_cursor;
                s.last_cursor = pos;
                if let Some(origin) = s.marquee_origin {
                    s.update_marquee(origin, PickKind::Preview, started);
                    return;
                }
                let ev = ViewportInput::PointerMove {
                    pos,
                    delta,
                    modifiers: s.modifiers,
                };
                if s.dispatch_input(&ev) {
                    s.track_live_input("drag", started);
                }
                if s.active_scene == DemoScene::Cad2D {
                    s.update_2d_overlay(pos);
                }
                if !s.vp_main.dirty().is_empty() {
                    s.window.request_redraw();
                }
            }
            WindowEvent::MouseWheel { delta, .. } if !replay => {
                let started = Instant::now();
                s.transition.cancel();
                let (d, unit) = match delta {
                    MouseScrollDelta::LineDelta(x, y) => {
                        (Vec2::new(f64::from(x), f64::from(y)), ScrollUnit::Line)
                    }
                    MouseScrollDelta::PixelDelta(p) => (Vec2::new(p.x, p.y), ScrollUnit::Pixel),
                };
                let ev = ViewportInput::Scroll {
                    pos: s.last_cursor,
                    delta: d,
                    unit,
                    modifiers: s.modifiers,
                };
                if s.dispatch_input(&ev) {
                    s.track_live_input("zoom", started);
                }
                if s.active_scene == DemoScene::Cad2D {
                    s.update_2d_overlay(s.last_cursor);
                }
                s.window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                if let Err(error) = s.render() {
                    eprintln!("viewport rendering failed: {error}");
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }
}

fn bench_scene(
    meshes: impl IntoIterator<Item = Mesh3D>,
    edits: &mut Vec<(Chunk3DId, Vec3)>,
) -> (Scene3DLayer, Vec<ChunkMeta>) {
    let mut scene = Scene3DLayer::new(1);
    let mut meta = Vec::new();
    for (index, mesh) in meshes.into_iter().enumerate() {
        let origin = mesh.origin;
        let pick = mesh.pick_ids.first().copied().map(PickId::from_rg32u);
        let id = scene.insert(mesh);
        if index % 100 == 0 {
            edits.push((id, origin));
        }
        if let Some(pick) = pick {
            meta.push(ChunkMeta {
                pick_object: pick.object,
                chunk_id: id,
                gizmo_pos: origin,
                name: "Assembly part",
            });
        }
    }
    (scene, meta)
}

fn build_arch_3d() -> (Scene3DLayer, Vec<ChunkMeta>) {
    let mut scene = Scene3DLayer::new(1);
    let mut meta = Vec::new();
    // Use translucent dark edge lines so the surface colors show through.
    let edge = [12, 16, 24, 110];

    let mut add = |name: &'static str,
                   obj_id: u32,
                   origin: Vec3,
                   min: Vec3,
                   max: Vec3,
                   color: [u8; 4],
                   gizmo_pos: Vec3,
                   selected: bool| {
        let mut m = Mesh3D::at_origin(origin);
        m.push_box_with_edges(min, max, color, edge, PickId::new(1, obj_id, 0), true);
        let chunk_id = scene.insert(m);
        if selected {
            scene.set_selected(chunk_id, true);
        }
        meta.push(ChunkMeta {
            pick_object: obj_id,
            chunk_id,
            gizmo_pos,
            name,
        });
    };

    add(
        "Foundation Slab",
        1,
        Vec3::ZERO,
        Vec3::new(-4.5, -3.5, 0.0),
        Vec3::new(4.5, 3.5, 0.4),
        [150, 160, 175, 255],
        Vec3::new(0.0, 0.0, 0.4),
        false,
    );
    for (i, (cx, cy)) in [(-3.5, -2.5), (3.5, -2.5), (3.5, 2.5), (-3.5, 2.5)]
        .into_iter()
        .enumerate()
    {
        add(
            "Steel Column",
            10 + (i as u32),
            Vec3::new(cx, cy, 0.402),
            Vec3::new(-0.35, -0.35, 0.0),
            Vec3::new(0.35, 0.35, 3.596),
            [70, 130, 210, 255],
            Vec3::new(cx, cy, 2.2),
            false,
        );
    }
    add(
        "Mezzanine Deck",
        20,
        Vec3::new(0.0, 0.0, 4.0),
        Vec3::new(-4.0, -3.0, 0.0),
        Vec3::new(4.0, 3.0, 0.45),
        [190, 195, 205, 255],
        Vec3::new(0.0, 0.0, 4.45),
        false,
    );
    add(
        "Core Block",
        99,
        Vec3::new(-0.8, 0.4, 0.402),
        Vec3::new(-1.2, -1.0, 0.0),
        Vec3::new(1.2, 1.0, 2.6),
        [65, 195, 130, 255],
        Vec3::new(-0.8, 0.4, 3.0),
        true,
    );
    add(
        "Roof Mechanical Unit",
        30,
        Vec3::new(1.8, -0.8, 4.452),
        Vec3::new(-0.9, -0.8, 0.0),
        Vec3::new(0.9, 0.8, 1.2),
        [215, 95, 80, 255],
        Vec3::new(1.8, -0.8, 5.65),
        false,
    );

    (scene, meta)
}

fn build_instanced_100k() -> InstancedMesh3DLayer {
    let mut proto = Mesh3D::at_origin(Vec3::ZERO);
    proto.push_box(
        Vec3::new(-0.35, -0.35, 0.0),
        Vec3::new(0.35, 0.35, 1.0),
        [255, 255, 255, 255],
        PickId::NONE,
        false,
    );
    let mut instances = Vec::with_capacity(100_000);
    for i in 0..100_000u32 {
        let gx = (i % 316) as f32 - 158.0;
        let gy = (i / 316) as f32 - 158.0;
        let dist = (gx * gx + gy * gy).sqrt();
        let h = 1.2
            + (dist * 0.14).cos().abs() * 5.0
            + ((gx * 0.25).sin() * (gy * 0.25).cos()).abs() * 3.0;
        let r = (70.0 + (h / 9.2) * 170.0).min(255.0) as u8;
        let g = (130.0 + (1.0 - (h / 9.2).min(1.0)) * 90.0).min(255.0) as u8;
        instances.push(Instance3D::new(
            [gx * 1.3, gy * 1.3, 0.0],
            [1.0, 1.0, h],
            [r, g, 220, 255],
            PickId::new(5, i, 0),
        ));
    }
    let mut layer = InstancedMesh3DLayer::new(5, Vec3::ZERO, proto);
    layer.set_instances(instances);
    layer
}

fn build_cad_2d(origin: Vec2, linear: bool) -> Mesh2D {
    let mut mesh = Mesh2D::default();
    let style_grid = DrawStyle2D::new(
        [90, 180, 255, 220],
        linear,
        order_to_depth(10, 100),
        PickId::NONE,
    );
    for i in 0..500_000u32 {
        let bx = f64::from(i % 500) * 2.0;
        let by = f64::from(i / 500) * 1.0;
        let p0 = origin + Vec2::new(bx, by);
        let p1 = if i % 2 == 0 {
            p0 + Vec2::new(1.8, 0.0)
        } else {
            p0 + Vec2::new(0.0, 0.9)
        };
        if i % 25 == 0 {
            mesh.push_wide_line(
                p0,
                p1,
                2.5,
                DrawStyle2D::new(
                    [235, 220, 130, 255],
                    linear,
                    order_to_depth(30, 100),
                    PickId::new(4, i, 0),
                ),
            );
        } else {
            mesh.push_line(p0, p1, style_grid);
        }
    }
    mesh
}

fn main() {
    println!(
        "iletki-demo controls:\n  Middle/Right drag: Orbit (3D) / Pan (2D), Shift+drag: Pan 3D\n  Scroll: Zoom/Dolly | Left-click: 64-bit GPU Pick + JFA Outline\n  [1] 3D Arch  [2] 2D 500k UTM CAD  [3] 100k 3D Instances\n  [Q] 4-Way Quad-View  [C] Section Clip  [M] OnDemand/Continuous  [R] Reset"
    );
    let Ok(event_loop) = EventLoop::new() else {
        eprintln!("failed to create winit event loop");
        return;
    };
    event_loop.set_control_flow(ControlFlow::Wait);
    let bench = match latency::Options::parse() {
        Ok(options) => options,
        Err(error) => {
            eprintln!("{error}");
            return;
        }
    };
    let mut app = DemoApp { state: None, bench };
    if let Err(err) = event_loop.run_app(&mut app) {
        eprintln!("event loop error: {err}");
    }
}
