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
//! - Left click: 64-bit GPU Pick (highlights selected 3D chunk with Jump-Flood outline + moves gizmo)
//! - `1` : 3D Architectural Scene (Shaded with Edges + JFA Outline + Grid + Gizmo)
//! - `2` : 2D National-Grid CAD Stress (500,000 segments at UTM 661km / 4491km)
//! - `3` : 100,000-Instance 3D City Stress (1.2M triangles in 1 draw call)
//! - `Q` : Toggle 4-Way CAD Quad-View (Top / Persp / Front / Right in 1 CommandEncoder)
//! - `C` : Toggle 3D Section Clip Plane
//! - `M` : Toggle `OnDemand` (0 static draws on idle) vs `Continuous`
//! - `R` : Reset camera

#![allow(clippy::print_stdout, clippy::print_stderr)]

use iletki::gpu::wgpu;
use iletki::{
    Bounds2, Bounds3, Cad2DRig, Camera, CameraTransition, Chunk3DId, DirtyFlags, DrawStyle2D,
    Grid3DConfig, GridStyle, InfiniteGrid3DLayer, Instance3D, InstancedMesh3DLayer, Mesh2D, Mesh3D,
    Modifiers, OrbitRig, OutlineConfig, PassTargetDesc, PickId, Plane, PointerButton, Projection,
    Scene2DLayer, Scene3DLayer, ScrollUnit, StandardView, TransformGizmoLayer, Vec2, Vec3,
    ViewportConfig, ViewportHost, ViewportInput, ViewportLayer, ViewportMode, order_to_depth,
};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowAttributes, WindowId};

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
    last_pick_text: String,
    frame_count: u32,
    fps_timer: Instant,
    smoothed_ms: f64,
}

impl GpuState {
    fn new(window: Arc<Window>) -> Result<Self, String> {
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

        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("iletki-demo device"),
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
        let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
            wgpu::PresentMode::Mailbox
        } else {
            wgpu::PresentMode::Fifo
        };

        let mut surface_config = surface
            .get_default_config(&adapter, width, height)
            .ok_or_else(|| "surface reported no default configuration".to_string())?;
        surface_config.format = format;
        surface_config.present_mode = present_mode;
        surface.configure(&device, &surface_config);

        let make_vp = |px: [u32; 2]| {
            ViewportHost::new(
                &device,
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

        let (scene_3d, chunks_3d) = build_arch_3d();
        let grid_3d = InfiniteGrid3DLayer::new(2, Grid3DConfig::default());
        let mut gizmo_3d = TransformGizmoLayer::new(3, 1);
        gizmo_3d.origin = Some(Vec3::new(-0.8, 0.4, 3.0));

        let instanced_3d = build_instanced_100k();

        let origin_2d = Vec2::new(661_858.0, 4_491_159.0);
        let mut scene_2d = Scene2DLayer::new(4);
        let mut mesh_2d = build_cad_2d(origin_2d, format.is_srgb());
        mesh_2d.optimize_spatial();
        scene_2d.upload_static_mesh(&device, &queue, &mesh_2d);

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
            grid_3d,
            gizmo_3d,
            instanced_3d,
            scene_2d,
            origin_2d,
            active_scene: DemoScene::Arch3D,
            quad_view: false,
            clip_enabled: false,
            modifiers: Modifiers::default(),
            last_cursor: Vec2::ZERO,
            press_pos: None,
            last_pick_text: String::from("Click any 3D object to pick"),
            frame_count: 0,
            fps_timer: Instant::now(),
            smoothed_ms: 0.0,
        };
        state.reset_cameras();
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
            Plane::from_normal_point(Vec3::Y, Vec3::new(0.0, -0.35, 0.0))
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
        if self.quad_view && self.active_scene == DemoScene::Arch3D {
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

    fn handle_pick(&mut self, logical_pos: Vec2) {
        if self.quad_view {
            return;
        }
        let dpi = f64::from(self.window.scale_factor() as f32);
        let px = (logical_pos.x * dpi).round().max(0.0) as u32;
        let py = (logical_pos.y * dpi).round().max(0.0) as u32;
        let rect = [px.saturating_sub(2), py.saturating_sub(2), 5, 5];

        let hits = match self.active_scene {
            DemoScene::Arch3D => {
                let mut layers: [&mut dyn ViewportLayer<()>; 3] =
                    [&mut self.scene_3d, &mut self.grid_3d, &mut self.gizmo_3d];
                self.vp_main
                    .pick_rect_blocking(&self.device, &self.queue, &(), &mut layers, rect)
            }
            DemoScene::Instanced100k => {
                let mut layers: [&mut dyn ViewportLayer<()>; 1] = [&mut self.instanced_3d];
                self.vp_main
                    .pick_rect_blocking(&self.device, &self.queue, &(), &mut layers, rect)
            }
            DemoScene::Cad2D => {
                let mut layers: [&mut dyn ViewportLayer<()>; 1] = [&mut self.scene_2d];
                self.vp_main
                    .pick_rect_blocking(&self.device, &self.queue, &(), &mut layers, rect)
            }
        };

        if let Ok(hits) = hits {
            if let Some(hit) = hits.first() {
                if self.active_scene == DemoScene::Arch3D && hit.id.layer == 1 {
                    self.scene_3d.clear_selection();
                    if let Some(meta) = self
                        .chunks_3d
                        .iter()
                        .find(|m| m.pick_object == hit.id.object)
                    {
                        self.scene_3d.set_selected(meta.chunk_id, true);
                        self.gizmo_3d.origin = Some(meta.gizmo_pos);
                        self.last_pick_text = format!(
                            "Picked {} (obj={} face={})",
                            meta.name, hit.id.object, hit.id.sub
                        );
                    }
                    self.vp_main
                        .mark_dirty(DirtyFlags::OVERLAY.union(DirtyFlags::GIZMO));
                } else {
                    self.last_pick_text = format!(
                        "Picked layer={} obj={} sub={}",
                        hit.id.layer, hit.id.object, hit.id.sub
                    );
                }
            } else {
                if self.active_scene == DemoScene::Arch3D {
                    self.scene_3d.clear_selection();
                    self.gizmo_3d.origin = None;
                    self.vp_main
                        .mark_dirty(DirtyFlags::OVERLAY.union(DirtyFlags::GIZMO));
                }
                self.last_pick_text = String::from("Empty background");
            }
            self.update_title();
            self.window.request_redraw();
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
        self.window.set_title(&format!(
            "iletki-demo | {scene_name} | {mode}{clip} | {:.2} ms | {} | [1/2/3 Q C M R]",
            self.smoothed_ms, self.last_pick_text
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

    fn render(&mut self) -> Result<(), iletki::ViewportError> {
        let frame_start = Instant::now();
        let dt = self.last_frame_time.elapsed().as_secs_f64().min(0.1);
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
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => return Ok(()),
        };
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

        if self.quad_view && self.active_scene == DemoScene::Arch3D {
            {
                let mut layers: [&mut dyn ViewportLayer<()>; 3] =
                    [&mut self.scene_3d, &mut self.grid_3d, &mut self.gizmo_3d];
                self.vp_top.prepare_for_pass(
                    &self.device,
                    &self.queue,
                    &mut enc,
                    &(),
                    &mut layers,
                    self.pass_desc,
                )?;
                self.vp_main.prepare_for_pass(
                    &self.device,
                    &self.queue,
                    &mut enc,
                    &(),
                    &mut layers,
                    self.pass_desc,
                )?;
                self.vp_front.prepare_for_pass(
                    &self.device,
                    &self.queue,
                    &mut enc,
                    &(),
                    &mut layers,
                    self.pass_desc,
                )?;
                self.vp_right.prepare_for_pass(
                    &self.device,
                    &self.queue,
                    &mut enc,
                    &(),
                    &mut layers,
                    self.pass_desc,
                )?;
            }
            let qw = (w / 2) as f32;
            let qh = (h / 2) as f32;
            let rw = (w - w / 2) as f32;
            let rh = (h - h / 2) as f32;
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
            pass.set_viewport(0.0, 0.0, qw, qh, 0.0, 1.0);
            self.vp_top.paint_into_pass(&mut pass, self.pass_desc);
            pass.set_viewport(qw, 0.0, rw, qh, 0.0, 1.0);
            self.vp_main.paint_into_pass(&mut pass, self.pass_desc);
            pass.set_viewport(0.0, qh, qw, rh, 0.0, 1.0);
            self.vp_front.paint_into_pass(&mut pass, self.pass_desc);
            pass.set_viewport(qw, qh, rw, rh, 0.0, 1.0);
            self.vp_right.paint_into_pass(&mut pass, self.pass_desc);
        } else {
            match self.active_scene {
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
                    )?;
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
                    )?;
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
                    )?;
                }
            }
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
        }

        self.queue.submit(std::iter::once(enc.finish()));
        self.queue.present(output);

        let ms = frame_start.elapsed().as_secs_f64() * 1000.0;
        self.smoothed_ms = if self.smoothed_ms == 0.0 {
            ms
        } else {
            self.smoothed_ms * 0.9 + ms * 0.1
        };
        self.frame_count += 1;
        if self.fps_timer.elapsed().as_secs_f64() >= 0.35 {
            self.update_title();
            self.frame_count = 0;
            self.fps_timer = Instant::now();
        }
        if self.vp_main.mode() == ViewportMode::Continuous || self.transition.is_active() {
            self.window.request_redraw();
        }
        Ok(())
    }
}

struct DemoApp {
    state: Option<GpuState>,
}

impl ApplicationHandler for DemoApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let attrs = WindowAttributes::default()
            .with_title("iletki-demo")
            .with_inner_size(PhysicalSize::new(1280u32, 800u32));
        let Ok(window) = event_loop.create_window(attrs) else {
            event_loop.exit();
            return;
        };
        match GpuState::new(Arc::new(window)) {
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

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(s) = &mut self.state else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                s.resize(size);
                s.window.request_redraw();
            }
            WindowEvent::ModifiersChanged(m) => {
                let st = m.state();
                s.modifiers = Modifiers {
                    shift: st.shift_key(),
                    ctrl: st.control_key(),
                    alt: st.alt_key(),
                    meta: st.super_key(),
                };
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    match code {
                        KeyCode::Escape => event_loop.exit(),
                        KeyCode::Digit1 => {
                            s.active_scene = DemoScene::Arch3D;
                            s.reset_cameras();
                        }
                        KeyCode::Digit2 => {
                            s.active_scene = DemoScene::Cad2D;
                            s.reset_cameras();
                        }
                        KeyCode::Digit3 => {
                            s.active_scene = DemoScene::Instanced100k;
                            s.reset_cameras();
                        }
                        KeyCode::KeyQ => {
                            if s.active_scene == DemoScene::Arch3D {
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
            WindowEvent::MouseInput { state, button, .. } => {
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
                    if s.active_scene == DemoScene::Cad2D {
                        s.vp_main.on_input(&ev, &mut s.cad2d_rig);
                    } else {
                        s.vp_main.on_input(&ev, &mut s.orbit_rig);
                    }
                    s.window.request_redraw();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let dpi = s.window.scale_factor();
                let pos = Vec2::new(position.x / dpi, position.y / dpi);
                let delta = pos - s.last_cursor;
                s.last_cursor = pos;
                let ev = ViewportInput::PointerMove {
                    pos,
                    delta,
                    modifiers: s.modifiers,
                };
                if s.active_scene == DemoScene::Cad2D {
                    s.vp_main.on_input(&ev, &mut s.cad2d_rig);
                    s.update_2d_overlay(pos);
                } else {
                    s.vp_main.on_input(&ev, &mut s.orbit_rig);
                }
                if !s.vp_main.dirty().is_empty() {
                    s.window.request_redraw();
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
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
                if s.active_scene == DemoScene::Cad2D {
                    s.vp_main.on_input(&ev, &mut s.cad2d_rig);
                    s.update_2d_overlay(s.last_cursor);
                } else {
                    s.vp_main.on_input(&ev, &mut s.orbit_rig);
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

fn build_arch_3d() -> (Scene3DLayer, Vec<ChunkMeta>) {
    let mut scene = Scene3DLayer::new(1);
    let mut meta = Vec::new();
    // Translucent dark crease color blends cleanly with each surface's own hue (CAD/Fusion style).
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
    let mut app = DemoApp { state: None };
    if let Err(err) = event_loop.run_app(&mut app) {
        eprintln!("event loop error: {err}");
    }
}
