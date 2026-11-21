//! Headless GPU integration tests for `iletki-2d`:
//! - Split `hi/lo` rendering at national grid coordinates `(661_858.125, 4_491_159.875)`
//! - Incremental `PagedBuffer` chunk update without rebuilding unrelated chunks
//! - Dynamic overlay compositing with 0 static scene redraws
//! - 64-bit `PickId` GPU picking on 2D geometry at national grid coordinates

#![cfg(not(target_arch = "wasm32"))]

use iletki_2d::{DrawStyle2D, Mesh2D, Scene2DLayer, order_to_depth};
use iletki_core::{Camera, PickId, UploadBudget, Vec2, ViewportMode};
use iletki_wgpu::{ViewportConfig, ViewportHost, ViewportLayer, wgpu};
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

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

fn create_headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    create_headless_device_with_limits(wgpu::Limits::default())
}

fn create_headless_device_with_limits(
    required_limits: wgpu::Limits,
) -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
        ..wgpu::RequestAdapterOptions::default()
    }))
    .or_else(|_| {
        block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: true,
            ..wgpu::RequestAdapterOptions::default()
        }))
    })
    .ok()?;
    block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("iletki-2d test device"),
        required_limits,
        ..wgpu::DeviceDescriptor::default()
    }))
    .ok()
}

#[test]
fn picking_matches_geometry_across_pages_and_unpickable_chunks() {
    let Some((device, queue)) = create_headless_device_with_limits(wgpu::Limits {
        max_buffer_size: 16 * 1024,
        ..wgpu::Limits::default()
    }) else {
        return;
    };
    let mut host = ViewportHost::new(
        &device,
        ViewportConfig {
            initial_px: [64, 64],
            msaa_samples: 1,
            ..ViewportConfig::default()
        },
    );
    let wanted = PickId::new(1, 77, 2);
    let mut scene = Scene2DLayer::new(1);
    let mut mesh = Mesh2D::default();
    let style = |id| DrawStyle2D::new([255, 0, 0, 255], false, 0.5, id);
    for _ in 0..114 {
        mesh.push_rect_fill(
            Vec2::new(-1000.0, -10.0),
            Vec2::new(-990.0, 10.0),
            style(PickId::NONE),
        );
    }
    mesh.push_rect_fill(
        Vec2::new(-10.0, -10.0),
        Vec2::new(10.0, 10.0),
        style(wanted),
    );
    scene.upload_static_mesh(&device, &queue, &mesh);

    let pick_center = |host: &mut ViewportHost, scene: &mut Scene2DLayer| {
        let mut layers: [&mut dyn ViewportLayer<()>; 1] = [scene];
        let mut encoder = device.create_command_encoder(&Default::default());
        host.render_to_texture(&device, &queue, &mut encoder, &(), &mut layers);
        queue.submit([encoder.finish()]);
        host.pick_rect_blocking(&device, &queue, &(), &mut layers, [32, 32, 1, 1])
            .unwrap()
    };
    assert_eq!(
        pick_center(&mut host, &mut scene).first().map(|hit| hit.id),
        Some(wanted)
    );

    scene.set_static_mesh(&Mesh2D::default());
    let mut unpickable = Mesh2D::default();
    unpickable.push_rect_fill(
        Vec2::new(-30.0, -10.0),
        Vec2::new(-20.0, 10.0),
        style(PickId::NONE),
    );
    scene.insert_chunk(&unpickable);
    let mut pickable = Mesh2D::default();
    pickable.push_rect_fill(
        Vec2::new(-10.0, -10.0),
        Vec2::new(10.0, 10.0),
        style(wanted),
    );
    let handle = scene.insert_chunk(&pickable);
    assert_eq!(
        pick_center(&mut host, &mut scene).first().map(|hit| hit.id),
        Some(wanted)
    );

    scene.remove_chunk(handle);
    let replacement = scene.insert_chunk(&pickable);
    assert_ne!(replacement, handle);
    scene.remove_chunk(handle);
    pickable.triangle_pick_ids.clear();
    scene.update_chunk(handle, &pickable);
    assert_eq!(
        pick_center(&mut host, &mut scene).first().map(|hit| hit.id),
        Some(wanted)
    );
    assert_eq!(scene.update_chunk(replacement, &pickable), replacement);
    assert!(pick_center(&mut host, &mut scene).is_empty());
}

#[test]
fn national_grid_2d_render_incremental_update_overlay_and_pick() {
    let Some((device, queue)) = create_headless_device() else {
        return;
    };

    // National grid coordinates far from the origin (where plain f32 loses precision).
    let origin = Vec2::new(661_858.125, 4_491_159.875);

    let mut host = ViewportHost::new(
        &device,
        ViewportConfig {
            color_format: wgpu::TextureFormat::Rgba8Unorm,
            msaa_samples: 1,
            clear_color: [0.0, 0.0, 0.0, 1.0],
            mode: ViewportMode::OnDemand,
            initial_px: [128, 128],
            ..ViewportConfig::default()
        },
    );
    *host.camera_mut() = Camera::ortho_2d(origin, 10.0);

    let mut scene = Scene2DLayer::new(1);
    let left_id = PickId::new(1, 101, 0);
    let right_id = PickId::new(1, 202, 3);

    // Insert two independent incremental chunks (left green box, right blue box).
    let mut left_mesh = Mesh2D::default();
    left_mesh.push_rect_fill(
        origin + Vec2::new(-5.0, -3.0),
        origin + Vec2::new(-1.0, 3.0),
        DrawStyle2D::new([0, 255, 0, 255], false, order_to_depth(0, 10), left_id),
    );
    let left_handle = scene.insert_chunk(&left_mesh);

    let mut right_mesh = Mesh2D::default();
    right_mesh.push_rect_fill(
        origin + Vec2::new(1.0, -3.0),
        origin + Vec2::new(5.0, 3.0),
        DrawStyle2D::new([0, 0, 255, 255], false, order_to_depth(1, 10), right_id),
    );
    let _right_handle = scene.insert_chunk(&right_mesh);

    // Frame 1: renders both static chunks.
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut layers: [&mut dyn ViewportLayer<()>; 1] = [&mut scene];
        let out = host.render_to_texture(&device, &queue, &mut enc, &(), &mut layers);
        assert!(out.stats.static_rebuilt);
    }
    queue.submit(std::iter::once(enc.finish()));

    let img1 = host.readback_rgba_blocking(&device, &queue).unwrap();
    // x = 64 - 30 = 34 hits left box (-3 world units); x = 64 + 30 = 94 hits right box (+3).
    assert_eq!(img1.pixel(34, 64), Some([0, 255, 0, 255]));
    assert_eq!(img1.pixel(94, 64), Some([0, 0, 255, 255]));
    assert_eq!(img1.pixel(64, 64), Some([0, 0, 0, 255]));

    // Pick left and right chunks at national grid coordinates!
    let left_hits = {
        let mut layers: [&mut dyn ViewportLayer<()>; 1] = [&mut scene];
        host.pick_rect_blocking(&device, &queue, &(), &mut layers, [32, 62, 5, 5])
            .unwrap()
    };
    assert_eq!(left_hits.len(), 1);
    assert_eq!(left_hits[0].id, left_id);

    let right_hits = {
        let mut layers: [&mut dyn ViewportLayer<()>; 1] = [&mut scene];
        host.pick_rect_blocking(&device, &queue, &(), &mut layers, [92, 62, 5, 5])
            .unwrap()
    };
    assert_eq!(right_hits.len(), 1);
    assert_eq!(right_hits[0].id, right_id);

    // Incremental in-place update: turn ONLY the left chunk yellow.
    let mut left_yellow = Mesh2D::default();
    left_yellow.push_rect_fill(
        origin + Vec2::new(-5.0, -3.0),
        origin + Vec2::new(-1.0, 3.0),
        DrawStyle2D::new([255, 255, 0, 255], false, order_to_depth(0, 10), left_id),
    );
    let _left_handle = scene.update_chunk(left_handle, &left_yellow);

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut layers: [&mut dyn ViewportLayer<()>; 1] = [&mut scene];
        host.render_to_texture(&device, &queue, &mut enc, &(), &mut layers);
    }
    queue.submit(std::iter::once(enc.finish()));

    let img2 = host.readback_rgba_blocking(&device, &queue).unwrap();
    assert_eq!(img2.pixel(34, 64), Some([255, 255, 0, 255]));
    assert_eq!(img2.pixel(94, 64), Some([0, 0, 255, 255]));

    // Dynamic overlay (red wide horizontal line across center): MUST NOT rebuild static scene!
    let mut overlay = Mesh2D::default();
    overlay.push_wide_line(
        origin + Vec2::new(-6.0, 0.0),
        origin + Vec2::new(6.0, 0.0),
        6.0,
        DrawStyle2D::new([255, 0, 0, 255], false, 0.995, PickId::NONE),
    );
    scene.set_overlay(&overlay);

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let overlay_stats = {
        let mut layers: [&mut dyn ViewportLayer<()>; 1] = [&mut scene];
        host.render_to_texture(&device, &queue, &mut enc, &(), &mut layers)
            .stats
    };
    queue.submit(std::iter::once(enc.finish()));

    assert!(!overlay_stats.static_rebuilt);
    assert!(overlay_stats.dynamic_drawn);

    let img3 = host.readback_rgba_blocking(&device, &queue).unwrap();
    // Center is now red from wide overlay line, while above it (y=50) yellow & blue remain intact!
    assert_eq!(img3.pixel(64, 64), Some([255, 0, 0, 255]));
    assert_eq!(img3.pixel(34, 50), Some([255, 255, 0, 255]));
    assert_eq!(img3.pixel(94, 50), Some([0, 0, 255, 255]));
}

#[test]
fn progressive_uploads_refresh_every_viewport_cache() {
    let Some((device, queue)) = create_headless_device() else {
        return;
    };
    let config = ViewportConfig {
        initial_px: [128, 128],
        msaa_samples: 1,
        color_format: wgpu::TextureFormat::Rgba8Unorm,
        clear_color: [0.0, 0.0, 0.0, 1.0],
        upload_budget: UploadBudget {
            max_bytes_per_frame: 144,
            ..UploadBudget::default()
        },
        ..ViewportConfig::default()
    };
    let mut first = ViewportHost::new(&device, config);
    let mut second = ViewportHost::new(&device, config);
    let mut scene = Scene2DLayer::new(1);

    let render = |host: &mut ViewportHost, scene: &mut Scene2DLayer| {
        let mut encoder = device.create_command_encoder(&Default::default());
        let mut layers: [&mut dyn ViewportLayer<()>; 1] = [scene];
        let frame = host.render_to_texture(&device, &queue, &mut encoder, &(), &mut layers);
        let stats = frame.stats;
        queue.submit([encoder.finish()]);
        stats
    };
    render(&mut first, &mut scene);
    render(&mut second, &mut scene);

    for x in [-30.0, 30.0] {
        let mut mesh = Mesh2D::default();
        mesh.push_rect_fill(
            Vec2::new(x - 5.0, -5.0),
            Vec2::new(x + 5.0, 5.0),
            DrawStyle2D::new([255, 0, 0, 255], false, 0.5, PickId::NONE),
        );
        scene.insert_chunk(&mesh);
    }
    let partial = render(&mut first, &mut scene);
    assert!(partial.streaming_pending && partial.static_rebuilt);
    assert_eq!(
        first
            .readback_rgba_blocking(&device, &queue)
            .unwrap()
            .pixel(94, 64),
        Some([0, 0, 0, 255])
    );

    let complete = render(&mut second, &mut scene);
    assert!(!complete.streaming_pending && complete.static_rebuilt);
    assert!(render(&mut first, &mut scene).static_rebuilt);
    assert_eq!(
        first
            .readback_rgba_blocking(&device, &queue)
            .unwrap()
            .pixel(94, 64),
        Some([255, 0, 0, 255])
    );
    assert!(!render(&mut first, &mut scene).static_rebuilt);
    assert!(!render(&mut second, &mut scene).static_rebuilt);
}
