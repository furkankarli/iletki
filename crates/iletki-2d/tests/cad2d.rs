//! Headless GPU integration tests for `iletki-2d`:
//! - Split `hi/lo` rendering at national grid coordinates `(661_858.125, 4_491_159.875)`
//! - Incremental `PagedBuffer` chunk update without rebuilding unrelated chunks
//! - Dynamic overlay compositing with 0 static scene redraws
//! - 64-bit `PickId` GPU picking on 2D geometry at national grid coordinates

#![cfg(not(target_arch = "wasm32"))]

use iletki_2d::{DrawStyle2D, Mesh2D, Scene2DLayer, order_to_depth};
use iletki_core::{Camera, DirtyFlags, PickId, Vec2, ViewportMode};
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
        ..wgpu::DeviceDescriptor::default()
    }))
    .ok()
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
    host.mark_dirty(DirtyFlags::STATIC_SCENE);

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
    host.mark_dirty(DirtyFlags::OVERLAY);

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
