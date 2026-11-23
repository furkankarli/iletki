//! Headless GPU integration tests for `iletki-3d`:
//! - Camera-relative Reversed-Z 3D cube at national-grid coordinates `(661_858, 4_491_159, 120)`
//! - 3D section clip plane slicing (`Camera::set_clip_planes`)
//! - 64-bit sub-object face picking + exact `f64` world point reconstruction from Reversed-Z depth
//! - Infinite anti-aliased ground grid and screen-size-invariant 3D transform gizmo axis picking

#![cfg(not(target_arch = "wasm32"))]

use iletki_3d::{
    GizmoAxis, Grid3DConfig, InfiniteGrid3DLayer, Mesh3D, Scene3DLayer, TransformGizmoLayer,
};
use iletki_core::{Bounds3, Camera, DirtyFlags, PickId, Plane, Vec3, ViewportMode};
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
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
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
        label: Some("iletki-3d test device"),
        ..wgpu::DeviceDescriptor::default()
    }))
    .ok()
}

#[test]
fn large_world_3d_cube_clip_plane_picking_grid_and_gizmo() {
    let Some((device, queue)) = create_headless_device() else {
        return;
    };

    // Place 3D scene far from origin (national grid coordinates) to verify camera-relative RTE.
    let origin = Vec3::new(661_858.0, 4_491_159.0, 120.0);
    let eye = origin + Vec3::new(0.0, -8.0, 0.0);

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
    )
    .unwrap();
    *host.camera_mut() =
        Camera::perspective_3d(eye, origin, Vec3::Z, std::f64::consts::FRAC_PI_3, 0.05);

    let mut scene = Scene3DLayer::new(10);
    let mut cube = Mesh3D::at_origin(origin);
    cube.push_box(
        Vec3::new(-2.0, -2.0, -2.0),
        Vec3::new(2.0, 2.0, 2.0),
        [40, 220, 80, 255],
        PickId::new(10, 777, 0),
        true,
    );
    scene.insert(cube);

    let mut grid = InfiniteGrid3DLayer::new(11, Grid3DConfig::default());
    let mut gizmo = TransformGizmoLayer::new(12, 999);

    // 1. Render cube + grid (no gizmo yet).
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut layers: [&mut dyn ViewportLayer<()>; 3] = [&mut scene, &mut grid, &mut gizmo];
        host.render_to_texture(&device, &queue, &mut enc, &(), &mut layers);
    }
    queue.submit(std::iter::once(enc.finish()));

    let img = host.readback_rgba_blocking(&device, &queue).unwrap();
    let center = img.pixel(64, 64).unwrap();
    // Green channel dominates on the lit cube center.
    assert!(center[1] > 40 && center[1] > center[0], "center={center:?}");

    // 2. Pick cube center: should hit the -Y front face (sub == 5) at world y = origin.y - 2.0!
    let hits = {
        let mut layers: [&mut dyn ViewportLayer<()>; 3] = [&mut scene, &mut grid, &mut gizmo];
        host.pick_rect_blocking(&device, &queue, &(), &mut layers, [62, 62, 5, 5])
            .unwrap()
    };
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, PickId::new(10, 777, 5));
    let hit_pos = hits[0].world_pos.expect("depth reconstructs world_pos");
    assert!(
        (hit_pos.y - (origin.y - 2.0)).abs() < 0.05,
        "expected front face y={}, got {:?}",
        origin.y - 2.0,
        hit_pos
    );

    // 3. Apply a section clip plane keeping only `y >= origin.y` (slices off front half!).
    let clip = Plane::from_normal_point(Vec3::Y, origin).unwrap();
    host.camera_mut().set_clip_planes(&[clip]);
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut layers: [&mut dyn ViewportLayer<()>; 3] = [&mut scene, &mut grid, &mut gizmo];
        host.render_to_texture(&device, &queue, &mut enc, &(), &mut layers);
    }
    queue.submit(std::iter::once(enc.finish()));

    // Picking the center now hits the BACK face (+Y, sub == 4) at y = origin.y + 2.0!
    let clipped_hits = {
        let mut layers: [&mut dyn ViewportLayer<()>; 3] = [&mut scene, &mut grid, &mut gizmo];
        host.pick_rect_blocking(&device, &queue, &(), &mut layers, [62, 62, 5, 5])
            .unwrap()
    };
    assert_eq!(clipped_hits.len(), 1);
    assert_eq!(clipped_hits[0].id, PickId::new(10, 777, 4));
    let back_pos = clipped_hits[0].world_pos.unwrap();
    assert!(
        (back_pos.y - (origin.y + 2.0)).abs() < 0.05,
        "expected back face y={}, got {:?}",
        origin.y + 2.0,
        back_pos
    );

    // 4. Enable 3D transform gizmo at `origin` (with clip plane cleared) and pick its center handle!
    host.camera_mut().clear_clip_planes();
    gizmo.origin = Some(origin);
    // Position gizmo in front of the cube so gizmo pick wins depth test cleanly.
    gizmo.origin = Some(origin + Vec3::new(0.0, -3.5, 0.0));
    host.mark_dirty(DirtyFlags::GIZMO);
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut layers: [&mut dyn ViewportLayer<()>; 3] = [&mut scene, &mut grid, &mut gizmo];
        host.render_to_texture(&device, &queue, &mut enc, &(), &mut layers);
    }
    queue.submit(std::iter::once(enc.finish()));

    let gizmo_hits = {
        let mut layers: [&mut dyn ViewportLayer<()>; 3] = [&mut scene, &mut grid, &mut gizmo];
        host.pick_rect_blocking(&device, &queue, &(), &mut layers, [62, 62, 5, 5])
            .unwrap()
    };
    assert!(!gizmo_hits.is_empty());
    assert_eq!(gizmo_hits[0].id.layer, 12);
    assert_eq!(
        GizmoAxis::from_sub(gizmo_hits[0].id.sub),
        Some(GizmoAxis::Center)
    );
}

#[test]
fn removed_and_foreign_chunk_ids_cannot_modify_a_live_chunk() {
    let mut scene = Scene3DLayer::new(1);
    let old = scene.insert(Mesh3D::at_origin(Vec3::ZERO));
    scene.remove(old);
    let current = scene.insert(Mesh3D::at_origin(Vec3::ZERO));
    let mut other = Scene3DLayer::new(2);
    let foreign = other.insert(Mesh3D::at_origin(Vec3::ZERO));

    assert_ne!(old, current);
    scene.set_selected(current, true);
    scene.set_selected(old, false);
    scene.remove(old);
    scene.remove(foreign);
    assert!(scene.is_selected(current));
    assert!(!scene.is_selected(old));
}

#[test]
fn single_pixel_pick_keeps_the_section_face_at_fractional_dpi() {
    let Some((device, queue)) = create_headless_device() else {
        return;
    };
    let center = Vec3::new(1_000_000_000.25, -2_000_000_000.5, 20.0);
    let mut host = ViewportHost::new(&device, ViewportConfig::default()).unwrap();
    host.resize(&device, [127, 93], 1.75);
    *host.camera_mut() = Camera::perspective_3d(
        center - Vec3::Y * 2.0,
        center,
        Vec3::Z,
        std::f64::consts::FRAC_PI_3,
        0.1,
    );
    host.camera_mut()
        .set_clip_planes(&[Plane::from_normal_point(Vec3::Y, center).unwrap()]);

    let mut scene = Scene3DLayer::new(1);
    for (offset, object) in [(0.0, 8), (3.0, 9)] {
        let mut cube = Mesh3D::at_origin(center + Vec3::X * offset);
        cube.push_box(
            Vec3::new(-0.4, -1.95, -0.4),
            Vec3::new(0.4, 0.2, 0.4),
            [40, 220, 80, 255],
            PickId::new(1, object, 0),
            true,
        );
        scene.insert(cube);
    }
    let wide = host
        .pick_rect_blocking(&device, &queue, &(), &mut [&mut scene], [52, 35, 23, 23])
        .unwrap();
    let pixel = host
        .pick_rect_blocking(&device, &queue, &(), &mut [&mut scene], [63, 46, 1, 1])
        .unwrap();
    let [hit] = pixel.as_slice() else {
        panic!("expected one hit at the viewport center, got {pixel:?}");
    };
    assert_eq!(hit.id, PickId::new(1, 8, 4));
    assert!(
        wide.iter()
            .any(|other| other.id == hit.id && other.depth.to_bits() == hit.depth.to_bits())
    );
    let point = hit.world_pos.unwrap();
    assert!((point - (center + Vec3::Y * 0.2)).length() < 1e-5);

    // Direct Mesh3D edits can leave an AABB that excludes the uploaded triangles.
    let mut stale_bounds = Mesh3D::at_origin(center);
    stale_bounds.push_box(
        Vec3::new(-0.2, -0.2, -0.2),
        Vec3::new(0.2, 0.1, 0.2),
        [255; 4],
        PickId::new(1, 10, 0),
        false,
    );
    stale_bounds.local_bounds =
        Bounds3::from_corners(Vec3::new(500.0, 0.0, 0.0), Vec3::new(501.0, 1.0, 1.0));
    scene.insert(stale_bounds);
    let hits = host
        .pick_rect_blocking(&device, &queue, &(), &mut [&mut scene], [63, 46, 1, 1])
        .unwrap();
    assert_eq!(hits.first().unwrap().id, PickId::new(1, 10, 0));
}

#[test]
fn shared_scene_releases_bindings_for_closed_viewports() {
    let Some((device, queue)) = create_headless_device() else {
        return;
    };
    let mut scene = Scene3DLayer::new(1);
    scene.insert(Mesh3D::at_origin(Vec3::ZERO));
    let mut active = ViewportHost::new(&device, ViewportConfig::default()).unwrap();
    let mut closed = ViewportHost::new(&device, ViewportConfig::default()).unwrap();
    let mut layers: [&mut dyn ViewportLayer<()>; 1] = [&mut scene];
    let mut encoder = device.create_command_encoder(&Default::default());
    active.render_to_texture(&device, &queue, &mut encoder, &(), &mut layers);
    closed.render_to_texture(&device, &queue, &mut encoder, &(), &mut layers);
    queue.submit([encoder.finish()]);

    drop(closed);
    assert_eq!(scene.prune_closed_viewports(), 1);
    assert_eq!(scene.prune_closed_viewports(), 0);
    drop(active);
    assert_eq!(scene.prune_closed_viewports(), 1);
}
