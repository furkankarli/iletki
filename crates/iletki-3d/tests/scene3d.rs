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
    create_device_with_limits(wgpu::Limits::default())
}

fn create_device_with_limits(limits: wgpu::Limits) -> Option<(wgpu::Device, wgpu::Queue)> {
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
        required_limits: limits,
        ..wgpu::DeviceDescriptor::default()
    }))
    .ok()
}

#[test]
fn separate_vertex_streams_match_the_interleaved_fallback() {
    let Some((device, queue)) = create_headless_device() else {
        return;
    };
    let (fallback_device, fallback_queue) = create_device_with_limits(wgpu::Limits {
        max_vertex_buffers: 3,
        ..wgpu::Limits::default()
    })
    .unwrap();
    let render = |device: &wgpu::Device, queue: &wgpu::Queue| {
        let mut host = ViewportHost::new(
            device,
            ViewportConfig {
                initial_px: [192, 192],
                ..ViewportConfig::default()
            },
        )
        .unwrap();
        let origin = Vec3::new(661_858.0, 4_491_159.0, 120.0);
        *host.camera_mut() =
            Camera::ortho_3d(origin + Vec3::Z * 40.0, origin, Vec3::Y, 27.0, 0.05, 100.0);
        let mut scene = Scene3DLayer::new(1);
        for index in 0..144 {
            let offset = Vec3::new(
                f64::from(index % 12) * 2.0 - 11.0,
                f64::from(index / 12) * 2.0 - 11.0,
                0.0,
            );
            let mut mesh = Mesh3D::at_origin(origin + offset);
            mesh.push_box(
                Vec3::new(-0.6, -0.6, -0.5),
                Vec3::new(0.6, 0.6, 0.5),
                [90, 110, 230, 255],
                PickId::new(1, index + 1, 3),
                true,
            );
            scene.insert(mesh);
        }
        let mut encoder = device.create_command_encoder(&Default::default());
        host.render_to_texture(device, queue, &mut encoder, &(), &mut [&mut scene]);
        queue.submit([encoder.finish()]);
        let pixels = host.readback_rgba_blocking(device, queue).unwrap().pixels;
        let hits = host
            .pick_rect_blocking(device, queue, &(), &mut [&mut scene], [0, 0, 192, 192])
            .unwrap();
        assert_eq!(hits.len(), 144);
        (pixels, hits)
    };
    assert_eq!(
        render(&device, &queue),
        render(&fallback_device, &fallback_queue)
    );
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

#[cfg(feature = "profiling")]
#[test]
fn packed_faces_match_direct_across_views_and_chunk_edits() {
    let Some((device, queue)) = create_headless_device() else {
        return;
    };
    let center = Vec3::new(1_000_000_000.25, -2_000_000_000.5, 20.0);
    let mut scene = Scene3DLayer::new(1);
    let cube = |origin, object| {
        let mut mesh = Mesh3D::at_origin(origin);
        mesh.push_box(
            Vec3::new(-0.3, -0.3, -0.3),
            Vec3::new(0.3, 0.3, 0.3),
            [65, 195, 130, 255],
            PickId::new(1, object, 0),
            true,
        );
        mesh
    };
    let middle = scene.insert(cube(center, 1));
    for i in 0..289 {
        if i == 32 {
            // A large direct draw interrupts the first packed page. Its transforms
            // must stay independent of the repeated draws on either side.
            let mut large = cube(center + Vec3::new(0.1, 0.15, -0.25), 800);
            large
                .vertices
                .resize(65_538, iletki_3d::Vertex3D::default());
            scene.insert(large);
        }
        let x = f64::from(i % 17) - 8.0;
        let y = f64::from(i / 17) - 8.0;
        if i != 144 {
            scene.insert(cube(center + Vec3::new(x, y, 0.0), i + 2));
        }
    }
    let config = ViewportConfig {
        initial_px: [128, 128],
        upload_budget: iletki_core::UploadBudget::UNLIMITED,
        ..ViewportConfig::default()
    };
    let mut top = ViewportHost::new(&device, config).unwrap();
    let mut oblique = ViewportHost::new(&device, config).unwrap();
    *top.camera_mut() = Camera::ortho_3d(center + Vec3::Z * 24.0, center, Vec3::Y, 24.0, 0.1, 60.0);
    *oblique.camera_mut() = Camera::ortho_3d(
        center + Vec3::new(8.0, -9.0, 24.0),
        center,
        Vec3::Z,
        24.0,
        0.1,
        60.0,
    );
    top.camera_mut().set_clip_planes(&[
        Plane::from_normal_point(Vec3::X, center - Vec3::X * 3.15).unwrap(),
        Plane::from_normal_point(Vec3::Y, center - Vec3::Y * 4.2).unwrap(),
    ]);
    oblique.camera_mut().set_clip_planes(&[
        Plane::from_normal_point(Vec3::X, center - Vec3::X * 6.2).unwrap(),
        Plane::from_normal_point(-Vec3::Y, center + Vec3::Y * 3.1).unwrap(),
    ]);

    let mut images = |scene: &mut Scene3DLayer, packed| {
        scene.set_profile_batch_geometry(packed);
        top.mark_dirty(DirtyFlags::CAMERA);
        oblique.mark_dirty(DirtyFlags::CAMERA);
        let mut encoder = device.create_command_encoder(&Default::default());
        let top_frame = top.render_to_texture(&device, &queue, &mut encoder, &(), &mut [scene]);
        assert!(!top_frame.stats.streaming_pending);
        let oblique_frame =
            oblique.render_to_texture(&device, &queue, &mut encoder, &(), &mut [scene]);
        assert!(!oblique_frame.stats.streaming_pending);
        queue.submit([encoder.finish()]);
        [
            top.readback_rgba_blocking(&device, &queue).unwrap().pixels,
            oblique
                .readback_rgba_blocking(&device, &queue)
                .unwrap()
                .pixels,
        ]
    };
    let visible = images(&mut scene, true);
    assert!(visible == images(&mut scene, false));

    scene.set_visible(middle, false);
    let hidden = images(&mut scene, true);
    assert!(hidden != visible && hidden == images(&mut scene, false));

    scene.remove(middle);
    let replacement = scene.insert(cube(center + Vec3::X * 0.2, 500));
    scene.set_selected(replacement, true);
    let selected = images(&mut scene, true);
    assert!(selected != hidden && selected == images(&mut scene, false));

    scene.set_origin(replacement, center + Vec3::X * 10.0);
    let relocated = images(&mut scene, true);
    assert!(relocated != selected && relocated == images(&mut scene, false));
    scene.clear_selection();
    let cleared = images(&mut scene, true);
    assert!(cleared != relocated && cleared == images(&mut scene, false));

    let hits = top
        .pick_rect_blocking(&device, &queue, &(), &mut [&mut scene], [115, 62, 5, 5])
        .unwrap();
    assert_eq!(hits.first().unwrap().id, PickId::new(1, 500, 0));
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

    // Exercise grouped rejection while keeping a face within the shader's clip tolerance.
    for _ in 0..128 {
        let mut hidden = Mesh3D::at_origin(center - Vec3::Y);
        hidden.push_box(
            Vec3::new(-0.2, -0.2, -0.2),
            Vec3::new(0.2, 0.2, 0.2),
            [255; 4],
            PickId::new(1, 11, 0),
            false,
        );
        scene.insert(hidden);
    }
    let mut boundary = Mesh3D::at_origin(center);
    let y = -0.00005;
    boundary.push_quad(
        Vec3::new(-0.2, y, -0.2),
        Vec3::new(0.2, y, -0.2),
        Vec3::new(0.2, y, 0.2),
        Vec3::new(-0.2, y, 0.2),
        [255; 4],
        PickId::new(1, 12, 0),
    );
    let boundary = scene.insert(boundary);
    let hits = host
        .pick_rect_blocking(&device, &queue, &(), &mut [&mut scene], [63, 46, 1, 1])
        .unwrap();
    assert_eq!(hits.first().unwrap().id, PickId::new(1, 12, 0));
    scene.remove(boundary);

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
        Bounds3::from_corners(Vec3::new(500.0, -2.0, 0.0), Vec3::new(501.0, -1.0, 1.0));
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
