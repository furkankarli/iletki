//! Headless GPU integration tests for `iletki-3d`:
//! - Camera-relative Reversed-Z 3D cube at national-grid coordinates `(661_858, 4_491_159, 120)`
//! - 3D section clip plane slicing (`Camera::set_clip_planes`)
//! - 64-bit sub-object face picking + exact `f64` world point reconstruction from Reversed-Z depth
//! - Infinite anti-aliased ground grid and screen-size-invariant 3D transform gizmo axis picking

#![cfg(not(target_arch = "wasm32"))]

use iletki_3d::{
    GizmoAxis, Grid3DConfig, InfiniteGrid3DLayer, Material3D, MaterialFlags, MaterialId, Mesh3D,
    Scene3DLayer, TransformGizmoLayer,
};
use iletki_core::{
    Bounds3, Camera, DirtyFlags, DrawPhase, DrawPhaseMask, PickId, Plane, Vec3, ViewportId,
    ViewportMode,
};
use iletki_wgpu::{
    DEPTH_FORMAT, DEPTH_STENCIL_FORMAT, PaintCtx, PrepareCtx, ViewportConfig, ViewportHost,
    ViewportLayer, wgpu,
};
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
                upload_budget: iletki_core::UploadBudget::UNLIMITED,
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
    let (sep_pixels, sep_hits) = render(&device, &queue);
    let (int_pixels, int_hits) = render(&fallback_device, &fallback_queue);
    assert_eq!(
        sep_hits, int_hits,
        "separate vs interleaved pick hits differed"
    );
    assert!(
        sep_pixels == int_pixels,
        "separate vs interleaved rendered pixels differed"
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
    let mut host = ViewportHost::new(
        &device,
        ViewportConfig {
            upload_budget: iletki_core::UploadBudget::UNLIMITED,
            ..ViewportConfig::default()
        },
    )
    .unwrap();
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

#[cfg(feature = "profiling")]
#[test]
fn indexed_flat_and_smooth_geometry_match_unindexed_with_exact_picks() {
    let Some((device, queue)) = create_headless_device() else {
        return;
    };

    let build_scene = |indexed: bool| {
        let mut scene = Scene3DLayer::new(1);
        scene.set_profile_indexed_geometry(indexed);
        let origin = Vec3::new(661_858.0, 4_491_159.0, 120.0);

        // 1. Flat-eligible packed boxes with edges, transparency, depth ties and sub-face IDs.
        for i in 0..144u32 {
            let offset = Vec3::new(
                f64::from(i % 12) * 2.2 - 12.1,
                f64::from(i / 12) * 2.2 - 12.1,
                0.0,
            );
            let mut mesh = Mesh3D::at_origin(origin + offset);
            let alpha = if i % 17 == 0 { 180 } else { 255 };
            mesh.push_box_with_edges(
                Vec3::new(-0.7, -0.7, -0.5),
                Vec3::new(0.7, 0.7, 0.5),
                [
                    (80 + (i * 23) % 150) as u8,
                    (90 + (i * 41) % 140) as u8,
                    (110 + (i * 59) % 130) as u8,
                    alpha,
                ],
                [20, 24, 30, 220],
                PickId::new(1, i + 1, 0),
                i % 3 == 0,
            );
            scene.insert(mesh);
        }

        // 2. Smooth-shaded cylinder (ineligible for flat provoking; uses exact-record indexing).
        let mut smooth = Mesh3D::at_origin(origin + Vec3::new(0.0, 0.0, 1.5));
        let segs = 24usize;
        let pick = PickId::new(1, 900, 0).to_rg32u();
        for sg in 0..segs {
            let a0 = std::f32::consts::TAU * (sg as f32) / (segs as f32);
            let a1 = std::f32::consts::TAU * ((sg + 1) as f32) / (segs as f32);
            let n0 = [a0.cos(), a0.sin(), 0.0];
            let n1 = [a1.cos(), a1.sin(), 0.0];
            let r = 3.0f32;
            let color = [220, 140, 80, 255];
            let v00 = iletki_3d::Vertex3D::new([n0[0] * r, n0[1] * r, -1.0], n0, color);
            let v10 = iletki_3d::Vertex3D::new([n1[0] * r, n1[1] * r, -1.0], n1, color);
            let v11 = iletki_3d::Vertex3D::new([n1[0] * r, n1[1] * r, 1.0], n1, color);
            let v01 = iletki_3d::Vertex3D::new([n0[0] * r, n0[1] * r, 1.0], n0, color);
            smooth
                .vertices
                .extend_from_slice(&[v00, v10, v11, v00, v11, v01]);
            smooth.pick_ids.resize(smooth.vertices.len(), pick);
        }
        smooth.local_bounds =
            Bounds3::from_corners(Vec3::new(-3.0, -3.0, -1.0), Vec3::new(3.0, 3.0, 1.0));
        scene.insert(smooth);
        (scene, origin)
    };

    let render_and_pick = |indexed: bool, with_section: bool| {
        let (mut scene, origin) = build_scene(indexed);
        let mut host = ViewportHost::new(
            &device,
            ViewportConfig {
                initial_px: [256, 256],
                upload_budget: iletki_core::UploadBudget::UNLIMITED,
                ..ViewportConfig::default()
            },
        )
        .unwrap();
        let mut cam = Camera::perspective_3d(
            origin + Vec3::new(22.0, -24.0, 20.0),
            origin,
            Vec3::Z,
            50.0_f64.to_radians(),
            0.1,
        );
        if with_section {
            let plane =
                Plane::from_normal_point(Vec3::X, origin + Vec3::new(0.3, 0.0, 0.0)).unwrap();
            cam.set_clip_planes(&[plane]);
        }
        *host.camera_mut() = cam;
        let mut enc = device.create_command_encoder(&Default::default());
        host.render_to_texture(&device, &queue, &mut enc, &(), &mut [&mut scene]);
        queue.submit([enc.finish()]);
        let image = host.readback_rgba_blocking(&device, &queue).unwrap();
        let hits = host
            .pick_rect_blocking(&device, &queue, &(), &mut [&mut scene], [0, 0, 256, 256])
            .unwrap();
        (image, hits)
    };

    for with_section in [false, true] {
        let (ref_img, ref_hits) = render_and_pick(false, with_section);
        let (idx_img, idx_hits) = render_and_pick(true, with_section);
        assert!(
            !ref_hits.is_empty(),
            "expected pick hits (section={with_section})"
        );
        // Provoking-vertex flat indexing rotates each triangle's vertex order
        // (a,b,c) -> (b,c,a)/(c,a,b) so each triangle's provoking corner lands at the
        // backend's provoking slot. Software rasterizers (Lavapipe, WARP, llvmpipe) and
        // tiled GPUs (Apple Metal) can differ by ~1 ULP in interpolated depth from the
        // rotated edge equations while returning the exact same PickId set and world positions.
        assert_eq!(
            idx_hits.len(),
            ref_hits.len(),
            "indexed pick count must match (section={with_section})"
        );
        assert!(
            idx_hits.windows(2).all(|w| w[0].depth >= w[1].depth),
            "indexed pick hits must stay sorted nearest-first (section={with_section})"
        );
        let ref_by_id: std::collections::HashMap<_, _> =
            ref_hits.iter().map(|hit| (hit.id, hit)).collect();
        assert_eq!(
            ref_by_id.len(),
            ref_hits.len(),
            "reference pick hits must have unique IDs (section={with_section})"
        );
        for idx in &idx_hits {
            let r = ref_by_id.get(&idx.id).unwrap_or_else(|| {
                panic!(
                    "indexed pick returned unexpected PickId {:?} (section={with_section})",
                    idx.id
                )
            });
            assert!(
                (idx.depth - r.depth).abs() <= 1e-5,
                "indexed pick depth diverged for {:?}: {} vs {} (section={with_section})",
                idx.id,
                idx.depth,
                r.depth
            );
            match (idx.world_pos, r.world_pos) {
                (Some(a), Some(b)) => assert!(
                    (a - b).length() <= 1e-3,
                    "indexed pick world_pos diverged for {:?}: {a:?} vs {b:?} (section={with_section})",
                    idx.id
                ),
                (None, None) => {}
                (a, b) => panic!(
                    "indexed pick world_pos presence mismatch for {:?}: {a:?} vs {b:?} (section={with_section})",
                    idx.id
                ),
            }
        }
        let mut max_delta = 0u8;
        let mut diff_gt1 = 0usize;
        for (a, b) in ref_img
            .pixels
            .as_chunks::<4>()
            .0
            .iter()
            .zip(idx_img.pixels.as_chunks::<4>().0.iter())
        {
            let d = a[0]
                .abs_diff(b[0])
                .max(a[1].abs_diff(b[1]))
                .max(a[2].abs_diff(b[2]))
                .max(a[3].abs_diff(b[3]));
            max_delta = max_delta.max(d);
            if d > 1 {
                diff_gt1 += 1;
            }
        }
        assert!(
            max_delta <= 2 && diff_gt1 * 10000 <= (256 * 256),
            "render parity exceeded tolerance (section={with_section}): max_delta={max_delta}, diff_gt1={diff_gt1}"
        );
    }
}

#[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
#[test]
fn storage_transforms_match_uniform_fallback_and_handle_edits() {
    let Some((device, queue)) = create_headless_device() else {
        return;
    };

    let run = |use_storage: bool| {
        let mut scene = Scene3DLayer::new(1);
        scene.set_profile_storage_transforms(use_storage);
        // Large national-grid coordinates that require split f64 hi/lo precision.
        let origin = Vec3::new(1_000_000_033.125, -500_000_017.25, 250.5);
        let mut ids = Vec::new();
        for i in 0..320u32 {
            let x = f64::from(i % 20) * 1.4 - 13.3;
            let y = f64::from(i / 20) * 1.4 - 10.5;
            let mut mesh = Mesh3D::at_origin(origin + Vec3::new(x, y, 0.0));
            mesh.push_box_with_edges(
                Vec3::new(-0.45, -0.45, 0.0),
                Vec3::new(0.45, 0.45, 1.0),
                [130, 175, 210, 255],
                [24, 30, 36, 220],
                PickId::new(1, i + 1, 0),
                false,
            );
            ids.push(scene.insert(mesh));
        }

        let mut host = ViewportHost::new(
            &device,
            ViewportConfig {
                initial_px: [256, 256],
                upload_budget: iletki_core::UploadBudget::UNLIMITED,
                ..ViewportConfig::default()
            },
        )
        .unwrap();

        let render_frame = |host: &mut ViewportHost, scene: &mut Scene3DLayer| {
            let mut enc = device.create_command_encoder(&Default::default());
            host.render_to_texture(&device, &queue, &mut enc, &(), &mut [scene]);
            queue.submit([enc.finish()]);
        };

        // Frame 1: initial full-scene wide view (builds retained draw list).
        *host.camera_mut() = Camera::perspective_3d(
            origin + Vec3::new(26.0, -28.0, 24.0),
            origin,
            Vec3::Z,
            50.0_f64.to_radians(),
            0.1,
        );
        render_frame(&mut host, &mut scene);

        // Frame 2: camera move (exercises retained O(1) hit) + edits: move, hide, select, remove + reuse.
        scene.set_origin(ids[10], origin + Vec3::new(0.0, 0.0, 3.0));
        scene.set_visible(ids[25], false);
        scene.set_selected(ids[10], true);
        scene.set_selected(ids[150], true);
        scene.remove(ids[0]);
        let mut replacement = Mesh3D::at_origin(origin + Vec3::new(1.0, 1.0, 4.5));
        replacement.push_box_with_edges(
            Vec3::new(-0.8, -0.8, 0.0),
            Vec3::new(0.8, 0.8, 1.4),
            [235, 90, 60, 255],
            [20, 20, 20, 255],
            PickId::new(1, 9999, 0),
            false,
        );
        let _reused = scene.insert(replacement);

        *host.camera_mut() = Camera::perspective_3d(
            origin + Vec3::new(20.0, -24.0, 22.0),
            origin + Vec3::new(0.0, 0.0, 1.0),
            Vec3::Z,
            50.0_f64.to_radians(),
            0.1,
        );
        render_frame(&mut host, &mut scene);
        let img = host.readback_rgba_blocking(&device, &queue).unwrap();
        let hits = host
            .pick_rect_blocking(&device, &queue, &(), &mut [&mut scene], [0, 0, 256, 256])
            .unwrap();
        (img, hits)
    };

    let (ref_img, ref_hits) = run(false);
    let (sto_img, sto_hits) = run(true);

    assert!(!ref_hits.is_empty());
    assert!(
        sto_hits.iter().any(|h| h.id == PickId::new(1, 9999, 0)),
        "reused slot replacement chunk must be pickable"
    );
    assert!(
        !sto_hits.iter().any(|h| h.id == PickId::new(1, 26, 0)),
        "hidden chunk must not be pickable"
    );
    assert!(
        !sto_hits.iter().any(|h| h.id == PickId::new(1, 1, 0)),
        "removed chunk must not be pickable after slot reuse"
    );
    assert_eq!(
        sto_hits, ref_hits,
        "storage picking must match uniform fallback bit-for-bit"
    );

    let mut max_delta = 0u8;
    let mut diff_gt1 = 0usize;
    for (a, b) in ref_img
        .pixels
        .as_chunks::<4>()
        .0
        .iter()
        .zip(sto_img.pixels.as_chunks::<4>().0.iter())
    {
        let d = a[0]
            .abs_diff(b[0])
            .max(a[1].abs_diff(b[1]))
            .max(a[2].abs_diff(b[2]))
            .max(a[3].abs_diff(b[3]));
        max_delta = max_delta.max(d);
        if d > 1 {
            diff_gt1 += 1;
        }
    }
    assert!(
        max_delta <= 2 && diff_gt1 * 10000 <= (256 * 256),
        "storage vs uniform render parity failed: max_delta={max_delta}, diff_gt1={diff_gt1}"
    );
}

#[test]
fn color_frame_selection_preview_matches_exact_marquee_pick() {
    use std::collections::{BTreeSet, HashMap};
    use std::task::Poll;

    let Some((device, queue)) = create_headless_device() else {
        return;
    };

    let mut host = ViewportHost::new(
        &device,
        ViewportConfig {
            initial_px: [256, 256],
            upload_budget: iletki_core::UploadBudget::UNLIMITED,
            ..ViewportConfig::default()
        },
    )
    .unwrap();
    if device.adapter_info().backend == iletki_wgpu::wgpu::Backend::Gl {
        assert!(!host.supports_color_frame_selection(&device));
        return;
    }
    assert!(
        host.supports_color_frame_selection(&device),
        "Vulkan/DX12/Metal must support Rg16Uint 4x MSAA color-frame selection"
    );

    let mut scene = Scene3DLayer::new(1);
    let mut grid = InfiniteGrid3DLayer::new(2, Grid3DConfig::default());
    let origin = Vec3::new(661_858.0, 4_491_159.0, 120.0);
    let mut id_by_object = HashMap::new();
    let mut chunk_ids = Vec::new();
    for i in 0..160u32 {
        let x = f64::from(i % 16) * 1.5 - 11.25;
        let y = f64::from(i / 16) * 1.5 - 6.75;
        let mut mesh = Mesh3D::at_origin(origin + Vec3::new(x, y, 0.0));
        let obj = i + 1;
        mesh.push_box_with_edges(
            Vec3::new(-0.4, -0.4, 0.0),
            Vec3::new(0.4, 0.4, 1.0),
            [110, 165, 210, 255],
            [20, 26, 34, 220],
            PickId::new(1, obj, 0),
            false,
        );
        // Leave one chunk unpickable to verify pickable mask parity.
        if i == 5 {
            mesh.pick_ids.clear();
        }
        let cid = scene.insert(mesh);
        id_by_object.insert(obj, cid);
        chunk_ids.push(cid);
    }
    scene.set_visible(chunk_ids[12], false);

    let mut cam = Camera::perspective_3d(
        origin + Vec3::new(16.0, -20.0, 18.0),
        origin,
        Vec3::Z,
        50.0_f64.to_radians(),
        0.1,
    );
    let plane = Plane::from_normal_point(Vec3::X, origin + Vec3::new(-1.0, 0.0, 0.0)).unwrap();
    cam.set_clip_planes(&[plane]);
    *host.camera_mut() = cam;

    let read_preview = |host: &mut ViewportHost,
                        scene: &mut Scene3DLayer,
                        grid: &mut InfiniteGrid3DLayer,
                        rect: [u32; 4]| {
        host.set_color_frame_selection_rect(Some(rect));
        let mut enc = device.create_command_encoder(&Default::default());
        let out = host.render_to_texture(&device, &queue, &mut enc, &(), &mut [scene, grid]);
        let static_rebuilt = out.stats.static_rebuilt;
        queue.submit([enc.finish()]);
        let mut pending = host
            .take_color_frame_selection_preview(&device)
            .expect("expected color-frame selection preview");
        assert_eq!(pending.snapshot().rect_px, rect);
        assert_eq!(pending.snapshot().viewport_id, host.viewport_id());
        let slots = loop {
            match pending.try_read() {
                Poll::Ready(res) => break res.unwrap(),
                Poll::Pending => std::thread::yield_now(),
            }
        };
        let chunks: BTreeSet<_> = slots
            .into_iter()
            .filter_map(|s| scene.chunk_id_at_slot(s as usize))
            .map(|cid| format!("{cid:?}"))
            .collect();
        (chunks, static_rebuilt)
    };

    let rect1 = [64, 64, 128, 128];
    let (preview1, rebuilt1) = read_preview(&mut host, &mut scene, &mut grid, rect1);
    assert!(rebuilt1, "first preview frame must populate MRT ID target");
    assert!(!preview1.is_empty(), "expected non-empty marquee preview");

    // Second drag rect on static camera must reuse the cached MRT ID target without rebuilding static scene.
    let rect2 = [80, 72, 96, 96];
    let (preview2, rebuilt2) = read_preview(&mut host, &mut scene, &mut grid, rect2);
    assert!(
        !rebuilt2,
        "subsequent static marquee drag must reuse cached MRT IDs without redrawing geometry"
    );

    // Clear preview and run exact release pick to verify superset / parity and lifecycle counters.
    host.set_color_frame_selection_rect(None);
    let exact_hits = host
        .pick_rect_blocking(&device, &queue, &(), &mut [&mut scene, &mut grid], rect2)
        .unwrap();
    let exact_chunks: BTreeSet<_> = exact_hits
        .iter()
        .filter(|h| h.id.layer == 1)
        .filter_map(|h| id_by_object.get(&h.id.object))
        .map(|cid| format!("{cid:?}"))
        .collect();
    assert!(!exact_chunks.is_empty());
    assert!(
        exact_chunks.is_subset(&preview2),
        "4x MSAA preview must cover all exact 1x release picks: missing {:?}",
        exact_chunks.difference(&preview2).collect::<Vec<_>>()
    );
    assert!(
        !preview2.contains(&format!("{:?}", chunk_ids[5])),
        "unpickable chunk must not appear in color-frame preview"
    );
    assert!(
        !preview2.contains(&format!("{:?}", chunk_ids[12])),
        "hidden chunk must not appear in color-frame preview"
    );
    let stats = host.selection_preview_stats();
    assert_eq!(stats.submitted, 2);
    assert_eq!(stats.ready, 2);
}

#[test]
fn gpu_culling_and_hiz_match_cpu_reference_across_views_sections_and_edits() {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let Ok(adapter) = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
        ..wgpu::RequestAdapterOptions::default()
    })) else {
        return;
    };
    if !adapter
        .features()
        .contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT)
    {
        return;
    }
    let clip_feat = adapter.features() & wgpu::Features::CLIP_DISTANCES;
    let Ok((gpu_device, gpu_queue)) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("iletki-3d gpu cull test"),
        required_features: wgpu::Features::MULTI_DRAW_INDIRECT_COUNT | clip_feat,
        ..wgpu::DeviceDescriptor::default()
    })) else {
        return;
    };
    let Ok((ref_device, ref_queue)) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("iletki-3d cpu cull reference"),
        required_features: clip_feat,
        ..wgpu::DeviceDescriptor::default()
    })) else {
        return;
    };

    let origin = Vec3::new(100_000.0, 200_000.0, 10.0);
    let build_scene = || {
        let mut scene = Scene3DLayer::new(1);
        let mut ids = Vec::new();
        // Large opaque occluder wall at y = 0 hiding the 80 distinct boxes behind it from -Y.
        let mut wall = Mesh3D::at_origin(origin);
        wall.push_box_with_edges(
            Vec3::new(-12.0, -0.3, -1.0),
            Vec3::new(12.0, 0.3, 8.0),
            [160, 150, 140, 255],
            [25, 25, 30, 255],
            PickId::new(1, 1, 0),
            false,
        );
        ids.push(scene.insert(wall));

        // 1040 distinct-sized non-repeated opaque boxes behind the wall (y > 2.0) to cross the
        // production GPU culling activation threshold (`MIN_GPU_CULL_ITEMS = 1024`).
        for i in 0..1040u32 {
            let col = i % 26;
            let row = (i / 26) % 20;
            let tier = i / (26 * 20);
            let x = f64::from(col) * 0.72 - 9.0;
            let y = 2.5 + f64::from(row) * 0.75;
            let z = f64::from(tier) * 2.2;
            let h = 0.8 + f64::from(i % 7) * 0.14;
            let mut m = Mesh3D::at_origin(origin + Vec3::new(x, y, z));
            m.push_box_with_edges(
                Vec3::new(-0.24, -0.24, 0.0),
                Vec3::new(0.24 + f64::from(i % 5) * 0.02, 0.24, h),
                [
                    (80 + (i * 17) % 160) as u8,
                    (90 + (i * 29) % 150) as u8,
                    (110 + (i * 43) % 130) as u8,
                    255,
                ],
                [20, 24, 30, 240],
                PickId::new(1, i + 10, 0),
                false,
            );
            ids.push(scene.insert(m));
        }

        // Translucent glass pane on the +X side in front of an opaque target box behind it.
        let mut glass = Mesh3D::at_origin(origin + Vec3::new(16.0, 0.0, 0.0));
        glass.push_box(
            Vec3::new(-2.0, -0.1, 0.0),
            Vec3::new(2.0, 0.1, 4.0),
            [120, 200, 240, 110],
            PickId::new(1, 500, 0),
            false,
        );
        ids.push(scene.insert(glass));

        let mut behind_glass = Mesh3D::at_origin(origin + Vec3::new(16.0, 3.0, 0.0));
        behind_glass.push_box(
            Vec3::new(-1.0, -1.0, 0.5),
            Vec3::new(1.0, 1.0, 3.5),
            [230, 60, 40, 255],
            PickId::new(1, 501, 0),
            false,
        );
        ids.push(scene.insert(behind_glass));

        (scene, ids)
    };

    let make_host = |device: &wgpu::Device| {
        ViewportHost::new_with_adapter(
            device,
            &adapter,
            ViewportConfig {
                initial_px: [257, 193], // Odd dimensions to exercise power-of-two Hi-Z padding
                mode: ViewportMode::Continuous,
                upload_budget: iletki_core::UploadBudget::UNLIMITED,
                ..ViewportConfig::default()
            },
        )
        .unwrap()
    };

    let (mut gpu_scene, gpu_ids) = build_scene();
    let (mut ref_scene, ref_ids) = build_scene();
    let mut gpu_front = make_host(&gpu_device);
    let mut gpu_back = make_host(&gpu_device);
    let mut ref_front = make_host(&ref_device);
    let mut ref_back = make_host(&ref_device);

    // Camera 1: directly in front of the opaque wall (80 boxes behind wall are occluded).
    let cam_front = Camera::perspective_3d(
        origin + Vec3::new(0.0, -6.5, 3.0),
        origin + Vec3::new(0.0, 4.0, 2.0),
        Vec3::Z,
        55.0_f64.to_radians(),
        0.1,
    );
    // Camera 2: looking from behind (+Y) so all 80 boxes are visible (independent viewport!).
    let cam_back = Camera::perspective_3d(
        origin + Vec3::new(2.0, 24.0, 12.0),
        origin + Vec3::new(0.0, 6.0, 1.5),
        Vec3::Z,
        55.0_f64.to_radians(),
        0.1,
    );
    // Camera 3: stepped around the wall so previously-occluded boxes suddenly become visible!
    let cam_reveal = Camera::perspective_3d(
        origin + Vec3::new(-18.0, -2.0, 7.0),
        origin + Vec3::new(0.0, 6.0, 1.5),
        Vec3::Z,
        55.0_f64.to_radians(),
        0.1,
    );
    // Camera 4: looking directly through the translucent glass pane at the red box behind it.
    let cam_glass = Camera::perspective_3d(
        origin + Vec3::new(16.0, -5.0, 2.0),
        origin + Vec3::new(16.0, 3.0, 2.0),
        Vec3::Z,
        45.0_f64.to_radians(),
        0.1,
    );
    // Camera 5: front view with a section clip plane slicing through the wall to expose boxes!
    let mut cam_section = cam_front.clone();
    cam_section.set_clip_planes(&[Plane::from_normal_point(
        Vec3::X,
        origin + Vec3::new(0.0, 0.0, 0.0),
    )
    .unwrap()]);

    let assert_images_match =
        |label: &str, a: &iletki_wgpu::RgbaImage, b: &iletki_wgpu::RgbaImage| {
            assert_eq!((a.width, a.height), (b.width, b.height));
            let mut max_delta = 0u8;
            let mut diff_gt2 = 0usize;
            for (pa, pb) in a
                .pixels
                .as_chunks::<4>()
                .0
                .iter()
                .zip(b.pixels.as_chunks::<4>().0)
            {
                let d = pa
                    .iter()
                    .zip(pb)
                    .map(|(&x, &y)| x.abs_diff(y))
                    .max()
                    .unwrap_or(0);
                max_delta = max_delta.max(d);
                if d > 2 {
                    diff_gt2 += 1;
                }
            }
            assert!(
                diff_gt2 == 0 && max_delta <= 2,
                "{label}: GPU cull vs CPU ref mismatch: max_delta={max_delta}, diff_gt2={diff_gt2}"
            );
        };

    let render_img = |device: &wgpu::Device,
                      queue: &wgpu::Queue,
                      host: &mut ViewportHost,
                      scene: &mut Scene3DLayer| {
        let mut enc = device.create_command_encoder(&Default::default());
        host.render_to_texture(device, queue, &mut enc, &(), &mut [scene]);
        queue.submit([enc.finish()]);
        host.readback_rgba_blocking(device, queue).unwrap()
    };

    // Frame 0 & Frame 1: dual-viewport shared scene in a single CommandEncoder (converges Hi-Z).
    *gpu_front.camera_mut() = cam_front.clone();
    *gpu_back.camera_mut() = cam_back.clone();
    *ref_front.camera_mut() = cam_front.clone();
    *ref_back.camera_mut() = cam_back.clone();
    for step in 0..2 {
        let mut g_enc = gpu_device.create_command_encoder(&Default::default());
        gpu_front.render_to_texture(
            &gpu_device,
            &gpu_queue,
            &mut g_enc,
            &(),
            &mut [&mut gpu_scene],
        );
        gpu_back.render_to_texture(
            &gpu_device,
            &gpu_queue,
            &mut g_enc,
            &(),
            &mut [&mut gpu_scene],
        );
        gpu_queue.submit([g_enc.finish()]);

        let mut r_enc = ref_device.create_command_encoder(&Default::default());
        ref_front.render_to_texture(
            &ref_device,
            &ref_queue,
            &mut r_enc,
            &(),
            &mut [&mut ref_scene],
        );
        ref_back.render_to_texture(
            &ref_device,
            &ref_queue,
            &mut r_enc,
            &(),
            &mut [&mut ref_scene],
        );
        ref_queue.submit([r_enc.finish()]);

        if step == 1 {
            assert!(<Scene3DLayer as ViewportLayer<()>>::uses_gpu_culling(
                &gpu_scene,
                gpu_front.viewport_id()
            ));
            assert!(<Scene3DLayer as ViewportLayer<()>>::uses_gpu_culling(
                &gpu_scene,
                gpu_back.viewport_id()
            ));
            assert!(!<Scene3DLayer as ViewportLayer<()>>::uses_gpu_culling(
                &ref_scene,
                ref_front.viewport_id()
            ));
            let g_img = gpu_front
                .readback_rgba_blocking(&gpu_device, &gpu_queue)
                .unwrap();
            let r_img = ref_front
                .readback_rgba_blocking(&ref_device, &ref_queue)
                .unwrap();
            assert_images_match("front-occluded", &g_img, &r_img);

            let gb_img = gpu_back
                .readback_rgba_blocking(&gpu_device, &gpu_queue)
                .unwrap();
            let rb_img = ref_back
                .readback_rgba_blocking(&ref_device, &ref_queue)
                .unwrap();
            assert_images_match("back-visible", &gb_img, &rb_img);
        }
    }

    // Immediate reveal frame after boxes were evicted from Phase 1: ZERO pop-in allowed!
    *gpu_front.camera_mut() = cam_reveal.clone();
    *ref_front.camera_mut() = cam_reveal.clone();
    let g_rev = render_img(&gpu_device, &gpu_queue, &mut gpu_front, &mut gpu_scene);
    let r_rev = render_img(&ref_device, &ref_queue, &mut ref_front, &mut ref_scene);
    assert_images_match("reveal-zero-popin", &g_rev, &r_rev);

    // Live edits: move, hide, remove + section clip plane!
    gpu_scene.set_origin(gpu_ids[5], origin + Vec3::new(-4.0, 2.0, 3.0));
    ref_scene.set_origin(ref_ids[5], origin + Vec3::new(-4.0, 2.0, 3.0));
    gpu_scene.set_visible(gpu_ids[10], false);
    ref_scene.set_visible(ref_ids[10], false);
    gpu_scene.remove(gpu_ids[15]);
    ref_scene.remove(ref_ids[15]);

    *gpu_front.camera_mut() = cam_section.clone();
    *ref_front.camera_mut() = cam_section.clone();
    let g_sec = render_img(&gpu_device, &gpu_queue, &mut gpu_front, &mut gpu_scene);
    let r_sec = render_img(&ref_device, &ref_queue, &mut ref_front, &mut ref_scene);
    assert_images_match("section-and-edits", &g_sec, &r_sec);

    // Bit-exact GPU picking parity under section clip + edits!
    let g_pick = gpu_front
        .pick_rect_blocking(
            &gpu_device,
            &gpu_queue,
            &(),
            &mut [&mut gpu_scene],
            [32, 32, 160, 128],
        )
        .unwrap();
    let r_pick = ref_front
        .pick_rect_blocking(
            &ref_device,
            &ref_queue,
            &(),
            &mut [&mut ref_scene],
            [32, 32, 160, 128],
        )
        .unwrap();
    assert_eq!(
        g_pick, r_pick,
        "picking must be 100% bit-exact with GPU culling"
    );

    // Translucent glass test (run 2 frames so Hi-Z converges): red box behind glass must be visible!
    *gpu_front.camera_mut() = cam_glass.clone();
    *ref_front.camera_mut() = cam_glass;
    for _ in 0..2 {
        let g_glass = render_img(&gpu_device, &gpu_queue, &mut gpu_front, &mut gpu_scene);
        let center = g_glass
            .pixel(g_glass.width / 2, g_glass.height / 2)
            .unwrap();
        assert!(
            center[0] > 90,
            "red box behind translucent glass was falsely occluded by Hi-Z: center={center:?}"
        );
    }

    // Compose Milestone 3 MRT selection preview with Milestone 4 two-phase GPU culling!
    *gpu_front.camera_mut() = cam_reveal;
    gpu_front.set_color_frame_selection_rect(Some([40, 40, 160, 120]));
    let mut enc = gpu_device.create_command_encoder(&Default::default());
    gpu_front.render_to_texture(
        &gpu_device,
        &gpu_queue,
        &mut enc,
        &(),
        &mut [&mut gpu_scene],
    );
    gpu_queue.submit([enc.finish()]);
    let mut pending = gpu_front
        .take_color_frame_selection_preview(&gpu_device)
        .expect("MRT selection preview must compose with two-phase GPU culling");
    let preview_slots = loop {
        match pending.try_read() {
            std::task::Poll::Ready(res) => break res.unwrap(),
            std::task::Poll::Pending => std::thread::yield_now(),
        }
    };
    assert!(!preview_slots.is_empty());
    gpu_front.set_color_frame_selection_rect(None);

    // Exercise quarter-res 16x16 Hi-Z across tiny and odd viewport dimensions (1x1, 3x5, 17x13).
    for dims in [[1, 1], [3, 5], [17, 13], [64, 37]] {
        gpu_front.resize(&gpu_device, dims, 1.0);
        let img = render_img(&gpu_device, &gpu_queue, &mut gpu_front, &mut gpu_scene);
        assert_eq!((img.width, img.height), (dims[0], dims[1]));
    }
}

#[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
#[test]
fn profiling_splits_phase1_and_phase2_pipeline_statistics() {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let Ok(adapter) = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
        ..wgpu::RequestAdapterOptions::default()
    })) else {
        return;
    };
    let required =
        wgpu::Features::MULTI_DRAW_INDIRECT_COUNT | wgpu::Features::PIPELINE_STATISTICS_QUERY;
    if !adapter.features().contains(required) {
        return;
    }
    let Ok((device, queue)) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("iletki-3d p2 stats test"),
        required_features: required,
        ..wgpu::DeviceDescriptor::default()
    })) else {
        return;
    };

    let mut scene = Scene3DLayer::new(1);
    for i in 0..1030u32 {
        let mut m = Mesh3D::at_origin(Vec3::new(
            f64::from(i % 32) * 1.1,
            f64::from(i / 32) * 1.1,
            0.0,
        ));
        let alpha = if i == 1029 { 120 } else { 255 };
        m.push_box(
            Vec3::new(-0.3, -0.3, 0.0),
            Vec3::new(0.3 + f64::from(i % 4) * 0.05, 0.3, 1.0),
            [140, 170, 200, alpha],
            PickId::new(1, i + 1, 0),
            false,
        );
        scene.insert(m);
    }

    let mut host = ViewportHost::new_with_adapter(
        &device,
        &adapter,
        ViewportConfig {
            initial_px: [256, 256],
            mode: ViewportMode::Continuous,
            upload_budget: iletki_core::UploadBudget::UNLIMITED,
            ..ViewportConfig::default()
        },
    )
    .unwrap();
    *host.camera_mut() = Camera::perspective_3d(
        Vec3::new(18.0, -18.0, 20.0),
        Vec3::new(16.0, 16.0, 0.5),
        Vec3::Z,
        55.0_f64.to_radians(),
        0.1,
    );

    // Warm one frame so Hi-Z visibility converges, then capture Phase 1 + Phase 2 stats.
    for frame in 0..2 {
        if frame == 1 {
            iletki_wgpu::profiling::begin(&device);
        }
        let mut enc = device.create_command_encoder(&Default::default());
        host.render_to_texture(&device, &queue, &mut enc, &(), &mut [&mut scene]);
        queue.submit([enc.finish()]);
        if frame == 1 {
            let cap = iletki_wgpu::profiling::finish(&device, &queue).unwrap();
            let p1_fs = cap
                .counters
                .iter()
                .find(|(k, _)| *k == "gpu.scene.fs_invocations")
                .map(|(_, v)| *v);
            let p2_fs = cap
                .counters
                .iter()
                .find(|(k, _)| *k == "gpu.scene_p2.fs_invocations")
                .map(|(_, v)| *v);
            assert!(
                p1_fs.is_some_and(|v| v > 0),
                "expected non-zero Phase 1 FS invocations, got {p1_fs:?}"
            );
            assert!(
                p2_fs.is_some(),
                "expected Phase 2 pipeline statistics counter to be emitted"
            );
        }
    }
}

#[test]
fn operational_resilience_optional_features_block_resizes_and_teardown() {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let Ok(adapter) = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
        ..wgpu::RequestAdapterOptions::default()
    })) else {
        return;
    };
    let is_gl = adapter.get_info().backend == wgpu::Backend::Gl;

    let build_scene = || {
        let mut scene = Scene3DLayer::new(1);
        let mut ids = Vec::with_capacity(1040);
        for i in 0..1040u32 {
            let x = f64::from(i % 32) * 1.15 - 17.8;
            let y = f64::from(i / 32) * 1.15 - 17.8;
            let mut m = Mesh3D::at_origin(Vec3::new(x, y, 0.0));
            m.push_box_with_edges(
                Vec3::new(-0.35, -0.35, 0.0),
                Vec3::new(0.35 + f64::from(i % 3) * 0.04, 0.35, 0.9),
                [105, 165, 220, 255],
                [24, 30, 38, 220],
                PickId::new(1, i + 1, 0),
                false,
            );
            ids.push(scene.insert(m));
        }
        (scene, ids)
    };
    let test_camera = || {
        let mut cam = Camera::perspective_3d(
            Vec3::new(22.0, -24.0, 22.0),
            Vec3::new(0.0, 0.0, 0.45),
            Vec3::Z,
            52.0_f64.to_radians(),
            0.1,
        );
        let plane = Plane::from_normal_point(Vec3::X, Vec3::new(-2.0, 0.0, 0.0)).unwrap();
        cam.set_clip_planes(&[plane]);
        cam
    };

    // 1. Optional features explicitly disabled (CLIP_DISTANCES and MULTI_DRAW_INDIRECT_COUNT off).
    {
        let Ok((device, queue)) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("iletki-3d resilience minimal features"),
            required_features: wgpu::Features::empty(),
            ..wgpu::DeviceDescriptor::default()
        })) else {
            return;
        };
        let (mut scene, ids) = build_scene();
        scene.set_selected(ids[528], true);
        let mut host = ViewportHost::new_with_adapter(
            &device,
            &adapter,
            ViewportConfig {
                initial_px: [256, 256],
                msaa_samples: 4,
                upload_budget: iletki_core::UploadBudget::UNLIMITED,
                ..ViewportConfig::default()
            },
        )
        .unwrap();
        *host.camera_mut() = test_camera();
        let mut enc = device.create_command_encoder(&Default::default());
        host.render_to_texture(&device, &queue, &mut enc, &(), &mut [&mut scene]);
        queue.submit([enc.finish()]);
        let img = host.readback_rgba_blocking(&device, &queue).unwrap();
        assert_eq!([img.width, img.height], [256, 256]);
        assert!(
            img.pixels.as_chunks::<4>().0.iter().any(|px| px[2] > 60),
            "minimal-feature device must render shaded geometry without optional features"
        );
        let hits = host
            .pick_rect_blocking(&device, &queue, &(), &mut [&mut scene], [64, 64, 128, 128])
            .unwrap();
        assert!(
            !hits.is_empty(),
            "minimal-feature device must return exact picks"
        );
    }

    // 2. Repeated resize across 256 px block boundaries with active Hi-Z, MRT selection & JFA outline,
    //    plus multi-viewport teardown and `prune_closed_viewports()` memory recovery.
    let adapter2 = if is_gl {
        let inst =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let Ok(a) = block_on(inst.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
            ..wgpu::RequestAdapterOptions::default()
        })) else {
            return;
        };
        a
    } else {
        adapter
    };
    let optional = adapter2.features()
        & (wgpu::Features::MULTI_DRAW_INDIRECT_COUNT | wgpu::Features::CLIP_DISTANCES);
    let Ok((device, queue)) = block_on(adapter2.request_device(&wgpu::DeviceDescriptor {
        label: Some("iletki-3d resilience full features"),
        required_features: optional,
        ..wgpu::DeviceDescriptor::default()
    })) else {
        return;
    };

    let (mut scene, ids) = build_scene();
    scene.set_selected(ids[528], true);
    scene.set_selected(ids[529], true);

    let make_host = |px: [u32; 2]| {
        let mut h = ViewportHost::new_with_adapter(
            &device,
            &adapter2,
            ViewportConfig {
                initial_px: px,
                msaa_samples: 4,
                upload_budget: iletki_core::UploadBudget::UNLIMITED,
                ..ViewportConfig::default()
            },
        )
        .unwrap();
        *h.camera_mut() = test_camera();
        h
    };

    let mut host = make_host([128, 128]);
    let supports_mrt = host.supports_color_frame_selection(&device);
    let sizes: [[u32; 2]; 9] = [
        [128, 128],
        [256, 256],
        [257, 257],
        [512, 512],
        [513, 300],
        [192, 192],
        [1, 1],
        [257, 193],
        [320, 240],
    ];
    let mut prev_gen = host.target_generation();
    let mut saw_gen_bump = false;
    for [w, h] in sizes {
        host.resize(&device, [w, h], 1.0);
        if supports_mrt && w >= 64 && h >= 64 {
            let rw = (w / 2).min(128);
            let rh = (h / 2).min(128);
            host.set_color_frame_selection_rect(Some([(w - rw) / 2, (h - rh) / 2, rw, rh]));
        } else {
            host.set_color_frame_selection_rect(None);
        }
        let mut enc = device.create_command_encoder(&Default::default());
        host.render_to_texture(&device, &queue, &mut enc, &(), &mut [&mut scene]);
        queue.submit([enc.finish()]);
        let cur_gen = host.target_generation();
        assert!(
            cur_gen >= prev_gen,
            "target generation must be monotonic across resizes"
        );
        if cur_gen > prev_gen {
            saw_gen_bump = true;
            prev_gen = cur_gen;
        }
        if supports_mrt && w >= 64 && h >= 64 {
            let mut pending = host
                .take_color_frame_selection_preview(&device)
                .expect("expected MRT preview after resize");
            let slots = loop {
                match pending.try_read() {
                    Poll::Ready(res) => break res.unwrap(),
                    Poll::Pending => std::thread::yield_now(),
                }
            };
            assert!(
                !slots.is_empty(),
                "expected non-empty MRT preview at {w}x{h}"
            );
        }
        let img = host.readback_rgba_blocking(&device, &queue).unwrap();
        assert_eq!([img.width, img.height], [w, h]);
    }
    assert!(
        saw_gen_bump,
        "crossing 256 px block boundaries must advance target_generation"
    );

    // 3. Multi-viewport teardown and `prune_closed_viewports()` memory recovery.
    host.set_color_frame_selection_rect(None);
    let mut extra = vec![
        make_host([160, 160]),
        make_host([160, 160]),
        make_host([160, 160]),
    ];
    {
        let mut enc = device.create_command_encoder(&Default::default());
        host.render_to_texture(&device, &queue, &mut enc, &(), &mut [&mut scene]);
        for vp in &mut extra {
            vp.render_to_texture(&device, &queue, &mut enc, &(), &mut [&mut scene]);
        }
        queue.submit([enc.finish()]);
    }
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    drop(extra);
    let pruned = scene.prune_closed_viewports();
    assert!(
        pruned >= 3,
        "expected at least 3 closed viewport bindings pruned, got {pruned}"
    );
    assert_eq!(scene.prune_closed_viewports(), 0);

    // Surviving viewport continues rendering and picking cleanly after sibling teardown.
    let mut enc = device.create_command_encoder(&Default::default());
    host.render_to_texture(&device, &queue, &mut enc, &(), &mut [&mut scene]);
    queue.submit([enc.finish()]);
    let hits = host
        .pick_rect_blocking(&device, &queue, &(), &mut [&mut scene], [80, 60, 160, 120])
        .unwrap();
    assert!(
        !hits.is_empty(),
        "surviving viewport must continue picking after sibling teardown"
    );
}

#[test]
fn material_ssbo_and_fallback_shade_update_section_and_pick() {
    let Some((device, queue)) = create_headless_device() else {
        return;
    };

    let mut host = ViewportHost::new(
        &device,
        ViewportConfig {
            color_format: wgpu::TextureFormat::Rgba8Unorm,
            msaa_samples: 4,
            clear_color: [0.05, 0.05, 0.06, 1.0],
            mode: ViewportMode::OnDemand,
            upload_budget: iletki_core::UploadBudget::UNLIMITED,
            initial_px: [240, 120],
            ..ViewportConfig::default()
        },
    )
    .unwrap();
    *host.camera_mut() = Camera::ortho_3d(
        Vec3::new(0.0, -10.0, 0.0),
        Vec3::ZERO,
        Vec3::Z,
        4.5,
        0.1,
        50.0,
    );

    let mut scene = Scene3DLayer::new(1);
    let make_white_box = |x: f64, obj: u32| {
        let mut m = Mesh3D::at_origin(Vec3::new(x, 0.0, 0.0));
        m.push_box(
            Vec3::new(-0.9, -0.9, -0.9),
            Vec3::new(0.9, 0.9, 0.9),
            [230, 230, 230, 255],
            PickId::new(1, obj, 0),
            false,
        );
        m
    };
    let left_id = scene.insert(make_white_box(-3.0, 10));
    let center_id = scene.insert(make_white_box(0.0, 20));
    let right_id = scene.insert(make_white_box(3.0, 30));

    let red_mat = scene.insert_material(Material3D::tinted([1.0, 0.1, 0.1, 1.0]));
    let green_unlit = scene.insert_material(Material3D {
        emissive: [0.0, 0.15, 0.0],
        ..Material3D::unlit([0.05, 0.9, 0.1])
    });
    assert_eq!(scene.chunk_material(left_id), Some(MaterialId::DEFAULT));
    scene.set_chunk_material(center_id, red_mat);
    scene.set_chunk_material(right_id, green_unlit);
    assert_eq!(scene.chunk_material(center_id), Some(red_mat));
    assert_eq!(scene.chunk_material(right_id), Some(green_unlit));

    // Hostile material updates/removals on default or out-of-bounds handles are safely ignored.
    assert!(!scene.update_material(MaterialId::DEFAULT, Material3D::unlit([1.0, 0.0, 0.0])));
    assert!(!scene.remove_material(MaterialId::DEFAULT));
    let hostile = scene.insert_material(Material3D {
        base_color_factor: [f32::NAN, f32::INFINITY, -1.0, f32::NAN],
        roughness: f32::NAN,
        metallic: f32::INFINITY,
        opacity: f32::NAN,
        backface_tint: f32::NEG_INFINITY,
        emissive: [f32::NAN, 0.0, 0.0],
        flags: MaterialFlags::NONE,
    });
    assert!(scene.remove_material(hostile));

    let render = |host: &mut ViewportHost, scene: &mut Scene3DLayer| {
        let mut enc = device.create_command_encoder(&Default::default());
        host.render_to_texture(&device, &queue, &mut enc, &(), &mut [scene]);
        queue.submit([enc.finish()]);
        host.readback_rgba_blocking(&device, &queue).unwrap()
    };

    // Frame 1: left neutral gray/white, center red-tinted, right unlit green.
    let img1 = render(&mut host, &mut scene);
    let left_px = img1.pixel(50, 60).unwrap();
    let center_px = img1.pixel(120, 60).unwrap();
    let right_px = img1.pixel(190, 60).unwrap();
    assert!(
        left_px[0].abs_diff(left_px[1]) <= 6
            && left_px[1].abs_diff(left_px[2]) <= 6
            && left_px[0] > 90,
        "default material box must remain neutral gray/white, got {left_px:?}"
    );
    assert!(
        center_px[0] > center_px[1].saturating_add(70)
            && center_px[0] > center_px[2].saturating_add(70),
        "red material box must shade red, got {center_px:?}"
    );
    assert!(
        right_px[1] > right_px[0].saturating_add(120)
            && right_px[1] > right_px[2].saturating_add(120),
        "unlit green material box must shade green, got {right_px:?}"
    );

    // Frame 2: live in-place update_material switches center from red to blue without geometry rebuild.
    assert!(scene.update_material(red_mat, Material3D::tinted([0.1, 0.2, 1.0, 1.0])));
    let img2 = render(&mut host, &mut scene);
    let center_blue = img2.pixel(120, 60).unwrap();
    assert!(
        center_blue[2] > center_blue[0].saturating_add(70)
            && center_blue[2] > center_blue[1].saturating_add(50),
        "updated material box must shade blue, got {center_blue:?}"
    );

    // Frame 3: live section clip plane + MRT preview (when supported) + exact 64-bit pick.
    if let Some(plane) = Plane::from_normal_point(Vec3::Z, Vec3::ZERO) {
        host.camera_mut().set_clip_planes(&[plane]);
    }
    let supports_mrt = host.supports_color_frame_selection(&device);
    if supports_mrt {
        host.set_color_frame_selection_rect(Some([10, 10, 220, 100]));
    }
    let img3 = render(&mut host, &mut scene);
    // Upper half (z > 0) stays rendered, lower half (z < 0) is cleanly clipped to background.
    let upper_center = img3.pixel(120, 45).unwrap();
    let lower_center = img3.pixel(120, 75).unwrap();
    assert!(
        upper_center[2] > upper_center[0].saturating_add(50),
        "unclipped upper half of blue box must remain blue, got {upper_center:?}"
    );
    assert!(
        lower_center[0] < 30 && lower_center[1] < 30 && lower_center[2] < 35,
        "clipped lower half must reveal background, got {lower_center:?}"
    );
    if supports_mrt {
        let mut pending = host.take_color_frame_selection_preview(&device).unwrap();
        let slots = loop {
            match pending.try_read() {
                Poll::Ready(res) => break res.unwrap(),
                Poll::Pending => std::thread::yield_now(),
            }
        };
        assert_eq!(
            slots.len(),
            3,
            "MRT preview with custom materials must report all 3 visible boxes, got {slots:?}"
        );
        host.set_color_frame_selection_rect(None);
    }
    let hits = host
        .pick_rect_blocking(&device, &queue, &(), &mut [&mut scene], [10, 10, 220, 50])
        .unwrap();
    let hit_objs: Vec<u32> = hits.iter().map(|h| h.id.object).collect();
    assert!(
        hit_objs.contains(&10) && hit_objs.contains(&20) && hit_objs.contains(&30),
        "exact pick must find all 3 boxes under custom materials, got {hit_objs:?}"
    );

    // Frame 4: removing material resets center chunk cleanly back to default neutral shading.
    host.camera_mut().set_clip_planes(&[]);
    assert!(scene.remove_material(red_mat));
    assert_eq!(scene.chunk_material(center_id), Some(MaterialId::DEFAULT));
    let img4 = render(&mut host, &mut scene);
    let center_reset = img4.pixel(120, 60).unwrap();
    assert!(
        center_reset[0].abs_diff(center_reset[1]) <= 6
            && center_reset[1].abs_diff(center_reset[2]) <= 6
            && center_reset[0] > 90,
        "center box must revert to neutral default shading after remove_material, got {center_reset:?}"
    );
}

struct StencilToggleLayer {
    enabled: bool,
}

impl ViewportLayer<()> for StencilToggleLayer {
    fn id(&self) -> u16 {
        42
    }

    fn phases(&self) -> DrawPhaseMask {
        DrawPhaseMask::NONE
    }

    fn requires_stencil(&self, _viewport_id: ViewportId) -> bool {
        self.enabled
    }

    fn prepare(&mut self, _ctx: &mut PrepareCtx<'_, ()>) -> bool {
        false
    }

    fn paint(&self, _phase: DrawPhase, _pass: &mut wgpu::RenderPass<'_>, _ctx: &PaintCtx<'_, ()>) {}
}

#[test]
fn lazy_stencil_depth_format_works_with_3d_scene_grid_and_gizmo() {
    let Some((device, queue)) = create_headless_device() else {
        return;
    };

    let mut host = ViewportHost::new(
        &device,
        ViewportConfig {
            color_format: wgpu::TextureFormat::Rgba8Unorm,
            msaa_samples: 4,
            upload_budget: iletki_core::UploadBudget::UNLIMITED,
            initial_px: [160, 160],
            ..ViewportConfig::default()
        },
    )
    .unwrap();
    *host.camera_mut() = Camera::perspective_3d(
        Vec3::new(4.0, -5.0, 4.0),
        Vec3::ZERO,
        Vec3::Z,
        45_f64.to_radians(),
        0.1,
    );

    let mut scene = Scene3DLayer::new(1);
    let mut mesh = Mesh3D::at_origin(Vec3::ZERO);
    mesh.push_box_with_edges(
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(1.0, 1.0, 1.0),
        [180, 200, 220, 255],
        [20, 25, 30, 255],
        PickId::new(1, 77, 0),
        true,
    );
    let chunk = scene.insert(mesh);
    let mat = scene.insert_material(Material3D::tinted([0.9, 0.5, 0.2, 1.0]));
    scene.set_chunk_material(chunk, mat);
    scene.set_selected(chunk, true);
    let mut grid = InfiniteGrid3DLayer::new(2, Grid3DConfig::default());
    let mut gizmo = TransformGizmoLayer::new(3, 900);
    gizmo.origin = Some(Vec3::ZERO);
    let mut toggle = StencilToggleLayer { enabled: false };

    let render_all = |host: &mut ViewportHost,
                      scene: &mut Scene3DLayer,
                      grid: &mut InfiniteGrid3DLayer,
                      gizmo: &mut TransformGizmoLayer,
                      toggle: &mut StencilToggleLayer| {
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut layers: [&mut dyn ViewportLayer<()>; 4] = [scene, grid, gizmo, toggle];
            host.render_to_texture(&device, &queue, &mut enc, &(), &mut layers);
        }
        queue.submit([enc.finish()]);
        host.readback_rgba_blocking(&device, &queue).unwrap()
    };

    // Pass 1: plain Depth32Float.
    let img_depth = render_all(&mut host, &mut scene, &mut grid, &mut gizmo, &mut toggle);
    assert_eq!(host.depth_format(), DEPTH_FORMAT);

    // Pass 2: lazily upgrade to Depth32FloatStencil8 across Scene3D + Grid + Gizmo + Outline.
    toggle.enabled = true;
    let img_stencil = render_all(&mut host, &mut scene, &mut grid, &mut gizmo, &mut toggle);
    if host.supports_stencil(&device) {
        assert_eq!(host.depth_format(), DEPTH_STENCIL_FORMAT);
    }
    assert_eq!([img_depth.width, img_depth.height], [160, 160]);
    assert_eq!(img_depth.pixels.len(), img_stencil.pixels.len());

    // Exact 64-bit picking remains healthy while viewport is on Depth32FloatStencil8.
    let hits = {
        let mut layers: [&mut dyn ViewportLayer<()>; 4] =
            [&mut scene, &mut grid, &mut gizmo, &mut toggle];
        host.pick_rect_blocking(&device, &queue, &(), &mut layers, [40, 40, 80, 80])
            .unwrap()
    };
    assert!(
        hits.iter().any(|h| h.id.object == 77),
        "scene box must remain pickable while viewport depth format is stencil-enabled"
    );

    // Pass 3: downgrade back to Depth32Float cleanly.
    toggle.enabled = false;
    let _img_back = render_all(&mut host, &mut scene, &mut grid, &mut gizmo, &mut toggle);
    assert_eq!(host.depth_format(), DEPTH_FORMAT);
}
