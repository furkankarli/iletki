//! Headless GPU integration tests for `iletki-3d`:
//! - Camera-relative Reversed-Z 3D cube at national-grid coordinates `(661_858, 4_491_159, 120)`
//! - 3D section clip plane slicing (`Camera::set_clip_planes`)
//! - 64-bit sub-object face picking + exact `f64` world point reconstruction from Reversed-Z depth
//! - Infinite anti-aliased ground grid and screen-size-invariant 3D transform gizmo axis picking

#![cfg(not(target_arch = "wasm32"))]

use iletki_3d::{
    DiagnosticView, GizmoAxis, Grid3DConfig, GtaoConfig, InfiniteGrid3DLayer, Material3D,
    MaterialFlags, MaterialId, Mesh3D, Scene3DLayer, SectionCapConfig, SectionCapPattern,
    TransformGizmoLayer, VisualStyle,
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
    let required_features = adapter.features() & wgpu::Features::DEPTH32FLOAT_STENCIL8;
    block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("iletki-3d test device"),
        required_features,
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

fn create_device_with_clip_distances() -> Option<(wgpu::Device, wgpu::Queue)> {
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
    if !adapter.features().contains(wgpu::Features::CLIP_DISTANCES) {
        return None;
    }
    let required_features = wgpu::Features::CLIP_DISTANCES
        | (adapter.features() & wgpu::Features::DEPTH32FLOAT_STENCIL8);
    block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("iletki-3d clip_distances test device"),
        required_features,
        required_limits: wgpu::Limits::default(),
        ..wgpu::DeviceDescriptor::default()
    }))
    .ok()
}

fn verify_section_caps_on_device(device: &wgpu::Device, queue: &wgpu::Queue, label: &str) {
    let mut host = ViewportHost::new(
        device,
        ViewportConfig {
            color_format: wgpu::TextureFormat::Rgba8Unorm,
            msaa_samples: 4,
            clear_color: [0.05, 0.05, 0.07, 1.0],
            upload_budget: iletki_core::UploadBudget::UNLIMITED,
            initial_px: [200, 200],
            ..ViewportConfig::default()
        },
    )
    .ok()
    .into_iter()
    .next();
    let Some(mut host) = host.take() else {
        return;
    };
    if !host.supports_stencil(device) {
        return;
    }

    let mut scene = Scene3DLayer::new(1);
    let mut box_mesh = Mesh3D::at_origin(Vec3::ZERO);
    box_mesh.push_box(
        Vec3::new(-1.5, -1.5, -2.0),
        Vec3::new(1.5, 1.5, 2.0),
        [185, 185, 185, 255],
        PickId::new(1, 501, 0),
        false,
    );
    let chunk = scene.insert(box_mesh);

    let render = |host: &mut ViewportHost, scene: &mut Scene3DLayer| {
        let mut enc = device.create_command_encoder(&Default::default());
        host.render_to_texture(device, queue, &mut enc, &(), &mut [scene]);
        queue.submit([enc.finish()]);
        host.readback_rgba_blocking(device, queue).ok()
    };

    // View from +Z looking down into a Z <= 0 cut plane (kept side is -Z).
    *host.camera_mut() = Camera::perspective_3d(
        Vec3::new(0.0, 0.0, 5.5),
        Vec3::ZERO,
        Vec3::Y,
        50_f64.to_radians(),
        0.1,
    );
    if let Some(plane_top) = Plane::from_normal_point(-Vec3::Z, Vec3::ZERO) {
        host.camera_mut().set_clip_planes(&[plane_top]);
    }

    // 1. Caps disabled (zero-tax default): stays Depth32Float and interior shows neutral gray.
    let img_off = render(&mut host, &mut scene);
    assert_eq!(host.depth_format(), DEPTH_FORMAT, "{label}: caps off depth");
    if let Some(img) = img_off
        && let Some(px) = img.pixel(100, 100)
    {
        assert!(
            px[0].abs_diff(px[1]) <= 8 && px[1].abs_diff(px[2]) <= 8 && px[0] > 50,
            "{label}: uncapped cut interior must show neutral gray back-face, got {px:?}"
        );
    }

    // 2. Enable solid red section cap: upgrades to Depth32FloatStencil8 and fills center red.
    scene.set_section_caps(SectionCapConfig::solid([0.95, 0.1, 0.1, 1.0]));
    let img_solid = render(&mut host, &mut scene);
    assert_eq!(
        host.depth_format(),
        DEPTH_STENCIL_FORMAT,
        "{label}: caps on depth"
    );
    if let Some(img) = img_solid {
        if let Some(center) = img.pixel(100, 100) {
            assert!(
                center[0] > 200 && center[1] < 55 && center[2] < 55,
                "{label}: solid red cap must fill cut center, got {center:?}"
            );
        }
        if let Some(corner) = img.pixel(5, 5) {
            assert!(
                corner[0] < 30 && corner[1] < 30 && corner[2] < 35,
                "{label}: background outside solid must stay dark, got {corner:?}"
            );
        }
    }

    // 3. View from the KEPT side (-Z looking toward +Z): cut face is not viewer-facing -> no cap.
    *host.camera_mut() = Camera::perspective_3d(
        Vec3::new(0.0, 0.0, -5.5),
        Vec3::ZERO,
        Vec3::Y,
        50_f64.to_radians(),
        0.1,
    );
    if let Some(plane_top) = Plane::from_normal_point(-Vec3::Z, Vec3::ZERO) {
        host.camera_mut().set_clip_planes(&[plane_top]);
    }
    if let Some(img_back) = render(&mut host, &mut scene)
        && let Some(center) = img_back.pixel(100, 100)
    {
        assert!(
            center[0].abs_diff(center[2]) <= 12 && center[0] > 60,
            "{label}: kept-side exterior face must not be capped red, got {center:?}"
        );
    }

    // 4. Opposing parallel clip planes (slab cut: -0.5 <= z <= +0.5).
    //    Pass A must clip ONLY by the active plane so the far back face at z=-2 still stamps stencil!
    *host.camera_mut() = Camera::perspective_3d(
        Vec3::new(0.0, 0.0, 5.5),
        Vec3::ZERO,
        Vec3::Y,
        50_f64.to_radians(),
        0.1,
    );
    if let (Some(p_hi), Some(p_lo)) = (
        Plane::from_normal_point(-Vec3::Z, Vec3::new(0.0, 0.0, 0.5)),
        Plane::from_normal_point(Vec3::Z, Vec3::new(0.0, 0.0, -0.5)),
    ) {
        host.camera_mut().set_clip_planes(&[p_hi, p_lo]);
    }
    if let Some(img_slab) = render(&mut host, &mut scene)
        && let Some(center) = img_slab.pixel(100, 100)
    {
        assert!(
            center[0] > 200 && center[1] < 55 && center[2] < 55,
            "{label}: opposing parallel slab cut must still cap cleanly, got {center:?}"
        );
    }

    // 5. Two perpendicular clip planes (corner cutaway: x >= 0 AND y >= 0, viewed from (-4, -4, 0)).
    *host.camera_mut() = Camera::perspective_3d(
        Vec3::new(-4.5, -4.5, 0.0),
        Vec3::new(0.5, 0.5, 0.0),
        Vec3::Z,
        55_f64.to_radians(),
        0.1,
    );
    if let (Some(px_plane), Some(py_plane)) = (
        Plane::from_normal_point(Vec3::X, Vec3::ZERO),
        Plane::from_normal_point(Vec3::Y, Vec3::ZERO),
    ) {
        host.camera_mut().set_clip_planes(&[px_plane, py_plane]);
    }
    if let Some(img_corner) = render(&mut host, &mut scene) {
        let mut red_left = 0usize;
        let mut red_right = 0usize;
        for y in 70..130 {
            for x in 30..98 {
                if let Some(p) = img_corner.pixel(x, y)
                    && p[0] > 190
                    && p[1] < 60
                    && p[2] < 60
                {
                    red_left += 1;
                }
            }
            for x in 102..170 {
                if let Some(p) = img_corner.pixel(x, y)
                    && p[0] > 190
                    && p[1] < 60
                    && p[2] < 60
                {
                    red_right += 1;
                }
            }
        }
        assert!(
            red_left > 100 && red_right > 100,
            "{label}: corner cutaway must cap both x=0 and y=0 faces (left={red_left}, right={red_right})"
        );
    }

    // 6. Hatch vs CrossHatch patterns on a clean Z <= 0 cut.
    *host.camera_mut() = Camera::perspective_3d(
        Vec3::new(0.0, 0.0, 5.5),
        Vec3::ZERO,
        Vec3::Y,
        50_f64.to_radians(),
        0.1,
    );
    if let Some(plane_top) = Plane::from_normal_point(-Vec3::Z, Vec3::ZERO) {
        host.camera_mut().set_clip_planes(&[plane_top]);
    }
    let count_white_and_blue = |img: &iletki_wgpu::RgbaImage| -> (usize, usize) {
        let mut white = 0usize;
        let mut blue = 0usize;
        for y in 60..140 {
            for x in 60..140 {
                if let Some(p) = img.pixel(x, y) {
                    if p[0] > 210 && p[1] > 210 && p[2] > 210 {
                        white += 1;
                    } else if p[2] > p[0].saturating_add(80) && p[2] > 150 {
                        blue += 1;
                    }
                }
            }
        }
        (white, blue)
    };
    scene.set_section_caps(SectionCapConfig {
        hatch_width_px: 2.5,
        ..SectionCapConfig::hatch([0.96, 0.96, 0.96, 1.0], [0.05, 0.1, 0.96, 1.0], 0.35)
    });
    let (hatch_white, hatch_blue) = render(&mut host, &mut scene)
        .map(|img| count_white_and_blue(&img))
        .unwrap_or((0, 0));
    assert!(
        hatch_white > 1000 && hatch_blue > 150,
        "{label}: hatch cap must show both white base ({hatch_white}) and blue hatch lines ({hatch_blue})"
    );

    scene.set_section_caps(SectionCapConfig {
        hatch_width_px: 2.5,
        ..SectionCapConfig::cross_hatch([0.96, 0.96, 0.96, 1.0], [0.05, 0.1, 0.96, 1.0], 0.35)
    });
    assert_eq!(scene.section_caps().pattern, SectionCapPattern::CrossHatch);
    let (cross_white, cross_blue) = render(&mut host, &mut scene)
        .map(|img| count_white_and_blue(&img))
        .unwrap_or((0, 0));
    assert!(
        cross_white > 500 && cross_blue > hatch_blue,
        "{label}: cross-hatch must add perpendicular blue lines ({cross_blue} > {hatch_blue}, white={cross_white})"
    );

    // 6b. Hollow thick-walled room (outer CCW box + inner inward-facing cavity):
    //     Two-sided stencil wrap must cap ONLY the solid walls and leave the hollow interior open!
    let mut hollow = Mesh3D::at_origin(Vec3::ZERO);
    hollow.push_box(
        Vec3::new(-1.5, -1.5, -2.0),
        Vec3::new(1.5, 1.5, 2.0),
        [185, 185, 185, 255],
        PickId::new(1, 502, 0),
        false,
    );
    let inner_start = hollow.vertices.len();
    hollow.push_box(
        Vec3::new(-0.7, -0.7, -1.5),
        Vec3::new(0.7, 0.7, 1.5),
        [160, 160, 160, 255],
        PickId::new(1, 503, 0),
        false,
    );
    // Flip inner cavity triangles to inward-facing (hollow void inside solid shell).
    if let Some(inner) = hollow.vertices.get_mut(inner_start..) {
        for tri in inner.as_chunks_mut::<3>().0 {
            tri.swap(1, 2);
            for v in tri.iter_mut() {
                v.normal = [-v.normal[0], -v.normal[1], -v.normal[2]];
            }
        }
    }
    scene.remove(chunk);
    let hollow_chunk = scene.insert(hollow);
    scene.set_section_caps(SectionCapConfig::solid([0.95, 0.1, 0.1, 1.0]));
    if let Some(img_hollow) = render(&mut host, &mut scene) {
        // Center (x=0, y=0) is inside the hollow cavity -> stencil == 0 -> NOT capped red.
        if let Some(cavity) = img_hollow.pixel(100, 100) {
            assert!(
                cavity[0].abs_diff(cavity[2]) <= 15 && cavity[0] < 200,
                "{label}: hollow cavity interior must stay uncapped gray, got {cavity:?}"
            );
        }
        // Solid wall cross-section (between 0.7 and 1.5 -> ~35 px off center) MUST be capped red.
        let wall_red = (128..146)
            .filter_map(|x| img_hollow.pixel(x, 100))
            .filter(|p| p[0] > 200 && p[1] < 55 && p[2] < 55)
            .count();
        assert!(
            wall_red >= 8,
            "{label}: solid wall cross-section around hollow cavity must be capped red, got {wall_red} px"
        );
    }

    // 7. Large-world UTM coordinates + Ortho3D + Transparent InfiniteGrid3DLayer + MRT + Pick.
    scene.remove(hollow_chunk);
    let utm = Vec3::new(661_858.0, 4_491_159.0, 120.0);
    let mut utm_mesh = Mesh3D::at_origin(utm);
    utm_mesh.push_box(
        Vec3::new(-2.0, -2.0, -2.0),
        Vec3::new(2.0, 2.0, 2.0),
        [180, 180, 180, 255],
        PickId::new(1, 777, 0),
        false,
    );
    let _utm_chunk = scene.insert(utm_mesh);
    scene.set_section_caps(SectionCapConfig::solid([0.1, 0.92, 0.2, 1.0]));
    *host.camera_mut() = Camera::ortho_3d(utm + Vec3::Z * 10.0, utm, Vec3::Y, 6.0, 0.1, 50.0);
    if let Some(utm_plane) = Plane::from_normal_point(-Vec3::Z, utm) {
        host.camera_mut().set_clip_planes(&[utm_plane]);
    }
    let mut grid = InfiniteGrid3DLayer::new(2, Grid3DConfig::default());
    let supports_mrt = host.supports_color_frame_selection(device);
    if supports_mrt {
        host.set_color_frame_selection_rect(Some([50, 50, 100, 100]));
    }
    {
        let mut enc = device.create_command_encoder(&Default::default());
        let mut layers: [&mut dyn ViewportLayer<()>; 2] = [&mut scene, &mut grid];
        host.render_to_texture(device, queue, &mut enc, &(), &mut layers);
        queue.submit([enc.finish()]);
    }
    if let Ok(img_utm) = host.readback_rgba_blocking(device, queue)
        && let Some(center) = img_utm.pixel(100, 100)
    {
        assert!(
            center[1] > 190 && center[0] < 60 && center[2] < 80,
            "{label}: UTM Ortho3D cap must fill cut center green, got {center:?}"
        );
    }
    if supports_mrt {
        if let Some(mut pending) = host.take_color_frame_selection_preview(device) {
            let slots = loop {
                match pending.try_read() {
                    Poll::Ready(res) => break res.ok(),
                    Poll::Pending => std::thread::yield_now(),
                }
            };
            assert!(
                slots.as_ref().is_some_and(|s| !s.is_empty()),
                "{label}: MRT preview with active section caps must still report visible chunk"
            );
        }
        host.set_color_frame_selection_rect(None);
    }
    if let Ok(hits) =
        host.pick_rect_blocking(device, queue, &(), &mut [&mut scene], [80, 80, 40, 40])
    {
        assert!(
            hits.iter().any(|h| h.id.object == 777),
            "{label}: exact 64-bit pick must still hit UTM box under active section caps"
        );
    }

    // 8. Disabling section caps cleanly downgrades viewport back to Depth32Float.
    scene.set_section_caps(SectionCapConfig::default());
    let _img_disabled = render(&mut host, &mut scene);
    assert_eq!(
        host.depth_format(),
        DEPTH_FORMAT,
        "{label}: disabling caps must downgrade depth format back to Depth32Float"
    );
}

#[test]
fn section_caps_solid_hatch_multiplane_and_utm_precision() {
    if let Some((device, queue)) = create_headless_device() {
        verify_section_caps_on_device(&device, &queue, "discard_fallback");
    }
    if let Some((device, queue)) = create_device_with_clip_distances() {
        verify_section_caps_on_device(&device, &queue, "clip_distances");
    }
}

#[test]
fn section_caps_multi_viewport_and_continuous_mode_isolated() {
    let Some((device, queue)) = create_headless_device() else {
        return;
    };
    let mut host_a = ViewportHost::new(
        &device,
        ViewportConfig {
            mode: ViewportMode::Continuous,
            color_format: wgpu::TextureFormat::Rgba8Unorm,
            msaa_samples: 4,
            clear_color: [0.05, 0.05, 0.07, 1.0],
            upload_budget: iletki_core::UploadBudget::UNLIMITED,
            initial_px: [128, 128],
            ..ViewportConfig::default()
        },
    )
    .unwrap();
    let mut host_b = ViewportHost::new(
        &device,
        ViewportConfig {
            mode: ViewportMode::OnDemand,
            color_format: wgpu::TextureFormat::Rgba8Unorm,
            msaa_samples: 4,
            clear_color: [0.05, 0.05, 0.07, 1.0],
            upload_budget: iletki_core::UploadBudget::UNLIMITED,
            initial_px: [128, 128],
            ..ViewportConfig::default()
        },
    )
    .unwrap();
    if !host_a.supports_stencil(&device) {
        return;
    }

    // Viewport A: sliced at z <= 0, viewing cut face from +Z (should show bright red cap).
    *host_a.camera_mut() = Camera::perspective_3d(
        Vec3::new(0.0, 0.0, 5.0),
        Vec3::ZERO,
        Vec3::Y,
        50_f64.to_radians(),
        0.1,
    );
    host_a
        .camera_mut()
        .set_clip_planes(&[Plane::from_normal_point(-Vec3::Z, Vec3::ZERO).unwrap()]);

    // Viewport B: sliced at x >= 0, viewing from -X (different clip plane in the same encoder!).
    *host_b.camera_mut() = Camera::perspective_3d(
        Vec3::new(-5.0, 0.0, 0.0),
        Vec3::ZERO,
        Vec3::Z,
        50_f64.to_radians(),
        0.1,
    );
    host_b
        .camera_mut()
        .set_clip_planes(&[Plane::from_normal_point(Vec3::X, Vec3::ZERO).unwrap()]);

    let mut scene = Scene3DLayer::new(1);
    let mut mesh = Mesh3D::at_origin(Vec3::ZERO);
    mesh.push_box(
        Vec3::new(-1.5, -1.5, -1.5),
        Vec3::new(1.5, 1.5, 1.5),
        [185, 185, 185, 255],
        PickId::new(1, 900, 0),
        false,
    );
    scene.insert(mesh);
    scene.set_section_caps(SectionCapConfig::solid([0.95, 0.1, 0.1, 1.0]));

    // Render BOTH viewports into one shared CommandEncoder (quad-view pattern).
    let mut encoder = device.create_command_encoder(&Default::default());
    host_a.render_to_texture(&device, &queue, &mut encoder, &(), &mut [&mut scene]);
    host_b.render_to_texture(&device, &queue, &mut encoder, &(), &mut [&mut scene]);
    queue.submit([encoder.finish()]);

    let img_a = host_a.readback_rgba_blocking(&device, &queue).unwrap();
    let img_b = host_b.readback_rgba_blocking(&device, &queue).unwrap();
    let px_a = img_a.pixel(64, 64).unwrap();
    let px_b = img_b.pixel(64, 64).unwrap();
    assert!(
        px_a[0] > 200 && px_a[1] < 55 && px_a[2] < 55,
        "Continuous MSAA viewport A must show solid red cap, got {px_a:?}"
    );
    assert!(
        px_b[0] > 200 && px_b[1] < 55 && px_b[2] < 55,
        "OnDemand viewport B with distinct clip plane in same encoder must also cap cleanly, got {px_b:?}"
    );
    assert_eq!(host_a.depth_format(), DEPTH_STENCIL_FORMAT);
    assert_eq!(host_b.depth_format(), DEPTH_STENCIL_FORMAT);

    // Also verify 1x MSAA + 144 repeated packed boxes (exercises packed/repeated/GPU-cull stencil paths).
    let mut host_1x = ViewportHost::new(
        &device,
        ViewportConfig {
            color_format: wgpu::TextureFormat::Rgba8Unorm,
            msaa_samples: 1,
            clear_color: [0.05, 0.05, 0.07, 1.0],
            upload_budget: iletki_core::UploadBudget::UNLIMITED,
            initial_px: [160, 160],
            ..ViewportConfig::default()
        },
    )
    .unwrap();
    *host_1x.camera_mut() = Camera::ortho_3d(
        Vec3::new(0.0, 0.0, 12.0),
        Vec3::ZERO,
        Vec3::Y,
        16.0,
        0.1,
        50.0,
    );
    host_1x
        .camera_mut()
        .set_clip_planes(&[Plane::from_normal_point(-Vec3::Z, Vec3::ZERO).unwrap()]);
    let mut packed_scene = Scene3DLayer::new(2);
    for idx in 0..144 {
        let gx = f64::from(idx % 12) * 1.2 - 6.6;
        let gy = f64::from(idx / 12) * 1.2 - 6.6;
        let mut m = Mesh3D::at_origin(Vec3::new(gx, gy, 0.0));
        m.push_box(
            Vec3::new(-0.45, -0.45, -1.0),
            Vec3::new(0.45, 0.45, 1.0),
            [180, 180, 180, 255],
            PickId::new(1, idx + 1, 0),
            false,
        );
        packed_scene.insert(m);
    }
    packed_scene.set_section_caps(SectionCapConfig::solid([0.95, 0.1, 0.1, 1.0]));
    let mut enc = device.create_command_encoder(&Default::default());
    host_1x.render_to_texture(&device, &queue, &mut enc, &(), &mut [&mut packed_scene]);
    queue.submit([enc.finish()]);
    let img_1x = host_1x.readback_rgba_blocking(&device, &queue).unwrap();
    let red_pixels = img_1x
        .pixels
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|p| p[0] > 200 && p[1] < 55 && p[2] < 55)
        .count();
    assert!(
        red_pixels > 2000,
        "1x MSAA 144-box repeated/packed scene must cap cross-sections cleanly, got {red_pixels} red pixels"
    );
}

#[test]
fn wboit_order_independent_blending_ghost_and_opacity_at_4x_and_1x_msaa() {
    let Some((device, queue)) = create_headless_device() else {
        return;
    };

    for msaa in [4u32, 1u32] {
        let make_host = || {
            let mut host = ViewportHost::new(
                &device,
                ViewportConfig {
                    color_format: wgpu::TextureFormat::Rgba8Unorm,
                    msaa_samples: msaa,
                    clear_color: [0.04, 0.04, 0.05, 1.0],
                    upload_budget: iletki_core::UploadBudget::UNLIMITED,
                    initial_px: [160, 160],
                    ..ViewportConfig::default()
                },
            )
            .unwrap();
            *host.camera_mut() = Camera::perspective_3d(
                Vec3::new(0.0, 0.0, 6.0),
                Vec3::ZERO,
                Vec3::Y,
                48_f64.to_radians(),
                0.1,
            );
            host
        };
        let make_pane = |z: f64, color: [u8; 4], obj: u32| {
            let mut m = Mesh3D::at_origin(Vec3::new(0.0, 0.0, z));
            m.push_box(
                Vec3::new(-1.4, -1.4, -0.1),
                Vec3::new(1.4, 1.4, 0.1),
                color,
                PickId::new(1, obj, 0),
                false,
            );
            m
        };

        // 1. Order independence: (Front Red, Back Blue) vs (Back Blue, Front Red) must match!
        let render_pair = |front_first: bool| {
            let mut host = make_host();
            let mut scene = Scene3DLayer::new(1);
            scene.set_oit_enabled(true);
            assert!(scene.oit_enabled());
            if front_first {
                scene.insert(make_pane(0.5, [240, 30, 30, 130], 1));
                scene.insert(make_pane(-1.8, [30, 50, 245, 130], 2));
            } else {
                scene.insert(make_pane(-1.8, [30, 50, 245, 130], 2));
                scene.insert(make_pane(0.5, [240, 30, 30, 130], 1));
            }
            let mut enc = device.create_command_encoder(&Default::default());
            host.render_to_texture(&device, &queue, &mut enc, &(), &mut [&mut scene]);
            queue.submit([enc.finish()]);
            host.readback_rgba_blocking(&device, &queue).unwrap()
        };
        let img_ab = render_pair(true);
        let img_ba = render_pair(false);
        let px_ab = img_ab.pixel(80, 80).unwrap();
        let px_ba = img_ba.pixel(80, 80).unwrap();
        for ch in 0..3 {
            assert!(
                px_ab[ch].abs_diff(px_ba[ch]) <= 2,
                "msaa={msaa}: WBOIT must be order-independent (ab={px_ab:?}, ba={px_ba:?})"
            );
        }
        assert!(
            px_ab[0] > px_ab[2] && px_ab[0] > 90 && px_ab[2] > 18,
            "msaa={msaa}: front red must dominate while back blue shines through, got {px_ab:?}"
        );

        // 2. Opaque occlusion + per-chunk ghost & opacity + live toggle back to opaque + picking.
        let mut host = make_host();
        let mut scene = Scene3DLayer::new(2);
        let front_id = scene.insert(make_pane(0.8, [230, 230, 230, 255], 10));
        let back_id = scene.insert(make_pane(-1.2, [20, 235, 30, 255], 20));
        assert!(!scene.is_chunk_ghost(front_id));
        assert_eq!(scene.chunk_opacity(front_id), Some(1.0));

        let render_scene = |host: &mut ViewportHost, scene: &mut Scene3DLayer| {
            let mut enc = device.create_command_encoder(&Default::default());
            host.render_to_texture(&device, &queue, &mut enc, &(), &mut [scene]);
            queue.submit([enc.finish()]);
            host.readback_rgba_blocking(&device, &queue).unwrap()
        };

        // Baseline opaque: front gray completely hides back green.
        let px_opaque = render_scene(&mut host, &mut scene).pixel(80, 80).unwrap();
        assert!(
            px_opaque[0].abs_diff(px_opaque[1]) <= 8 && px_opaque[0] > 80,
            "msaa={msaa}: opaque front must hide back green, got {px_opaque:?}"
        );

        // Per-chunk ghost: back green shines clearly through ghosted front box!
        scene.set_chunk_ghost(front_id, true);
        assert!(scene.is_chunk_ghost(front_id));
        let px_ghost = render_scene(&mut host, &mut scene).pixel(80, 80).unwrap();
        assert!(
            px_ghost[1] > px_ghost[0].saturating_add(15),
            "msaa={msaa}: ghosted front box must reveal green box behind it, got {px_ghost:?}"
        );

        // Exact 64-bit pick still hits the ghosted front chunk.
        let hits = host
            .pick_rect_blocking(&device, &queue, &(), &mut [&mut scene], [70, 70, 20, 20])
            .unwrap();
        assert!(
            hits.iter().any(|h| h.id.object == 10),
            "msaa={msaa}: ghosted chunk must remain pickable"
        );

        // Per-chunk opacity: 0.3 -> green visible; 0.0 -> full green; 1.0 -> restores opaque gray.
        scene.set_chunk_ghost(front_id, false);
        scene.set_chunk_opacity(front_id, 0.3);
        let px_op30 = render_scene(&mut host, &mut scene).pixel(80, 80).unwrap();
        assert!(
            px_op30[1] > px_op30[0].saturating_add(25),
            "msaa={msaa}: 0.3 opacity front must reveal back green, got {px_op30:?}"
        );

        scene.set_chunk_opacity(front_id, 0.0);
        let px_op0 = render_scene(&mut host, &mut scene).pixel(80, 80).unwrap();
        assert!(
            px_op0[1] > 120 && px_op0[0] < 40,
            "msaa={msaa}: 0.0 opacity front must vanish completely, got {px_op0:?}"
        );

        // Also test opaque occlusion in front of a translucent chunk.
        scene.set_chunk_opacity(front_id, 1.0);
        scene.set_chunk_opacity(back_id, 0.4);
        let px_restored = render_scene(&mut host, &mut scene).pixel(80, 80).unwrap();
        assert!(
            px_restored[0].abs_diff(px_restored[1]) <= 8 && px_restored[0] > 80,
            "msaa={msaa}: restored opaque front must occlude translucent back, got {px_restored:?}"
        );
        scene.set_chunk_opacity(back_id, 1.0);
        let _px_clean = render_scene(&mut host, &mut scene);
    }

    // 3. Coexistence: Section Caps + WBOIT in the same frame & PostOpaque hook.
    let mut host = ViewportHost::new(
        &device,
        ViewportConfig {
            color_format: wgpu::TextureFormat::Rgba8Unorm,
            msaa_samples: 4,
            clear_color: [0.04, 0.04, 0.05, 1.0],
            upload_budget: iletki_core::UploadBudget::UNLIMITED,
            initial_px: [200, 140],
            ..ViewportConfig::default()
        },
    )
    .unwrap();
    if host.supports_stencil(&device) {
        *host.camera_mut() = Camera::ortho_3d(
            Vec3::new(0.0, 0.0, 8.0),
            Vec3::ZERO,
            Vec3::Y,
            4.5,
            0.1,
            30.0,
        );
        host.camera_mut()
            .set_clip_planes(&[Plane::from_normal_point(-Vec3::Z, Vec3::ZERO).unwrap()]);
        let mut scene = Scene3DLayer::new(3);
        // Left: solid gray box sliced at z <= 0 -> capped bright red.
        let mut left = Mesh3D::at_origin(Vec3::new(-1.4, 0.0, 0.0));
        left.push_box(
            Vec3::new(-1.0, -1.0, -1.5),
            Vec3::new(1.0, 1.0, 1.5),
            [180, 180, 180, 255],
            PickId::new(1, 31, 0),
            false,
        );
        scene.insert(left);
        // Right: translucent blue box sliced at z <= 0 -> NOT capped, rendered via WBOIT blue.
        let mut right = Mesh3D::at_origin(Vec3::new(1.4, 0.0, 0.0));
        right.push_box(
            Vec3::new(-1.0, -1.0, -1.5),
            Vec3::new(1.0, 1.0, 1.5),
            [40, 80, 245, 255],
            PickId::new(1, 32, 0),
            false,
        );
        let right_id = scene.insert(right);
        let glass = scene.insert_material(Material3D::translucent([0.2, 0.4, 1.0, 0.45]));
        scene.set_chunk_material(right_id, glass);
        scene.set_section_caps(SectionCapConfig::solid([0.95, 0.1, 0.1, 1.0]));

        let mut enc = device.create_command_encoder(&Default::default());
        host.render_to_texture(&device, &queue, &mut enc, &(), &mut [&mut scene]);
        queue.submit([enc.finish()]);
        let img = host.readback_rgba_blocking(&device, &queue).unwrap();
        let left_px = img.pixel(55, 70).unwrap();
        let right_px = img.pixel(145, 70).unwrap();
        assert!(
            left_px[0] > 200 && left_px[1] < 55 && left_px[2] < 55,
            "left cut solid must receive red stencil cap alongside WBOIT, got {left_px:?}"
        );
        assert!(
            right_px[2] > right_px[0].saturating_add(40),
            "right translucent box must render blue via WBOIT without red cap, got {right_px:?}"
        );
    }
}

#[test]
fn cad_visual_styles_and_diagnostic_views_render_and_pick_cleanly() {
    let Some((device, queue)) = create_headless_device() else {
        return;
    };
    let mut host = ViewportHost::new(
        &device,
        ViewportConfig {
            color_format: wgpu::TextureFormat::Rgba8Unorm,
            msaa_samples: 4,
            clear_color: [0.05, 0.06, 0.08, 1.0],
            upload_budget: iletki_core::UploadBudget::UNLIMITED,
            initial_px: [240, 160],
            ..ViewportConfig::default()
        },
    )
    .unwrap();
    *host.camera_mut() = Camera::ortho_3d(
        Vec3::new(0.0, 0.0, 10.0),
        Vec3::ZERO,
        Vec3::Y,
        5.0,
        0.1,
        30.0,
    );

    let mut scene = Scene3DLayer::new(1);
    let make_edged_box = |x: f64, z: f64, color: [u8; 4], obj: u32| {
        let mut m = Mesh3D::at_origin(Vec3::new(x, 0.0, z));
        m.push_box_with_edges(
            Vec3::new(-0.9, -0.9, -0.5),
            Vec3::new(0.9, 0.9, 0.5),
            color,
            [15, 15, 20, 255],
            PickId::new(1, obj, 0),
            true,
        );
        m
    };
    // Left box (near z=+1.5), center box (z=0.0) with a second overlapping box directly behind it
    // (for XRay & Overdraw), and an inverted back-facing quad on the right (for Backfaces).
    let _left = scene.insert(make_edged_box(-2.4, 1.5, [210, 180, 140, 255], 101));
    let _center_front = scene.insert(make_edged_box(0.0, 0.2, [190, 200, 210, 255], 102));
    let _center_back = scene.insert(make_edged_box(0.0, -2.5, [25, 230, 40, 255], 103));

    // Right chunk: back-facing (CW when viewed from +Z) quad spanning x in [1.5, 3.3], y in [-0.9, 0.9].
    let mut back_quad = Mesh3D::at_origin(Vec3::new(2.4, 0.0, -1.5));
    let c = [200, 200, 200, 255];
    let pid = PickId::new(1, 104, 0);
    // CW order viewed from +Z -> front_facing == false!
    back_quad.push_triangle(
        Vec3::new(-0.9, -0.9, 0.0),
        Vec3::new(-0.9, 0.9, 0.0),
        Vec3::new(0.9, 0.9, 0.0),
        c,
        pid,
    );
    back_quad.push_triangle(
        Vec3::new(-0.9, -0.9, 0.0),
        Vec3::new(0.9, 0.9, 0.0),
        Vec3::new(0.9, -0.9, 0.0),
        c,
        pid,
    );
    let _right_back = scene.insert(back_quad);

    let render = |host: &mut ViewportHost, scene: &mut Scene3DLayer| {
        let mut enc = device.create_command_encoder(&Default::default());
        host.render_to_texture(&device, &queue, &mut enc, &(), &mut [scene]);
        queue.submit([enc.finish()]);
        host.readback_rgba_blocking(&device, &queue).unwrap()
    };

    // Pixel columns in 240x160 ortho (view_height=5.0 -> 32 px/unit, width=7.5 units):
    // x=-2.4 -> px ~43, x=0.0 -> px 120, x=+2.4 -> px ~197; y=0 -> py 80.
    // 1. ShadedWithEdges vs Shaded (edges present on perimeter of center box).
    assert_eq!(scene.visual_style(), VisualStyle::ShadedWithEdges);
    assert_eq!(scene.diagnostic_view(), DiagnosticView::None);
    let img_edges = render(&mut host, &mut scene);
    scene.set_visual_style(VisualStyle::Shaded);
    let img_shaded = render(&mut host, &mut scene);
    assert_ne!(
        img_edges.pixels, img_shaded.pixels,
        "ShadedWithEdges and Shaded must differ along CAD edges"
    );

    // 2. Technical (Hidden-Line): bright paper face fill + dark ink edges.
    scene.set_visual_style(VisualStyle::Technical);
    let img_tech = render(&mut host, &mut scene);
    let tech_face = img_tech.pixel(120, 80).unwrap();
    assert!(
        tech_face[0] > 180 && tech_face[1] > 180 && tech_face[2] > 180,
        "Technical style face must render bright architectural paper, got {tech_face:?}"
    );

    // 3. Wireframe: interior of center box opens to dark background while border has bright wire edges.
    scene.set_visual_style(VisualStyle::Wireframe);
    let img_wire = render(&mut host, &mut scene);
    let wire_interior = img_wire.pixel(120, 80).unwrap();
    assert!(
        wire_interior[0] < 35 && wire_interior[1] < 35 && wire_interior[2] < 40,
        "Wireframe interior must stay transparent background, got {wire_interior:?}"
    );
    let wire_bright_px = img_wire
        .pixels
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|p| p[2] > 130 && p[0] > 100)
        .count();
    assert!(
        wire_bright_px > 40,
        "Wireframe style must draw visible bright wireframe edges, got {wire_bright_px} px"
    );

    // 4. XRay: translucent front box reveals bright green box directly behind it!
    scene.set_visual_style(VisualStyle::XRay);
    scene.set_xray_opacity(0.32);
    let img_xray = render(&mut host, &mut scene);
    let xray_center = img_xray.pixel(120, 80).unwrap();
    assert!(
        xray_center[1] > 60 && xray_center[1] > xray_center[0],
        "XRay must reveal rear green box through front gray box, got {xray_center:?}"
    );

    // 5. DiagnosticView::Normals: +Z facing face maps normal (0, 0, 1) -> (0.5, 0.5, 1.0) = blue-dominant!
    scene.set_diagnostic_view(DiagnosticView::Normals);
    let img_norm = render(&mut host, &mut scene);
    let norm_px = img_norm.pixel(120, 80).unwrap();
    assert!(
        norm_px[2] > 200 && norm_px[0].abs_diff(128) <= 20 && norm_px[1].abs_diff(128) <= 20,
        "Normals view on +Z face must encode (0.5, 0.5, 1.0), got {norm_px:?}"
    );

    // 6. DiagnosticView::Facets: renders crisp geometric normals without panics or blank output.
    scene.set_diagnostic_view(DiagnosticView::Facets);
    let img_facets = render(&mut host, &mut scene);
    let facet_px = img_facets.pixel(120, 80).unwrap();
    assert!(
        facet_px[0] > 80 && facet_px[1] > 80 && facet_px[2] > 80,
        "Facets view must shade geometric facets cleanly, got {facet_px:?}"
    );

    // 7. DiagnosticView::Backfaces: front box = blue, inverted CW right quad = red-orange!
    scene.set_diagnostic_view(DiagnosticView::Backfaces);
    let img_bf = render(&mut host, &mut scene);
    let bf_front = img_bf.pixel(43, 80).unwrap();
    let bf_back = img_bf.pixel(197, 80).unwrap();
    assert!(
        bf_front[2] > bf_front[0].saturating_add(40),
        "Backfaces view must shade front-facing CCW box blue, got {bf_front:?}"
    );
    assert!(
        bf_back[0] > bf_back[2].saturating_add(50),
        "Backfaces view must shade back-facing CW quad red-orange, got {bf_back:?}"
    );

    // 8. DiagnosticView::Overdraw: 2 stacked boxes at center (4 surfaces) > 1 box at left (2 surfaces) > background!
    scene.set_diagnostic_view(DiagnosticView::Overdraw);
    let img_od = render(&mut host, &mut scene);
    let od_bg = img_od.pixel(4, 4).unwrap();
    let od_single = img_od.pixel(43, 80).unwrap();
    let od_double = img_od.pixel(120, 80).unwrap();
    assert!(
        od_double[0] > od_single[0] && od_single[0] > od_bg[0],
        "Overdraw intensity must increase strictly with layer count (bg={od_bg:?}, single={od_single:?}, double={od_double:?})"
    );

    // 9. DiagnosticView::Chunks: distinct chunks render distinct hashed palette colors.
    scene.set_diagnostic_view(DiagnosticView::Chunks);
    let img_chunks = render(&mut host, &mut scene);
    let ch_left = img_chunks.pixel(43, 80).unwrap();
    let ch_center = img_chunks.pixel(120, 80).unwrap();
    let color_diff = ch_left[0].abs_diff(ch_center[0]) as u32
        + ch_left[1].abs_diff(ch_center[1]) as u32
        + ch_left[2].abs_diff(ch_center[2]) as u32;
    assert!(
        color_diff > 30,
        "Chunks view must assign distinct palette colors (left={ch_left:?}, center={ch_center:?})"
    );

    // 10. DiagnosticView::Depth: near left box (z=+1.5) strictly brighter than far right quad (z=-1.5).
    scene.set_diagnostic_view(DiagnosticView::Depth);
    let img_depth = render(&mut host, &mut scene);
    let d_near = img_depth.pixel(43, 80).unwrap();
    let d_far = img_depth.pixel(197, 80).unwrap();
    assert!(
        d_near[0] > d_far[0].saturating_add(12),
        "Depth view must shade nearer geometry brighter than farther geometry (near={d_near:?}, far={d_far:?})"
    );

    // Exact 64-bit picking remains fully operational while in diagnostic/style modes.
    let hits = host
        .pick_rect_blocking(&device, &queue, &(), &mut [&mut scene], [10, 40, 220, 80])
        .unwrap();
    let hit_ids: Vec<u32> = hits.iter().map(|h| h.id.object).collect();
    assert!(
        hit_ids.contains(&101) && hit_ids.contains(&102) && hit_ids.contains(&104),
        "exact pick must work cleanly across visual styles and diagnostics, got {hit_ids:?}"
    );

    // Returning to defaults restores MRT color-frame selection capability cleanly.
    scene.set_diagnostic_view(DiagnosticView::None);
    scene.set_visual_style(VisualStyle::ShadedWithEdges);
    assert!(ViewportLayer::<()>::supports_color_frame_selection(&scene));
    let _img_default = render(&mut host, &mut scene);

    // Multi-viewport X-Ray isolation + repeated-geometry OIT (24 identical boxes at different
    // sizes across two ViewportHosts sharing one Scene3DLayer).
    let mut host_b = ViewportHost::new(
        &device,
        ViewportConfig {
            color_format: wgpu::TextureFormat::Rgba8Unorm,
            msaa_samples: 1,
            clear_color: [0.05, 0.06, 0.08, 1.0],
            upload_budget: iletki_core::UploadBudget::UNLIMITED,
            initial_px: [128, 96],
            ..ViewportConfig::default()
        },
    )
    .unwrap();
    *host_b.camera_mut() = Camera::ortho_3d(
        Vec3::new(0.0, 0.0, 12.0),
        Vec3::ZERO,
        Vec3::Y,
        6.0,
        0.1,
        30.0,
    );
    let mut rep_scene = Scene3DLayer::new(7);
    for i in 0..24u32 {
        let mut b = Mesh3D::at_origin(Vec3::new(
            (i % 6) as f64 * 0.9 - 2.25,
            (i / 6) as f64 * 0.9 - 1.35,
            if i % 2 == 0 { 0.4 } else { -0.4 },
        ));
        b.push_box(
            Vec3::new(-0.35, -0.35, -0.25),
            Vec3::new(0.35, 0.35, 0.25),
            [180, 210, 240, 255],
            PickId::new(7, i + 1, 0),
            false,
        );
        rep_scene.insert(b);
    }
    rep_scene.set_visual_style(VisualStyle::XRay);
    let img_a = render(&mut host, &mut rep_scene);
    let img_b = render(&mut host_b, &mut rep_scene);
    let lit_a = img_a
        .pixels
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|px| px[2] > 35)
        .count();
    let lit_b = img_b
        .pixels
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|px| px[2] > 35)
        .count();
    assert!(
        lit_a > 200 && lit_b > 100,
        "shared Scene3DLayer XRay must render cleanly in both 4x and 1x viewports (lit_a={lit_a}, lit_b={lit_b})"
    );
}

#[test]
fn gtao_darkens_concave_corners_preserves_background_and_coexists_with_caps_and_oit() {
    let Some((device, queue)) = create_headless_device() else {
        return;
    };
    let is_gl = device.adapter_info().backend == wgpu::Backend::Gl;

    // Build an open trihedral concave CAD corner (floor + back wall + left wall) with visible sky.
    let build_corner_scene = || {
        let mut scene = Scene3DLayer::new(1);
        let mat_color = [215, 215, 215, 255];
        let mut floor = Mesh3D::at_origin(Vec3::ZERO);
        floor.push_box(
            Vec3::new(-0.4, -0.25, -0.4),
            Vec3::new(3.6, 0.0, 3.6),
            mat_color,
            PickId::new(1, 501, 0),
            false,
        );
        let mut back_wall = Mesh3D::at_origin(Vec3::ZERO);
        back_wall.push_box(
            Vec3::new(-0.4, 0.0, -0.4),
            Vec3::new(3.6, 2.2, 0.0),
            mat_color,
            PickId::new(1, 502, 0),
            false,
        );
        let mut left_wall = Mesh3D::at_origin(Vec3::ZERO);
        left_wall.push_box(
            Vec3::new(-0.4, 0.0, 0.0),
            Vec3::new(0.0, 2.2, 3.6),
            mat_color,
            PickId::new(1, 503, 0),
            false,
        );
        let floor_id = scene.insert(floor);
        let _back_id = scene.insert(back_wall);
        let _left_id = scene.insert(left_wall);
        (scene, floor_id)
    };

    let project_px = |cam: &Camera, world: Vec3, w: u32, h: u32| -> (u32, u32) {
        let vp = cam.view_proj_relative(iletki_core::Vec2::new(f64::from(w), f64::from(h)));
        let rel = world - cam.position_3d();
        let clip = vp.mul_vec4(iletki_core::Vec4::new(rel.x, rel.y, rel.z, 1.0));
        let ndc_x = clip.x / clip.w;
        let ndc_y = clip.y / clip.w;
        let px = (((ndc_x + 1.0) * 0.5) * f64::from(w))
            .round()
            .clamp(0.0, f64::from(w - 1)) as u32;
        let py = (((1.0 - ndc_y) * 0.5) * f64::from(h))
            .round()
            .clamp(0.0, f64::from(h - 1)) as u32;
        (px, py)
    };

    let avg_luma = |img: &iletki_wgpu::RgbaImage, cx: u32, cy: u32| -> f32 {
        let mut sum = 0.0f32;
        let mut count = 0.0f32;
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let x = (cx as i32 + dx).clamp(0, img.width as i32 - 1) as u32;
                let y = (cy as i32 + dy).clamp(0, img.height as i32 - 1) as u32;
                if let Some(px) = img.pixel(x, y) {
                    sum += f32::from(px[0]) * 0.299
                        + f32::from(px[1]) * 0.587
                        + f32::from(px[2]) * 0.114;
                    count += 1.0;
                }
            }
        }
        sum / count.max(1.0)
    };

    let configs = [
        (
            "persp_4x",
            4u32,
            Camera::perspective_3d(
                Vec3::new(3.8, 3.0, 4.4),
                Vec3::new(1.0, 0.5, 1.0),
                Vec3::Y,
                std::f64::consts::FRAC_PI_4,
                0.1,
            ),
        ),
        (
            "persp_1x",
            1u32,
            Camera::perspective_3d(
                Vec3::new(3.8, 3.0, 4.4),
                Vec3::new(1.0, 0.5, 1.0),
                Vec3::Y,
                std::f64::consts::FRAC_PI_4,
                0.1,
            ),
        ),
        (
            "ortho_4x",
            4u32,
            Camera::ortho_3d(
                Vec3::new(4.2, 3.4, 4.8),
                Vec3::new(1.0, 0.5, 1.0),
                Vec3::Y,
                5.4,
                0.1,
                40.0,
            ),
        ),
    ];

    for (label, msaa, cam) in configs {
        let mut host = ViewportHost::new(
            &device,
            ViewportConfig {
                color_format: wgpu::TextureFormat::Rgba8Unorm,
                msaa_samples: msaa,
                clear_color: [0.06, 0.08, 0.12, 1.0],
                upload_budget: iletki_core::UploadBudget::UNLIMITED,
                initial_px: [240, 160],
                ..ViewportConfig::default()
            },
        )
        .unwrap();
        *host.camera_mut() = cam.clone();
        let (mut scene, _floor_id) = build_corner_scene();
        assert!(!scene.gtao().enabled);

        let render = |host: &mut ViewportHost, scene: &mut Scene3DLayer| {
            let mut enc = device.create_command_encoder(&Default::default());
            host.render_to_texture(&device, &queue, &mut enc, &(), &mut [scene]);
            queue.submit([enc.finish()]);
            host.readback_rgba_blocking(&device, &queue).unwrap()
        };

        let img_base = render(&mut host, &mut scene);

        let mut gtao_cfg = GtaoConfig::with_radius(1.4);
        gtao_cfg.intensity = 1.25;
        gtao_cfg.power = 1.6;
        scene.set_gtao(gtao_cfg);
        assert!(scene.gtao().enabled);
        let img_ao = render(&mut host, &mut scene);

        let corner_px = project_px(&cam, Vec3::new(0.18, 0.02, 0.18), 240, 160);
        let open_px = project_px(&cam, Vec3::new(2.9, 0.02, 2.9), 240, 160);
        let bg_base = img_base.pixel(4, 4).unwrap();
        let bg_ao = img_ao.pixel(4, 4).unwrap();
        assert_eq!(
            bg_base, bg_ao,
            "{label}: GTAO must preserve background pixels bit-for-bit"
        );

        let corner_base = avg_luma(&img_base, corner_px.0, corner_px.1);
        let corner_ao = avg_luma(&img_ao, corner_px.0, corner_px.1);
        let open_base = avg_luma(&img_base, open_px.0, open_px.1);
        let open_ao = avg_luma(&img_ao, open_px.0, open_px.1);

        if !is_gl {
            let corner_drop = corner_base - corner_ao;
            let open_drop = open_base - open_ao;
            assert!(
                corner_drop > 18.0,
                "{label}: GTAO must clearly darken concave corner (base={corner_base:.1}, ao={corner_ao:.1}, px={corner_px:?})"
            );
            assert!(
                open_drop < 12.0 && corner_drop > open_drop * 2.0,
                "{label}: open flat floor must remain bright without false self-shadowing (open_drop={open_drop:.1}, corner_drop={corner_drop:.1})"
            );
        } else {
            assert_eq!(
                img_base.pixels, img_ao.pixels,
                "{label}: GL fallback must gracefully no-op without corrupting output"
            );
        }

        // Exact 64-bit picking remains intact with GTAO active.
        let hits = host
            .pick_rect_blocking(&device, &queue, &(), &mut [&mut scene], [0, 0, 240, 160])
            .unwrap();
        let hit_ids: Vec<u32> = hits.iter().map(|h| h.id.object).collect();
        assert!(
            hit_ids.contains(&501) && hit_ids.contains(&502) && hit_ids.contains(&503),
            "{label}: picking must succeed with GTAO enabled, got {hit_ids:?}"
        );

        // Toggling GTAO off restores exact baseline pixels and releases GPU resources.
        scene.set_gtao(GtaoConfig::default());
        assert!(!scene.gtao().enabled);
        let img_off = render(&mut host, &mut scene);
        assert_eq!(
            img_base.pixels, img_off.pixels,
            "{label}: disabling GTAO must restore baseline pixels exactly"
        );
    }

    // Resilience & Coexistence: block-boundary resizes (including 1x1), Continuous mode,
    // Section Caps + WBOIT + GTAO all enabled simultaneously in one frame.
    let mut host = ViewportHost::new(
        &device,
        ViewportConfig {
            color_format: wgpu::TextureFormat::Rgba8Unorm,
            msaa_samples: 4,
            clear_color: [0.06, 0.08, 0.12, 1.0],
            mode: ViewportMode::Continuous,
            upload_budget: iletki_core::UploadBudget::UNLIMITED,
            initial_px: [240, 160],
            ..ViewportConfig::default()
        },
    )
    .unwrap();
    *host.camera_mut() = Camera::perspective_3d(
        Vec3::new(4.0, 3.0, 4.5),
        Vec3::new(0.8, 0.5, 0.8),
        Vec3::Y,
        std::f64::consts::FRAC_PI_4,
        0.1,
    );
    host.camera_mut()
        .set_clip_planes(&[Plane::from_normal_point(
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(2.2, 0.0, 0.0),
        )
        .unwrap()]);
    let (mut scene, _floor_id) = build_corner_scene();
    let mut glass_pane = Mesh3D::at_origin(Vec3::new(1.2, 0.8, 1.2));
    glass_pane.push_box(
        Vec3::new(-0.3, -0.5, -0.05),
        Vec3::new(0.3, 0.5, 0.05),
        [40, 120, 240, 255],
        PickId::new(1, 504, 0),
        false,
    );
    let glass_id = scene.insert(glass_pane);
    let glass_mat = scene.insert_material(Material3D::translucent([0.2, 0.5, 1.0, 0.45]));
    scene.set_chunk_material(glass_id, glass_mat);
    if host.supports_stencil(&device) {
        scene.set_section_caps(SectionCapConfig::solid([0.9, 0.2, 0.15, 1.0]));
    }
    scene.set_gtao(GtaoConfig::with_radius(1.2));

    for size in [[257, 193], [1, 1], [240, 160]] {
        host.resize(&device, size, 1.0);
        let mut enc = device.create_command_encoder(&Default::default());
        host.render_to_texture(&device, &queue, &mut enc, &(), &mut [&mut scene]);
        queue.submit([enc.finish()]);
    }
    let img_combo = host.readback_rgba_blocking(&device, &queue).unwrap();
    assert_eq!((img_combo.width, img_combo.height), (240, 160));
    let non_bg = img_combo
        .pixels
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|px| px[0] > 30 || px[1] > 30 || px[2] > 50)
        .count();
    assert!(
        non_bg > 2000,
        "combined Section Caps + WBOIT + GTAO in Continuous mode must render cleanly (non_bg={non_bg})"
    );
}

#[test]
fn technical_style_draws_automatic_creases_on_edgeless_meshes_and_coexists_with_gtao() {
    let Some((device, queue)) = create_headless_device() else {
        return;
    };
    let is_gl = device.adapter_info().backend == wgpu::Backend::Gl;

    // Two edge-less solid boxes (slab + stacked upper block, zero explicit edge geometry).
    let mut scene = Scene3DLayer::new(1);
    let mut slab = Mesh3D::at_origin(Vec3::ZERO);
    slab.push_box(
        Vec3::new(-3.0, -3.0, 0.0),
        Vec3::new(3.0, 3.0, 0.4),
        [180, 185, 195, 255],
        PickId::new(1, 601, 0),
        false,
    );
    let mut block = Mesh3D::at_origin(Vec3::new(0.0, 0.0, 0.402));
    block.push_box(
        Vec3::new(-1.0, -1.0, 0.0),
        Vec3::new(1.0, 1.0, 1.8),
        [210, 100, 90, 255],
        PickId::new(1, 602, 0),
        false,
    );
    scene.insert(slab);
    scene.insert(block);

    let mut host = ViewportHost::new(
        &device,
        ViewportConfig {
            color_format: wgpu::TextureFormat::Rgba8UnormSrgb,
            msaa_samples: 4,
            clear_color: [0.08, 0.09, 0.11, 1.0],
            upload_budget: iletki_core::UploadBudget::UNLIMITED,
            initial_px: [256, 192],
            ..ViewportConfig::default()
        },
    )
    .unwrap();
    *host.camera_mut() = Camera::perspective_3d(
        Vec3::new(5.5, -6.0, 4.6),
        Vec3::new(0.0, 0.0, 0.9),
        Vec3::Z,
        50.0_f64.to_radians(),
        0.1,
    );

    let render = |host: &mut ViewportHost, scene: &mut Scene3DLayer| {
        let mut enc = device.create_command_encoder(&Default::default());
        host.render_to_texture(&device, &queue, &mut enc, &(), &mut [scene]);
        queue.submit([enc.finish()]);
        host.readback_rgba_blocking(&device, &queue).unwrap()
    };

    scene.set_visual_style(VisualStyle::Technical);
    let img_tech = render(&mut host, &mut scene);
    let bg = img_tech.pixel(4, 4).unwrap();
    let paper_px = img_tech
        .pixels
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|px| px[0] > 180 && px[1] > 180 && px[2] > 180)
        .count();
    assert!(
        paper_px > 2500,
        "Technical mode must preserve bright paper faces (paper_px={paper_px})"
    );

    if !is_gl {
        // Count dark ink pixels inside the bounding region of the geometry (strictly darker than
        // architectural paper and distinct from clear background pixels).
        let mut ink_px = 0usize;
        for y in 20..172u32 {
            for x in 24..232u32 {
                let px = img_tech.pixel(x, y).unwrap();
                let is_bg = px[0].abs_diff(bg[0]) <= 3
                    && px[1].abs_diff(bg[1]) <= 3
                    && px[2].abs_diff(bg[2]) <= 3;
                let luma = (u32::from(px[0]) + u32::from(px[1]) + u32::from(px[2])) / 3;
                if !is_bg && luma < 150 {
                    ink_px += 1;
                }
            }
        }
        assert!(
            ink_px > 180,
            "Technical mode must synthesize automatic silhouette/crease ink lines on edge-less meshes (ink_px={ink_px})"
        );

        // Coexistence: Technical + GTAO together must keep paper bright while adding corner AO.
        scene.set_gtao(GtaoConfig::with_radius(1.4));
        let img_tech_gtao = render(&mut host, &mut scene);
        assert_eq!(
            img_tech.pixel(4, 4).unwrap(),
            img_tech_gtao.pixel(4, 4).unwrap(),
            "Technical + GTAO must keep background pixels exact"
        );
        assert_ne!(
            img_tech.pixels, img_tech_gtao.pixels,
            "Technical + GTAO must visibly add ambient shading under the ink lines"
        );
    }
}

#[test]
fn gtao_cut_cavity_preserves_msaa_edges_and_monotonic_falloff() {
    let Some((device, queue)) = create_headless_device() else {
        return;
    };
    if device.adapter_info().backend == wgpu::Backend::Gl {
        return;
    }

    // Reproduce the clipped hollow-box CAD inspection case:
    // - Mezzanine Deck sliced by Y = -0.35 (shallow diagonal cut lip in 4x MSAA)
    // - Green Core Block sliced by Y = -0.35 (thin left inner wall next to wide interior floor)
    let mut scene = Scene3DLayer::new(1);
    let mut deck = Mesh3D::at_origin(Vec3::new(0.0, 0.0, 4.0));
    deck.push_box(
        Vec3::new(-4.0, -3.0, 0.0),
        Vec3::new(4.0, 3.0, 0.45),
        [190, 195, 205, 255],
        PickId::new(1, 701, 0),
        false,
    );
    let mut core = Mesh3D::at_origin(Vec3::new(-0.8, 0.4, 0.402));
    core.push_box(
        Vec3::new(-1.2, -1.0, 0.0),
        Vec3::new(1.2, 1.0, 2.6),
        [65, 195, 130, 255],
        PickId::new(1, 702, 0),
        false,
    );
    scene.insert(deck);
    scene.insert(core);

    let clip = Plane::from_normal_point(Vec3::Y, Vec3::new(0.0, -0.35, 0.0)).unwrap();
    let render_pair = |cam: Camera, size: [u32; 2], scene: &mut Scene3DLayer| {
        let mut host = ViewportHost::new(
            &device,
            ViewportConfig {
                color_format: wgpu::TextureFormat::Rgba8UnormSrgb,
                msaa_samples: 4,
                clear_color: [0.09, 0.09, 0.10, 1.0],
                upload_budget: iletki_core::UploadBudget::UNLIMITED,
                initial_px: size,
                ..ViewportConfig::default()
            },
        )
        .unwrap();
        let mut c = cam;
        c.set_clip_planes(&[clip]);
        *host.camera_mut() = c;

        scene.set_gtao(GtaoConfig::default());
        let mut enc = device.create_command_encoder(&Default::default());
        host.render_to_texture(&device, &queue, &mut enc, &(), &mut [scene]);
        queue.submit([enc.finish()]);
        let img_off = host.readback_rgba_blocking(&device, &queue).unwrap();

        scene.set_gtao(GtaoConfig::with_radius(1.35));
        let mut enc = device.create_command_encoder(&Default::default());
        host.render_to_texture(&device, &queue, &mut enc, &(), &mut [scene]);
        queue.submit([enc.finish()]);
        let img_on = host.readback_rgba_blocking(&device, &queue).unwrap();
        (img_off, img_on)
    };

    // 1. Close-up on the sliced green Core Block cavity: verify monotonic horizontal falloff
    //    moving rightward away from the thin left inner wall (zero phantom double-shadow bands).
    let cam_core = Camera::perspective_3d(
        Vec3::new(-0.55, -2.4, 2.55),
        Vec3::new(-0.8, 0.4, 0.85),
        Vec3::Z,
        48.0_f64.to_radians(),
        0.05,
    );
    let (core_off, core_on) = render_pair(cam_core, [320, 240], &mut scene);
    // Find interior green floor columns (bright green floor in baseline, distinct from darker
    // vertical side wall) across rows y in 125..165.
    let mut floor_prof = Vec::new();
    for x in 0..320u32 {
        let mut sum = 0.0f32;
        let mut cnt = 0u32;
        for y in 125..165u32 {
            let base = core_off.pixel(x, y).unwrap();
            let px = core_on.pixel(x, y).unwrap();
            if base[1] > 140 && base[1] > base[0].saturating_add(60) {
                sum += f32::from(px[1]);
                cnt += 1;
            }
        }
        if cnt == 40 {
            floor_prof.push((x, sum / 40.0));
        }
    }
    assert!(
        floor_prof.len() > 60,
        "expected wide interior green floor slice, found {} columns",
        floor_prof.len()
    );
    // Start 4 px right of the left wall-floor corner; check that luminance rises smoothly without
    // phantom secondary shadow dips (> 1.5 luma levels below prior peak).
    let mut peak = 0.0f32;
    let mut max_dip = 0.0f32;
    for &(_x, g) in floor_prof.iter().skip(4).take(100) {
        if g > peak {
            peak = g;
        } else {
            max_dip = max_dip.max(peak - g);
        }
    }
    assert!(
        max_dip < 1.5,
        "cut-cavity GTAO must not produce phantom double-shadow dips (max_dip={max_dip:.2} luma)"
    );

    // 2. Close-up on the sliced grey Mezzanine Deck shallow diagonal lip: verify 4x MSAA edge
    //    pixels smoothly interpolate the applied AO factor (`g_on / g_off`) horizontally along the
    //    diagonal boundary without binary zig-zag cliffs, with symmetric even-vs-odd row parity
    //    and blotch-free smoothness inside the shallow cavity.
    let cam_deck = Camera::perspective_3d(
        Vec3::new(0.4, -4.8, 4.95),
        Vec3::new(0.0, 0.2, 4.15),
        Vec3::Z,
        45.0_f64.to_radians(),
        0.05,
    );
    let (deck_off, deck_on) = render_pair(cam_deck, [320, 160], &mut scene);
    let mut msaa_edges_checked = 0usize;
    let mut max_ratio_jump = 0.0f32;
    let mut even_ratios = Vec::new();
    let mut odd_ratios = Vec::new();
    let mut cavity_scan = Vec::new();
    for x in 60..259u32 {
        // Locate the top-deck -> cavity MSAA edge row `y` at column `x`.
        let mut saw_top = false;
        for y in 10..130u32 {
            let g_off = f32::from(deck_off.pixel(x, y).unwrap()[1]);
            if g_off > 170.0 {
                saw_top = true;
            } else if saw_top && g_off < 165.0 {
                // Compare applied AO ratio on the same row `y` between adjacent columns `x` and `x+1`
                // whenever both are inside the shallow-slope transition (g_off > 60).
                let g_off_next = f32::from(deck_off.pixel(x + 1, y).unwrap()[1]);
                if g_off > 60.0 && g_off_next > 60.0 {
                    let r0 = f32::from(deck_on.pixel(x, y).unwrap()[1]) / g_off;
                    let r1 = f32::from(deck_on.pixel(x + 1, y).unwrap()[1]) / g_off_next;
                    msaa_edges_checked += 1;
                    max_ratio_jump = max_ratio_jump.max((r1 - r0).abs());
                    if y % 2 == 0 {
                        even_ratios.push(r0);
                    } else {
                        odd_ratios.push(r0);
                    }
                }
                // Sample 5 px inside the shallow cut cavity to verify zero blotchy noise.
                if y + 5 < 155 {
                    let g_cav = f32::from(deck_on.pixel(x, y + 5).unwrap()[1]);
                    cavity_scan.push(g_cav);
                }
                break;
            }
        }
    }
    assert!(
        msaa_edges_checked > 40,
        "expected diagonal cut-lip MSAA transitions, found {msaa_edges_checked}"
    );
    assert!(
        max_ratio_jump < 0.14,
        "4x MSAA GTAO composite must transition smoothly across shallow diagonal cut lips (max_ratio_jump={max_ratio_jump:.3})"
    );
    if !even_ratios.is_empty() && !odd_ratios.is_empty() {
        let even_mean = even_ratios.iter().sum::<f32>() / even_ratios.len() as f32;
        let odd_mean = odd_ratios.iter().sum::<f32>() / odd_ratios.len() as f32;
        assert!(
            (even_mean - odd_mean).abs() < 0.05,
            "full-res XeGTAO must eliminate even/odd row sawtooth on shallow diagonal cut lips (even={even_mean:.3}, odd={odd_mean:.3})"
        );
    }
    // Check local detrended smoothness along the cavity scanline (max |g[i] - 0.5*(g[i-3]+g[i+3])|):
    // Hilbert 3x3 blotches exceeded 8-12 luma; 4x4 stratified A-Trous stays silky smooth.
    if cavity_scan.len() > 20 {
        let mut sq_sum = 0.0f32;
        let mut cnt = 0usize;
        for w in cavity_scan.windows(7) {
            let trend = 0.5 * (w[0] + w[6]);
            let dev = w[3] - trend;
            sq_sum += dev * dev;
            cnt += 1;
        }
        let rms = (sq_sum / cnt.max(1) as f32).sqrt();
        assert!(
            rms < 2.2,
            "cut-cavity GTAO must be free of blotchy spatial noise (detrended rms={rms:.2} luma)"
        );
    }
}

fn verify_antialiased_section_styles_on_device(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
) {
    let is_gl = device.adapter_info().backend == wgpu::Backend::Gl;
    let host = ViewportHost::new(
        device,
        ViewportConfig {
            color_format: wgpu::TextureFormat::Rgba8UnormSrgb,
            msaa_samples: 4,
            clear_color: [0.07, 0.09, 0.12, 1.0],
            upload_budget: iletki_core::UploadBudget::UNLIMITED,
            initial_px: [256, 192],
            ..ViewportConfig::default()
        },
    );
    assert!(host.is_ok(), "{label}: viewport host creation must succeed");
    let Ok(mut host) = host else {
        return;
    };
    let clip = Plane::from_normal_point(Vec3::new(0.22, 1.0, 0.0), Vec3::new(0.0, -0.35, 0.0));
    assert!(clip.is_some(), "{label}: clip plane must be valid");
    let Some(clip) = clip else {
        return;
    };
    let mut cam = Camera::perspective_3d(
        Vec3::new(4.8, -5.8, 3.8),
        Vec3::new(0.0, 0.2, 1.1),
        Vec3::Z,
        46.0_f64.to_radians(),
        0.05,
    );
    cam.set_clip_planes(&[clip]);
    *host.camera_mut() = cam;

    // Include both an edged box and an edge-less box (plus several identical boxes to exercise
    // packed/indexed-flat and storage/repeated `_section` style pipelines).
    let mut scene = Scene3DLayer::new(1);
    let mut slab = Mesh3D::at_origin(Vec3::ZERO);
    slab.push_box_with_edges(
        Vec3::new(-3.2, -2.6, 0.0),
        Vec3::new(3.2, 2.6, 0.45),
        [185, 190, 200, 255],
        [15, 18, 24, 150],
        PickId::new(1, 801, 0),
        true,
    );
    scene.insert(slab);
    for idx in 0..8u32 {
        let mut block = Mesh3D::at_origin(Vec3::new(f64::from(idx % 4) * 1.4 - 2.1, 0.3, 0.452));
        block.push_box(
            Vec3::new(-0.45, -0.9, 0.0),
            Vec3::new(0.45, 0.9, 1.6),
            [110, 175, 145, 255],
            PickId::new(1, 810 + idx, 0),
            false,
        );
        scene.insert(block);
    }

    let render = |host: &mut ViewportHost, scene: &mut Scene3DLayer| {
        let mut enc = device.create_command_encoder(&Default::default());
        host.render_to_texture(device, queue, &mut enc, &(), &mut [scene]);
        queue.submit([enc.finish()]);
        let img = host.readback_rgba_blocking(device, queue);
        assert!(img.is_ok(), "{label}: readback must succeed");
        img.ok()
    };

    // 1. Clipped Technical + GTAO: exercises `vs_style_*_section` + `fs_style_section` +
    //    `fs_style_edge_section` + 4x MSAA `technical_edges.wgsl` + full-res XeGTAO.
    scene.set_visual_style(VisualStyle::Technical);
    scene.set_gtao(GtaoConfig::with_radius(1.0));
    let Some(img_tech) = render(&mut host, &mut scene) else {
        return;
    };
    let Some(bg) = img_tech.pixel(4, 4) else {
        return;
    };
    let paper_px = img_tech
        .pixels
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|px| px[0] > 180 && px[1] > 180 && px[2] > 180)
        .count();
    assert!(
        paper_px > 2500,
        "{label}: clipped Technical style must render bright paper faces (paper_px={paper_px})"
    );
    if !is_gl {
        let mut ink_px = 0usize;
        for y in 20..176u32 {
            for x in 16..240u32 {
                if let Some(px) = img_tech.pixel(x, y) {
                    let is_bg = px[0].abs_diff(bg[0]) <= 3
                        && px[1].abs_diff(bg[1]) <= 3
                        && px[2].abs_diff(bg[2]) <= 3;
                    let luma = (u32::from(px[0]) + u32::from(px[1]) + u32::from(px[2])) / 3;
                    if !is_bg && luma < 150 {
                        ink_px += 1;
                    }
                }
            }
        }
        assert!(
            ink_px > 120,
            "{label}: clipped Technical style must draw antialiased ink edges (ink_px={ink_px})"
        );
    }

    // 2. Clipped Facets (`dpdx`/`dpdy` uniform control flow in `fs_style_section`) + Overdraw + Wireframe.
    scene.set_gtao(GtaoConfig::default());
    scene.set_diagnostic_view(DiagnosticView::Facets);
    let Some(img_facets) = render(&mut host, &mut scene) else {
        return;
    };
    let facets_lit = img_facets
        .pixels
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|px| px[0] > 80 && px[1] > 80 && px[2] > 80)
        .count();
    assert!(
        facets_lit > 2500,
        "{label}: clipped Facets diagnostic must render cleanly (facets_lit={facets_lit})"
    );

    scene.set_diagnostic_view(DiagnosticView::Overdraw);
    let Some(img_overdraw) = render(&mut host, &mut scene) else {
        return;
    };
    let overdraw_lit = img_overdraw
        .pixels
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|px| px[0] > bg[0].saturating_add(15))
        .count();
    assert!(
        overdraw_lit > 1500,
        "{label}: clipped Overdraw diagnostic must accumulate cleanly (overdraw_lit={overdraw_lit})"
    );

    scene.set_diagnostic_view(DiagnosticView::None);
    scene.set_visual_style(VisualStyle::Wireframe);
    let Some(img_wire) = render(&mut host, &mut scene) else {
        return;
    };
    let wire_lit = img_wire
        .pixels
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|px| px[2] > 120 && px[0] > 90)
        .count();
    assert!(
        wire_lit > 80,
        "{label}: clipped Wireframe style must draw antialiased section edges (wire_lit={wire_lit})"
    );

    // 3. Toggling clip plane off and back on must reuse cached style pipelines without errors.
    host.camera_mut().clear_clip_planes();
    scene.set_visual_style(VisualStyle::Technical);
    let Some(img_unclipped) = render(&mut host, &mut scene) else {
        return;
    };
    assert_ne!(
        img_unclipped.pixels, img_tech.pixels,
        "{label}: clearing clip plane must reveal full unclipped Technical geometry"
    );
    host.camera_mut().set_clip_planes(&[clip]);
    let _img_reclipped = render(&mut host, &mut scene);
}

#[test]
fn antialiased_section_styles_render_cleanly_with_clip_distances_and_sampled_fallback() {
    if let Some((device, queue)) = create_headless_device() {
        verify_antialiased_section_styles_on_device(&device, &queue, "sampled_fallback");
    }
    if let Some((device, queue)) = create_device_with_clip_distances() {
        verify_antialiased_section_styles_on_device(&device, &queue, "clip_distances");
    }
}
