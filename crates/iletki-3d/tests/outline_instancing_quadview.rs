//! Headless GPU integration & stress tests for advanced 3D viewport rendering:
//! 1. Jump-Flood Algorithm (JFA) screen-space selection outlines in both `OnDemand` and
//!    `Continuous` modes.
//! 2. 100,000-instance GPU instancing (`InstancedMesh3DLayer`) in a single draw call with
//!    per-instance 64-bit GPU picking.
//! 3. Simultaneous 4-way CAD Quad-View (Top / Front / Right / Perspective) sharing one scene in a
//!    SINGLE `wgpu::CommandEncoder` without uniform clobbering.
//! 4. 3D *Shaded with Edges* (`depth_offset` wireframe over solid faces) with section clip planes.

#![cfg(not(target_arch = "wasm32"))]

use iletki_3d::{
    Grid3DConfig, InfiniteGrid3DLayer, Instance3D, InstancedMesh3DLayer, Mesh3D, Scene3DLayer,
    TransformGizmoLayer,
};
use iletki_core::{Camera, DirtyFlags, OutlineConfig, PickId, Plane, Vec3, ViewportMode};
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
        power_preference: wgpu::PowerPreference::HighPerformance,
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
        label: Some("iletki-3d advanced rendering test device"),
        ..wgpu::DeviceDescriptor::default()
    }))
    .ok()
}

#[test]
fn jump_flood_selection_outline_renders_exterior_halo_in_both_modes() {
    let Some((device, queue)) = create_headless_device() else {
        return;
    };

    for mode in [ViewportMode::OnDemand, ViewportMode::Continuous] {
        let mut host = ViewportHost::new(
            &device,
            ViewportConfig {
                color_format: wgpu::TextureFormat::Rgba8Unorm,
                msaa_samples: 1,
                clear_color: [0.0, 0.0, 0.0, 1.0],
                mode,
                initial_px: [128, 128],
                outline: OutlineConfig {
                    color: [1.0, 0.45, 0.0, 1.0],
                    width_px: 4.0,
                },
                ..ViewportConfig::default()
            },
        )
        .unwrap();
        *host.camera_mut() = Camera::perspective_3d(
            Vec3::new(0.0, -7.0, 0.0),
            Vec3::ZERO,
            Vec3::Z,
            std::f64::consts::FRAC_PI_3,
            0.05,
        );

        let mut scene = Scene3DLayer::new(1);
        let mut cube = Mesh3D::at_origin(Vec3::ZERO);
        cube.push_box(
            Vec3::new(-1.2, -1.2, -1.2),
            Vec3::new(1.2, 1.2, 1.2),
            [30, 210, 60, 255],
            PickId::new(1, 10, 0),
            false,
        );
        let id = scene.insert(cube);
        scene.set_selected(id, true);

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut layers: [&mut dyn ViewportLayer<()>; 1] = [&mut scene];
            host.render_to_texture(&device, &queue, &mut enc, &(), &mut layers);
        }
        queue.submit(std::iter::once(enc.finish()));

        let img = host.readback_rgba_blocking(&device, &queue).unwrap();

        // 1. Interior center pixel must remain green (outline is exterior-only!).
        let center = img.pixel(64, 64).unwrap();
        assert!(
            center[1] > 40 && center[1] > center[0],
            "mode={mode:?} center should stay green, got {center:?}"
        );

        // 2. Corner background pixel must stay black.
        let corner = img.pixel(2, 2).unwrap();
        assert!(
            corner[0] < 5 && corner[1] < 5 && corner[2] < 5,
            "mode={mode:?} corner should stay black, got {corner:?}"
        );

        // 3. Count vivid orange halo pixels along the horizontal midline (r > 180, g in 50..160, b < 40).
        let mut orange_halo_px = 0u32;
        for x in 0..128 {
            let px = img.pixel(x, 64).unwrap();
            if px[0] > 180 && px[1] > 50 && px[1] < 160 && px[2] < 40 {
                orange_halo_px += 1;
            }
        }
        assert!(
            orange_halo_px >= 6,
            "mode={mode:?} expected >= 6 orange JFA halo pixels across midline (left+right), got {orange_halo_px}"
        );
    }
}

#[test]
fn instanced_100k_3d_boxes_single_draw_call_and_pick() {
    let Some((device, queue)) = create_headless_device() else {
        return;
    };

    let mut host = ViewportHost::new(
        &device,
        ViewportConfig {
            color_format: wgpu::TextureFormat::Rgba8Unorm,
            msaa_samples: 1,
            clear_color: [0.05, 0.05, 0.08, 1.0],
            mode: ViewportMode::Continuous,
            initial_px: [256, 256],
            ..ViewportConfig::default()
        },
    )
    .unwrap();
    *host.camera_mut() = Camera::perspective_3d(
        Vec3::new(0.0, -6.0, 0.0),
        Vec3::ZERO,
        Vec3::Z,
        std::f64::consts::FRAC_PI_3,
        0.05,
    );

    let mut proto = Mesh3D::at_origin(Vec3::ZERO);
    proto.push_box(
        Vec3::new(-0.4, -0.4, -0.4),
        Vec3::new(0.4, 0.4, 0.4),
        [255, 255, 255, 255],
        PickId::NONE,
        false,
    );

    let hero_pick = PickId::new(9, 424_242, 7);
    let mut instances = Vec::with_capacity(100_000);
    // Instance 0: hero box right at the origin facing the camera.
    instances.push(Instance3D::new(
        [0.0, 0.0, 0.0],
        [2.0, 2.0, 2.0],
        [240, 40, 40, 255],
        hero_pick,
    ));
    // Remaining 99,999 instances spread behind the hero in a 3D architectural grid.
    for i in 1..100_000u32 {
        let gx = (i % 100) as f32 - 50.0;
        let gy = ((i / 100) % 100) as f32 + 4.0;
        let gz = (i / 10_000) as f32 * 2.5 - 10.0;
        instances.push(Instance3D::new(
            [gx * 2.0, gy * 2.0, gz],
            [0.8, 0.8, 0.8],
            [80, 150, 220, 255],
            PickId::new(9, i, 0),
        ));
    }

    let mut inst_layer = InstancedMesh3DLayer::new(9, Vec3::ZERO, proto);
    inst_layer.set_instances(instances);

    // Warm up upload + render 100,000 instances (1.2M triangles) in ONE draw call!
    let t0 = std::time::Instant::now();
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut layers: [&mut dyn ViewportLayer<()>; 1] = [&mut inst_layer];
        let out = host.render_to_texture(&device, &queue, &mut enc, &(), &mut layers);
        assert_eq!(out.stats.layer_draws, 1);
    }
    queue.submit(std::iter::once(enc.finish()));
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(inst_layer.instance_count(), 100_000);
    eprintln!("100,000 instanced 3D boxes (1.2M tris) upload+render: {elapsed_ms:.2} ms");

    let img = host.readback_rgba_blocking(&device, &queue).unwrap();
    let center = img.pixel(128, 128).unwrap();
    assert!(
        center[0] > 60 && center[0] > center[1] && center[0] > center[2],
        "expected red hero box at center, got {center:?}"
    );

    // Per-instance 64-bit GPU picking on the hero box.
    let hits = {
        let mut layers: [&mut dyn ViewportLayer<()>; 1] = [&mut inst_layer];
        host.pick_rect_blocking(&device, &queue, &(), &mut layers, [126, 126, 5, 5])
            .unwrap()
    };
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, hero_pick);
    let pos = hits[0].world_pos.expect("reconstructed depth");
    assert!(
        (pos.y - (-0.8)).abs() < 0.08,
        "expected hero front face y=-0.8, got {pos:?}"
    );
}

#[test]
fn quad_view_four_viewports_share_one_scene_in_single_encoder() {
    let Some((device, queue)) = create_headless_device() else {
        return;
    };

    let make_host = |cam: Camera| {
        let mut h = ViewportHost::new(
            &device,
            ViewportConfig {
                color_format: wgpu::TextureFormat::Rgba8Unorm,
                msaa_samples: 1,
                clear_color: [0.08, 0.09, 0.11, 1.0],
                mode: ViewportMode::OnDemand,
                initial_px: [128, 128],
                ..ViewportConfig::default()
            },
        )
        .unwrap();
        *h.camera_mut() = cam;
        h
    };

    // Four distinct CAD viewpoints: Top, Front, Right, Perspective.
    let mut vp_top = make_host(Camera::ortho_3d(
        Vec3::new(0.0, 0.0, 12.0),
        Vec3::ZERO,
        Vec3::Y,
        10.0,
        0.1,
        100.0,
    ));
    let mut vp_front = make_host(Camera::ortho_3d(
        Vec3::new(0.0, -12.0, 0.0),
        Vec3::ZERO,
        Vec3::Z,
        10.0,
        0.1,
        100.0,
    ));
    let mut vp_right = make_host(Camera::ortho_3d(
        Vec3::new(12.0, 0.0, 0.0),
        Vec3::ZERO,
        Vec3::Z,
        10.0,
        0.1,
        100.0,
    ));
    let mut vp_persp = make_host(Camera::perspective_3d(
        Vec3::new(7.0, -7.0, 6.0),
        Vec3::ZERO,
        Vec3::Z,
        std::f64::consts::FRAC_PI_3,
        0.05,
    ));

    // Verify all 4 hosts have distinct ViewportIds.
    let ids = [vp_top.id(), vp_front.id(), vp_right.id(), vp_persp.id()];
    for i in 0..4 {
        for j in (i + 1)..4 {
            assert_ne!(ids[i], ids[j]);
        }
    }

    // Build ONE asymmetric scene (red X-arm, green Y-arm, blue Z-tower).
    let mut scene = Scene3DLayer::new(1);
    let mut red_x = Mesh3D::at_origin(Vec3::new(2.0, 0.0, 0.0));
    red_x.push_box_with_edges(
        Vec3::new(-1.5, -0.6, -0.6),
        Vec3::new(1.5, 0.6, 0.6),
        [220, 50, 50, 255],
        [20, 20, 20, 255],
        PickId::new(1, 1, 0),
        false,
    );
    let red_id = scene.insert(red_x);
    scene.set_selected(red_id, true);

    let mut green_y = Mesh3D::at_origin(Vec3::new(0.0, 2.5, 0.0));
    green_y.push_box(
        Vec3::new(-0.6, -1.5, -0.6),
        Vec3::new(0.6, 1.5, 0.6),
        [50, 210, 80, 255],
        PickId::new(1, 2, 0),
        false,
    );
    scene.insert(green_y);

    let mut blue_z = Mesh3D::at_origin(Vec3::new(0.0, 0.0, 2.5));
    blue_z.push_box(
        Vec3::new(-0.6, -0.6, -1.5),
        Vec3::new(0.6, 0.6, 1.5),
        [60, 120, 235, 255],
        PickId::new(1, 3, 0),
        false,
    );
    scene.insert(blue_z);

    let mut grid = InfiniteGrid3DLayer::new(2, Grid3DConfig::default());
    let mut gizmo = TransformGizmoLayer::new(3, 1);
    gizmo.origin = Some(Vec3::new(-2.0, -2.0, 0.0));

    // Record ALL 4 viewports into a SINGLE CommandEncoder and submit ONCE!
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut layers: [&mut dyn ViewportLayer<()>; 3] = [&mut scene, &mut grid, &mut gizmo];
        vp_top.render_to_texture(&device, &queue, &mut enc, &(), &mut layers);
        vp_front.render_to_texture(&device, &queue, &mut enc, &(), &mut layers);
        vp_right.render_to_texture(&device, &queue, &mut enc, &(), &mut layers);
        vp_persp.render_to_texture(&device, &queue, &mut enc, &(), &mut layers);
    }
    queue.submit(std::iter::once(enc.finish()));

    let imgs = [
        vp_top.readback_rgba_blocking(&device, &queue).unwrap(),
        vp_front.readback_rgba_blocking(&device, &queue).unwrap(),
        vp_right.readback_rgba_blocking(&device, &queue).unwrap(),
        vp_persp.readback_rgba_blocking(&device, &queue).unwrap(),
    ];

    // Every pair of views must differ significantly (no uniform clobbering!).
    for i in 0..4 {
        for j in (i + 1)..4 {
            let mut diff_pixels = 0u32;
            for (a, b) in imgs[i]
                .pixels
                .as_chunks::<4>()
                .0
                .iter()
                .zip(imgs[j].pixels.as_chunks::<4>().0.iter())
            {
                if a[0].abs_diff(b[0]) > 12 || a[1].abs_diff(b[1]) > 12 || a[2].abs_diff(b[2]) > 12
                {
                    diff_pixels += 1;
                }
            }
            assert!(
                diff_pixels > 500,
                "view {i} and view {j} look too similar ({diff_pixels} diff pixels) — possible uniform clobbering"
            );
        }
    }
}

#[test]
fn shaded_with_edges_depth_offset_and_clip_plane() {
    let Some((device, queue)) = create_headless_device() else {
        return;
    };

    let mut host = ViewportHost::new(
        &device,
        ViewportConfig {
            color_format: wgpu::TextureFormat::Rgba8Unorm,
            msaa_samples: 1,
            clear_color: [0.1, 0.2, 0.4, 1.0],
            mode: ViewportMode::OnDemand,
            initial_px: [128, 128],
            ..ViewportConfig::default()
        },
    )
    .unwrap();
    // Look at the cube diagonally so a vertical front edge runs right through the middle column!
    *host.camera_mut() = Camera::perspective_3d(
        Vec3::new(4.0, -4.0, 2.0),
        Vec3::ZERO,
        Vec3::Z,
        std::f64::consts::FRAC_PI_3,
        0.05,
    );

    let mut scene = Scene3DLayer::new(1);
    let mut cube = Mesh3D::at_origin(Vec3::ZERO);
    cube.push_box_with_edges(
        Vec3::new(-1.5, -1.5, -1.5),
        Vec3::new(1.5, 1.5, 1.5),
        [230, 230, 230, 255],
        [5, 5, 5, 255],
        PickId::NONE,
        false,
    );
    scene.insert(cube);

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut layers: [&mut dyn ViewportLayer<()>; 1] = [&mut scene];
        host.render_to_texture(&device, &queue, &mut enc, &(), &mut layers);
    }
    queue.submit(std::iter::once(enc.finish()));

    let img = host.readback_rgba_blocking(&device, &queue).unwrap();
    let mut bright_face_px = 0u32;
    let mut dark_edge_px = 0u32;
    for y in 20..108 {
        for x in 20..108 {
            let px = img.pixel(x, y).unwrap();
            if px[0] > 80 && px[1] > 80 && px[2] > 80 {
                bright_face_px += 1;
            }
            if px[0] < 15 && px[1] < 15 && px[2] < 15 {
                dark_edge_px += 1;
            }
        }
    }
    assert!(
        bright_face_px > 500,
        "expected lit faces, got {bright_face_px}"
    );
    assert!(
        dark_edge_px > 30,
        "expected crisp depth-offset dark edges over solid faces, got {dark_edge_px}"
    );

    // Clip everything with `z < -10` (slices entire cube + edges away).
    let clip = Plane::from_normal_point(Vec3::Z, Vec3::new(0.0, 0.0, 10.0)).unwrap();
    host.camera_mut().set_clip_planes(&[clip]);
    host.mark_dirty(DirtyFlags::CAMERA);
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut layers: [&mut dyn ViewportLayer<()>; 1] = [&mut scene];
        host.render_to_texture(&device, &queue, &mut enc, &(), &mut layers);
    }
    queue.submit(std::iter::once(enc.finish()));
    let clipped = host.readback_rgba_blocking(&device, &queue).unwrap();
    let center = clipped.pixel(64, 64).unwrap();
    // Should see clear color [26, 51, 102] — neither faces nor edges leak through!
    assert!(
        center[2] > 80 && center[0] < 40,
        "expected clear background after clipping edges+faces, got {center:?}"
    );
}

#[test]
fn coplanar_clip_plane_has_zero_striping_and_srgb_target_has_no_double_gamma() {
    let Some((device, queue)) = create_headless_device() else {
        return;
    };

    // 1. Face lying EXACTLY on the clip plane y = 0.0 must render 100% solid (no stipple noise).
    let mut host = ViewportHost::new(
        &device,
        ViewportConfig {
            color_format: wgpu::TextureFormat::Rgba8UnormSrgb,
            msaa_samples: 1,
            clear_color: [0.09, 0.11, 0.15, 1.0],
            mode: ViewportMode::OnDemand,
            initial_px: [128, 128],
            ..ViewportConfig::default()
        },
    )
    .unwrap();
    *host.camera_mut() = Camera::perspective_3d(
        Vec3::new(0.0, -5.0, 0.0),
        Vec3::ZERO,
        Vec3::Z,
        std::f64::consts::FRAC_PI_3,
        0.05,
    );
    let clip = Plane::from_normal_point(Vec3::Y, Vec3::ZERO).unwrap();
    host.camera_mut().set_clip_planes(&[clip]);

    // Box whose back face lies EXACTLY on y = 0.0 (front half y < 0 is clipped away).
    let mut scene = Scene3DLayer::new(1);
    let mut box_mesh = Mesh3D::at_origin(Vec3::new(0.0, -0.8, 0.0));
    box_mesh.push_box(
        Vec3::new(-1.5, -0.8, -1.5),
        Vec3::new(1.5, 0.8, 1.5),
        [215, 80, 70, 255],
        PickId::NONE,
        false,
    );
    scene.insert(box_mesh);

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut layers: [&mut dyn ViewportLayer<()>; 1] = [&mut scene];
        host.render_to_texture(&device, &queue, &mut enc, &(), &mut layers);
    }
    queue.submit(std::iter::once(enc.finish()));

    let img = host.readback_rgba_blocking(&device, &queue).unwrap();

    // Verify sRGB clear color is dark slate (~[23, 28, 38]), NOT double-gamma washed-out (~[85, 95, 108]).
    let bg = img.pixel(2, 2).unwrap();
    assert!(
        bg[0] < 32 && bg[1] < 38 && bg[2] < 48,
        "sRGB clear color was double-gamma encoded: {bg:?}"
    );

    // Every pixel in the center 24x24 patch of the coplanar y=0 face must be solid red-tinted
    // (zero background-colored stipple holes!).
    let mut holes = 0u32;
    for y in 52..76 {
        for x in 52..76 {
            let px = img.pixel(x, y).unwrap();
            if px[0] <= px[2] {
                holes += 1;
            }
        }
    }
    assert_eq!(
        holes, 0,
        "coplanar face on clip plane y=0.0 had {holes} stipple/discard holes"
    );
}

#[test]
fn multisampled_outline_preserves_edge_coverage_at_large_screen_coordinates() {
    // Omitting the rectangle hint keeps the same mask pipeline and forces full-screen processing.
    struct FullScreenMask<'a>(&'a mut Scene3DLayer);
    impl ViewportLayer<()> for FullScreenMask<'_> {
        fn id(&self) -> u16 {
            ViewportLayer::<()>::id(self.0)
        }

        fn phases(&self) -> iletki_core::DrawPhaseMask {
            ViewportLayer::<()>::phases(self.0)
        }

        fn revision(&self) -> iletki_wgpu::LayerRevision {
            ViewportLayer::<()>::revision(self.0)
        }

        fn prepare(&mut self, ctx: &mut iletki_wgpu::PrepareCtx<'_>) -> bool {
            self.0.prepare(ctx)
        }

        fn supports_multisampled_outline_mask(&self) -> bool {
            ViewportLayer::<()>::supports_multisampled_outline_mask(self.0)
        }

        fn paint_multisampled_outline_mask(
            &self,
            pass: &mut wgpu::RenderPass<'_>,
            ctx: &iletki_wgpu::PaintCtx<'_>,
        ) {
            self.0.paint_multisampled_outline_mask(pass, ctx);
        }

        fn paint(
            &self,
            phase: iletki_core::DrawPhase,
            pass: &mut wgpu::RenderPass<'_>,
            ctx: &iletki_wgpu::PaintCtx<'_>,
        ) {
            self.0.paint(phase, pass, ctx);
        }
    }

    let Some((device, queue)) = create_headless_device() else {
        return;
    };
    for mode in [ViewportMode::OnDemand, ViewportMode::Continuous] {
        let mut host = ViewportHost::new(
            &device,
            ViewportConfig {
                color_format: wgpu::TextureFormat::Rgba8Unorm,
                initial_px: [4096, 64],
                msaa_samples: 4,
                mode,
                clear_color: [0.0, 0.0, 0.0, 1.0],
                outline: OutlineConfig {
                    color: [1.0, 0.0, 0.0, 1.0],
                    width_px: 3.0,
                },
                ..Default::default()
            },
        )
        .unwrap();
        *host.camera_mut() = Camera::ortho_3d(
            Vec3::new(0.0, -8.0, 0.0),
            Vec3::ZERO,
            Vec3::Z,
            8.0,
            0.05,
            100.0,
        );
        let mut scene = Scene3DLayer::new(1);
        // With eight pixels per world unit, the right edges both bisect a physical pixel.
        for x in [-192.0, 192.0] {
            let mut mesh = Mesh3D::at_origin(Vec3::new(x, 0.0, 0.0));
            mesh.push_quad(
                Vec3::new(-4.0, 0.0, -3.0),
                Vec3::new(0.0625, 0.0, -3.0),
                Vec3::new(0.0625, 0.0, 3.0),
                Vec3::new(-4.0, 0.0, 3.0),
                [0, 255, 0, 255],
                PickId::NONE,
            );
            let chunk = scene.insert(mesh);
            scene.set_selected(chunk, true);
        }

        let mut encoder = device.create_command_encoder(&Default::default());
        host.render_to_texture(&device, &queue, &mut encoder, &(), &mut [&mut scene]);
        queue.submit([encoder.finish()]);
        let image = host.readback_rgba_blocking(&device, &queue).unwrap();
        let [left, edge, outside, far] = [511, 512, 513, 519].map(|x| image.pixel(x, 32).unwrap());
        assert!(left[0] < 8 && left[1] > 40, "{left:?}");
        assert!((95..=160).contains(&edge[0]) && edge[1] > 20, "{edge:?}");
        assert!(outside[0] > 235 && outside[1] < 8, "{outside:?}");
        assert!(far[0] < 8, "{far:?}");

        for x in 511..=519 {
            let [red, _, _, _] = image.pixel(x, 32).unwrap();
            let [far_red, _, _, _] = image.pixel(x + 3072, 32).unwrap();
            assert!(
                red.abs_diff(far_red) <= 2,
                "{mode:?}, {x}: {red} versus {far_red}"
            );
        }

        host.mark_dirty(DirtyFlags::SELECTION);
        let mut encoder = device.create_command_encoder(&Default::default());
        host.render_to_texture(
            &device,
            &queue,
            &mut encoder,
            &(),
            &mut [&mut FullScreenMask(&mut scene)],
        );
        queue.submit([encoder.finish()]);
        let full_screen = host.readback_rgba_blocking(&device, &queue).unwrap();
        assert!(
            image.pixels == full_screen.pixels,
            "{mode:?}: scissor changed rendered pixels"
        );
    }
}
