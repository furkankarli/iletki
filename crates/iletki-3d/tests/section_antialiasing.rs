//! GPU checks for live 3D section boundaries against pre-clipped geometry.

#![cfg(not(target_arch = "wasm32"))]

use iletki_3d::{Instance3D, InstancedMesh3DLayer, Mesh3D, Scene3DLayer};
use iletki_core::{Camera, PickId, Plane, Vec3, ViewportMode};
use iletki_wgpu::{RgbaImage, ViewportConfig, ViewportError, ViewportHost, ViewportLayer, wgpu};
use std::future::Future;
use std::pin::pin;
use std::task::{Context, Poll, Waker};

const SIZE: u32 = 128;
const COLOR: [u8; 4] = [0, 255, 0, 255];
const PICK_ID: PickId = PickId::new(1, 99, 7);

fn block_on<F: Future>(future: F) -> F::Output {
    let mut context = Context::from_waker(Waker::noop());
    let mut future = pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

fn adapter() -> Option<wgpu::Adapter> {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = block_on(instance.request_adapter(&Default::default())).ok();
    if let Ok(expected) = std::env::var("ILETKI_EXPECT_BACKEND") {
        let actual = adapter.as_ref().map(|adapter| adapter.get_info().backend);
        assert_eq!(
            actual.map(|backend| format!("{backend:?}").to_ascii_lowercase()),
            Some(expected.to_ascii_lowercase()),
            "requested section test backend is missing"
        );
    }
    adapter
}

fn device_features(adapter: &wgpu::Adapter) -> impl Iterator<Item = wgpu::Features> {
    let raster = adapter.features() & wgpu::Features::CLIP_DISTANCES;
    let test_raster = adapter.get_info().backend != wgpu::Backend::Gl && !raster.is_empty();
    std::iter::once(wgpu::Features::empty()).chain(test_raster.then_some(raster))
}

fn config(samples: u32, mode: ViewportMode) -> ViewportConfig {
    ViewportConfig {
        color_format: wgpu::TextureFormat::Rgba8Unorm,
        clear_color: [0.0, 0.0, 0.0, 1.0],
        initial_px: [SIZE; 2],
        msaa_samples: samples,
        mode,
        ..Default::default()
    }
}

fn render(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    host: &mut ViewportHost,
    layer: &mut dyn ViewportLayer<()>,
    camera: Camera,
) -> Result<RgbaImage, ViewportError> {
    *host.camera_mut() = camera;
    let mut encoder = device.create_command_encoder(&Default::default());
    host.render_to_texture(device, queue, &mut encoder, &(), &mut [layer]);
    queue.submit([encoder.finish()]);
    host.readback_rgba_blocking(device, queue)
}

#[cfg(feature = "profiling")]
#[test]
fn packed_live_cut_matches_direct_faces_edges_selection_and_pick() {
    let Some(adapter) = adapter() else { return };
    let dynamic_limit = wgpu::Limits::default().max_dynamic_uniform_buffers_per_pipeline_layout;
    for (features, limit) in
        device_features(&adapter).flat_map(|features| [(features, dynamic_limit), (features, 0)])
    {
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: features,
            required_limits: wgpu::Limits {
                max_dynamic_uniform_buffers_per_pipeline_layout: limit,
                ..Default::default()
            },
            ..Default::default()
        }))
        .unwrap();
        let mut settings = config(4, ViewportMode::OnDemand);
        settings.upload_budget = iletki_core::UploadBudget::UNLIMITED;
        let mut host = ViewportHost::new_with_adapter(&device, &adapter, settings).unwrap();
        let mut scene = Scene3DLayer::new(1);
        let mut camera = head_on_camera();
        camera.set_clip_planes(&[Plane::from_normal_point(
            Vec3::new(1.0, 0.4, 0.15),
            Vec3::new(0.06, 0.0, 0.0),
        )
        .unwrap()]);
        let mut selection_updates = Vec::new();
        let mut motion = None;
        let mut replaced = None;
        let mut hidden = None;
        let mut removed = None;
        for i in (0..7).chain((0..282).map(|index| 7 + index * 37 % 282)) {
            // Put a fully clipped chunk between two visible packed ranges, then
            // put hidden and removed chunks in separate gaps on the same page.
            let x = if matches!(i, 0 | 2 | 3 | 5 | 6) {
                1.8
            } else {
                (f64::from(i / 17) - 8.0) * 0.45
            };
            let z = (f64::from(i % 17) - 8.0) * 0.45;
            let mut mesh = Mesh3D::at_origin(Vec3::new(x, 0.0, z));
            mesh.push_box_with_edges(
                Vec3::new(-0.19, -0.15, -0.19),
                Vec3::new(0.19, 0.15, 0.19),
                [70, 200, 130, 255],
                [20, 30, 30, 150],
                PickId::new(1, i + 1, 0),
                true,
            );
            if i == 1 {
                // Exceed the small-object cutoff so this clipped chunk forms a staged gap.
                mesh.push_box_with_edges(
                    Vec3::new(-0.1, -0.1, -0.1),
                    Vec3::new(0.1, 0.1, 0.1),
                    [185, 70, 175, 255],
                    [20, 30, 30, 150],
                    PickId::new(1, i + 1, 0),
                    true,
                );
            }
            if i == 80 {
                mesh.pick_ids.clear();
            }
            if i == 81 {
                mesh.pick_ids.fill([0, 0]);
            }
            if i == 246 {
                // A caller can supply bounds that predate an edit to the public vertex list.
                mesh.local_bounds = iletki_core::Bounds3::from_corners(
                    Vec3::new(-3.4, -0.15, -0.19),
                    Vec3::new(-3.2, 0.15, 0.19),
                );
            }
            let chunk = scene.insert(mesh);
            if i == 3 {
                hidden = Some(chunk);
            }
            if i == 4 {
                motion = Some(chunk);
            }
            if i == 5 {
                removed = Some(chunk);
            }
            if i == 39 {
                replaced = Some(chunk);
            }
            if i % 11 != 0 {
                scene.set_selected(chunk, true);
            }
            if (100..109).contains(&i) {
                selection_updates.push(chunk);
            }
        }
        render(&device, &queue, &mut host, &mut scene, camera.clone()).unwrap();
        for id in selection_updates {
            scene.set_selected(id, false);
        }
        render(&device, &queue, &mut host, &mut scene, camera.clone()).unwrap();
        scene.set_visible(hidden.unwrap(), false);
        scene.set_origin(motion.unwrap(), Vec3::new(2.25, -0.35, -1.8));
        scene.remove(replaced.unwrap());
        let mut moved_mesh = Mesh3D::at_origin(Vec3::new(2.25, -0.35, 1.8));
        moved_mesh.push_box_with_edges(
            Vec3::new(-0.19, -0.15, -0.19),
            Vec3::new(0.19, 0.15, 0.19),
            [185, 70, 175, 255],
            [20, 30, 30, 150],
            PickId::new(1, 290, 0),
            true,
        );
        scene.insert(moved_mesh);
        scene.remove(removed.unwrap());
        iletki_wgpu::profiling::begin(&device);
        let image = render(&device, &queue, &mut host, &mut scene, camera.clone()).unwrap();
        let capture = iletki_wgpu::profiling::finish(&device, &queue).unwrap();
        if features.contains(wgpu::Features::CLIP_DISTANCES) {
            for counter in [
                "scene.section_bridged_triangles",
                "scene.section_bridged_edge_vertices",
                "scene.section_deferred_chunks",
            ] {
                assert!(
                    capture
                        .counters
                        .iter()
                        .any(|&(label, value)| label == counter && value > 0),
                    "fixture did not exercise {counter} with dynamic uniform limit {limit}"
                );
            }
        }
        let area = [0, 0, SIZE, SIZE];
        let packed = host
            .pick_rect_blocking(&device, &queue, &(), &mut [&mut scene], area)
            .unwrap();
        scene.set_profile_section_bridging(false);
        let unbridged = render(&device, &queue, &mut host, &mut scene, camera.clone()).unwrap();
        assert!(
            image.pixels == unbridged.pixels,
            "bridged section differs with {features:?} and dynamic uniform limit {limit}"
        );
        scene.set_profile_section_bridging(true);
        scene.set_profile_section_groups(false);
        let per_chunk = render(&device, &queue, &mut host, &mut scene, camera.clone()).unwrap();
        assert!(
            image.pixels == per_chunk.pixels,
            "grouped section differs with {features:?} and dynamic uniform limit {limit}"
        );
        scene.set_profile_section_culling(false);
        let unculled = render(&device, &queue, &mut host, &mut scene, camera.clone()).unwrap();
        assert!(
            image.pixels == unculled.pixels,
            "culled section differs with {features:?} and dynamic uniform limit {limit}"
        );
        scene.set_profile_section_culling(true);
        scene.set_profile_section_groups(true);
        scene.set_profile_batch_geometry(false);
        let direct = render(&device, &queue, &mut host, &mut scene, camera).unwrap();
        assert!(
            image.pixels == direct.pixels,
            "packed section differs with {features:?} and dynamic uniform limit {limit}"
        );
        let picks = host
            .pick_rect_blocking(&device, &queue, &(), &mut [&mut scene], area)
            .unwrap();
        assert!(!packed.is_empty());
        assert_eq!(
            packed
                .iter()
                .map(|hit| (hit.id, hit.depth.to_bits(), hit.world_pos))
                .collect::<Vec<_>>(),
            picks
                .iter()
                .map(|hit| (hit.id, hit.depth.to_bits(), hit.world_pos))
                .collect::<Vec<_>>()
        );
    }
}

fn clip_triangle(triangle: [Vec3; 3], plane: Plane) -> Vec<Vec3> {
    let mut clipped = Vec::new();
    for (a, b) in triangle
        .iter()
        .copied()
        .zip(triangle.iter().copied().cycle().skip(1))
    {
        let da = plane.normal.dot(a) + plane.d + 1e-4;
        let db = plane.normal.dot(b) + plane.d + 1e-4;
        if (da >= 0.0) != (db >= 0.0) {
            clipped.push(a + (b - a) * (da / (da - db)));
        }
        if db >= 0.0 {
            clipped.push(b);
        }
    }
    clipped
}

fn inclined_quad(plane: Option<Plane>) -> Option<Scene3DLayer> {
    let normal = Vec3::new(1.0, -1.0, 0.8).normalized()?;
    let u = normal.cross(Vec3::Z).normalized()? * 2.1;
    let v = normal.cross(u).normalized()? * 2.0;
    let [a, b, c, d] = [-u - v, u - v, u + v, -u + v];
    let mut mesh = Mesh3D::at_origin(Vec3::ZERO);
    for triangle in [[a, b, c], [a, c, d]] {
        let polygon =
            plane.map_or_else(|| triangle.to_vec(), |plane| clip_triangle(triangle, plane));
        if let [first, rest @ ..] = polygon.as_slice() {
            for (&next, &last) in rest.iter().zip(rest.iter().skip(1)) {
                mesh.push_triangle(*first, next, last, COLOR, PICK_ID);
            }
        }
    }
    let mut layer = Scene3DLayer::new(PICK_ID.layer);
    layer.insert(mesh);
    Some(layer)
}

fn green(image: &RgbaImage, x: u32, y: u32) -> u8 {
    image.pixel(x, y).map_or(0, |[_, green, _, _]| green)
}

fn assert_reference(
    uncut: &RgbaImage,
    pixel: &RgbaImage,
    active: &RgbaImage,
    reference: &RgbaImage,
) {
    let mut count = 0;
    let mut active_error = 0.0;
    let mut pixel_error = 0.0;
    for y in 2..SIZE - 2 {
        for x in 2..SIZE - 2 {
            let full = green(uncut, x, y);
            if full < 64
                || (y - 2..=y + 2).any(|row| {
                    (x - 2..=x + 2).any(|col| green(uncut, col, row).abs_diff(full) > 12)
                })
            {
                continue;
            }
            let coverage = |image| f64::from(green(image, x, y)) / f64::from(full);
            let expected = coverage(reference);
            if !(0.10..=0.90).contains(&expected) {
                continue;
            }
            count += 1;
            active_error += (coverage(active) - expected).abs();
            pixel_error += (coverage(pixel) - expected).abs();
        }
    }
    assert!(count > 10, "fixture only has {count} partial cut pixels");
    assert!(
        active_error / f64::from(count) < 0.035 && active_error < pixel_error * 0.25,
        "cut coverage error {active_error:.2} versus {pixel_error:.2} for {count} pixels"
    );
}

fn head_on_camera() -> Camera {
    Camera::ortho_3d(
        Vec3::new(0.0, -8.0, 0.0),
        Vec3::ZERO,
        Vec3::Z,
        8.0,
        0.05,
        100.0,
    )
}

#[test]
fn section_coverage_matches_geometry_in_four_views() {
    let Some(adapter) = adapter() else { return };
    let info = adapter.get_info();
    let caps = adapter.get_downlevel_capabilities();
    let sampled = caps
        .flags
        .contains(wgpu::DownlevelFlags::MULTISAMPLED_SHADING);
    let plane =
        Plane::from_normal_point(Vec3::new(0.54, 0.73, 0.42), Vec3::new(0.1, -0.13, 0.16)).unwrap();
    let cameras = [
        Camera::perspective_3d(Vec3::new(4.5, -5.7, 3.7), Vec3::ZERO, Vec3::Z, 0.75, 0.05),
        Camera::ortho_3d(
            Vec3::new(0.0, 0.0, 8.0),
            Vec3::ZERO,
            Vec3::Y,
            5.5,
            0.05,
            100.0,
        ),
        head_on_camera(),
        Camera::ortho_3d(
            Vec3::new(8.0, 0.0, 0.0),
            Vec3::ZERO,
            Vec3::Z,
            5.5,
            0.05,
            100.0,
        ),
    ];

    for features in device_features(&adapter) {
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: features,
            ..Default::default()
        }))
        .unwrap();
        let antialiased = info.backend != wgpu::Backend::Gl && (sampled || !features.is_empty());
        let mut stock = inclined_quad(None).unwrap();
        let mut reference = inclined_quad(Some(plane)).unwrap();
        for mode in [ViewportMode::OnDemand, ViewportMode::Continuous] {
            let mut pixel = ViewportHost::new(&device, config(4, mode)).unwrap();
            let mut active =
                ViewportHost::new_with_adapter(&device, &adapter, config(4, mode)).unwrap();
            for view in &cameras {
                let uncut = render(&device, &queue, &mut active, &mut stock, view.clone()).unwrap();
                let expected =
                    render(&device, &queue, &mut pixel, &mut reference, view.clone()).unwrap();
                let mut cut = view.clone();
                cut.set_clip_planes(&[plane]);
                let aa = render(&device, &queue, &mut active, &mut stock, cut.clone()).unwrap();
                let base = render(&device, &queue, &mut pixel, &mut stock, cut).unwrap();
                if antialiased {
                    assert_reference(&uncut, &base, &aa, &expected);
                } else {
                    assert_eq!(aa.pixels, base.pixels);
                }
                let restored =
                    render(&device, &queue, &mut active, &mut stock, view.clone()).unwrap();
                assert_eq!(
                    restored.pixels, uncut.pixels,
                    "clip toggle left stale pixels"
                );
            }
        }
        let mut pixel = ViewportHost::new(&device, config(1, ViewportMode::OnDemand)).unwrap();
        let mut active =
            ViewportHost::new_with_adapter(&device, &adapter, config(1, ViewportMode::OnDemand))
                .unwrap();
        let mut cut = head_on_camera();
        cut.set_clip_planes(&[plane]);
        assert_eq!(
            render(&device, &queue, &mut active, &mut stock, cut.clone())
                .unwrap()
                .pixels,
            render(&device, &queue, &mut pixel, &mut stock, cut)
                .unwrap()
                .pixels,
        );
    }
}

#[test]
fn section_clips_cad_lines_and_instances() {
    let Some(adapter) = adapter() else { return };
    if adapter.get_info().backend == wgpu::Backend::Gl {
        return;
    }
    let sampled = adapter
        .get_downlevel_capabilities()
        .flags
        .contains(wgpu::DownlevelFlags::MULTISAMPLED_SHADING);
    for features in device_features(&adapter) {
        if !sampled && features.is_empty() {
            continue;
        }
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: features,
            ..Default::default()
        }))
        .unwrap();
        let mut host =
            ViewportHost::new_with_adapter(&device, &adapter, config(4, ViewportMode::OnDemand))
                .unwrap();
        let mid = SIZE / 2;
        let offset = 0.5 * 8.0 / f64::from(SIZE);
        let mut camera = head_on_camera();
        let plane = Plane::from_normal_point(Vec3::X, Vec3::new(offset, 0.0, 0.0)).unwrap();
        camera.set_clip_planes(&[plane]);

        let mut proto = Mesh3D::at_origin(Vec3::ZERO);
        proto.push_quad(
            Vec3::new(-3.0, 0.0, -3.0),
            Vec3::new(3.0, 0.0, -3.0),
            Vec3::new(3.0, 0.0, 3.0),
            Vec3::new(-3.0, 0.0, 3.0),
            COLOR,
            PICK_ID,
        );
        let mut instances = InstancedMesh3DLayer::new(PICK_ID.layer, Vec3::ZERO, proto);
        instances.set_instances(vec![Instance3D::new([0.0; 3], [1.0; 3], COLOR, PICK_ID)]);
        let image = render(&device, &queue, &mut host, &mut instances, camera.clone()).unwrap();
        let [left, partial, full] = [mid - 1, mid, mid + 1].map(|x| green(&image, x, mid));
        assert!(
            left == 0 && partial > 8 && partial.saturating_add(8) < full,
            "{left}, {partial}, {full}"
        );
        assert!(
            host.pick_rect_blocking(
                &device,
                &queue,
                &(),
                &mut [&mut instances],
                [mid - 1, mid, 1, 1]
            )
            .unwrap()
            .is_empty()
        );
        let hits = host
            .pick_rect_blocking(
                &device,
                &queue,
                &(),
                &mut [&mut instances],
                [mid + 1, mid, 1, 1],
            )
            .unwrap();
        assert!(matches!(hits.as_slice(), [hit] if hit.id == PICK_ID));

        let mut edge = Mesh3D::at_origin(Vec3::ZERO);
        edge.push_edge_line(
            Vec3::new(-3.0, 0.0, -offset),
            Vec3::new(3.0, 0.0, -offset),
            COLOR,
        );
        let mut scene = Scene3DLayer::new(PICK_ID.layer);
        scene.insert(edge);
        let image = render(&device, &queue, &mut host, &mut scene, camera.clone()).unwrap();
        let green_column = |image: &RgbaImage, x| {
            (mid - 1..=mid + 1)
                .map(|y| u16::from(green(image, x, y)))
                .sum::<u16>()
        };
        let left = green_column(&image, mid - 1);
        let full = green_column(&image, mid + 1);
        assert!(left == 0 && full > 0, "{features:?}: {left}, {full}");
        if features.is_empty() {
            // Default line rasterization can place the endpoint coverage at a whole pixel.
            // Check that a moving plane still removes the previous boundary column.
            let boundary = green_column(&image, mid);
            assert!(boundary <= full, "sampled line: {boundary}, {full}");
            let shifted =
                Plane::from_normal_point(Vec3::X, Vec3::new(offset * 2.5, 0.0, 0.0)).unwrap();
            camera.set_clip_planes(&[shifted]);
            let image = render(&device, &queue, &mut host, &mut scene, camera).unwrap();
            assert_eq!(green_column(&image, mid), 0);
            assert_eq!(green_column(&image, mid + 2), full);
        } else {
            // Hardware applies the backend's own line endpoint rules to the clipped segment.
            let mut clipped_edge = Mesh3D::at_origin(Vec3::ZERO);
            clipped_edge.push_edge_line(
                Vec3::new(offset - 1e-4, 0.0, -offset),
                Vec3::new(3.0, 0.0, -offset),
                COLOR,
            );
            let mut reference = Scene3DLayer::new(PICK_ID.layer);
            reference.insert(clipped_edge);
            let reference =
                render(&device, &queue, &mut host, &mut reference, head_on_camera()).unwrap();
            for x in mid - 1..=mid + 1 {
                assert_eq!(green_column(&image, x), green_column(&reference, x));
            }
        }
    }
}
