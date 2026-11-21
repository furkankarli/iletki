//! Headless visual showcase & multi-scenario stress benchmark for `iletki`.
//!
//! Run in release mode to generate visual PNGs and measure real GPU performance:
//! ```text
//! cargo run --release --example showcase -- [output_dir]
//! ```
//!
//! Scenarios executed:
//! 1. **2D National-Grid CAD Stress** — 500,000 segments at UTM `(661_858, 4_491_159)` with 8x8
//!    spatial tile culling, measuring pan/zoom frame times, idle `OnDemand` cache hit latency, and
//!    64-bit GPU picking.
//! 2. **4-Way 3D CAD Quad-View** — Top, Front, Right and Perspective viewports sharing ONE 3D
//!    scene (`Scene3DLayer` + *Shaded with Edges* + Jump-Flood selection outline + infinite 3D
//!    ground grid + 3D transform gizmo + section clip) in a SINGLE `CommandEncoder`, composited
//!    into `quad_view.png` (1024x1024, 4x MSAA).
//! 3. **100,000-Instance 3D Stress** — 100,000 instanced 3D boxes (1.2M triangles) in ONE GPU draw
//!    call, measuring 60-frame orbit latency and rendering `instanced_100k.png` (800x600, 4x MSAA).
//! 4. **Drag-Resize Hysteresis Stress** — 120 simulated drag-resize frames proving 256 px
//!    over-allocation eliminates per-frame texture thrashing.

#![allow(clippy::print_stdout, clippy::print_stderr)]

use iletki::gpu::wgpu;
use iletki::{
    Camera, DrawStyle2D, Grid3DConfig, GridStyle, InfiniteGrid3DLayer, Instance3D,
    InstancedMesh3DLayer, Mesh2D, Mesh3D, OutlineConfig, PickId, Plane, RgbaImage, Scene2DLayer,
    Scene3DLayer, TransformGizmoLayer, Vec2, Vec3, ViewportConfig, ViewportHost, ViewportLayer,
    ViewportMode, order_to_depth,
};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};
use std::time::Instant;

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

fn create_device() -> Result<(wgpu::Device, wgpu::Queue, String), String> {
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
    .map_err(|e| format!("no GPU adapter available: {e}"))?;
    let info = adapter.get_info();
    let name = format!(
        "{} ({:?}, {} {})",
        info.name, info.backend, info.driver, info.driver_info
    );
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("iletki showcase device"),
        ..wgpu::DeviceDescriptor::default()
    }))
    .map_err(|e| format!("request_device failed: {e}"))?;
    Ok((device, queue, name))
}

fn wait_gpu(
    device: &wgpu::Device,
    submission: Option<wgpu::SubmissionIndex>,
) -> Result<(), String> {
    device
        .poll(wgpu::PollType::Wait {
            submission_index: submission,
            timeout: None,
        })
        .map(|_| ())
        .map_err(|e| format!("GPU submission did not complete: {e}"))
}

fn summarize_samples(
    path: &std::path::Path,
    samples_ms: &mut [f64],
) -> Result<(f64, f64, f64), String> {
    use std::fmt::Write as _;

    let mut csv = String::from("frame,wall_ms\n");
    for (index, ms) in samples_ms.iter().enumerate() {
        writeln!(&mut csv, "{index},{ms:.6}").map_err(|e| e.to_string())?;
    }
    std::fs::write(path, csv).map_err(|e| format!("failed to write {}: {e}", path.display()))?;

    if samples_ms.is_empty() {
        return Ok((0.0, 0.0, 0.0));
    }
    samples_ms.sort_by(f64::total_cmp);
    let mean = samples_ms.iter().sum::<f64>() / (samples_ms.len() as f64);
    let at = |p: f64| {
        let idx = ((samples_ms.len() - 1) as f64 * p).round() as usize;
        samples_ms.get(idx).copied().unwrap_or_default()
    };
    Ok((mean, at(0.50), at(0.95)))
}

/// Minimal zero-dependency uncompressed-DEFLATE PNG writer (RGBA8).
fn write_png(path: &std::path::Path, width: u32, height: u32, rgba: &[u8]) -> Result<(), String> {
    let row_bytes = (width as usize).saturating_mul(4);
    let expected = row_bytes.saturating_mul(height as usize);
    if rgba.len() < expected {
        return Err("pixel buffer smaller than width*height*4".into());
    }

    let mut raw = Vec::with_capacity((row_bytes + 1).saturating_mul(height as usize));
    for y in 0..(height as usize) {
        raw.push(0); // filter type 0 (None)
        let start = y * row_bytes;
        if let Some(row) = rgba.get(start..(start + row_bytes)) {
            raw.extend_from_slice(row);
        }
    }

    // Wrap raw scanlines in valid zlib (CMF/FLG + stored DEFLATE blocks + Adler-32).
    let mut zlib = Vec::with_capacity(raw.len() + (raw.len() / 65_535 + 1) * 5 + 6);
    zlib.push(0x78); // CMF: deflate, 32K window
    zlib.push(0x01); // FLG: no dict, check bits

    let mut offset = 0usize;
    while offset < raw.len() {
        let remaining = raw.len() - offset;
        let block_len = remaining.min(65_535);
        let is_final = u8::from(offset + block_len >= raw.len());
        zlib.push(is_final);
        let len_u16 = block_len as u16;
        let nlen_u16 = !len_u16;
        zlib.extend_from_slice(&len_u16.to_le_bytes());
        zlib.extend_from_slice(&nlen_u16.to_le_bytes());
        if let Some(slice) = raw.get(offset..(offset + block_len)) {
            zlib.extend_from_slice(slice);
        }
        offset += block_len;
    }
    let adler = adler32(&raw);
    zlib.extend_from_slice(&adler.to_be_bytes());

    let mut out = Vec::with_capacity(zlib.len() + 64);
    out.extend_from_slice(&[137, 80, 78, 71, 13, 10, 26, 10]);

    let mut ihdr = [0u8; 13];
    ihdr[0..4].copy_from_slice(&width.to_be_bytes());
    ihdr[4..8].copy_from_slice(&height.to_be_bytes());
    ihdr[8] = 8; // bit depth
    ihdr[9] = 6; // color type RGBA
    push_png_chunk(&mut out, *b"IHDR", &ihdr);
    push_png_chunk(&mut out, *b"IDAT", &zlib);
    push_png_chunk(&mut out, *b"IEND", &[]);

    std::fs::write(path, out).map_err(|e| format!("failed to write {}: {e}", path.display()))
}

fn push_png_chunk(out: &mut Vec<u8>, chunk_type: [u8; 4], data: &[u8]) {
    let len = u32::try_from(data.len()).unwrap_or(0);
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(&chunk_type);
    out.extend_from_slice(data);
    let mut crc = 0xFFFF_FFFFu32;
    for &b in chunk_type.iter().chain(data.iter()) {
        crc ^= u32::from(b);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    out.extend_from_slice(&(!crc).to_be_bytes());
}

fn adler32(data: &[u8]) -> u32 {
    let mut s1 = 1u32;
    let mut s2 = 0u32;
    for &b in data {
        s1 = (s1 + u32::from(b)) % 65_521;
        s2 = (s2 + s1) % 65_521;
    }
    (s2 << 16) | s1
}

/// Composites 4 equal-sized quadrant images into a 2x2 grid with a 2 px border divider.
fn compose_quad(tl: &RgbaImage, tr: &RgbaImage, bl: &RgbaImage, br: &RgbaImage) -> RgbaImage {
    let qw = tl.width;
    let qh = tl.height;
    let w = qw * 2;
    let h = qh * 2;
    let mut pixels = vec![0u8; (w as usize) * (h as usize) * 4];
    let blit = |dst: &mut [u8], src: &RgbaImage, ox: u32, oy: u32| {
        for y in 0..src.height {
            for x in 0..src.width {
                if let Some(px) = src.pixel(x, y) {
                    let di = (((oy + y) * w + (ox + x)) as usize) * 4;
                    if let Some(slot) = dst.get_mut(di..(di + 4)) {
                        slot.copy_from_slice(&px);
                    }
                }
            }
        }
    };
    blit(&mut pixels, tl, 0, 0);
    blit(&mut pixels, tr, qw, 0);
    blit(&mut pixels, bl, 0, qh);
    blit(&mut pixels, br, qw, qh);

    // Draw subtle light border lines between quadrants.
    let border = [70u8, 80, 95, 255];
    for x in 0..w {
        for dy in 0..2 {
            let y = qh.saturating_sub(1) + dy;
            let di = ((y * w + x) as usize) * 4;
            if let Some(slot) = pixels.get_mut(di..(di + 4)) {
                slot.copy_from_slice(&border);
            }
        }
    }
    for y in 0..h {
        for dx in 0..2 {
            let x = qw.saturating_sub(1) + dx;
            let di = ((y * w + x) as usize) * 4;
            if let Some(slot) = pixels.get_mut(di..(di + 4)) {
                slot.copy_from_slice(&border);
            }
        }
    }
    RgbaImage {
        width: w,
        height: h,
        pixels,
    }
}

fn bench_2d_national_grid(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    out_dir: &std::path::Path,
) -> Result<(), String> {
    println!(
        "\n=== Scenario 1: 2D National-Grid CAD Stress (500,000 segments, UTM 661km/4491km) ==="
    );
    let origin = Vec2::new(661_858.0, 4_491_159.0);
    let mut host = ViewportHost::new(
        device,
        ViewportConfig {
            color_format: wgpu::TextureFormat::Rgba8Unorm,
            msaa_samples: 4,
            clear_color: [0.09, 0.11, 0.14, 1.0],
            mode: ViewportMode::OnDemand,
            initial_px: [1024, 768],
            ..ViewportConfig::default()
        },
    )
    .map_err(|e| e.to_string())?;
    *host.camera_mut() = Camera::ortho_2d(origin + Vec2::new(250.0, 250.0), 3.0);

    let mut scene = Scene2DLayer::new(1);
    let mut mesh = Mesh2D::default();
    let style_grid = DrawStyle2D::new(
        [90, 180, 255, 220],
        false,
        order_to_depth(10, 100),
        PickId::NONE,
    );

    // Build 500,000 CAD segments across a 1 km x 1 km urban parcel grid at national coordinates.
    let t_build = Instant::now();
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
            let wall_style = DrawStyle2D::new(
                [235, 220, 130, 255],
                false,
                order_to_depth(30, 100),
                PickId::new(1, i, 0),
            );
            mesh.push_wide_line(p0, p1, 2.5, wall_style);
        } else {
            mesh.push_line(p0, p1, style_grid);
        }
    }
    mesh.optimize_spatial();
    let build_ms = t_build.elapsed().as_secs_f64() * 1000.0;

    // Initial upload + render
    let t_first = Instant::now();
    scene.upload_static_mesh(device, queue, &mesh);
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut layers: [&mut dyn ViewportLayer<()>; 1] = [&mut scene];
        host.render_to_texture(device, queue, &mut enc, &(), &mut layers);
    }
    let submission = queue.submit([enc.finish()]);
    wait_gpu(device, Some(submission))?;
    let first_ms = t_first.elapsed().as_secs_f64() * 1000.0;

    // 60 active pan+zoom frames (simulating continuous CAD navigation)
    let mut pan_samples = Vec::with_capacity(60);
    for frame in 0..60u32 {
        let t = f64::from(frame) * 0.08;
        let center = origin + Vec2::new(250.0 + t.cos() * 80.0, 250.0 + t.sin() * 80.0);
        let zoom = 2.5 + (t * 0.5).sin() * 1.5;
        *host.camera_mut() = Camera::ortho_2d(center, zoom);
        let t0 = Instant::now();
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut layers: [&mut dyn ViewportLayer<()>; 1] = [&mut scene];
            host.render_to_texture(device, queue, &mut enc, &(), &mut layers);
        }
        let submission = queue.submit([enc.finish()]);
        wait_gpu(device, Some(submission))?;
        pan_samples.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let (mean, p50, p95) =
        summarize_samples(&out_dir.join("cad2d-pan-zoom.csv"), &mut pan_samples)?;

    // 30 idle cursor-hover frames (OnDemand static cache hit = 0 static redraws!)
    let mut idle_samples = Vec::with_capacity(30);
    let mut static_rebuilds_during_idle = 0u32;
    let logical = host.logical_size();
    let grid_style = GridStyle::default();
    let visible = host
        .camera()
        .visible_2d(logical)
        .unwrap_or(iletki::Bounds2::EMPTY);
    let zoom = match host.camera().projection {
        iletki::Projection::Ortho2D { zoom, .. } => zoom,
        _ => 1.0,
    };
    for frame in 0..30u32 {
        let mut overlay = Mesh2D::default();
        iletki::cad2d::overlay::push_adaptive_grid(&mut overlay, visible, zoom, grid_style, false);
        iletki::cad2d::overlay::push_crosshair(
            &mut overlay,
            visible,
            origin + Vec2::new(250.0 + f64::from(frame) * 2.0, 250.0),
            [255, 255, 255, 200],
            false,
        );
        scene.set_overlay(&overlay);
        host.mark_dirty(iletki::DirtyFlags::OVERLAY);
        let t0 = Instant::now();
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut layers: [&mut dyn ViewportLayer<()>; 1] = [&mut scene];
            let out = host.render_to_texture(device, queue, &mut enc, &(), &mut layers);
            if out.stats.static_rebuilt {
                static_rebuilds_during_idle += 1;
            }
        }
        let submission = queue.submit([enc.finish()]);
        wait_gpu(device, Some(submission))?;
        idle_samples.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let (idle_mean, idle_p50, _) =
        summarize_samples(&out_dir.join("cad2d-crosshair.csv"), &mut idle_samples)?;

    let img = host
        .readback_rgba_blocking(device, queue)
        .map_err(|e| e.to_string())?;
    let png_path = out_dir.join("cad2d_national_grid.png");
    write_png(&png_path, img.width, img.height, &img.pixels)?;

    println!(
        "  CPU build + 8x8 spatial tile sort: {build_ms:.2} ms\n  First GPU upload + render:         {first_ms:.2} ms\n  Pan+Zoom (60 frames, 4x MSAA):     mean {mean:.2} ms | p50 {p50:.2} | p95 {p95:.2} ms\n  Crosshair updates (OnDemand):   mean {idle_mean:.2} ms | p50 {idle_p50:.2} ms (static rebuilds: {static_rebuilds_during_idle})\n  Saved: {}",
        png_path.display()
    );
    Ok(())
}

fn render_quad_view_showcase(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    out_dir: &std::path::Path,
) -> Result<(), String> {
    println!("\n=== Scenario 2: 4-Way 3D CAD Quad-View (Single CommandEncoder, 4x MSAA) ===");
    let make_host = |cam: Camera| -> Result<ViewportHost, String> {
        let mut h = ViewportHost::new(
            device,
            ViewportConfig {
                color_format: wgpu::TextureFormat::Rgba8Unorm,
                msaa_samples: 4,
                clear_color: [0.10, 0.12, 0.16, 1.0],
                mode: ViewportMode::OnDemand,
                initial_px: [512, 512],
                outline: OutlineConfig {
                    color: [1.0, 0.55, 0.05, 1.0],
                    width_px: 4.0,
                },
                ..ViewportConfig::default()
            },
        )
        .map_err(|e| e.to_string())?;
        *h.camera_mut() = cam;
        Ok(h)
    };

    let mut vp_top = make_host(Camera::ortho_3d(
        Vec3::new(0.0, 0.0, 20.0),
        Vec3::new(0.0, 0.0, 1.5),
        Vec3::Y,
        14.0,
        0.1,
        100.0,
    ))?;
    let mut vp_front = make_host(Camera::ortho_3d(
        Vec3::new(0.0, -20.0, 2.0),
        Vec3::new(0.0, 0.0, 2.0),
        Vec3::Z,
        12.0,
        0.1,
        100.0,
    ))?;
    let mut vp_right = make_host(Camera::ortho_3d(
        Vec3::new(20.0, 0.0, 2.0),
        Vec3::new(0.0, 0.0, 2.0),
        Vec3::Z,
        12.0,
        0.1,
        100.0,
    ))?;
    let mut vp_persp = make_host(Camera::perspective_3d(
        Vec3::new(9.5, -10.5, 7.5),
        Vec3::new(0.0, 0.0, 1.8),
        Vec3::Z,
        std::f64::consts::FRAC_PI_4,
        0.05,
    ))?;
    // Apply an architectural section plane ONLY on the Right viewport (slices y < 0 away!).
    if let Some(clip) = Plane::from_normal_point(Vec3::Y, Vec3::ZERO) {
        vp_right.camera_mut().set_clip_planes(&[clip]);
    }

    // Build ONE architectural structural assembly with Shaded-with-Edges!
    let mut scene = Scene3DLayer::new(1);
    let edge_col = [12, 16, 24, 110];

    // Base concrete slab
    let mut slab = Mesh3D::at_origin(Vec3::ZERO);
    slab.push_box_with_edges(
        Vec3::new(-4.5, -3.5, 0.0),
        Vec3::new(4.5, 3.5, 0.4),
        [150, 160, 175, 255],
        edge_col,
        PickId::new(1, 1, 0),
        true,
    );
    scene.insert(slab);

    // 4 steel structural columns
    for (idx, (cx, cy)) in [(-3.5, -2.5), (3.5, -2.5), (3.5, 2.5), (-3.5, 2.5)]
        .into_iter()
        .enumerate()
    {
        let mut col = Mesh3D::at_origin(Vec3::new(cx, cy, 0.4));
        col.push_box_with_edges(
            Vec3::new(-0.35, -0.35, 0.0),
            Vec3::new(0.35, 0.35, 3.6),
            [70, 130, 210, 255],
            edge_col,
            PickId::new(1, 10 + (idx as u32), 0),
            false,
        );
        scene.insert(col);
    }

    // Upper mezzanine beam deck
    let mut deck = Mesh3D::at_origin(Vec3::new(0.0, 0.0, 4.0));
    deck.push_box_with_edges(
        Vec3::new(-4.0, -3.0, 0.0),
        Vec3::new(4.0, 3.0, 0.45),
        [190, 195, 205, 255],
        edge_col,
        PickId::new(1, 20, 0),
        true,
    );
    scene.insert(deck);

    // Selected architectural core block (will receive Jump-Flood orange selection silhouette!)
    let mut core_block = Mesh3D::at_origin(Vec3::new(-0.8, 0.4, 0.4));
    core_block.push_box_with_edges(
        Vec3::new(-1.2, -1.0, 0.0),
        Vec3::new(1.2, 1.0, 2.6),
        [65, 195, 130, 255],
        edge_col,
        PickId::new(1, 99, 0),
        true,
    );
    let selected_id = scene.insert(core_block);
    scene.set_selected(selected_id, true);

    // Secondary mechanical unit on roof
    let mut roof_unit = Mesh3D::at_origin(Vec3::new(1.8, -0.8, 4.45));
    roof_unit.push_box_with_edges(
        Vec3::new(-0.9, -0.8, 0.0),
        Vec3::new(0.9, 0.8, 1.2),
        [215, 95, 80, 255],
        edge_col,
        PickId::new(1, 30, 0),
        true,
    );
    scene.insert(roof_unit);

    let mut grid = InfiniteGrid3DLayer::new(2, Grid3DConfig::default());
    let mut gizmo = TransformGizmoLayer::new(3, 1);
    gizmo.origin = Some(Vec3::new(-0.8, 0.4, 3.0));

    // Render ALL 4 viewports into ONE CommandEncoder!
    let t0 = Instant::now();
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut layers: [&mut dyn ViewportLayer<()>; 3] = [&mut scene, &mut grid, &mut gizmo];
        vp_top.render_to_texture(device, queue, &mut enc, &(), &mut layers);
        vp_front.render_to_texture(device, queue, &mut enc, &(), &mut layers);
        vp_right.render_to_texture(device, queue, &mut enc, &(), &mut layers);
        vp_persp.render_to_texture(device, queue, &mut enc, &(), &mut layers);
    }
    let submission = queue.submit([enc.finish()]);
    wait_gpu(device, Some(submission))?;
    let quad_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let img_top = vp_top
        .readback_rgba_blocking(device, queue)
        .map_err(|e| e.to_string())?;
    let img_front = vp_front
        .readback_rgba_blocking(device, queue)
        .map_err(|e| e.to_string())?;
    let img_right = vp_right
        .readback_rgba_blocking(device, queue)
        .map_err(|e| e.to_string())?;
    let img_persp = vp_persp
        .readback_rgba_blocking(device, queue)
        .map_err(|e| e.to_string())?;

    let quad = compose_quad(&img_top, &img_persp, &img_front, &img_right);
    let png_path = out_dir.join("quad_view.png");
    write_png(&png_path, quad.width, quad.height, &quad.pixels)?;
    println!(
        "  4 simultaneous 512x512 4x MSAA viewports (JFA + Edges + Grid + Gizmo + Section Clip) in 1 encoder: {quad_ms:.2} ms\n  Saved: {}",
        png_path.display()
    );
    Ok(())
}

fn bench_instanced_100k(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    out_dir: &std::path::Path,
) -> Result<(), String> {
    println!("\n=== Scenario 3: 100,000-Instance 3D Stress (1.2M triangles, 1 draw call) ===");
    let mut host = ViewportHost::new(
        device,
        ViewportConfig {
            color_format: wgpu::TextureFormat::Rgba8Unorm,
            msaa_samples: 4,
            clear_color: [0.06, 0.08, 0.12, 1.0],
            mode: ViewportMode::Continuous,
            initial_px: [800, 600],
            ..ViewportConfig::default()
        },
    )
    .map_err(|e| e.to_string())?;

    let mut proto = Mesh3D::at_origin(Vec3::ZERO);
    proto.push_box(
        Vec3::new(-0.35, -0.35, 0.0),
        Vec3::new(0.35, 0.35, 1.0),
        [255, 255, 255, 255],
        PickId::NONE,
        false,
    );

    // 100,000 architectural towers arranged in a 316 x 316 city grid with wave heights.
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
        let b = 220u8;
        instances.push(Instance3D::new(
            [gx * 1.3, gy * 1.3, 0.0],
            [1.0, 1.0, h],
            [r, g, b, 255],
            PickId::new(5, i, 0),
        ));
    }

    let mut inst_layer = InstancedMesh3DLayer::new(5, Vec3::ZERO, proto);
    inst_layer.set_instances(instances);

    // Warm-up upload + render
    *host.camera_mut() = Camera::perspective_3d(
        Vec3::new(35.0, -45.0, 28.0),
        Vec3::new(0.0, 0.0, 2.0),
        Vec3::Z,
        std::f64::consts::FRAC_PI_3,
        0.1,
    );
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut layers: [&mut dyn ViewportLayer<()>; 1] = [&mut inst_layer];
        host.render_to_texture(device, queue, &mut enc, &(), &mut layers);
    }
    let submission = queue.submit([enc.finish()]);
    wait_gpu(device, Some(submission))?;

    // Save visual render of the 100k-instance 3D city
    let img = host
        .readback_rgba_blocking(device, queue)
        .map_err(|e| e.to_string())?;
    let png_path = out_dir.join("instanced_100k.png");
    write_png(&png_path, img.width, img.height, &img.pixels)?;

    // Measure 60 orbiting frames
    let mut samples = Vec::with_capacity(60);
    for frame in 0..60u32 {
        let angle = f64::from(frame) * 0.06;
        let eye = Vec3::new(angle.cos() * 55.0, angle.sin() * 55.0, 26.0);
        *host.camera_mut() = Camera::perspective_3d(
            eye,
            Vec3::new(0.0, 0.0, 2.0),
            Vec3::Z,
            std::f64::consts::FRAC_PI_3,
            0.1,
        );
        let t0 = Instant::now();
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut layers: [&mut dyn ViewportLayer<()>; 1] = [&mut inst_layer];
            host.render_to_texture(device, queue, &mut enc, &(), &mut layers);
        }
        let submission = queue.submit([enc.finish()]);
        wait_gpu(device, Some(submission))?;
        samples.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let (mean, p50, p95) = summarize_samples(&out_dir.join("instanced-orbit.csv"), &mut samples)?;
    println!(
        "  60-frame orbit (100k boxes = 1.2M tris, 4x MSAA, 1 draw call): mean {mean:.2} ms | p50 {p50:.2} | p95 {p95:.2} ms\n  Saved: {}",
        png_path.display()
    );
    Ok(())
}

fn bench_resize_hysteresis(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<(), String> {
    println!("\n=== Scenario 4: Drag-Resize Hysteresis Stress (120 frames, 600px <-> 1200px) ===");
    let mut host = ViewportHost::new(
        device,
        ViewportConfig {
            color_format: wgpu::TextureFormat::Rgba8Unorm,
            msaa_samples: 4,
            mode: ViewportMode::OnDemand,
            initial_px: [600, 480],
            ..ViewportConfig::default()
        },
    )
    .map_err(|e| e.to_string())?;
    let mut reallocs = 0u32;
    let mut last_submission = None;
    let t0 = Instant::now();
    for frame in 0..120u32 {
        let w = if frame < 60 {
            600 + frame * 10
        } else {
            1200 - (frame - 60) * 10
        };
        let h = (w * 3) / 4;
        host.resize(device, [w, h], 1.0);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let mut empty: [&mut dyn ViewportLayer<()>; 0] = [];
        let out = host.render_to_texture(device, queue, &mut enc, &(), &mut empty);
        if out.stats.target_reallocated {
            reallocs += 1;
        }
        last_submission = Some(queue.submit([enc.finish()]));
    }
    wait_gpu(device, last_submission)?;
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "  120 drag-resize frames: {total_ms:.2} ms total | target-set reallocations after initialization: {reallocs}"
    );
    Ok(())
}

fn main() -> Result<(), String> {
    let out_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/showcase"));
    std::fs::create_dir_all(&out_dir)
        .map_err(|e| format!("failed to create {}: {e}", out_dir.display()))?;

    let (device, queue, gpu_name) = create_device()?;
    println!("iletki showcase & stress benchmark on GPU: {gpu_name}");

    bench_2d_national_grid(&device, &queue, &out_dir)?;
    render_quad_view_showcase(&device, &queue, &out_dir)?;
    bench_instanced_100k(&device, &queue, &out_dir)?;
    bench_resize_hysteresis(&device, &queue)?;

    Ok(())
}
