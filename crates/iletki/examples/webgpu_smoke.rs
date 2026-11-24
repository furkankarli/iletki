//! Runs a viewport render and picking check inside a WebGPU-enabled browser.

#[cfg(not(target_arch = "wasm32"))]
fn main() {}

#[cfg(target_arch = "wasm32")]
fn main() -> Result<(), iletki::gpu::wgpu::web_sys::wasm_bindgen::JsValue> {
    web::start()
}

#[cfg(target_arch = "wasm32")]
mod web {
    use iletki::gpu::wgpu;
    use iletki::{
        Camera, DrawStyle2D, Mesh2D, Mesh3D, PassTargetDesc, PickId, Plane, Scene2DLayer,
        Scene3DLayer, UploadBudget, Vec2, Vec3, ViewportConfig, ViewportHost, ViewportLayer,
    };
    use std::collections::{BTreeSet, HashMap};
    use wgpu::web_sys::{
        js_sys::{self, Promise, Reflect, futures::future_to_promise},
        wasm_bindgen::JsValue,
    };

    const SIZE: u32 = 64;
    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
    const COLOR: [u8; 4] = [0, 255, 0, 255];
    const ID: PickId = PickId::new(7, 0x1234_5678, 0x9abc);

    pub(super) fn start() -> Result<(), JsValue> {
        let completion = future_to_promise(async {
            run()
                .await
                .map(|report| JsValue::from_str(&report))
                .map_err(|error| JsValue::from_str(&error))
        });
        Reflect::set(
            &js_sys::global(),
            &JsValue::from_str("iletkiWebGpuSmoke"),
            completion.as_ref(),
        )?;
        Ok(())
    }

    async fn run() -> Result<String, String> {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::BROWSER_WEBGPU;
        let instance = wgpu::Instance::new(descriptor);
        let adapter = instance
            .request_adapter(&Default::default())
            .await
            .map_err(|e| e.to_string())?;
        let info = adapter.get_info();
        if info.backend != wgpu::Backend::BrowserWebGpu {
            return Err(format!("unexpected rendering backend: {:?}", info.backend));
        }
        let (device, queue) = adapter
            .request_device(&Default::default())
            .await
            .map_err(|e| e.to_string())?;

        let mut host = ViewportHost::new(
            &device,
            ViewportConfig {
                color_format: FORMAT,
                initial_px: [SIZE; 2],
                ..Default::default()
            },
        )
        .map_err(|e| e.to_string())?;
        let mut mesh = Mesh2D::default();
        mesh.push_rect_fill(
            Vec2::new(-12.0, -12.0),
            Vec2::new(12.0, 12.0),
            DrawStyle2D::new(COLOR, false, 0.5, ID),
        );
        let mut layer = Scene2DLayer::new(ID.layer);
        layer.set_static_mesh(&mesh);
        let layers: &mut [&mut dyn ViewportLayer<()>] = &mut [&mut layer];
        let pixel = render_pixels(&device, &queue, &mut host, layers, SIZE / 2, 1).await?;
        if pixel != COLOR {
            return Err(format!("unexpected WebGPU center pixel: {pixel:?}"));
        }
        let hits = host
            .pick_rect_async(&device, &queue, &(), layers, [SIZE / 2, SIZE / 2, 1, 1])
            .await
            .map_err(|e| e.to_string())?;
        if !matches!(hits.as_slice(), [hit] if hit.id == ID && (hit.depth - 0.5).abs() < 1e-5) {
            return Err(format!("unexpected WebGPU pick: {hits:?}"));
        }
        check_section(&adapter, &device, &queue).await?;
        check_modern_3d_paths(&adapter, &device, &queue).await?;
        Ok(format!(
            "{:?} ({}) rendered {COLOR:?}, picked {ID:?} and verified 4× section, outline, flat indexing, Q32.32 UTM transforms and Rg16Uint MRT selection preview",
            info.backend, info.name
        ))
    }

    async fn check_section(
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), String> {
        let mut host = ViewportHost::new_with_adapter(
            device,
            adapter,
            ViewportConfig {
                color_format: FORMAT,
                clear_color: [0.0, 0.0, 0.0, 1.0],
                initial_px: [SIZE; 2],
                ..Default::default()
            },
        )
        .map_err(|e| e.to_string())?;
        let mut camera = Camera::ortho_3d(
            Vec3::new(0.0, -8.0, 0.0),
            Vec3::ZERO,
            Vec3::Z,
            8.0,
            0.05,
            100.0,
        );
        let section = Plane::from_normal_point(Vec3::X, Vec3::new(0.0625, 0.0, 0.0))
            .ok_or("invalid test section plane")?;
        camera.set_clip_planes(&[section]);
        *host.camera_mut() = camera;

        let mut mesh = Mesh3D::at_origin(Vec3::ZERO);
        mesh.push_quad(
            Vec3::new(-3.0, 0.0, -3.0),
            Vec3::new(3.0, 0.0, -3.0),
            Vec3::new(3.0, 0.0, 3.0),
            Vec3::new(-3.0, 0.0, 3.0),
            COLOR,
            ID,
        );
        let mut scene = Scene3DLayer::new(ID.layer);
        let chunk = scene.insert(mesh);
        let pixels =
            render_pixels(device, queue, &mut host, &mut [&mut scene], SIZE / 2 - 1, 3).await?;
        let (Some(&clipped), Some(&cut), Some(&solid)) =
            (pixels.get(1), pixels.get(5), pixels.get(9))
        else {
            return Err("missing WebGPU section pixels".to_owned());
        };
        if clipped != 0 || cut < 8 || cut.saturating_add(8) >= solid {
            return Err(format!("unexpected WebGPU section pixels: {pixels:?}"));
        }
        scene.set_selected(chunk, true);
        let pixels =
            render_pixels(device, queue, &mut host, &mut [&mut scene], SIZE / 2 - 1, 3).await?;
        let (Some(&outside), Some(&edge), Some(&inside)) =
            (pixels.first(), pixels.get(4), pixels.get(8))
        else {
            return Err("missing WebGPU outline pixels".to_owned());
        };
        if outside < 220 || !(95..=160).contains(&edge) || inside > 16 {
            return Err(format!("unexpected WebGPU outline pixels: {pixels:?}"));
        }

        scene.set_selected(chunk, false);
        host.camera_mut().clear_clip_planes();
        // Enough visible geometry to exercise the packed path. Keep these quads behind the
        // original surface so the center retains its expected shading and pick ID.
        let mut packed_chunks = vec![chunk];
        for i in 1..=128 {
            let mut mesh = Mesh3D::at_origin(Vec3::new(0.0, f64::from(i) * 0.01, 0.0));
            mesh.push_quad(
                Vec3::new(-0.5, 0.0, -0.5),
                Vec3::new(0.5, 0.0, -0.5),
                Vec3::new(0.5, 0.0, 0.5),
                Vec3::new(-0.5, 0.0, 0.5),
                [255, 0, 0, 255],
                PickId::new(ID.layer, i, 0),
            );
            packed_chunks.push(scene.insert(mesh));
        }
        let center =
            render_pixels(device, queue, &mut host, &mut [&mut scene], SIZE / 2, 1).await?;
        if !matches!(center.as_slice(), [r, g, b, 255] if *g > 32 && g > r && g > b) {
            return Err(format!(
                "unexpected WebGPU packed geometry pixel: {center:?}"
            ));
        }
        let hits = host
            .pick_rect_async(
                device,
                queue,
                &(),
                &mut [&mut scene],
                [SIZE / 2, SIZE / 2, 1, 1],
            )
            .await
            .map_err(|error| error.to_string())?;
        if !matches!(hits.as_slice(), [hit] if hit.id == ID) {
            return Err(format!("unexpected WebGPU packed geometry pick: {hits:?}"));
        }
        host.camera_mut().set_clip_planes(&[section]);
        let packed_cut =
            render_pixels(device, queue, &mut host, &mut [&mut scene], SIZE / 2 - 1, 3).await?;
        if packed_cut.get(1) != Some(&clipped)
            || packed_cut.get(5) != Some(&cut)
            || packed_cut.get(9) != Some(&solid)
        {
            return Err(format!(
                "unexpected WebGPU packed section pixels: {packed_cut:?}"
            ));
        }
        for id in packed_chunks {
            scene.set_selected(id, true);
        }
        let packed_outline =
            render_pixels(device, queue, &mut host, &mut [&mut scene], SIZE / 2 - 1, 3).await?;
        if packed_outline != pixels {
            return Err(format!(
                "unexpected WebGPU packed outline: {packed_outline:?}"
            ));
        }
        Ok(())
    }

    async fn check_modern_3d_paths(
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), String> {
        let mut host = ViewportHost::new_with_adapter(
            device,
            adapter,
            ViewportConfig {
                color_format: FORMAT,
                clear_color: [0.0, 0.0, 0.0, 1.0],
                initial_px: [SIZE; 2],
                msaa_samples: 4,
                upload_budget: UploadBudget::UNLIMITED,
                ..Default::default()
            },
        )
        .map_err(|e| e.to_string())?;
        if !host.supports_color_frame_selection(device) {
            return Err(
                "WebGPU core must support Rg16Uint 4× MSAA color-frame selection preview"
                    .to_owned(),
            );
        }

        // Large national-grid coordinates exercise Q32.32 storage-buffer transforms; flat-shaded
        // boxes with face sub-IDs exercise provoking-vertex flat indexing.
        let origin = Vec3::new(661_858.0, 4_491_159.0, 120.0);
        let mut scene = Scene3DLayer::new(ID.layer);
        let mut id_by_object = HashMap::new();
        let mut hidden_id = None;
        let mut unpickable_id = None;
        for i in 0..16u32 {
            let x = f64::from(i % 4) * 1.4 - 2.1;
            let y = f64::from(i / 4) * 1.4 - 2.1;
            let mut mesh = Mesh3D::at_origin(origin + Vec3::new(x, y, 0.0));
            let obj = i + 1;
            mesh.push_box_with_edges(
                Vec3::new(-0.45, -0.45, 0.0),
                Vec3::new(0.45, 0.45, 1.0),
                [90, 185, 225, 255],
                [20, 26, 34, 255],
                PickId::new(ID.layer, obj, 0),
                true,
            );
            if i == 1 {
                mesh.pick_ids.clear();
            }
            let cid = scene.insert(mesh);
            id_by_object.insert(obj, cid);
            if i == 0 {
                scene.set_visible(cid, false);
                hidden_id = Some(cid);
            } else if i == 1 {
                unpickable_id = Some(cid);
            }
        }
        *host.camera_mut() = Camera::perspective_3d(
            origin + Vec3::new(5.5, -6.5, 6.2),
            origin + Vec3::new(0.0, 0.0, 0.4),
            Vec3::Z,
            50.0_f64.to_radians(),
            0.1,
        );

        let target = PassTargetDesc {
            format: FORMAT,
            sample_count: 1,
            depth_format: None,
            blend_alpha: false,
        };
        let run_preview = |host: &mut ViewportHost,
                           scene: &mut Scene3DLayer,
                           rect: [u32; 4]|
         -> Result<(iletki::PendingSlotPreview, bool), String> {
            host.set_color_frame_selection_rect(Some(rect));
            let mut encoder = device.create_command_encoder(&Default::default());
            let stats = host
                .prepare_for_pass(device, queue, &mut encoder, &(), &mut [scene], target)
                .map_err(|e| e.to_string())?;
            queue.submit([encoder.finish()]);
            let pending = host
                .take_color_frame_selection_preview(device)
                .ok_or_else(|| "expected color-frame selection preview on WebGPU".to_owned())?;
            Ok((pending, stats.static_rebuilt))
        };

        // First drag populates the 4× MSAA Rg16Uint attachment and runs compute bitset reduction.
        let rect1 = [8, 8, 48, 48];
        let (pending1, rebuilt1) = run_preview(&mut host, &mut scene, rect1)?;
        if !rebuilt1 {
            return Err("first WebGPU MRT preview must populate the static ID target".to_owned());
        }
        let slots1 = pending1.await.map_err(|e| e.to_string())?;
        if slots1.is_empty() {
            return Err("expected non-empty WebGPU MRT selection preview".to_owned());
        }
        let chunks1: BTreeSet<_> = slots1
            .into_iter()
            .filter_map(|slot| scene.chunk_id_at_slot(slot as usize))
            .map(|cid| format!("{cid:?}"))
            .collect();
        if let Some(hidden) = hidden_id
            && chunks1.contains(&format!("{hidden:?}"))
        {
            return Err("hidden chunk appeared in WebGPU MRT preview".to_owned());
        }
        if let Some(unpickable) = unpickable_id
            && chunks1.contains(&format!("{unpickable:?}"))
        {
            return Err("unpickable chunk appeared in WebGPU MRT preview".to_owned());
        }

        // Second drag on static camera reuses the cached MRT attachment without rebuilding static.
        let rect2 = [16, 16, 32, 32];
        let (pending2, rebuilt2) = run_preview(&mut host, &mut scene, rect2)?;
        if rebuilt2 {
            return Err(
                "subsequent static WebGPU MRT preview must reuse cached IDs without redraw"
                    .to_owned(),
            );
        }
        let slots2 = pending2.await.map_err(|e| e.to_string())?;
        let chunks2: BTreeSet<_> = slots2
            .into_iter()
            .filter_map(|slot| scene.chunk_id_at_slot(slot as usize))
            .map(|cid| format!("{cid:?}"))
            .collect();

        // Verify shaded color output at large UTM coordinates + exact 64-bit pick parity.
        host.set_color_frame_selection_rect(None);
        let center =
            render_pixels(device, queue, &mut host, &mut [&mut scene], SIZE / 2, 1).await?;
        if !matches!(center.as_slice(), [r, g, b, 255] if *b > 32 && b >= r && *g > 24) {
            return Err(format!(
                "unexpected WebGPU UTM flat-indexed center pixel: {center:?}"
            ));
        }
        let exact_hits = host
            .pick_rect_async(device, queue, &(), &mut [&mut scene], rect2)
            .await
            .map_err(|e| e.to_string())?;
        if exact_hits.is_empty() {
            return Err("expected non-empty exact pick at UTM coordinates".to_owned());
        }
        if !exact_hits
            .iter()
            .any(|hit| hit.id.layer == ID.layer && hit.id.sub > 0)
        {
            return Err(format!(
                "expected non-zero flat face sub-ID in UTM exact pick: {exact_hits:?}"
            ));
        }
        let exact_chunks: BTreeSet<_> = exact_hits
            .iter()
            .filter(|hit| hit.id.layer == ID.layer)
            .filter_map(|hit| id_by_object.get(&hit.id.object))
            .map(|cid| format!("{cid:?}"))
            .collect();
        if !exact_chunks.is_subset(&chunks2) {
            return Err(format!(
                "WebGPU 4× MSAA MRT preview missed exact pick chunks: exact={exact_chunks:?}, preview={chunks2:?}"
            ));
        }
        let stats = host.selection_preview_stats();
        if stats.submitted != 2 || stats.ready != 2 {
            return Err(format!(
                "unexpected WebGPU selection preview stats: {stats:?}"
            ));
        }
        Ok(())
    }

    async fn render_pixels(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        host: &mut ViewportHost,
        layers: &mut [&mut dyn ViewportLayer<()>],
        x: u32,
        width: u32,
    ) -> Result<Vec<u8>, String> {
        let target = PassTargetDesc {
            format: FORMAT,
            sample_count: 1,
            depth_format: None,
            blend_alpha: false,
        };

        let mut encoder = device.create_command_encoder(&Default::default());
        host.prepare_for_pass(device, queue, &mut encoder, &(), layers, target)
            .map_err(|e| e.to_string())?;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("WebGPU smoke output"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            host.paint_into_pass(&mut pass, target);
        }
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("WebGPU smoke pixels"),
            size: u64::from(width) * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                origin: wgpu::Origin3d {
                    x,
                    y: SIZE / 2,
                    z: 0,
                },
                ..texture.as_image_copy()
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: Default::default(),
            },
            wgpu::Extent3d {
                width,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);
        Promise::new(&mut |resolve, reject| {
            readback
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    let _ = match result {
                        Ok(()) => resolve.call0(&JsValue::UNDEFINED),
                        Err(error) => reject
                            .call1(&JsValue::UNDEFINED, &JsValue::from_str(&error.to_string())),
                    };
                });
        })
        .await
        .map_err(|e| format!("pixel mapping failed: {e:?}"))?;

        let pixels = readback
            .get_mapped_range(..)
            .map_err(|e| e.to_string())?
            .to_vec();
        readback.unmap();
        Ok(pixels)
    }
}
