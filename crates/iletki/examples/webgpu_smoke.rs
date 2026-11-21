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
        DrawStyle2D, Mesh2D, PassTargetDesc, PickId, Scene2DLayer, Vec2, ViewportConfig,
        ViewportHost, ViewportLayer,
    };
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
        let target = PassTargetDesc {
            format: FORMAT,
            sample_count: 1,
            depth_format: None,
            blend_alpha: false,
        };

        let mut encoder = device.create_command_encoder(&Default::default());
        host.prepare_for_pass(&device, &queue, &mut encoder, &(), layers, target)
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
            label: Some("center pixel"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                origin: wgpu::Origin3d {
                    x: SIZE / 2,
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
                width: 1,
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

        let pixel = readback.get_mapped_range(..).map_err(|e| e.to_string())?;
        if *pixel != COLOR {
            return Err(format!("unexpected WebGPU center pixel: {:?}", &*pixel));
        }
        drop(pixel);
        readback.unmap();

        let hits = host
            .pick_rect_async(&device, &queue, &(), layers, [SIZE / 2, SIZE / 2, 1, 1])
            .await
            .map_err(|e| e.to_string())?;
        if !matches!(hits.as_slice(), [hit] if hit.id == ID && (hit.depth - 0.5).abs() < 1e-5) {
            return Err(format!("unexpected WebGPU pick: {hits:?}"));
        }
        Ok(format!(
            "{:?} ({}) rendered {COLOR:?} and picked {ID:?}",
            info.backend, info.name
        ))
    }
}
