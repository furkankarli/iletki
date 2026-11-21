//! Draw and pick a rectangle inside an egui render pass.

use std::future::Future;
use std::task::{Context, Poll, Waker};
use std::time::Duration;

use iletki::gpu::wgpu;
use iletki::{
    DrawStyle2D, Mesh2D, PassTargetDesc, PickId, Scene2DLayer, Vec2, ViewportConfig, ViewportError,
    ViewportHost, ViewportLayer,
};
use iletki_egui::{EguiViewportResource, ViewportPaintCallback, egui, egui_pass_target, egui_wgpu};

const SIZE: u32 = 64;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const COLOR: [u8; 4] = [0, 255, 0, 255];
const ID: PickId = PickId::new(1, 0x1234_5678, 0x9abc);

struct Viewport {
    host: ViewportHost,
    layer: Scene2DLayer,
    prepare_error: Option<ViewportError>,
}

impl EguiViewportResource for Viewport {
    fn host(&self) -> &ViewportHost {
        &self.host
    }

    fn host_mut(&mut self) -> &mut ViewportHost {
        &mut self.host
    }

    fn pass_target(&self) -> PassTargetDesc {
        egui_pass_target(FORMAT, false)
    }

    fn prepare_viewport(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let target = self.pass_target();
        let layers: &mut [&mut dyn ViewportLayer<()>] = &mut [&mut self.layer];
        self.prepare_error = self
            .host
            .prepare_for_pass(device, queue, encoder, &(), layers, target)
            .err();
    }
}

fn block_on<T>(future: impl Future<Output = T>) -> T {
    let mut context = Context::from_waker(Waker::noop());
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = block_on(instance.request_adapter(&Default::default()))?;
    let (device, queue) = block_on(adapter.request_device(&Default::default()))?;

    let mut mesh = Mesh2D::default();
    mesh.push_rect_fill(
        Vec2::new(-12.0, -12.0),
        Vec2::new(12.0, 12.0),
        DrawStyle2D::new(COLOR, false, 0.5, ID),
    );
    let mut layer = Scene2DLayer::new(ID.layer);
    layer.set_static_mesh(&mesh);
    let host = ViewportHost::new(
        &device,
        ViewportConfig {
            color_format: FORMAT,
            initial_px: [SIZE; 2],
            ..Default::default()
        },
    )?;
    let mut renderer = egui_wgpu::Renderer::new(&device, FORMAT, Default::default());
    renderer.callback_resources.insert(Viewport {
        host,
        layer,
        prepare_error: None,
    });

    let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::Vec2::splat(SIZE as f32));
    let paint = ViewportPaintCallback::<Viewport>::new(rect, 1.0).into_paint_callback(rect);
    let jobs = [egui::ClippedPrimitive {
        clip_rect: rect,
        primitive: egui::epaint::Primitive::Callback(paint),
    }];
    let screen = egui_wgpu::ScreenDescriptor {
        size_in_pixels: [SIZE; 2],
        pixels_per_point: 1.0,
    };
    let mut encoder = device.create_command_encoder(&Default::default());
    let extra = renderer.update_buffers(&device, &queue, &mut encoder, &jobs, &screen);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("egui output"),
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
        let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
        renderer.render(&mut pass.forget_lifetime(), &jobs, &screen);
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
    let submission = queue.submit(extra.into_iter().chain([encoder.finish()]));

    let viewport = renderer
        .callback_resources
        .get_mut::<Viewport>()
        .ok_or("missing viewport")?;
    if let Some(error) = viewport.prepare_error.take() {
        return Err(error.into());
    }
    let (sender, receiver) = std::sync::mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    let timeout = Some(Duration::from_secs(10));
    device.poll(wgpu::PollType::Wait {
        submission_index: Some(submission),
        timeout,
    })?;
    receiver.recv_timeout(Duration::from_secs(10))??;
    let pixel = readback.get_mapped_range(..)?;
    if *pixel != COLOR {
        return Err(format!("unexpected egui center pixel: {:?}", &*pixel).into());
    }
    drop(pixel);
    readback.unmap();

    let layers: &mut [&mut dyn ViewportLayer<()>] = &mut [&mut viewport.layer];
    let hits = viewport.host.pick_rect_blocking(
        &device,
        &queue,
        &(),
        layers,
        [SIZE / 2, SIZE / 2, 1, 1],
    )?;
    if !matches!(hits.as_slice(), [hit] if hit.id == ID) {
        return Err(format!("unexpected pick: {hits:?}").into());
    }
    let info = adapter.get_info();
    println!(
        "{:?} ({}) rendered {COLOR:?} and picked {ID:?}",
        info.backend, info.name
    );
    Ok(())
}
