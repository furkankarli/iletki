//! Headless `wgpu` integration tests for `iletki-wgpu`:
//! - OnDemand static-cache reuse (zero static redraw on idle and overlay-only updates)
//! - Exact RGBA pixel verification of static scene + dynamic overlay compositing
//! - BorrowedPass blitting into an external target
//! - 64-bit `PickId` + Reversed-Z depth picking

#![cfg(not(target_arch = "wasm32"))]

use iletki_core::{DirtyFlags, DrawPhase, DrawPhaseMask, PickId, ViewportMode};
use iletki_wgpu::{
    CAMERA_WGSL, DEPTH_FORMAT, PICK_ID_FORMAT, PaintCtx, PassTargetDesc, PrepareCtx,
    ViewportConfig, ViewportError, ViewportHost, ViewportLayer, wgpu,
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
    create_headless_device_with_limits(wgpu::Limits::default())
}

fn create_headless_device_with_limits(
    required_limits: wgpu::Limits,
) -> Option<(wgpu::Device, wgpu::Queue)> {
    let expected = std::env::var_os("ILETKI_EXPECT_BACKEND");
    let result = request_headless_device(required_limits, expected.as_deref());
    assert!(
        expected.is_none() || result.is_ok(),
        "could not use the expected {expected:?} backend: {:?}",
        result.as_ref().err()
    );
    result.ok()
}

#[allow(clippy::print_stderr)] // Include the selected adapter in the CI smoke test output.
fn request_headless_device(
    required_limits: wgpu::Limits,
    expected: Option<&std::ffi::OsStr>,
) -> Result<(wgpu::Device, wgpu::Queue), String> {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let options = wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        ..wgpu::RequestAdapterOptions::default()
    };
    let adapter = block_on(instance.request_adapter(&options))
        .or_else(|_| {
            block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                force_fallback_adapter: true,
                ..options
            }))
        })
        .map_err(|error| error.to_string())?;
    let info = adapter.get_info();
    if let Some(expected) = expected {
        assert_eq!(info.backend.to_str(), expected, "unexpected GPU backend");
        eprintln!(
            "Backend {} using {} ({:?})",
            info.backend, info.name, info.device_type
        );
    }
    block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("iletki test device"),
        required_limits,
        ..wgpu::DeviceDescriptor::default()
    }))
    .map_err(|error| error.to_string())
}

#[test]
fn gpu_configuration_returns_errors_before_creating_pipelines() {
    let Some((device, queue)) = create_headless_device() else {
        return;
    };
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let format = wgpu::TextureFormat::Rgba8Uint;
    let invalid_color = ViewportHost::new(
        &device,
        ViewportConfig {
            color_format: format,
            ..ViewportConfig::default()
        },
    );
    assert_eq!(
        invalid_color.unwrap_err(),
        ViewportError::UnsupportedColorFormat(format)
    );

    let config = ViewportConfig::default();
    let invalid_samples = ViewportHost::new(
        &device,
        ViewportConfig {
            msaa_samples: 2,
            ..config
        },
    );
    assert_eq!(
        invalid_samples.unwrap_err(),
        ViewportError::UnsupportedSampleCount {
            format: config.color_format,
            sample_count: 2,
        }
    );

    let mut host = ViewportHost::new(&device, config).unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    let pass = PassTargetDesc {
        format: config.color_format,
        sample_count: 1,
        depth_format: Some(format),
        blend_alpha: false,
    };
    assert_eq!(
        host.prepare_for_pass(&device, &queue, &mut encoder, &(), &mut [], pass),
        Err(ViewportError::UnsupportedDepthFormat(format)),
    );
    assert_eq!(
        host.prepare_for_pass(
            &device,
            &queue,
            &mut encoder,
            &(),
            &mut [],
            PassTargetDesc {
                sample_count: 0,
                depth_format: None,
                ..pass
            }
        ),
        Err(ViewportError::UnsupportedSampleCount {
            format: pass.format,
            sample_count: 0
        }),
    );
    assert!(block_on(scope.pop()).is_none());
}

#[test]
fn target_sizing_and_readback_respect_device_limits() {
    let Some((device, queue)) = create_headless_device_with_limits(wgpu::Limits {
        max_texture_dimension_2d: 500,
        max_buffer_size: 16 * 1024,
        ..wgpu::Limits::default()
    }) else {
        return;
    };
    let mut host = ViewportHost::new(
        &device,
        ViewportConfig {
            initial_px: [128, 128],
            ..ViewportConfig::default()
        },
    )
    .unwrap();
    host.resize(&device, [513, 128], 1.0);
    assert_eq!(host.size().active_px, [500, 128]);
    assert_eq!(host.size().allocated_px, [500, 256]);
    assert_eq!(
        host.size().uv_max.map(f32::to_bits),
        [1.0_f32, 0.5].map(f32::to_bits)
    );

    let mut encoder = device.create_command_encoder(&Default::default());
    let mut layers: [&mut dyn ViewportLayer<()>; 0] = [];
    host.render_to_texture(&device, &queue, &mut encoder, &(), &mut layers);
    queue.submit([encoder.finish()]);

    assert_eq!(
        host.readback_rgba_blocking(&device, &queue),
        Err(ViewportError::RegionTooLarge)
    );
    assert_eq!(
        host.pick_rect_blocking(&device, &queue, &(), &mut layers, [0, 0, 500, 128]),
        Err(ViewportError::RegionTooLarge)
    );
}

/// Test layer that draws a green quad in `Opaque`, an optional red center quad in `Overlay`, and
/// writes a 64-bit `PickId` in `Picking`.
struct SolidQuadLayer {
    opaque_pipe: Option<wgpu::RenderPipeline>,
    overlay_pipe: Option<wgpu::RenderPipeline>,
    pick_pipe: Option<wgpu::RenderPipeline>,
    draw_overlay: bool,
    pick_id: PickId,
}

impl SolidQuadLayer {
    fn new(pick_id: PickId) -> Self {
        Self {
            opaque_pipe: None,
            overlay_pipe: None,
            pick_pipe: None,
            draw_overlay: false,
            pick_id,
        }
    }
}

impl ViewportLayer<()> for SolidQuadLayer {
    fn id(&self) -> u16 {
        self.pick_id.layer
    }

    fn phases(&self) -> DrawPhaseMask {
        let base = DrawPhaseMask::OPAQUE.union(DrawPhaseMask::PICKING);
        if self.draw_overlay {
            base.union(DrawPhaseMask::OVERLAY)
        } else {
            base
        }
    }

    fn prepare(&mut self, ctx: &mut PrepareCtx<'_, ()>) -> bool {
        if self.opaque_pipe.is_some() {
            return false;
        }
        let [id_lo, id_hi] = self.pick_id.to_rg32u();
        let src = format!(
            "{CAMERA_WGSL}
            @vertex fn vs_big(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {{
                var pos = array<vec2<f32>, 6>(
                    vec2(-0.8, -0.8), vec2(0.8, -0.8), vec2(0.8, 0.8),
                    vec2(-0.8, -0.8), vec2(0.8, 0.8), vec2(-0.8, 0.8),
                );
                // Reversed-Z depth 0.5 (well in front of far=0.0).
                return vec4(pos[i], 0.5, 1.0);
            }}
            @vertex fn vs_center(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {{
                var pos = array<vec2<f32>, 6>(
                    vec2(-0.25, -0.25), vec2(0.25, -0.25), vec2(0.25, 0.25),
                    vec2(-0.25, -0.25), vec2(0.25, 0.25), vec2(-0.25, 0.25),
                );
                return vec4(pos[i], 0.8, 1.0);
            }}
            @fragment fn fs_green() -> @location(0) vec4<f32> {{ return vec4(0.0, 1.0, 0.0, 1.0); }}
            @fragment fn fs_red() -> @location(0) vec4<f32> {{ return vec4(1.0, 0.0, 0.0, 1.0); }}
            @fragment fn fs_pick() -> @location(0) vec2<u32> {{ return vec2({id_lo}u, {id_hi}u); }}
            "
        );
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("solid quad test"),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("solid quad layout"),
                bind_group_layouts: &[Some(ctx.camera_bind_group_layout)],
                immediate_size: 0,
            });
        let mk_color = |vs: &str, fs: &str| {
            ctx.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("solid quad color"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some(vs),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some(fs),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: ctx.color_format,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: ctx.depth_format,
                        depth_write_enabled: Some(true),
                        depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState {
                        count: ctx.msaa_samples,
                        ..wgpu::MultisampleState::default()
                    },
                    multiview_mask: None,
                    cache: None,
                })
        };
        self.opaque_pipe = Some(mk_color("vs_big", "fs_green"));
        self.overlay_pipe = Some(mk_color("vs_center", "fs_red"));
        self.pick_pipe = Some(
            ctx.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("solid quad pick"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_big"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_pick"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: PICK_ID_FORMAT,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: DEPTH_FORMAT,
                        depth_write_enabled: Some(true),
                        depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                }),
        );
        false
    }

    fn paint(&self, phase: DrawPhase, pass: &mut wgpu::RenderPass<'_>, _ctx: &PaintCtx<'_, ()>) {
        let pipe = match phase {
            DrawPhase::Opaque => self.opaque_pipe.as_ref(),
            DrawPhase::Overlay => self.overlay_pipe.as_ref(),
            DrawPhase::Picking => self.pick_pipe.as_ref(),
            _ => None,
        };
        if let Some(p) = pipe {
            pass.set_pipeline(p);
            pass.draw(0..6, 0..1);
        }
    }
}

#[test]
fn on_demand_caches_static_scene_and_picks_64bit_id() {
    let Some((device, queue)) = create_headless_device() else {
        // CI host without any GPU or software Vulkan/DX12/Metal adapter.
        return;
    };

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
    let expected_id = PickId::new(7, 42_001, 19);
    let mut layer = SolidQuadLayer::new(expected_id);

    // Frame 1: static scene builds into static_cache.
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let stats1 = {
        let mut layers: [&mut dyn ViewportLayer<()>; 1] = [&mut layer];
        let out = host.render_to_texture(&device, &queue, &mut enc, &(), &mut layers);
        assert_eq!(out.active_px, [128, 128]);
        assert!((out.uv_max[0] - 0.5).abs() < 1e-6);
        out.stats
    };
    queue.submit(std::iter::once(enc.finish()));
    assert!(stats1.static_rebuilt);
    assert!(!stats1.dynamic_drawn);

    let img1 = host.readback_rgba_blocking(&device, &queue).unwrap();
    // Center is green, corner is black.
    assert_eq!(img1.pixel(64, 64), Some([0, 255, 0, 255]));
    assert_eq!(img1.pixel(2, 2), Some([0, 0, 0, 255]));

    // Frame 2 (idle): zero static redraws, zero dynamic redraws!
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let stats2 = {
        let mut layers: [&mut dyn ViewportLayer<()>; 1] = [&mut layer];
        host.render_to_texture(&device, &queue, &mut enc, &(), &mut layers)
            .stats
    };
    queue.submit(std::iter::once(enc.finish()));
    assert!(!stats2.static_rebuilt);
    assert!(!stats2.dynamic_drawn);
    assert_eq!(stats2.layer_draws, 0);

    // Frame 3 (overlay-only dirty): static scene stays cached (0 opaque draws!), red overlay runs!
    layer.draw_overlay = true;
    host.mark_dirty(DirtyFlags::OVERLAY);
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let stats3 = {
        let mut layers: [&mut dyn ViewportLayer<()>; 1] = [&mut layer];
        host.render_to_texture(&device, &queue, &mut enc, &(), &mut layers)
            .stats
    };
    queue.submit(std::iter::once(enc.finish()));
    assert!(!stats3.static_rebuilt);
    assert!(stats3.dynamic_drawn);
    assert_eq!(stats3.layer_draws, 1);

    let img3 = host.readback_rgba_blocking(&device, &queue).unwrap();
    // Center is now red (overlay), mid-ring stays green (blitted from static_cache), corner black!
    assert_eq!(img3.pixel(64, 64), Some([255, 0, 0, 255]));
    assert_eq!(img3.pixel(30, 64), Some([0, 255, 0, 255]));
    assert_eq!(img3.pixel(2, 2), Some([0, 0, 0, 255]));

    // GPU 64-bit ID + depth picking at center.
    let hits = {
        let mut layers: [&mut dyn ViewportLayer<()>; 1] = [&mut layer];
        host.pick_rect_blocking(&device, &queue, &(), &mut layers, [60, 60, 9, 9])
            .unwrap()
    };
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, expected_id);
    assert!((hits[0].depth - 0.5).abs() < 1e-4);

    // Picking in empty corner returns no hits.
    let empty = {
        let mut layers: [&mut dyn ViewportLayer<()>; 1] = [&mut layer];
        host.pick_rect_blocking(&device, &queue, &(), &mut layers, [0, 0, 4, 4])
            .unwrap()
    };
    assert!(empty.is_empty());

    // Async cross-platform GPU picking path (also available on wasm32/WebGPU) matches blocking.
    let async_hits = {
        let mut layers: [&mut dyn ViewportLayer<()>; 1] = [&mut layer];
        block_on(host.pick_rect_async(&device, &queue, &(), &mut layers, [60, 60, 9, 9])).unwrap()
    };
    assert_eq!(async_hits.len(), 1);
    assert_eq!(async_hits[0].id, expected_id);
    assert!((async_hits[0].depth - 0.5).abs() < 1e-4);
}

#[test]
fn pick_can_start_before_render_and_outlive_the_viewport() {
    let Some((device, queue)) = create_headless_device() else {
        return;
    };
    let mut host = ViewportHost::new(
        &device,
        ViewportConfig {
            msaa_samples: 1,
            initial_px: [128, 128],
            ..ViewportConfig::default()
        },
    )
    .unwrap();
    let id = PickId::new(7, 98, 3);
    let mut layer = SolidQuadLayer::new(id);
    let mut layers: [&mut dyn ViewportLayer<()>; 1] = [&mut layer];
    let mut request = host
        .begin_pick_rect(&device, &queue, &(), &mut layers, [60, 60, 9, 9])
        .unwrap();
    let mut corner = host
        .begin_pick_rect(&device, &queue, &(), &mut layers, [0, 0, 9, 9])
        .unwrap();
    let empty = block_on(std::future::poll_fn(|_| corner.try_read())).unwrap();
    assert!(empty.is_empty());
    let mut reused = host
        .begin_pick_rect(&device, &queue, &(), &mut layers, [63, 63, 3, 3])
        .unwrap();
    let hits = block_on(std::future::poll_fn(|_| reused.try_read())).unwrap();
    assert_eq!(hits.first().map(|hit| hit.id), Some(id));

    host.resize(&device, [512, 384], 2.0);
    drop(host);
    drop(layer);

    let hits = block_on(std::future::poll_fn(|_| request.try_read())).unwrap();
    assert_eq!(hits.first().map(|hit| hit.id), Some(id));
    assert_eq!(
        request.try_read(),
        Poll::Ready(Err(ViewportError::PickAlreadyRead))
    );
}
