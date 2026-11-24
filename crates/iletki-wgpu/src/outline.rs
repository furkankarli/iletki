//! Jump-Flooding Algorithm (JFA) screen-space selection outline post-processing engine.
//!
//! Lazily allocated only when at least one active layer participates in [`DrawPhase::OutlineMask`].
//! Renders selected objects to an `R8Unorm` coverage mask, propagates nearest-seed offsets in
//! width-dependent JFA passes, and composites a smooth anti-aliased silhouette
//! outline of up to 15 px thickness onto the active viewport target.

use bytemuck::{Pod, Zeroable};
use iletki_core::OutlineConfig;
use wgpu::util::DeviceExt;

/// Standard texture format used by [`iletki_core::DrawPhase::OutlineMask`] pipelines.
pub const OUTLINE_MASK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct JfaParams {
    step_px: f32,
    width_px: f32,
    coverage_aa: u32,
    _pad: u32,
    color: [f32; 4],
    active_px: [f32; 2],
    _pad2: [f32; 2],
}

/// Lazily initialized GPU resources for the Jump-Flood selection outline pass.
pub(crate) struct OutlineEngine {
    seed_pipe: wgpu::RenderPipeline,
    step_pipe: wgpu::RenderPipeline,
    composite_pipe: wgpu::RenderPipeline,
    composite_format: wgpu::TextureFormat,
    composite_msaa: u32,
    // Separate uniform buffers for each pass so queue writes never clobber each other.
    step_buffers: [wgpu::Buffer; 5],
    composite_buffer: wgpu::Buffer,
    extent: [u32; 2],
    mask_view: wgpu::TextureView,
    multisampled_mask_view: Option<wgpu::TextureView>,
    ping_view: wgpu::TextureView,
    pong_view: wgpu::TextureView,
    seed_ping_bg: wgpu::BindGroup,
    seed_pong_bg: wgpu::BindGroup,
    step_ping_to_pong_bgs: [wgpu::BindGroup; 5],
    step_pong_to_ping_bgs: [wgpu::BindGroup; 5],
    composite_bg: wgpu::BindGroup,
}

impl std::fmt::Debug for OutlineEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OutlineEngine")
            .field("extent", &self.extent)
            .field("composite_format", &self.composite_format)
            .field("composite_msaa", &self.composite_msaa)
            .finish()
    }
}

impl OutlineEngine {
    pub(crate) fn ensure<'a>(
        slot: &'a mut Option<Self>,
        device: &wgpu::Device,
        allocated_px: [u32; 2],
        composite_format: wgpu::TextureFormat,
        composite_msaa: u32,
        depth_format: wgpu::TextureFormat,
        multisampled_mask: bool,
    ) -> &'a mut Self {
        let needs_rebuild = slot.as_ref().is_none_or(|e| {
            e.extent != allocated_px
                || e.composite_format != composite_format
                || e.composite_msaa != composite_msaa
                || e.multisampled_mask_view.is_some() != multisampled_mask
        });
        if needs_rebuild {
            *slot = Some(Self::new(
                device,
                allocated_px,
                composite_format,
                composite_msaa,
                depth_format,
                multisampled_mask,
            ));
        }
        slot.get_or_insert_with(|| {
            Self::new(
                device,
                allocated_px,
                composite_format,
                composite_msaa,
                depth_format,
                multisampled_mask,
            )
        })
    }

    fn new(
        device: &wgpu::Device,
        allocated_px: [u32; 2],
        composite_format: wgpu::TextureFormat,
        composite_msaa: u32,
        depth_format: wgpu::TextureFormat,
        multisampled_mask: bool,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("iletki outline jfa shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/outline_jfa.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("iletki outline jfa layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("iletki outline jfa pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        let make_jfa = |label: &'static str, fs: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipe_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_full"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fs),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rg16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        let seed_pipe = make_jfa("iletki jfa seed", "fs_seed");
        let step_pipe = make_jfa("iletki jfa step", "fs_step");
        let composite_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("iletki jfa composite"),
            layout: Some(&pipe_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_full"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_composite"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: composite_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: composite_msaa.max(1),
                ..wgpu::MultisampleState::default()
            },
            multiview_mask: None,
            cache: None,
        });

        let [width, height] = allocated_px;
        let size = wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        };
        let mk_tex = |label: &'static str, format: wgpu::TextureFormat| {
            let t = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            t.create_view(&wgpu::TextureViewDescriptor::default())
        };
        let mask_view = mk_tex("iletki outline mask", OUTLINE_MASK_FORMAT);
        let multisampled_mask_view = multisampled_mask.then(|| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("iletki multisampled outline mask"),
                    size,
                    mip_level_count: 1,
                    sample_count: 4,
                    dimension: wgpu::TextureDimension::D2,
                    format: OUTLINE_MASK_FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        });
        let ping_view = mk_tex("iletki jfa ping", wgpu::TextureFormat::Rg16Float);
        let pong_view = mk_tex("iletki jfa pong", wgpu::TextureFormat::Rg16Float);

        let mk_buf = || {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("iletki jfa params"),
                contents: bytemuck::bytes_of(&JfaParams::zeroed()),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        };
        let step_buffers = [mk_buf(), mk_buf(), mk_buf(), mk_buf(), mk_buf()];
        let composite_buffer = mk_buf();

        let mk_bg = |src: &wgpu::TextureView, buf: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("iletki jfa bg"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(src),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&mask_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buf.as_entire_binding(),
                    },
                ],
            })
        };

        let seed_ping_bg = mk_bg(&pong_view, &composite_buffer);
        let seed_pong_bg = mk_bg(&ping_view, &composite_buffer);
        let step_ping_to_pong_bgs = step_buffers.each_ref().map(|buf| mk_bg(&ping_view, buf));
        let step_pong_to_ping_bgs = step_buffers.each_ref().map(|buf| mk_bg(&pong_view, buf));
        let composite_bg = mk_bg(&ping_view, &composite_buffer);

        Self {
            seed_pipe,
            step_pipe,
            composite_pipe,
            composite_format,
            composite_msaa,
            step_buffers,
            composite_buffer,
            extent: allocated_px,
            mask_view,
            multisampled_mask_view,
            ping_view,
            pong_view,
            seed_ping_bg,
            seed_pong_bg,
            step_ping_to_pong_bgs,
            step_pong_to_ping_bgs,
            composite_bg,
        }
    }

    pub(crate) fn mask_color_views(&self) -> (&wgpu::TextureView, Option<&wgpu::TextureView>) {
        self.multisampled_mask_view
            .as_ref()
            .map_or((&self.mask_view, None), |view| {
                (view, Some(&self.mask_view))
            })
    }

    /// Runs seed initialization and the propagation needed for the requested width.
    pub(crate) fn run_jump_flood(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        active_px: [u32; 2],
        rect: [u32; 4],
        config: OutlineConfig,
    ) {
        let [aw, ah] = active_px;
        let [alloc_w, alloc_h] = self.extent;
        let width = if config.width_px.is_finite() {
            config.width_px.clamp(1.0, 15.0)
        } else {
            OutlineConfig::default().width_px
        };
        // Include the 0.65 px feather and up to 0.5 px of subpixel seed displacement.
        let steps: &[f32] = match width {
            ..=2.0 => &[2.0, 1.0],
            ..=6.0 => &[4.0, 2.0, 1.0],
            ..=14.0 => &[8.0, 4.0, 2.0, 1.0],
            _ => &[8.0, 4.0, 2.0, 1.0, 1.0],
        };

        let [cr, cg, cb, ca] = config.color;
        let target_color = if self.composite_format.is_srgb() {
            let lin = |c: f32| {
                let c = c.clamp(0.0, 1.0);
                if c <= 0.04045 {
                    c / 12.92
                } else {
                    ((c + 0.055) / 1.055).powf(2.4)
                }
            };
            [lin(cr), lin(cg), lin(cb), ca]
        } else {
            config.color
        };

        let params = |step_px| JfaParams {
            step_px,
            width_px: width,
            coverage_aa: u32::from(self.multisampled_mask_view.is_some()),
            _pad: 0,
            color: target_color,
            active_px: [aw as f32, ah as f32],
            _pad2: [0.0, 0.0],
        };
        for (&step, buf) in steps.iter().zip(&self.step_buffers) {
            queue.write_buffer(buf, 0, bytemuck::bytes_of(&params(step)));
        }
        queue.write_buffer(&self.composite_buffer, 0, bytemuck::bytes_of(&params(1.0)));

        // Choose the initial texture so the last propagation always writes into ping.
        let (seed_view, seed_bg) = if steps.len().is_multiple_of(2) {
            (&self.ping_view, &self.seed_ping_bg)
        } else {
            (&self.pong_view, &self.seed_pong_bg)
        };
        self.fullscreen_pass(
            encoder,
            "iletki jfa seed",
            seed_view,
            &self.seed_pipe,
            seed_bg,
            [alloc_w, alloc_h],
            rect,
        );

        for i in 0..steps.len() {
            let (dst, bg) = if (i + steps.len()).is_multiple_of(2) {
                (&self.pong_view, self.step_ping_to_pong_bgs.get(i))
            } else {
                (&self.ping_view, self.step_pong_to_ping_bgs.get(i))
            };
            if let Some(bg) = bg {
                self.fullscreen_pass(
                    encoder,
                    "iletki jfa step",
                    dst,
                    &self.step_pipe,
                    bg,
                    [alloc_w, alloc_h],
                    rect,
                );
            }
        }
    }

    /// Composites the smooth Jump-Flood outline halo inside an open color render pass.
    pub(crate) fn composite_into_pass(&self, pass: &mut wgpu::RenderPass<'_>, rect: [u32; 4]) {
        let [x, y, w, h] = rect;
        let [alloc_w, alloc_h] = self.extent;
        pass.set_viewport(0.0, 0.0, alloc_w as f32, alloc_h as f32, 0.0, 1.0);
        pass.set_scissor_rect(x, y, w, h);
        pass.set_pipeline(&self.composite_pipe);
        pass.set_bind_group(0, &self.composite_bg, &[]);
        pass.draw(0..3, 0..1);
    }

    #[allow(clippy::too_many_arguments)]
    fn fullscreen_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        label: &'static str,
        dst_view: &wgpu::TextureView,
        pipeline: &wgpu::RenderPipeline,
        bg: &wgpu::BindGroup,
        [alloc_w, alloc_h]: [u32; 2],
        [x, y, w, h]: [u32; 4],
    ) {
        let gpu_timer = crate::trace::pass(label);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dst_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: -128.0,
                        g: -128.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: gpu_timer.as_ref().and_then(|timer| timer.writes()),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_viewport(0.0, 0.0, alloc_w as f32, alloc_h as f32, 0.0, 1.0);
        pass.set_scissor_rect(x, y, w, h);
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.draw(0..3, 0..1);
    }
}
