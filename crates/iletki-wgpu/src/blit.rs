//! Fullscreen single-triangle textured blit pipeline for static-cache compositing and borrowed
//! render-pass output.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlitParams {
    uv_max: [f32; 2],
    _pad: [f32; 2],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PipelineKey {
    format: wgpu::TextureFormat,
    sample_count: u32,
    depth_format: Option<wgpu::TextureFormat>,
    blend_premultiplied: bool,
}

/// Manages fullscreen textured blit pipelines keyed by target format and MSAA sample count.
pub(crate) struct BlitResources {
    shader: wgpu::ShaderModule,
    layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    sampler: wgpu::Sampler,
    /// Constant `[1.0, 1.0]` UV params for internal 1:1 copies between same-sized targets.
    identity_params: wgpu::Buffer,
    /// Dynamic `uv_max` params for cropping over-allocated targets into external host passes.
    crop_params: wgpu::Buffer,
    pipelines: Vec<(PipelineKey, wgpu::RenderPipeline)>,
}

impl std::fmt::Debug for BlitResources {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlitResources")
            .field("cached_pipelines", &self.pipelines.len())
            .finish()
    }
}

impl BlitResources {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("iletki blit shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/blit.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("iletki blit layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("iletki blit pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("iletki blit sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..wgpu::SamplerDescriptor::default()
        });
        let identity_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("iletki blit identity params"),
            contents: bytemuck::bytes_of(&BlitParams {
                uv_max: [1.0, 1.0],
                _pad: [0.0, 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let crop_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("iletki blit crop params"),
            contents: bytemuck::bytes_of(&BlitParams {
                uv_max: [1.0, 1.0],
                _pad: [0.0, 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        Self {
            shader,
            layout,
            pipeline_layout,
            sampler,
            identity_params,
            crop_params,
            pipelines: Vec::new(),
        }
    }

    pub(crate) fn write_crop_uv_max(&self, queue: &wgpu::Queue, uv_max: [f32; 2]) {
        let params = BlitParams {
            uv_max,
            _pad: [0.0, 0.0],
        };
        queue.write_buffer(&self.crop_params, 0, bytemuck::bytes_of(&params));
    }

    pub(crate) fn create_identity_bind_group(
        &self,
        device: &wgpu::Device,
        texture_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        self.create_bg(device, texture_view, &self.identity_params)
    }

    pub(crate) fn create_crop_bind_group(
        &self,
        device: &wgpu::Device,
        texture_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        self.create_bg(device, texture_view, &self.crop_params)
    }

    fn create_bg(
        &self,
        device: &wgpu::Device,
        texture_view: &wgpu::TextureView,
        params: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("iletki blit bind group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
            ],
        })
    }

    pub(crate) fn ensure_pipeline(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        sample_count: u32,
        depth_format: Option<wgpu::TextureFormat>,
        blend_premultiplied: bool,
    ) {
        let key = PipelineKey {
            format,
            sample_count,
            depth_format,
            blend_premultiplied,
        };
        if self.pipelines.iter().any(|(k, _)| *k == key) {
            return;
        }
        let depth_stencil = depth_format.map(|df| wgpu::DepthStencilState {
            format: df,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });
        let blend = blend_premultiplied.then_some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING);
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("iletki blit pipeline"),
            layout: Some(&self.pipeline_layout),
            vertex: wgpu::VertexState {
                module: &self.shader,
                entry_point: Some("vs_blit"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &self.shader,
                entry_point: Some("fs_blit"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..wgpu::PrimitiveState::default()
            },
            depth_stencil,
            multisample: wgpu::MultisampleState {
                count: key.sample_count,
                ..wgpu::MultisampleState::default()
            },
            multiview_mask: None,
            cache: None,
        });
        self.pipelines.push((key, pipeline));
    }

    pub(crate) fn get_pipeline(
        &self,
        format: wgpu::TextureFormat,
        sample_count: u32,
        depth_format: Option<wgpu::TextureFormat>,
        blend_premultiplied: bool,
    ) -> Option<&wgpu::RenderPipeline> {
        let key = PipelineKey {
            format,
            sample_count,
            depth_format,
            blend_premultiplied,
        };
        self.pipelines
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, p)| p)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DepthCopyParams {
    origin: [u32; 2],
    _pad: [u32; 2],
}

/// Copies a sub-rectangle of a `Depth32Float` texture into a compact `R32Float` texture for
/// WebGPU-compliant partial depth readback during picking.
pub(crate) struct DepthCopyPipeline {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    params_buffer: wgpu::Buffer,
    scratch: Option<([u32; 2], wgpu::Texture, wgpu::TextureView)>,
}

impl DepthCopyPipeline {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("iletki depth copy shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/depth_copy.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("iletki depth copy layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
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
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("iletki depth copy pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("iletki depth copy pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_depth_copy"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_depth_copy"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R32Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("iletki depth copy params"),
            contents: bytemuck::bytes_of(&DepthCopyParams::zeroed()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        Self {
            layout,
            pipeline,
            params_buffer,
            scratch: None,
        }
    }

    pub(crate) fn copy_subrect(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        src_depth_view: &wgpu::TextureView,
        origin: [u32; 2],
        size: [u32; 2],
    ) -> wgpu::Texture {
        let [width, height] = size;
        let params = DepthCopyParams {
            origin,
            _pad: [0, 0],
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        let make_scratch = || {
            let dst = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("iletki pick depth r32f"),
                size: wgpu::Extent3d {
                    width: width.max(1),
                    height: height.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let dst_view = dst.create_view(&wgpu::TextureViewDescriptor::default());
            (size, dst, dst_view)
        };
        // Bound persistent GPU storage independently of occasional large selections.
        let temporary;
        let (_, dst, dst_view) = if width.saturating_mul(height) <= 512 * 512 {
            if self
                .scratch
                .as_ref()
                .is_some_and(|(cached, _, _)| *cached != size)
            {
                self.scratch = None;
            }
            self.scratch.get_or_insert_with(make_scratch)
        } else {
            temporary = make_scratch();
            &temporary
        };
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("iletki depth copy bg"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src_depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });
        let gpu_timer = crate::trace::pass("iletki depth subrect copy");
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("iletki depth subrect copy"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dst_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: gpu_timer.as_ref().map(|timer| timer.writes()),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
        drop(pass);
        dst.clone()
    }
}
