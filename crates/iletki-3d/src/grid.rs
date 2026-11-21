//! Infinite anti-aliased 3D ground grid with Reversed-Z exact depth writing.

use bytemuck::{Pod, Zeroable};
use iletki_core::{DrawPhase, DrawPhaseMask, ViewportId};
use iletki_wgpu::wgpu;
use iletki_wgpu::wgpu::util::DeviceExt;
use iletki_wgpu::{
    CAMERA_WGSL, DEPTH_FORMAT, LayerRevision, PaintCtx, PrepareCtx, ViewportLayer, ViewportLifetime,
};

/// Ground plane orientation for [`InfiniteGrid3DLayer`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GridPlane {
    /// Horizontal XY plane (`z = 0`) for Z-up scenes.
    #[default]
    XyZUp,
    /// Horizontal XZ plane (`y = 0`, standard for Y-up game engines).
    XzYUp,
}

/// Visual and spacing configuration for [`InfiniteGrid3DLayer`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grid3DConfig {
    /// Ground plane orientation.
    pub plane: GridPlane,
    /// Minor grid line spacing in world units (default `1.0`).
    pub minor_spacing: f32,
    /// Major grid line spacing in world units (default `10.0`).
    pub major_spacing: f32,
    /// Distance at which the grid fades smoothly to transparent (default `250.0`).
    pub fade_distance: f32,
    /// Minor line RGBA `[0..1]`.
    pub minor_color: [f32; 4],
    /// Major line RGBA `[0..1]`.
    pub major_color: [f32; 4],
    /// Primary axis (X) RGBA `[0..1]`.
    pub axis_u_color: [f32; 4],
    /// Secondary axis (Y in Z-up, Z in Y-up) RGBA `[0..1]`.
    pub axis_v_color: [f32; 4],
}

impl Default for Grid3DConfig {
    fn default() -> Self {
        Self {
            plane: GridPlane::XyZUp,
            minor_spacing: 1.0,
            major_spacing: 10.0,
            fade_distance: 250.0,
            minor_color: [1.0, 1.0, 1.0, 0.14],
            major_color: [1.0, 1.0, 1.0, 0.30],
            axis_u_color: [0.90, 0.25, 0.25, 0.85],
            axis_v_color: [0.25, 0.82, 0.35, 0.85],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GridUniforms {
    cam_pos_rel_plane: [f32; 3],
    minor_spacing: f32,
    major_spacing: f32,
    fade_distance: f32,
    plane_mode: u32,
    _pad: u32,
    minor_color: [f32; 4],
    major_color: [f32; 4],
    axis_u_color: [f32; 4],
    axis_v_color: [f32; 4],
}

struct GridViewportBinding {
    id: ViewportId,
    lifetime: ViewportLifetime,
    buf: wgpu::Buffer,
    bg: wgpu::BindGroup,
}

struct GridGpu {
    format: wgpu::TextureFormat,
    msaa: u32,
    layout: wgpu::BindGroupLayout,
    per_viewport: Vec<GridViewportBinding>,
    pipeline: wgpu::RenderPipeline,
}

/// Analytic infinite anti-aliased 3D ground grid layer rendered in [`DrawPhase::Transparent`].
pub struct InfiniteGrid3DLayer {
    layer_id: u16,
    revision: LayerRevision,
    prepared_config: Grid3DConfig,
    /// Active grid visual/spacing configuration.
    pub config: Grid3DConfig,
    /// Toggle grid rendering without removing the layer.
    pub visible: bool,
    gpu: Option<GridGpu>,
}

impl std::fmt::Debug for InfiniteGrid3DLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InfiniteGrid3DLayer")
            .field("layer_id", &self.layer_id)
            .field("config", &self.config)
            .field("visible", &self.visible)
            .finish()
    }
}

impl InfiniteGrid3DLayer {
    /// Creates an infinite 3D ground grid layer.
    #[must_use]
    pub fn new(layer_id: u16, config: Grid3DConfig) -> Self {
        Self {
            layer_id,
            revision: LayerRevision::new(),
            prepared_config: config,
            config,
            visible: true,
            gpu: None,
        }
    }

    fn ensure_gpu<W>(&mut self, ctx: &PrepareCtx<'_, W>) {
        if self
            .gpu
            .as_ref()
            .is_some_and(|g| g.format == ctx.color_format && g.msaa == ctx.msaa_samples)
        {
            return;
        }
        let wgsl = format!("{CAMERA_WGSL}\n{}", include_str!("shaders/grid3d.wgsl"));
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("iletki-3d grid shader"),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });
        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("iletki-3d grid layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("iletki-3d grid pipeline layout"),
                bind_group_layouts: &[Some(ctx.camera_bind_group_layout), Some(&layout)],
                immediate_size: 0,
            });
        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("iletki-3d grid pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_grid"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_grid"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: ctx.color_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(false),
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
            });
        self.gpu = Some(GridGpu {
            format: ctx.color_format,
            msaa: ctx.msaa_samples,
            layout,
            per_viewport: Vec::new(),
            pipeline,
        });
    }
}

impl<W> ViewportLayer<W> for InfiniteGrid3DLayer {
    fn id(&self) -> u16 {
        self.layer_id
    }

    fn phases(&self) -> DrawPhaseMask {
        if self.visible {
            DrawPhaseMask::TRANSPARENT
        } else {
            DrawPhaseMask::NONE
        }
    }

    fn revision(&self) -> LayerRevision {
        self.revision
    }

    fn prepare(&mut self, ctx: &mut PrepareCtx<'_, W>) -> bool {
        if let Some(gpu) = &mut self.gpu {
            gpu.per_viewport
                .retain(|binding| binding.lifetime.is_alive());
        }
        if self.prepared_config != self.config {
            self.prepared_config = self.config;
            self.revision.mark_static();
        }
        if !self.visible {
            return false;
        }
        self.ensure_gpu(ctx);
        let Some(gpu) = &mut self.gpu else {
            return false;
        };
        let vp_id = ctx.viewport_id;
        if !gpu.per_viewport.iter().any(|b| b.id == vp_id) {
            let buf = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("iletki-3d grid uniforms"),
                    contents: bytemuck::bytes_of(&GridUniforms::zeroed()),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("iletki-3d grid bg"),
                layout: &gpu.layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            });
            gpu.per_viewport.push(GridViewportBinding {
                id: vp_id,
                lifetime: ctx.viewport_lifetime(),
                buf,
                bg,
            });
        }
        if let Some(binding) = gpu.per_viewport.iter().find(|b| b.id == vp_id) {
            let cam = ctx.camera.position_3d();
            let u = GridUniforms {
                cam_pos_rel_plane: cam.to_f32(),
                minor_spacing: self.config.minor_spacing.max(1e-4),
                major_spacing: self.config.major_spacing.max(1e-4),
                fade_distance: self.config.fade_distance.max(1.0),
                plane_mode: match self.config.plane {
                    GridPlane::XyZUp => 0,
                    GridPlane::XzYUp => 1,
                },
                _pad: 0,
                minor_color: self.config.minor_color,
                major_color: self.config.major_color,
                axis_u_color: self.config.axis_u_color,
                axis_v_color: self.config.axis_v_color,
            };
            ctx.queue
                .write_buffer(&binding.buf, 0, bytemuck::bytes_of(&u));
        }
        false
    }

    fn paint(&self, phase: DrawPhase, pass: &mut wgpu::RenderPass<'_>, ctx: &PaintCtx<'_, W>) {
        if !self.visible || phase != DrawPhase::Transparent {
            return;
        }
        let Some(gpu) = &self.gpu else {
            return;
        };
        let Some(binding) = gpu.per_viewport.iter().find(|b| b.id == ctx.viewport_id) else {
            return;
        };
        pass.set_pipeline(&gpu.pipeline);
        pass.set_bind_group(1, &binding.bg, &[]);
        pass.draw(0..3, 0..1);
    }
}
