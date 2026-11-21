//! Screen-size-invariant 3D transform gizmo rendered in [`DrawPhase::Gizmo`] with 64-bit
//! per-axis GPU picking (`PickId::sub`).

use crate::scene::ObjectUniforms;
use crate::vertex::{Mesh3D, Vertex3D};
use iletki_core::{DrawPhase, DrawPhaseMask, PickId, Projection, Vec3, ViewportId};
use iletki_wgpu::wgpu;
use iletki_wgpu::wgpu::util::DeviceExt;
use iletki_wgpu::{
    CAMERA_WGSL, DEPTH_FORMAT, LayerRevision, PICK_ID_FORMAT, PaintCtx, PrepareCtx, ViewportLayer,
};

/// Sub-object axis identifier encoded in [`PickId::sub`] for [`TransformGizmoLayer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum GizmoAxis {
    /// World +X axis arrow (red).
    X = 1,
    /// World +Y axis arrow (green).
    Y = 2,
    /// World +Z axis arrow (blue).
    Z = 3,
    /// Center origin cube (white/yellow).
    Center = 4,
}

impl GizmoAxis {
    /// Decodes a [`PickId::sub`] value into a [`GizmoAxis`].
    #[must_use]
    pub const fn from_sub(sub: u16) -> Option<Self> {
        match sub {
            1 => Some(Self::X),
            2 => Some(Self::Y),
            3 => Some(Self::Z),
            4 => Some(Self::Center),
            _ => None,
        }
    }
}

struct GizmoViewportBinding {
    id: ViewportId,
    buf: wgpu::Buffer,
    bg: wgpu::BindGroup,
}

struct GizmoGpu {
    format: wgpu::TextureFormat,
    msaa: u32,
    vertex_buf: wgpu::Buffer,
    pick_buf: wgpu::Buffer,
    vertex_count: u32,
    object_layout: wgpu::BindGroupLayout,
    per_viewport: Vec<GizmoViewportBinding>,
    color_pipe: wgpu::RenderPipeline,
    pick_pipe: wgpu::RenderPipeline,
}

/// Interactive 3D RGB translation gizmo anchored at a `f64` world origin.
///
/// Renders in [`DrawPhase::Gizmo`] (after a fresh depth clear so it always stays visible in front
/// of 3D geometry while properly depth-sorting its own 3 axes) and in [`DrawPhase::Picking`] with
/// distinct [`GizmoAxis`] sub-IDs.
///
/// Follows standard Blender / Unity / Godot viewport sizing: fixed logical pixel height
/// ([`TransformGizmoLayer::size_px`], default `92.0` pt) regardless of window size or camera
/// distance, with an optional [`TransformGizmoLayer::max_world_scale`] clamp.
pub struct TransformGizmoLayer {
    layer_id: u16,
    revision: LayerRevision,
    prepared_origin: Option<Vec3>,
    prepared_scale: (u64, Option<u64>),
    object_id: u32,
    /// World-space `f64` position of the gizmo (set `None` to hide).
    pub origin: Option<Vec3>,
    /// Desired visual length of each axis arrow in logical points (default `92.0`).
    pub size_px: f64,
    /// Optional upper bound on the gizmo's world-space radius when zoomed very far out.
    pub max_world_scale: Option<f64>,
    gpu: Option<GizmoGpu>,
}

impl std::fmt::Debug for TransformGizmoLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransformGizmoLayer")
            .field("layer_id", &self.layer_id)
            .field("object_id", &self.object_id)
            .field("origin", &self.origin)
            .field("size_px", &self.size_px)
            .field("max_world_scale", &self.max_world_scale)
            .finish()
    }
}

impl TransformGizmoLayer {
    /// Creates a new transform gizmo layer with `layer_id` and `object_id` embedded into its
    /// [`PickId`]s.
    #[must_use]
    pub fn new(layer_id: u16, object_id: u32) -> Self {
        Self {
            layer_id,
            revision: LayerRevision::new(),
            prepared_origin: None,
            prepared_scale: (92.0_f64.to_bits(), None),
            object_id,
            origin: None,
            size_px: 92.0,
            max_world_scale: None,
            gpu: None,
        }
    }

    fn build_geometry(&self) -> Mesh3D {
        let mut m = Mesh3D::at_origin(Vec3::ZERO);
        let id = |axis: GizmoAxis| PickId::new(self.layer_id, self.object_id, axis as u16);

        m.push_axis_arrow(
            Vec3::X,
            1.0,
            0.022,
            0.074,
            [235, 65, 65, 255],
            id(GizmoAxis::X),
        );
        m.push_axis_arrow(
            Vec3::Y,
            1.0,
            0.022,
            0.074,
            [65, 215, 85, 255],
            id(GizmoAxis::Y),
        );
        m.push_axis_arrow(
            Vec3::Z,
            1.0,
            0.022,
            0.074,
            [65, 135, 245, 255],
            id(GizmoAxis::Z),
        );
        m.push_box(
            Vec3::new(-0.068, -0.068, -0.068),
            Vec3::new(0.068, 0.068, 0.068),
            [245, 220, 90, 255],
            id(GizmoAxis::Center),
            false,
        );
        m
    }

    fn ensure_gpu<W>(&mut self, ctx: &PrepareCtx<'_, W>) {
        if self
            .gpu
            .as_ref()
            .is_some_and(|g| g.format == ctx.color_format && g.msaa == ctx.msaa_samples)
        {
            return;
        }
        let mesh = self.build_geometry();
        let vertex_count = u32::try_from(mesh.vertices.len()).unwrap_or(0);
        let vertex_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("iletki-3d gizmo vertices"),
                contents: bytemuck::cast_slice(&mesh.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let pick_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("iletki-3d gizmo picks"),
                contents: bytemuck::cast_slice(&mesh.pick_ids),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let wgsl = format!("{CAMERA_WGSL}\n{}", include_str!("shaders/mesh3d.wgsl"));
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("iletki-3d gizmo shader"),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });
        let object_layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("iletki-3d gizmo object layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
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
                label: Some("iletki-3d gizmo pipeline layout"),
                bind_group_layouts: &[Some(ctx.camera_bind_group_layout), Some(&object_layout)],
                immediate_size: 0,
            });

        let vert_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex3D>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![
                0 => Float32x3,
                1 => Float32x3,
                2 => Unorm8x4
            ],
        };
        let pick_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<[u32; 2]>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![3 => Uint32x2],
        };

        let color_pipe = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("iletki-3d gizmo color"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_mesh"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Some(vert_layout.clone())],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_unlit"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: ctx.color_format,
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
                multisample: wgpu::MultisampleState {
                    count: ctx.msaa_samples,
                    ..wgpu::MultisampleState::default()
                },
                multiview_mask: None,
                cache: None,
            });

        let pick_pipe = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("iletki-3d gizmo pick"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_mesh_pick"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Some(vert_layout), Some(pick_layout)],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_pick_unclipped"),
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
            });

        self.gpu = Some(GizmoGpu {
            format: ctx.color_format,
            msaa: ctx.msaa_samples,
            vertex_buf,
            pick_buf,
            vertex_count,
            object_layout,
            per_viewport: Vec::new(),
            color_pipe,
            pick_pipe,
        });
    }
}

impl<W> ViewportLayer<W> for TransformGizmoLayer {
    fn id(&self) -> u16 {
        self.layer_id
    }

    fn phases(&self) -> DrawPhaseMask {
        if self.origin.is_some() {
            DrawPhaseMask::GIZMO.union(DrawPhaseMask::PICKING)
        } else {
            DrawPhaseMask::NONE
        }
    }

    fn revision(&self) -> LayerRevision {
        self.revision
    }

    fn prepare(&mut self, ctx: &mut PrepareCtx<'_, W>) -> bool {
        let scale = (
            self.size_px.to_bits(),
            self.max_world_scale.map(f64::to_bits),
        );
        if self.prepared_origin != self.origin || self.prepared_scale != scale {
            self.prepared_origin = self.origin;
            self.prepared_scale = scale;
            self.revision.mark_dynamic();
        }
        let Some(origin) = self.origin else {
            return false;
        };
        self.ensure_gpu(ctx);
        let Some(gpu) = &mut self.gpu else {
            return false;
        };
        let vp_id = ctx.viewport_id;
        if !gpu.per_viewport.iter().any(|b| b.id == vp_id) {
            let buf = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("iletki-3d gizmo uniforms"),
                    contents: bytemuck::bytes_of(&ObjectUniforms {
                        offset_rel: [0.0, 0.0, 0.0],
                        scale: 1.0,
                    }),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("iletki-3d gizmo bg"),
                layout: &gpu.object_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            });
            gpu.per_viewport
                .push(GizmoViewportBinding { id: vp_id, buf, bg });
        }
        if let Some(binding) = gpu.per_viewport.iter().find(|b| b.id == vp_id) {
            let cam_pos = ctx.camera.position_3d();
            let offset = origin - cam_pos;
            let vh = ctx.logical_size.y.max(1.0);
            let frac = (self.size_px.max(8.0) / vh).clamp(0.02, 0.45);
            let world_span = match ctx.camera.projection {
                Projection::Perspective3D {
                    fov_y, rotation, ..
                } => {
                    // Use depth along camera forward axis so gizmo stays exact pixel size even
                    // near viewport corners without perspective edge magnification.
                    let forward = rotation.rotate(Vec3::new(0.0, 0.0, -1.0));
                    let depth = offset
                        .dot(forward)
                        .abs()
                        .max(offset.length() * 0.25)
                        .max(1e-3);
                    2.0 * depth * (0.5 * fov_y).tan()
                }
                Projection::Ortho3D { view_height, .. } => view_height,
                Projection::Ortho2D { zoom, .. } => {
                    if zoom > 0.0 {
                        vh / zoom
                    } else {
                        1.0
                    }
                }
            };
            let mut world_scale = (world_span * frac).max(1e-5);
            if let Some(cap) = self.max_world_scale
                && cap.is_finite()
                && cap > 0.0
            {
                world_scale = world_scale.min(cap);
            }
            let u = ObjectUniforms {
                offset_rel: offset.to_f32(),
                scale: world_scale as f32,
            };
            ctx.queue
                .write_buffer(&binding.buf, 0, bytemuck::bytes_of(&u));
        }
        false
    }

    fn paint(&self, phase: DrawPhase, pass: &mut wgpu::RenderPass<'_>, ctx: &PaintCtx<'_, W>) {
        if self.origin.is_none() {
            return;
        }
        let Some(gpu) = &self.gpu else {
            return;
        };
        let Some(binding) = gpu.per_viewport.iter().find(|b| b.id == ctx.viewport_id) else {
            return;
        };
        match phase {
            DrawPhase::Gizmo => {
                pass.set_pipeline(&gpu.color_pipe);
                pass.set_bind_group(1, &binding.bg, &[]);
                pass.set_vertex_buffer(0, gpu.vertex_buf.slice(..));
                pass.draw(0..gpu.vertex_count, 0..1);
            }
            DrawPhase::Picking => {
                pass.set_pipeline(&gpu.pick_pipe);
                pass.set_bind_group(1, &binding.bg, &[]);
                pass.set_vertex_buffer(0, gpu.vertex_buf.slice(..));
                pass.set_vertex_buffer(1, gpu.pick_buf.slice(..));
                pass.draw(0..gpu.vertex_count, 0..1);
            }
            _ => {}
        }
    }
}
