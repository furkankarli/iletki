//! GPU-instanced 3D layer (`InstancedMesh3DLayer`) for drawing 100,000+ repeated CAD blocks,
//! structural columns, bolts, trees or markers in a single draw call with per-instance 64-bit GPU
//! picking and multi-viewport support.

use crate::section::SectionAa;
use crate::vertex::{Mesh3D, Vertex3D};
use bytemuck::{Pod, Zeroable};
use iletki_core::{DrawPhase, DrawPhaseMask, PickId, Vec3, ViewportId};
use iletki_wgpu::wgpu;
use iletki_wgpu::wgpu::util::DeviceExt;
use iletki_wgpu::{
    CAMERA_WGSL, DEPTH_FORMAT, LayerRevision, PICK_ID_FORMAT, PaintCtx, PrepareCtx, ViewportLayer,
    ViewportLifetime,
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BatchUniforms {
    offset_rel: [f32; 3],
    _pad: f32,
}

/// One GPU instance payload (36 bytes): local offset relative to the batch's `f64` world origin,
/// scale, RGBA color and 64-bit [`PickId`].
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable, PartialEq)]
pub struct Instance3D {
    /// Local offset from [`InstancedMesh3DLayer::origin`].
    pub offset: [f32; 3],
    /// Non-uniform scale `(sx, sy, sz)`.
    pub scale: [f32; 3],
    /// RGBA color bytes.
    pub color: [u8; 4],
    /// Packed 64-bit [`PickId`].
    pub pick_id: [u32; 2],
}

impl Instance3D {
    /// Creates an instance from a local offset relative to the batch origin.
    #[must_use]
    pub fn new(offset: [f32; 3], scale: [f32; 3], color: [u8; 4], pick_id: PickId) -> Self {
        Self {
            offset,
            scale,
            color,
            pick_id: pick_id.to_rg32u(),
        }
    }
}

struct InstPipelines {
    format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
    msaa: u32,
    batch_layout: wgpu::BindGroupLayout,
    lit: wgpu::RenderPipeline,
    section_lit: Option<wgpu::RenderPipeline>,
    pick: wgpu::RenderPipeline,
}

struct ViewportBinding {
    id: ViewportId,
    lifetime: ViewportLifetime,
    buf: wgpu::Buffer,
    bg: wgpu::BindGroup,
}

/// Draws a single prototype [`Mesh3D`] repeated across arbitrarily many [`Instance3D`]s in **one
/// GPU draw call**, supporting `f64` world origin, 6 section clip planes, per-instance 64-bit GPU
/// picking and simultaneous multi-viewport sharing.
pub struct InstancedMesh3DLayer {
    layer_id: u16,
    revision: LayerRevision,
    prepared_origin: Option<Vec3>,
    /// Shared `f64` world origin of all instances in this batch.
    pub origin: Vec3,
    prototype: Option<Mesh3D>,
    pending_instances: Option<Vec<Instance3D>>,
    vertex_buf: Option<wgpu::Buffer>,
    vertex_count: u32,
    instance_buf: Option<wgpu::Buffer>,
    instance_count: u32,
    pipelines: Option<InstPipelines>,
    use_antialiased_section: bool,
    per_viewport: Vec<ViewportBinding>,
}

impl std::fmt::Debug for InstancedMesh3DLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InstancedMesh3DLayer")
            .field("layer_id", &self.layer_id)
            .field("origin", &self.origin)
            .field("vertex_count", &self.vertex_count)
            .field("instance_count", &self.instance_count)
            .finish()
    }
}

impl InstancedMesh3DLayer {
    /// Creates a new instanced 3D layer anchored at `origin` with prototype geometry `prototype`.
    #[must_use]
    pub fn new(layer_id: u16, origin: Vec3, prototype: Mesh3D) -> Self {
        Self {
            layer_id,
            revision: LayerRevision::new(),
            prepared_origin: None,
            origin: if origin.is_finite() {
                origin
            } else {
                Vec3::ZERO
            },
            prototype: Some(prototype),
            pending_instances: None,
            vertex_buf: None,
            vertex_count: 0,
            instance_buf: None,
            instance_count: 0,
            pipelines: None,
            use_antialiased_section: false,
            per_viewport: Vec::new(),
        }
    }

    /// Replaces the instance list (uploaded on the next `prepare`).
    pub fn set_instances(&mut self, instances: Vec<Instance3D>) {
        self.pending_instances = Some(instances);
        self.revision.mark_static();
    }

    /// Returns the number of uploaded instances.
    #[must_use]
    pub const fn instance_count(&self) -> u32 {
        self.instance_count
    }

    fn ensure_pipelines<W>(&mut self, ctx: &PrepareCtx<'_, W>) {
        let section_aa = SectionAa::for_viewport(ctx);
        if self.pipelines.as_ref().is_some_and(|p| {
            p.format == ctx.color_format
                && p.depth_format == ctx.depth_format
                && p.msaa == ctx.msaa_samples
                && (section_aa.is_none() || p.section_lit.is_some())
        }) {
            return;
        }
        let wgsl = format!(
            "{CAMERA_WGSL}\n{}",
            include_str!("shaders/instanced3d.wgsl")
        );
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("iletki-3d instanced shader"),
                source: wgpu::ShaderSource::Wgsl(wgsl.as_str().into()),
            });
        let batch_layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("iletki-3d instanced batch layout"),
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
                label: Some("iletki-3d instanced pipeline layout"),
                bind_group_layouts: &[Some(ctx.camera_bind_group_layout), Some(&batch_layout)],
                immediate_size: 0,
            });
        let vert_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex3D>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
        };
        let inst_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Instance3D>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![
                2 => Float32x3,
                3 => Float32x3,
                4 => Unorm8x4,
                5 => Uint32x2
            ],
        };

        let make_lit = |shader: &wgpu::ShaderModule, vertex, fragment| {
            ctx.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("iletki-3d instanced lit"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: shader,
                        entry_point: Some(vertex),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[Some(vert_layout.clone()), Some(inst_layout.clone())],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: shader,
                        entry_point: Some(fragment),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: ctx.color_format,
                            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        cull_mode: None,
                        ..wgpu::PrimitiveState::default()
                    },
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
        let lit = make_lit(&shader, "vs_inst_color", "fs_inst_lit");

        let section_lit = section_aa.map(|aa| {
            let shader = aa.shader(
                ctx.device,
                &wgsl,
                include_str!("shaders/instanced3d_sampled.wgsl"),
                include_str!("shaders/instanced3d_raster.wgsl"),
            );
            match aa {
                SectionAa::Sampled => make_lit(&shader, "vs_inst_sampled", "fs_inst_sampled"),
                SectionAa::Raster => make_lit(&shader, "vs_inst_raster", "fs_inst_raster"),
            }
        });

        let pick = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("iletki-3d instanced pick"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_inst_pick"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Some(vert_layout), Some(inst_layout)],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_inst_pick"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: PICK_ID_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..wgpu::PrimitiveState::default()
                },
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

        self.per_viewport.clear();
        self.pipelines = Some(InstPipelines {
            format: ctx.color_format,
            depth_format: ctx.depth_format,
            msaa: ctx.msaa_samples,
            batch_layout,
            lit,
            section_lit,
            pick,
        });
    }
}

impl<W> ViewportLayer<W> for InstancedMesh3DLayer {
    fn id(&self) -> u16 {
        self.layer_id
    }

    fn phases(&self) -> DrawPhaseMask {
        if self.instance_count > 0
            || self
                .pending_instances
                .as_ref()
                .is_some_and(|v| !v.is_empty())
        {
            DrawPhaseMask::OPAQUE.union(DrawPhaseMask::PICKING)
        } else {
            DrawPhaseMask::NONE
        }
    }

    fn revision(&self) -> LayerRevision {
        self.revision
    }

    fn prepare(&mut self, ctx: &mut PrepareCtx<'_, W>) -> bool {
        self.per_viewport
            .retain(|binding| binding.lifetime.is_alive());
        if self.prepared_origin != Some(self.origin) {
            self.prepared_origin = Some(self.origin);
            self.revision.mark_static();
        }
        self.ensure_pipelines(ctx);
        self.use_antialiased_section =
            ctx.supports_section_antialiasing() && !ctx.camera.clip_planes().is_empty();
        let Some(p) = &self.pipelines else {
            return false;
        };

        if let Some(proto) = &self.prototype
            && !ctx.uploads.reserve(size_of_val(proto.vertices.as_slice()))
        {
            return true;
        }
        if let Some(proto) = self.prototype.take() {
            self.revision.mark_static();
            self.vertex_count = u32::try_from(proto.vertices.len()).unwrap_or(0);
            if self.vertex_count > 0 {
                self.vertex_buf = Some(ctx.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("iletki-3d instanced proto vertices"),
                        contents: bytemuck::cast_slice(&proto.vertices),
                        usage: wgpu::BufferUsages::VERTEX,
                    },
                ));
            }
            ctx.uploads.finish_upload();
        }

        let pending = self
            .pending_instances
            .as_ref()
            .is_some_and(|instances| !ctx.uploads.reserve(size_of_val(instances.as_slice())));
        if !pending && let Some(insts) = self.pending_instances.take() {
            self.revision.mark_static();
            self.instance_count = u32::try_from(insts.len()).unwrap_or(0);
            if self.instance_count > 0 {
                self.instance_buf = Some(ctx.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("iletki-3d instances"),
                        contents: bytemuck::cast_slice(&insts),
                        usage: wgpu::BufferUsages::VERTEX,
                    },
                ));
            } else {
                self.instance_buf = None;
            }
            ctx.uploads.finish_upload();
        }

        // Per-ViewportId uniform buffer so 4-way quad views never clobber each other!
        let vp_id = ctx.viewport_id;
        if !self.per_viewport.iter().any(|b| b.id == vp_id) {
            let buf = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("iletki-3d instanced batch uniforms"),
                    contents: bytemuck::bytes_of(&BatchUniforms::zeroed()),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("iletki-3d instanced batch bg"),
                layout: &p.batch_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            });
            self.per_viewport.push(ViewportBinding {
                id: vp_id,
                lifetime: ctx.viewport_lifetime(),
                buf,
                bg,
            });
        }

        if let Some(binding) = self.per_viewport.iter().find(|b| b.id == vp_id) {
            let offset_rel = (self.origin - ctx.camera.position_3d()).to_f32();
            let u = BatchUniforms {
                offset_rel,
                _pad: 0.0,
            };
            ctx.queue
                .write_buffer(&binding.buf, 0, bytemuck::bytes_of(&u));
        }
        pending
    }

    fn paint(&self, phase: DrawPhase, pass: &mut wgpu::RenderPass<'_>, ctx: &PaintCtx<'_, W>) {
        if self.vertex_count == 0 || self.instance_count == 0 {
            return;
        }
        let (Some(p), Some(vbuf), Some(ibuf)) =
            (&self.pipelines, &self.vertex_buf, &self.instance_buf)
        else {
            return;
        };
        let Some(binding) = self.per_viewport.iter().find(|b| b.id == ctx.viewport_id) else {
            return;
        };
        match phase {
            DrawPhase::Opaque => {
                let lit = p
                    .section_lit
                    .as_ref()
                    .filter(|_| self.use_antialiased_section)
                    .unwrap_or(&p.lit);
                pass.set_pipeline(lit);
                pass.set_bind_group(1, &binding.bg, &[]);
                pass.set_vertex_buffer(0, vbuf.slice(..));
                pass.set_vertex_buffer(1, ibuf.slice(..));
                pass.draw(0..self.vertex_count, 0..self.instance_count);
            }
            DrawPhase::Picking => {
                pass.set_pipeline(&p.pick);
                pass.set_bind_group(1, &binding.bg, &[]);
                pass.set_vertex_buffer(0, vbuf.slice(..));
                pass.set_vertex_buffer(1, ibuf.slice(..));
                pass.draw(0..self.vertex_count, 0..self.instance_count);
            }
            _ => {}
        }
    }
}
