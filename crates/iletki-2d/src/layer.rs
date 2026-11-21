//! 2D CAD/vector mesh builders and [`Scene2DLayer`] implementing [`iletki_wgpu::ViewportLayer`].

use crate::paged_buffer::{PagedBuffer, SlotHandle};
use crate::spatial::{TileRange, bucket_plain, bucket_wide};
use crate::vertex::{DrawStyle2D, Vertex2D, WideVertex2D, push_wide_segment};
use iletki_core::{DrawPhase, DrawPhaseMask, PickId, Vec2};
use iletki_wgpu::{
    CAMERA_WGSL, DEPTH_FORMAT, LayerRevision, PICK_ID_FORMAT, PaintCtx, PrepareCtx, ViewportLayer,
    wgpu,
};

/// CPU-side batch of compact split-precision 2D primitives ready to upload into a [`Scene2DLayer`].
///
/// Parallel picking ID arrays are allocated lazily only when a primitive is pushed with a non-empty
/// [`PickId`], so drawings that do not need GPU picking incur zero extra RAM/VRAM cost.
#[derive(Debug, Clone, Default)]
pub struct Mesh2D {
    /// 1 px line list (`2` vertices per segment, 24 bytes/vertex).
    pub lines: Vec<Vertex2D>,
    /// Solid fill triangle list (`3` vertices per triangle, 24 bytes/vertex).
    pub triangles: Vec<Vertex2D>,
    /// Screen-space wide line list (`6` vertices per segment, 52 bytes/vertex).
    pub wide: Vec<WideVertex2D>,
    /// Optional parallel 64-bit pick IDs for [`Mesh2D::lines`].
    pub line_pick_ids: Vec<[u32; 2]>,
    /// Optional parallel 64-bit pick IDs for [`Mesh2D::triangles`].
    pub triangle_pick_ids: Vec<[u32; 2]>,
    /// Optional parallel 64-bit pick IDs for [`Mesh2D::wide`].
    pub wide_pick_ids: Vec<[u32; 2]>,
    /// Optional spatial tile ranges for frustum culling of [`Mesh2D::lines`].
    pub line_tiles: Vec<TileRange>,
    /// Optional spatial tile ranges for frustum culling of [`Mesh2D::triangles`].
    pub triangle_tiles: Vec<TileRange>,
    /// Optional spatial tile ranges for frustum culling of [`Mesh2D::wide`].
    pub wide_tiles: Vec<TileRange>,
}

impl Mesh2D {
    /// Returns `true` if all primitive lists are empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty() && self.triangles.is_empty() && self.wide.is_empty()
    }

    /// Clears all vertices, pick IDs and spatial tiles while keeping allocated capacities.
    pub fn clear(&mut self) {
        self.lines.clear();
        self.triangles.clear();
        self.wide.clear();
        self.line_pick_ids.clear();
        self.triangle_pick_ids.clear();
        self.wide_pick_ids.clear();
        self.line_tiles.clear();
        self.triangle_tiles.clear();
        self.wide_tiles.clear();
    }

    /// Reorders large opaque primitive lists into an `8 × 8` spatial grid (+ global oversized
    /// bucket) so [`Scene2DLayer`] skips off-screen geometry when zoomed into large CAD drawings.
    ///
    /// Safe to call on a background worker thread before handing the mesh to the GPU thread.
    pub fn optimize_spatial(&mut self) {
        self.line_tiles = bucket_plain(&mut self.lines, &mut self.line_pick_ids, 2);
        self.triangle_tiles = bucket_plain(&mut self.triangles, &mut self.triangle_pick_ids, 3);
        self.wide_tiles = bucket_wide(&mut self.wide, &mut self.wide_pick_ids);
    }

    fn record_pick_ids(
        pick_ids: &mut Vec<[u32; 2]>,
        prev_vert_count: usize,
        added: usize,
        pick_id: PickId,
    ) {
        if pick_id.is_none() && pick_ids.is_empty() {
            return;
        }
        if pick_ids.len() < prev_vert_count {
            pick_ids.resize(prev_vert_count, [0, 0]);
        }
        let packed = pick_id.to_rg32u();
        for _ in 0..added {
            pick_ids.push(packed);
        }
    }

    /// Pushes a 1 px world-space line segment.
    pub fn push_line(&mut self, a: Vec2, b: Vec2, style: DrawStyle2D) {
        if !a.is_finite() || !b.is_finite() {
            return;
        }
        let prev = self.lines.len();
        self.lines
            .push(Vertex2D::new(a, style.color, style.linear, style.depth));
        self.lines
            .push(Vertex2D::new(b, style.color, style.linear, style.depth));
        Self::record_pick_ids(&mut self.line_pick_ids, prev, 2, style.pick_id);
    }

    /// Pushes a screen-space wide line segment (`width_pt` logical screen points).
    pub fn push_wide_line(&mut self, a: Vec2, b: Vec2, width_pt: f32, style: DrawStyle2D) {
        let prev = self.wide.len();
        if push_wide_segment(&mut self.wide, a, b, width_pt, style) {
            Self::record_pick_ids(&mut self.wide_pick_ids, prev, 6, style.pick_id);
        }
    }

    /// Pushes a filled triangle.
    pub fn push_triangle(&mut self, a: Vec2, b: Vec2, c: Vec2, style: DrawStyle2D) {
        if !a.is_finite() || !b.is_finite() || !c.is_finite() {
            return;
        }
        let prev = self.triangles.len();
        self.triangles
            .push(Vertex2D::new(a, style.color, style.linear, style.depth));
        self.triangles
            .push(Vertex2D::new(b, style.color, style.linear, style.depth));
        self.triangles
            .push(Vertex2D::new(c, style.color, style.linear, style.depth));
        Self::record_pick_ids(&mut self.triangle_pick_ids, prev, 3, style.pick_id);
    }

    /// Pushes an axis-aligned filled rectangle spanning `min..=max`.
    pub fn push_rect_fill(&mut self, min: Vec2, max: Vec2, style: DrawStyle2D) {
        let p00 = min;
        let p10 = Vec2::new(max.x, min.y);
        let p11 = max;
        let p01 = Vec2::new(min.x, max.y);
        self.push_triangle(p00, p10, p11, style);
        self.push_triangle(p00, p11, p01, style);
    }

    /// Pushes a 1 px axis-aligned rectangle outline spanning `min..=max`.
    pub fn push_rect_outline(&mut self, min: Vec2, max: Vec2, style: DrawStyle2D) {
        let p00 = min;
        let p10 = Vec2::new(max.x, min.y);
        let p11 = max;
        let p01 = Vec2::new(min.x, max.y);
        self.push_line(p00, p10, style);
        self.push_line(p10, p11, style);
        self.push_line(p11, p01, style);
        self.push_line(p01, p00, style);
    }
}

/// Handle to an incrementally managed 2D geometry chunk inside [`Scene2DLayer`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Chunk2DHandle {
    lines: Option<SlotHandle>,
    triangles: Option<SlotHandle>,
    wide: Option<SlotHandle>,
    line_picks: Option<SlotHandle>,
    triangle_picks: Option<SlotHandle>,
    wide_picks: Option<SlotHandle>,
}

struct Pipelines {
    format: wgpu::TextureFormat,
    msaa: u32,
    fill_color: wgpu::RenderPipeline,
    wide_color: wgpu::RenderPipeline,
    line_color: wgpu::RenderPipeline,
    fill_pick: wgpu::RenderPipeline,
    wide_pick: wgpu::RenderPipeline,
    line_pick: wgpu::RenderPipeline,
}

/// High-precision 2D CAD/vector [`ViewportLayer`] with compact 24-byte split `hi/lo` vertices,
/// zero-copy bulk uploads, incremental paged GPU buffers, separate dynamic overlays and optional
/// 64-bit GPU picking.
pub struct Scene2DLayer {
    layer_id: u16,
    revision: LayerRevision,
    pipelines: Option<Pipelines>,
    // Static scene paged pools
    triangles: PagedBuffer<Vertex2D>,
    wide: PagedBuffer<WideVertex2D>,
    lines: PagedBuffer<Vertex2D>,
    // Optional parallel pick-ID pools (empty unless PickIds are provided)
    triangle_picks: PagedBuffer<[u32; 2]>,
    wide_picks: PagedBuffer<[u32; 2]>,
    line_picks: PagedBuffer<[u32; 2]>,
    // Dynamic overlay pools (redrawn cheaply on top of static cache)
    overlay_triangles: PagedBuffer<Vertex2D>,
    overlay_wide: PagedBuffer<WideVertex2D>,
    overlay_lines: PagedBuffer<Vertex2D>,
    has_overlay: bool,
}

impl std::fmt::Debug for Scene2DLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scene2DLayer")
            .field("layer_id", &self.layer_id)
            .field("triangles", &self.triangles)
            .field("wide", &self.wide)
            .field("lines", &self.lines)
            .field("has_overlay", &self.has_overlay)
            .finish()
    }
}

impl Scene2DLayer {
    /// Creates a new empty 2D scene layer with identifier `layer_id`.
    #[must_use]
    pub fn new(layer_id: u16) -> Self {
        Self {
            layer_id,
            revision: LayerRevision::new(),
            pipelines: None,
            triangles: PagedBuffer::new("iletki-2d fills", 3),
            wide: PagedBuffer::new("iletki-2d wide", 6),
            lines: PagedBuffer::new("iletki-2d lines", 2),
            triangle_picks: PagedBuffer::new("iletki-2d fill picks", 3),
            wide_picks: PagedBuffer::new("iletki-2d wide picks", 6),
            line_picks: PagedBuffer::new("iletki-2d line picks", 2),
            overlay_triangles: PagedBuffer::new("iletki-2d overlay fills", 3),
            overlay_wide: PagedBuffer::new("iletki-2d overlay wide", 6),
            overlay_lines: PagedBuffer::new("iletki-2d overlay lines", 2),
            has_overlay: false,
        }
    }

    /// Zero-copy immediate upload of a static 2D scene directly onto `device`/`queue`, reusing
    /// existing GPU page buffers when capacity fits and attaching any spatial tiles computed by
    /// [`Mesh2D::optimize_spatial`].
    ///
    /// Every viewport using this layer refreshes its cached scene on the next frame.
    pub fn upload_static_mesh(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: &Mesh2D,
    ) {
        self.revision.mark_static();
        self.triangles
            .upload_direct_tiled(device, queue, &mesh.triangles, &mesh.triangle_tiles);
        self.wide
            .upload_direct_tiled(device, queue, &mesh.wide, &mesh.wide_tiles);
        self.lines
            .upload_direct_tiled(device, queue, &mesh.lines, &mesh.line_tiles);
        self.triangle_picks
            .upload_direct(device, queue, &mesh.triangle_pick_ids);
        self.wide_picks
            .upload_direct(device, queue, &mesh.wide_pick_ids);
        self.line_picks
            .upload_direct(device, queue, &mesh.line_pick_ids);
    }

    /// Replaces the entire static 2D scene with `mesh` (queued for `UploadBudget`-throttled upload
    /// during `prepare`).
    pub fn set_static_mesh(&mut self, mesh: &Mesh2D) {
        self.revision.mark_static();
        self.triangles.replace_all(&mesh.triangles);
        self.wide.replace_all(&mesh.wide);
        self.lines.replace_all(&mesh.lines);
        self.triangle_picks.replace_all(&mesh.triangle_pick_ids);
        self.wide_picks.replace_all(&mesh.wide_pick_ids);
        self.line_picks.replace_all(&mesh.line_pick_ids);
    }

    /// Inserts an incremental static geometry chunk and returns its [`Chunk2DHandle`].
    pub fn insert_chunk(&mut self, mesh: &Mesh2D) -> Chunk2DHandle {
        self.revision.mark_static();
        Chunk2DHandle {
            lines: self.lines.insert(&mesh.lines),
            triangles: self.triangles.insert(&mesh.triangles),
            wide: self.wide.insert(&mesh.wide),
            line_picks: self.line_picks.insert(&mesh.line_pick_ids),
            triangle_picks: self.triangle_picks.insert(&mesh.triangle_pick_ids),
            wide_picks: self.wide_picks.insert(&mesh.wide_pick_ids),
        }
    }

    /// Updates an existing incremental chunk in place (or reallocates individual pools if grown).
    pub fn update_chunk(&mut self, handle: Chunk2DHandle, mesh: &Mesh2D) -> Chunk2DHandle {
        self.revision.mark_static();
        let lines = match handle.lines {
            Some(s) => self.lines.update(s, &mesh.lines),
            None => self.lines.insert(&mesh.lines),
        };
        let triangles = match handle.triangles {
            Some(s) => self.triangles.update(s, &mesh.triangles),
            None => self.triangles.insert(&mesh.triangles),
        };
        let wide = match handle.wide {
            Some(s) => self.wide.update(s, &mesh.wide),
            None => self.wide.insert(&mesh.wide),
        };
        let line_picks = match handle.line_picks {
            Some(s) => self.line_picks.update(s, &mesh.line_pick_ids),
            None => self.line_picks.insert(&mesh.line_pick_ids),
        };
        let triangle_picks = match handle.triangle_picks {
            Some(s) => self.triangle_picks.update(s, &mesh.triangle_pick_ids),
            None => self.triangle_picks.insert(&mesh.triangle_pick_ids),
        };
        let wide_picks = match handle.wide_picks {
            Some(s) => self.wide_picks.update(s, &mesh.wide_pick_ids),
            None => self.wide_picks.insert(&mesh.wide_pick_ids),
        };
        Chunk2DHandle {
            lines,
            triangles,
            wide,
            line_picks,
            triangle_picks,
            wide_picks,
        }
    }

    /// Removes an incremental chunk from the static scene without rebuilding other pages.
    pub fn remove_chunk(&mut self, handle: Chunk2DHandle) {
        self.revision.mark_static();
        if let Some(s) = handle.lines {
            self.lines.remove(s);
        }
        if let Some(s) = handle.triangles {
            self.triangles.remove(s);
        }
        if let Some(s) = handle.wide {
            self.wide.remove(s);
        }
        if let Some(s) = handle.line_picks {
            self.line_picks.remove(s);
        }
        if let Some(s) = handle.triangle_picks {
            self.triangle_picks.remove(s);
        }
        if let Some(s) = handle.wide_picks {
            self.wide_picks.remove(s);
        }
    }

    /// Zero-copy immediate upload of dynamic 2D overlay geometry.
    pub fn upload_overlay(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, mesh: &Mesh2D) {
        self.revision.mark_dynamic();
        self.overlay_triangles
            .upload_direct(device, queue, &mesh.triangles);
        self.overlay_wide.upload_direct(device, queue, &mesh.wide);
        self.overlay_lines.upload_direct(device, queue, &mesh.lines);
        self.has_overlay = !mesh.is_empty();
    }

    /// Replaces the dynamic 2D overlay geometry (crosshair, snaps, selection box, highlights).
    pub fn set_overlay(&mut self, mesh: &Mesh2D) {
        self.revision.mark_dynamic();
        self.overlay_triangles.replace_all(&mesh.triangles);
        self.overlay_wide.replace_all(&mesh.wide);
        self.overlay_lines.replace_all(&mesh.lines);
        self.has_overlay = !mesh.is_empty();
    }

    /// Clears all dynamic overlay geometry.
    pub fn clear_overlay(&mut self) {
        self.revision.mark_dynamic();
        self.overlay_triangles.clear();
        self.overlay_wide.clear();
        self.overlay_lines.clear();
        self.has_overlay = false;
    }

    fn ensure_pipelines<W>(&mut self, ctx: &PrepareCtx<'_, W>) {
        if self
            .pipelines
            .as_ref()
            .is_some_and(|p| p.format == ctx.color_format && p.msaa == ctx.msaa_samples)
        {
            return;
        }
        let wgsl = format!("{CAMERA_WGSL}\n{}", include_str!("shaders/cad2d.wgsl"));
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("iletki-2d shader"),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });
        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("iletki-2d layout"),
                bind_group_layouts: &[Some(ctx.camera_bind_group_layout)],
                immediate_size: 0,
            });
        let plain_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex2D>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![
                0 => Float32x2,
                1 => Float32x2,
                2 => Unorm8x4,
                3 => Float32
            ],
        };
        let wide_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<WideVertex2D>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![
                0 => Float32x2,
                1 => Float32x2,
                2 => Float32x2,
                3 => Float32x2,
                4 => Float32x2,
                5 => Float32,
                6 => Unorm8x4,
                7 => Float32
            ],
        };
        let pick_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<[u32; 2]>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![8 => Uint32x2],
        };

        let make = |label: &str,
                    topology: wgpu::PrimitiveTopology,
                    vs: &str,
                    fs: &str,
                    buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
                    format: wgpu::TextureFormat,
                    blend: Option<wgpu::BlendState>,
                    msaa: u32| {
            ctx.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some(vs),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers,
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some(fs),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format,
                            blend,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology,
                        ..wgpu::PrimitiveState::default()
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: DEPTH_FORMAT,
                        depth_write_enabled: Some(true),
                        depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState {
                        count: msaa,
                        ..wgpu::MultisampleState::default()
                    },
                    multiview_mask: None,
                    cache: None,
                })
        };

        let alpha = Some(wgpu::BlendState::ALPHA_BLENDING);
        let plain_only = [Some(plain_layout.clone())];
        let wide_only = [Some(wide_layout.clone())];
        let plain_and_pick = [Some(plain_layout), Some(pick_layout.clone())];
        let wide_and_pick = [Some(wide_layout), Some(pick_layout)];

        self.pipelines = Some(Pipelines {
            format: ctx.color_format,
            msaa: ctx.msaa_samples,
            fill_color: make(
                "iletki-2d fill",
                wgpu::PrimitiveTopology::TriangleList,
                "vs_plain",
                "fs_color",
                &plain_only,
                ctx.color_format,
                alpha,
                ctx.msaa_samples,
            ),
            wide_color: make(
                "iletki-2d wide",
                wgpu::PrimitiveTopology::TriangleList,
                "vs_wide",
                "fs_color",
                &wide_only,
                ctx.color_format,
                alpha,
                ctx.msaa_samples,
            ),
            line_color: make(
                "iletki-2d line",
                wgpu::PrimitiveTopology::LineList,
                "vs_plain",
                "fs_color",
                &plain_only,
                ctx.color_format,
                alpha,
                ctx.msaa_samples,
            ),
            fill_pick: make(
                "iletki-2d fill pick",
                wgpu::PrimitiveTopology::TriangleList,
                "vs_plain_pick",
                "fs_pick",
                &plain_and_pick,
                PICK_ID_FORMAT,
                None,
                1,
            ),
            wide_pick: make(
                "iletki-2d wide pick",
                wgpu::PrimitiveTopology::TriangleList,
                "vs_wide_pick",
                "fs_pick",
                &wide_and_pick,
                PICK_ID_FORMAT,
                None,
                1,
            ),
            line_pick: make(
                "iletki-2d line pick",
                wgpu::PrimitiveTopology::LineList,
                "vs_plain_pick",
                "fs_pick",
                &plain_and_pick,
                PICK_ID_FORMAT,
                None,
                1,
            ),
        });
    }
}

impl<W> ViewportLayer<W> for Scene2DLayer {
    fn id(&self) -> u16 {
        self.layer_id
    }

    fn phases(&self) -> DrawPhaseMask {
        let mut mask = DrawPhaseMask::OPAQUE;
        if !self.triangle_picks.is_empty()
            || !self.wide_picks.is_empty()
            || !self.line_picks.is_empty()
        {
            mask = mask.union(DrawPhaseMask::PICKING);
        }
        if self.has_overlay {
            mask = mask.union(DrawPhaseMask::OVERLAY);
        }
        mask
    }

    fn revision(&self) -> LayerRevision {
        self.revision
    }

    fn prepare(&mut self, ctx: &mut PrepareCtx<'_, W>) -> bool {
        self.ensure_pipelines(ctx);
        if self.triangles.has_pending_uploads()
            || self.wide.has_pending_uploads()
            || self.lines.has_pending_uploads()
        {
            self.revision.mark_static();
        }
        let mut pending = false;
        pending |= self.triangles.flush(ctx.device, ctx.queue, ctx.budget);
        pending |= self.wide.flush(ctx.device, ctx.queue, ctx.budget);
        pending |= self.lines.flush(ctx.device, ctx.queue, ctx.budget);
        pending |= self.triangle_picks.flush(ctx.device, ctx.queue, ctx.budget);
        pending |= self.wide_picks.flush(ctx.device, ctx.queue, ctx.budget);
        pending |= self.line_picks.flush(ctx.device, ctx.queue, ctx.budget);
        let _ = self.overlay_triangles.flush(
            ctx.device,
            ctx.queue,
            iletki_core::UploadBudget::UNLIMITED,
        );
        let _ =
            self.overlay_wide
                .flush(ctx.device, ctx.queue, iletki_core::UploadBudget::UNLIMITED);
        let _ =
            self.overlay_lines
                .flush(ctx.device, ctx.queue, iletki_core::UploadBudget::UNLIMITED);
        pending
    }

    fn paint(&self, phase: DrawPhase, pass: &mut wgpu::RenderPass<'_>, ctx: &PaintCtx<'_, W>) {
        let Some(p) = &self.pipelines else {
            return;
        };
        let visible = ctx.camera.visible_2d(ctx.logical_size);
        match phase {
            DrawPhase::Opaque => {
                if !self.triangles.is_empty() {
                    pass.set_pipeline(&p.fill_color);
                    self.triangles.draw_culled(pass, visible);
                }
                if !self.wide.is_empty() {
                    pass.set_pipeline(&p.wide_color);
                    self.wide.draw_culled(pass, visible);
                }
                if !self.lines.is_empty() {
                    pass.set_pipeline(&p.line_color);
                    self.lines.draw_culled(pass, visible);
                }
            }
            DrawPhase::Overlay => {
                if !self.overlay_triangles.is_empty() {
                    pass.set_pipeline(&p.fill_color);
                    self.overlay_triangles.draw_all(pass);
                }
                if !self.overlay_wide.is_empty() {
                    pass.set_pipeline(&p.wide_color);
                    self.overlay_wide.draw_all(pass);
                }
                if !self.overlay_lines.is_empty() {
                    pass.set_pipeline(&p.line_color);
                    self.overlay_lines.draw_all(pass);
                }
            }
            DrawPhase::Picking => {
                if !self.triangle_picks.is_empty() {
                    pass.set_pipeline(&p.fill_pick);
                    self.triangles
                        .draw_all_with_pick(&self.triangle_picks, pass);
                }
                if !self.wide_picks.is_empty() {
                    pass.set_pipeline(&p.wide_pick);
                    self.wide.draw_all_with_pick(&self.wide_picks, pass);
                }
                if !self.line_picks.is_empty() {
                    pass.set_pipeline(&p.line_pick);
                    self.lines.draw_all_with_pick(&self.line_picks, pass);
                }
            }
            _ => {}
        }
    }
}
