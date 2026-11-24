//! GPU-driven frustum, section-plane and conservative two-phase Reversed-Z Hi-Z culling.
//!
//! Capability-gated to native/WebGPU backends with [`wgpu::Features::MULTI_DRAW_INDIRECT_COUNT`],
//! compute shaders and vertex-stage storage buffers. Scenes made solely of repeated instanced
//! pages (or small scenes below [`MIN_GPU_CULL_CHUNKS`]) stay on the retained `StorageDraws` path.

use crate::face_batch::{GeometryPage, PageKind, PaintPipelines, VertexRange};
use crate::objects::FaceRanges;
use crate::storage_batch::{self, ChunkTransformGpu, ChunkTransformTable, ViewCameraOrigin};
use bytemuck::{Pod, Zeroable};
use iletki_core::{Bounds3, Camera, Frustum, MAX_CLIP_PLANES, Vec3};
use iletki_wgpu::wgpu;
use iletki_wgpu::wgpu::util::DeviceExt;
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::Arc;

const MIN_GPU_CULL_CHUNKS: usize = 1024;

const FLAG_TRUSTED: u32 = 1 << 0;
const FLAG_TRANSLUCENT: u32 = 1 << 1;
const FLAG_FACE_INDEXED: u32 = 1 << 2;
const FLAG_HAS_FACES: u32 = 1 << 3;
const FLAG_HAS_EDGES: u32 = 1 << 4;
const FLAG_EDGE_INDEXED: u32 = 1 << 5;

const DRAW_INDEXED_STRIDE: u64 = 20;
const DRAW_NON_INDEXED_STRIDE: u64 = 16;

const PHASE1_SUFFIX: &str = r#"
@compute @workgroup_size(64, 1, 1)
fn cull_phase1(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.item_count) {
        return;
    }
    let item = items[idx];
    let t = chunk_transforms[item.chunk_index];
    if ((t.flags & 1u) == 0u) {
        vis_state[idx] = 0u;
        return;
    }
    let offset_rel = sub_q32_32(t.whole, t.frac, u.cam_whole, u.cam_frac);
    let trusted = (item.flags & FLAG_TRUSTED) != 0u;
    if (trusted) {
        let bmin = item.local_min + offset_rel;
        let bmax = item.local_max + offset_rel;
        let mag_xyz = max(abs(bmin), abs(bmax));
        for (var p: u32 = 0u; p < u.frustum_count; p = p + 1u) {
            let pl = u.frustum[p];
            let far = vec3<f32>(
                select(bmin.x, bmax.x, pl.x >= 0.0),
                select(bmin.y, bmax.y, pl.y >= 0.0),
                select(bmin.z, bmax.z, pl.z >= 0.0),
            );
            let mag = abs(pl.x) * mag_xyz.x + abs(pl.y) * mag_xyz.y + abs(pl.z) * mag_xyz.z + abs(pl.w);
            let err = mag * F32_ERR_SCALE + 1e-6;
            if (dot(pl.xyz, far) + pl.w + err < 0.0) {
                vis_state[idx] = 0u;
                return;
            }
        }
        for (var p: u32 = 0u; p < u.clip_plane_count; p = p + 1u) {
            let pl = u.clip_planes[p];
            let far = vec3<f32>(
                select(bmin.x, bmax.x, pl.x >= 0.0),
                select(bmin.y, bmax.y, pl.y >= 0.0),
                select(bmin.z, bmax.z, pl.z >= 0.0),
            );
            let mag = abs(pl.x) * mag_xyz.x + abs(pl.y) * mag_xyz.y + abs(pl.z) * mag_xyz.z + abs(pl.w);
            let err = (mag + SECTION_TOLERANCE) * F32_ERR_SCALE;
            if (dot(pl.xyz, far) + pl.w + err < -SECTION_TOLERANCE) {
                vis_state[idx] = 0u;
                return;
            }
        }
    }

    let prev_vis = (vis_state[idx] & VIS_PREV) != 0u;
    let translucent = (item.flags & FLAG_TRANSLUCENT) != 0u;
    let has_faces = (item.flags & FLAG_HAS_FACES) != 0u;
    if (prev_vis && !translucent && has_faces) {
        emit_draw(
            item.face_bucket,
            item.face_indirect_base,
            item.face_first,
            item.face_count,
            (item.flags & FLAG_FACE_INDEXED) != 0u,
        );
        vis_state[idx] = VIS_PREV | VIS_DRAWN_P1 | VIS_PASSED_FS;
    } else {
        vis_state[idx] = VIS_PASSED_FS;
    }
}
"#;

const PHASE2_SUFFIX: &str = r#"
@group(1) @binding(0)
var hiz_tex: texture_2d<f32>;

@compute @workgroup_size(64, 1, 1)
fn cull_phase2(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.item_count) {
        return;
    }
    let state = vis_state[idx];
    if ((state & VIS_PASSED_FS) == 0u) {
        return;
    }
    let item = items[idx];
    let has_faces = (item.flags & FLAG_HAS_FACES) != 0u;
    let has_edges = (item.flags & FLAG_HAS_EDGES) != 0u;
    let translucent = (item.flags & FLAG_TRANSLUCENT) != 0u;
    let trusted = (item.flags & FLAG_TRUSTED) != 0u;

    var face_vis = true;
    var edge_vis = true;
    if (trusted) {
        let t = chunk_transforms[item.chunk_index];
        let offset_rel = sub_q32_32(t.whole, t.frac, u.cam_whole, u.cam_frac);
        let bmin = item.local_min + offset_rel;
        let bmax = item.local_max + offset_rel;

        var near_or_behind = false;
        var min_x = 1e30;
        var max_x = -1e30;
        var min_y = 1e30;
        var max_y = -1e30;
        var max_face_z = -1e30;
        var max_edge_z = -1e30;

        for (var c: u32 = 0u; c < 8u; c = c + 1u) {
            let corner = vec3<f32>(
                select(bmin.x, bmax.x, (c & 1u) != 0u),
                select(bmin.y, bmax.y, (c & 2u) != 0u),
                select(bmin.z, bmax.z, (c & 4u) != 0u),
            );
            let clip = u.view_proj * vec4<f32>(corner, 1.0);
            if (clip.w <= 1e-6 || clip.z > clip.w) {
                near_or_behind = true;
                break;
            }
            let inv_w = 1.0 / clip.w;
            let nx = clip.x * inv_w;
            let ny = clip.y * inv_w;
            let nz = clip.z * inv_w;
            let edge_nz = min((clip.z * 1.0008 + 1e-5) * inv_w, 1.0);
            min_x = min(min_x, nx);
            max_x = max(max_x, nx);
            min_y = min(min_y, ny);
            max_y = max(max_y, ny);
            max_face_z = max(max_face_z, nz);
            max_edge_z = max(max_edge_z, edge_nz);
        }

        if (!near_or_behind) {
            if (max_x < -1.0 || min_x > 1.0 || max_y < -1.0 || min_y > 1.0 || max_edge_z < 0.0) {
                face_vis = false;
                edge_vis = false;
            } else {
                let aw = u.active_w;
                let ah = u.active_h;
                let u0 = clamp(min_x * 0.5 + 0.5, 0.0, 1.0);
                let u1 = clamp(max_x * 0.5 + 0.5, 0.0, 1.0);
                let v0 = clamp(0.5 - max_y * 0.5, 0.0, 1.0);
                let v1 = clamp(0.5 - min_y * 0.5, 0.0, 1.0);
                let px0 = u32(max(floor(u0 * f32(aw)) - 1.0, 0.0));
                let py0 = u32(max(floor(v0 * f32(ah)) - 1.0, 0.0));
                let px1 = min(u32(ceil(u1 * f32(aw))) + 1u, aw - 1u);
                let py1 = min(u32(ceil(v1 * f32(ah))) + 1u, ah - 1u);

                let max_mip = max(textureNumLevels(hiz_tex), 1u) - 1u;
                var mip: u32 = 0u;
                loop {
                    if (mip >= max_mip) {
                        break;
                    }
                    let shift = mip + 1u;
                    if (((px1 >> shift) - (px0 >> shift)) <= 1u && ((py1 >> shift) - (py0 >> shift)) <= 1u) {
                        break;
                    }
                    mip = mip + 1u;
                }
                let shift = mip + 1u;
                let tx0 = i32(px0 >> shift);
                let tx1 = i32(px1 >> shift);
                let ty0 = i32(py0 >> shift);
                let ty1 = i32(py1 >> shift);
                let d00 = textureLoad(hiz_tex, vec2<i32>(tx0, ty0), i32(mip)).r;
                let d10 = textureLoad(hiz_tex, vec2<i32>(tx1, ty0), i32(mip)).r;
                let d01 = textureLoad(hiz_tex, vec2<i32>(tx0, ty1), i32(mip)).r;
                let d11 = textureLoad(hiz_tex, vec2<i32>(tx1, ty1), i32(mip)).r;
                let hiz_min = min(min(d00, d10), min(d01, d11));

                let face_z = max(max_face_z, 0.0) * 1.0002 + 1e-6;
                let edge_z = max(max_edge_z, 0.0) * 1.0002 + 1e-6;
                face_vis = face_z >= hiz_min;
                edge_vis = edge_z >= hiz_min;
            }
        }
    }

    vis_state[idx] = select(0u, VIS_PREV, face_vis && !translucent);

    if (has_faces && face_vis && (state & VIS_DRAWN_P1) == 0u) {
        emit_draw(
            item.face_bucket,
            item.face_indirect_base,
            item.face_first,
            item.face_count,
            (item.flags & FLAG_FACE_INDEXED) != 0u,
        );
    }
    if (has_edges && edge_vis) {
        emit_draw(
            item.edge_bucket,
            item.edge_indirect_base,
            item.edge_first,
            item.edge_count,
            (item.flags & FLAG_EDGE_INDEXED) != 0u,
        );
    }
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct CullViewUniforms {
    pub(crate) view_proj: [[f32; 4]; 4],
    pub(crate) frustum: [[f32; 4]; 6],
    pub(crate) clip_planes: [[f32; 4]; MAX_CLIP_PLANES],
    pub(crate) cam_whole: [i32; 3],
    pub(crate) frustum_count: u32,
    pub(crate) cam_frac: [u32; 3],
    pub(crate) clip_plane_count: u32,
    pub(crate) item_count: u32,
    pub(crate) active_w: u32,
    pub(crate) active_h: u32,
    pub(crate) _pad0: u32,
}

impl CullViewUniforms {
    pub(crate) fn build(
        camera: &Camera,
        logical_size: iletki_core::Vec2,
        dpi_scale: f32,
        active_px: [u32; 2],
        anchor: Vec3,
        cull_sections: bool,
        item_count: u32,
    ) -> Self {
        let cam_data = camera.to_uniform_data(logical_size, dpi_scale);
        let cam_pos = camera.position_3d();
        let origin = ViewCameraOrigin::from_vec3(cam_pos - anchor);

        let projection = camera.view_proj_relative(logical_size);
        let raw_frustum = Frustum::from_view_proj_reverse_z(projection);
        let mut frustum = [[0.0_f32; 4]; 6];
        let mut frustum_count = 0u32;
        for plane in raw_frustum.planes.into_iter().flatten() {
            let nx = plane.normal.x as f32;
            let ny = plane.normal.y as f32;
            let nz = plane.normal.z as f32;
            let d = plane.d as f32;
            if [nx, ny, nz, d].into_iter().all(f32::is_finite)
                && let Some(dst) = frustum.get_mut(frustum_count as usize)
            {
                *dst = [nx, ny, nz, d];
                frustum_count += 1;
            }
        }

        let mut clip_planes = [[0.0_f32; 4]; MAX_CLIP_PLANES];
        let mut clip_plane_count = 0u32;
        if cull_sections {
            for plane in camera.clip_planes() {
                let coeff = plane.to_camera_relative_f32(cam_pos);
                let [x, y, z, d] = coeff;
                if d.is_finite()
                    && [x, y, z]
                        .into_iter()
                        .all(|n| n == 0.0 || (n.is_normal() && n.abs() <= 1.0))
                    && let Some(dst) = clip_planes.get_mut(clip_plane_count as usize)
                {
                    *dst = coeff;
                    clip_plane_count += 1;
                }
            }
        }

        Self {
            view_proj: cam_data.view_proj_rel,
            frustum,
            clip_planes,
            cam_whole: origin.cam_whole,
            frustum_count,
            cam_frac: origin.cam_frac,
            clip_plane_count,
            item_count,
            active_w: active_px[0].max(1),
            active_h: active_px[1].max(1),
            _pad0: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct ChunkCullItem {
    local_min: [f32; 3],
    chunk_index: u32,
    local_max: [f32; 3],
    flags: u32,
    face_bucket: u32,
    face_first: u32,
    face_count: u32,
    face_indirect_base: u32,
    edge_bucket: u32,
    edge_first: u32,
    edge_count: u32,
    edge_indirect_base: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum BucketKind {
    OpaqueFace,
    TranslucentFace,
    Edge,
}

pub(crate) struct GpuCullBucket {
    pub(crate) page: Arc<GeometryPage>,
    pub(crate) kind: BucketKind,
    pub(crate) indexed: bool,
    pub(crate) flat: bool,
    pub(crate) indirect_base: u32,
    pub(crate) max_draws: u32,
}

/// Returns whether `device` supports compute GPU culling + `multi_draw_*_indirect_count`.
pub(crate) fn supports(device: &wgpu::Device) -> bool {
    if !storage_batch::supports(device) {
        return false;
    }
    if !device
        .features()
        .contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT)
    {
        return false;
    }
    let limits = device.limits();
    limits.max_compute_workgroups_per_dimension >= 1024
        && limits.max_compute_invocations_per_workgroup >= 64
        && limits.max_storage_buffers_per_shader_stage >= 6
}

/// Shared compute pipelines and bind group layouts for GPU culling.
pub(crate) struct GpuCullPipelines {
    group0_layout: wgpu::BindGroupLayout,
    hiz_layout: wgpu::BindGroupLayout,
    phase1_pipeline: wgpu::ComputePipeline,
    phase2_pipeline: wgpu::ComputePipeline,
}

impl GpuCullPipelines {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let storage_entry = |binding, read_only, min_bytes| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(min_bytes),
            },
            count: None,
        };
        let group0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("iletki-3d gpu cull group0 layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(size_of::<CullViewUniforms>() as u64),
                    },
                    count: None,
                },
                storage_entry(1, true, size_of::<ChunkTransformGpu>() as u64),
                storage_entry(2, true, size_of::<ChunkCullItem>() as u64),
                storage_entry(3, false, 4),
                storage_entry(4, false, DRAW_INDEXED_STRIDE),
                storage_entry(5, false, DRAW_NON_INDEXED_STRIDE),
                storage_entry(6, false, 4),
            ],
        });
        let hiz_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("iletki-3d gpu cull hiz layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });

        let common = include_str!("shaders/gpu_cull.wgsl");
        let p1_src = format!("{common}\n{PHASE1_SUFFIX}");
        let p2_src = format!("{common}\n{PHASE2_SUFFIX}");
        let p1_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("iletki-3d gpu cull phase1"),
            source: wgpu::ShaderSource::Wgsl(p1_src.into()),
        });
        let p2_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("iletki-3d gpu cull phase2"),
            source: wgpu::ShaderSource::Wgsl(p2_src.into()),
        });

        let p1_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("iletki-3d gpu cull phase1 layout"),
            bind_group_layouts: &[Some(&group0_layout)],
            immediate_size: 0,
        });
        let p2_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("iletki-3d gpu cull phase2 layout"),
            bind_group_layouts: &[Some(&group0_layout), Some(&hiz_layout)],
            immediate_size: 0,
        });

        let phase1_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("iletki-3d gpu cull phase1 pipeline"),
            layout: Some(&p1_layout),
            module: &p1_mod,
            entry_point: Some("cull_phase1"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let phase2_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("iletki-3d gpu cull phase2 pipeline"),
            layout: Some(&p2_layout),
            module: &p2_mod,
            entry_point: Some("cull_phase2"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            group0_layout,
            hiz_layout,
            phase1_pipeline,
            phase2_pipeline,
        }
    }
}

/// Chunk metadata snapshot passed to [`GpuCullScene::build`].
pub(crate) struct CullChunkInput<'a> {
    pub(crate) chunk_index: usize,
    pub(crate) local_bounds: Bounds3,
    pub(crate) trusted: bool,
    pub(crate) translucent: bool,
    pub(crate) vertex_buf: Option<&'a VertexRange>,
    pub(crate) face_ranges: Option<&'a FaceRanges>,
    pub(crate) edge_buf: Option<&'a VertexRange>,
}

/// Scene-wide GPU culling table shared across all viewports (rebuilt only on geometry changes).
pub(crate) struct GpuCullScene {
    pub(crate) generation: u64,
    pub(crate) buckets: Vec<GpuCullBucket>,
    pub(crate) item_count: u32,
    pub(crate) total_indexed: u32,
    pub(crate) total_non_indexed: u32,
    pub(crate) has_edges: bool,
    pub(crate) extra_chunks: Vec<usize>,
    items_buf: wgpu::Buffer,
}

impl GpuCullScene {
    pub(crate) fn build<'a>(
        device: &wgpu::Device,
        generation: u64,
        chunks: impl Iterator<Item = CullChunkInput<'a>>,
    ) -> Option<Self> {
        let mut buckets: Vec<GpuCullBucket> = Vec::new();
        let mut bucket_map: HashMap<(usize, BucketKind), u32> = HashMap::new();
        let mut items: Vec<ChunkCullItem> = Vec::new();
        let mut extra_chunks: Vec<usize> = Vec::new();
        let mut packed_chunk_count = 0usize;
        let mut has_edges = false;

        let mut get_or_add_bucket = |page: &Arc<GeometryPage>, kind: BucketKind| -> u32 {
            let key = (Arc::as_ptr(page) as usize, kind);
            if let Some(&id) = bucket_map.get(&key) {
                if let Some(bucket) = buckets.get_mut(id as usize) {
                    bucket.max_draws = bucket.max_draws.saturating_add(1);
                }
                return id;
            }
            let id = buckets.len() as u32;
            buckets.push(GpuCullBucket {
                page: Arc::clone(page),
                kind,
                indexed: page.indices().is_some(),
                flat: page.kind() == PageKind::IndexedFlat,
                indirect_base: 0,
                max_draws: 1,
            });
            bucket_map.insert(key, id);
            id
        };

        for input in chunks {
            let mut is_extra = false;
            let mut has_packed_item = false;
            let mut single_face_item: Option<usize> = None;

            if let Some(vbuf) = input.vertex_buf {
                match vbuf {
                    VertexRange::Packed { page, elements, .. } if page.repeated_elements() == 0 => {
                        let kind = if input.translucent {
                            BucketKind::TranslucentFace
                        } else {
                            BucketKind::OpaqueFace
                        };
                        let indexed_flag = if page.indices().is_some() {
                            FLAG_FACE_INDEXED
                        } else {
                            0
                        };
                        let trans_flag = if input.translucent {
                            FLAG_TRANSLUCENT
                        } else {
                            0
                        };
                        has_packed_item = true;
                        if let Some(ranges) = input.face_ranges {
                            for (sub_range, sub_bounds) in ranges.iter_ranges() {
                                let start = elements.start.saturating_add(sub_range.start);
                                let count = sub_range.end.saturating_sub(sub_range.start);
                                if count == 0 {
                                    continue;
                                }
                                let bucket = get_or_add_bucket(page, kind);
                                let trusted = input.trusted && !sub_bounds.is_empty();
                                let flags = FLAG_HAS_FACES
                                    | indexed_flag
                                    | trans_flag
                                    | if trusted { FLAG_TRUSTED } else { 0 };
                                items.push(ChunkCullItem {
                                    local_min: sub_bounds.min.to_f32(),
                                    chunk_index: input.chunk_index as u32,
                                    local_max: sub_bounds.max.to_f32(),
                                    flags,
                                    face_bucket: bucket,
                                    face_first: start,
                                    face_count: count,
                                    face_indirect_base: 0,
                                    edge_bucket: 0,
                                    edge_first: 0,
                                    edge_count: 0,
                                    edge_indirect_base: 0,
                                });
                            }
                        } else {
                            let count = elements.end.saturating_sub(elements.start);
                            if count > 0 {
                                let bucket = get_or_add_bucket(page, kind);
                                let trusted = input.trusted && !input.local_bounds.is_empty();
                                let flags = FLAG_HAS_FACES
                                    | indexed_flag
                                    | trans_flag
                                    | if trusted { FLAG_TRUSTED } else { 0 };
                                single_face_item = Some(items.len());
                                items.push(ChunkCullItem {
                                    local_min: input.local_bounds.min.to_f32(),
                                    chunk_index: input.chunk_index as u32,
                                    local_max: input.local_bounds.max.to_f32(),
                                    flags,
                                    face_bucket: bucket,
                                    face_first: elements.start,
                                    face_count: count,
                                    face_indirect_base: 0,
                                    edge_bucket: 0,
                                    edge_first: 0,
                                    edge_count: 0,
                                    edge_indirect_base: 0,
                                });
                            }
                        }
                    }
                    _ => {
                        is_extra = true;
                    }
                }
            }

            if let Some(ebuf) = input.edge_buf {
                match ebuf {
                    VertexRange::Packed { page, elements, .. } if page.repeated_elements() == 0 => {
                        let count = elements.end.saturating_sub(elements.start);
                        if count > 0 {
                            has_edges = true;
                            has_packed_item = true;
                            let bucket = get_or_add_bucket(page, BucketKind::Edge);
                            let edge_idx_flag = if page.indices().is_some() {
                                FLAG_EDGE_INDEXED
                            } else {
                                0
                            };
                            if let Some(item_idx) = single_face_item
                                && let Some(item) = items.get_mut(item_idx)
                            {
                                item.flags |= FLAG_HAS_EDGES | edge_idx_flag;
                                item.edge_bucket = bucket;
                                item.edge_first = elements.start;
                                item.edge_count = count;
                            } else {
                                let trusted = input.trusted && !input.local_bounds.is_empty();
                                let flags = FLAG_HAS_EDGES
                                    | edge_idx_flag
                                    | if trusted { FLAG_TRUSTED } else { 0 };
                                items.push(ChunkCullItem {
                                    local_min: input.local_bounds.min.to_f32(),
                                    chunk_index: input.chunk_index as u32,
                                    local_max: input.local_bounds.max.to_f32(),
                                    flags,
                                    face_bucket: 0,
                                    face_first: 0,
                                    face_count: 0,
                                    face_indirect_base: 0,
                                    edge_bucket: bucket,
                                    edge_first: elements.start,
                                    edge_count: count,
                                    edge_indirect_base: 0,
                                });
                            }
                        }
                    }
                    _ => {
                        has_edges = true;
                        is_extra = true;
                    }
                }
            }

            if has_packed_item {
                packed_chunk_count = packed_chunk_count.saturating_add(1);
            }
            if is_extra {
                extra_chunks.push(input.chunk_index);
            }
        }

        if packed_chunk_count < MIN_GPU_CULL_CHUNKS || items.is_empty() {
            return None;
        }

        // Assign contiguous non-overlapping indirect buffer ranges per bucket.
        let mut total_indexed = 0u32;
        let mut total_non_indexed = 0u32;
        for bucket in &mut buckets {
            if bucket.indexed {
                bucket.indirect_base = total_indexed;
                total_indexed = total_indexed.saturating_add(bucket.max_draws);
            } else {
                bucket.indirect_base = total_non_indexed;
                total_non_indexed = total_non_indexed.saturating_add(bucket.max_draws);
            }
        }

        for item in &mut items {
            if (item.flags & FLAG_HAS_FACES) != 0
                && let Some(bucket) = buckets.get(item.face_bucket as usize)
            {
                item.face_indirect_base = bucket.indirect_base;
            }
            if (item.flags & FLAG_HAS_EDGES) != 0
                && let Some(bucket) = buckets.get(item.edge_bucket as usize)
            {
                item.edge_indirect_base = bucket.indirect_base;
            }
        }

        let item_count = u32::try_from(items.len()).ok()?;
        let items_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("iletki-3d gpu cull items"),
            contents: bytemuck::cast_slice(&items),
            usage: wgpu::BufferUsages::STORAGE,
        });

        Some(Self {
            generation,
            buckets,
            item_count,
            total_indexed,
            total_non_indexed,
            has_edges,
            extra_chunks,
            items_buf,
        })
    }

    /// Replays Phase 1 indirect draws for `OpaqueFace` buckets.
    pub(crate) fn paint_phase1(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        view: &GpuCullView,
        storage_bg: &wgpu::BindGroup,
        pipelines: &PaintPipelines<'_>,
    ) {
        self.paint_buckets(pass, view, storage_bg, pipelines, BucketKind::OpaqueFace);
    }

    /// Replays Phase 2 indirect draws: newly-visible opaque faces, translucent faces, then edges.
    pub(crate) fn paint_phase2(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        view: &GpuCullView,
        storage_bg: &wgpu::BindGroup,
        face_pipelines: &PaintPipelines<'_>,
        edge_pipeline: &wgpu::RenderPipeline,
    ) {
        self.paint_buckets(
            pass,
            view,
            storage_bg,
            face_pipelines,
            BucketKind::OpaqueFace,
        );
        self.paint_buckets(
            pass,
            view,
            storage_bg,
            face_pipelines,
            BucketKind::TranslucentFace,
        );
        if self.has_edges {
            let edge_pipes = PaintPipelines {
                smooth: edge_pipeline,
                flat: None,
                repeated_smooth: None,
                repeated_flat: None,
                repeated_slots: None,
            };
            self.paint_buckets(pass, view, storage_bg, &edge_pipes, BucketKind::Edge);
        }
    }

    fn paint_buckets(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        view: &GpuCullView,
        storage_bg: &wgpu::BindGroup,
        pipelines: &PaintPipelines<'_>,
        target_kind: BucketKind,
    ) {
        let mut group_bound = false;
        let mut active_flat: Option<bool> = None;

        for (bucket_idx, bucket) in self.buckets.iter().enumerate() {
            if bucket.kind != target_kind || bucket.max_draws == 0 {
                continue;
            }
            if !group_bound {
                pass.set_bind_group(1, storage_bg, &[]);
                group_bound = true;
            }
            let use_flat = bucket.flat && pipelines.flat.is_some();
            if active_flat != Some(use_flat) {
                let pipeline = if use_flat {
                    pipelines.flat.unwrap_or(pipelines.smooth)
                } else {
                    pipelines.smooth
                };
                pass.set_pipeline(pipeline);
                active_flat = Some(use_flat);
            }
            let slot_binding = bucket.page.bind_vertices(pass);
            pass.set_vertex_buffer(slot_binding, bucket.page.slots().slice(..));
            let count_offset = (bucket_idx as u64).saturating_mul(4);
            if let Some(indices) = bucket.page.indices() {
                pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint16);
                let indirect_offset =
                    u64::from(bucket.indirect_base).saturating_mul(DRAW_INDEXED_STRIDE);
                pass.multi_draw_indexed_indirect_count(
                    &view.indexed_args_buf,
                    indirect_offset,
                    &view.bucket_counts_buf,
                    count_offset,
                    bucket.max_draws,
                );
            } else {
                let indirect_offset =
                    u64::from(bucket.indirect_base).saturating_mul(DRAW_NON_INDEXED_STRIDE);
                pass.multi_draw_indirect_count(
                    &view.non_indexed_args_buf,
                    indirect_offset,
                    &view.bucket_counts_buf,
                    count_offset,
                    bucket.max_draws,
                );
            }
        }
    }
}

/// Per-viewport persistent GPU culling buffers (independent Hi-Z visibility per split view).
pub(crate) struct GpuCullView {
    scene_generation: u64,
    transforms_generation: u64,
    uniform_buf: wgpu::Buffer,
    vis_state_buf: wgpu::Buffer,
    indexed_args_buf: wgpu::Buffer,
    non_indexed_args_buf: wgpu::Buffer,
    bucket_counts_buf: wgpu::Buffer,
    group0_bg: wgpu::BindGroup,
    cached_hiz: Option<(u64, wgpu::BindGroup)>,
}

impl GpuCullView {
    pub(crate) fn ensure_and_update<'a>(
        slot: &'a mut Option<Self>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipelines: &GpuCullPipelines,
        scene: &GpuCullScene,
        transforms: &ChunkTransformTable,
        uniforms: &CullViewUniforms,
    ) -> Option<&'a Self> {
        let transforms_buf = transforms.buffer()?;
        let rebuild = slot
            .as_ref()
            .is_none_or(|v| v.scene_generation != scene.generation);
        if rebuild {
            let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("iletki-3d gpu cull uniforms"),
                contents: bytemuck::bytes_of(uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            // Initialize every item to prev_vis = 1 so frame 0 renders full occluders with zero
            // pop-in and converges immediately by frame 1.
            let init_vis = vec![1u32; (scene.item_count as usize).max(1)];
            let vis_state_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("iletki-3d gpu cull visibility"),
                contents: bytemuck::cast_slice(&init_vis),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            let indexed_args_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("iletki-3d gpu cull indexed args"),
                size: u64::from(scene.total_indexed.max(1)).saturating_mul(DRAW_INDEXED_STRIDE),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
                mapped_at_creation: false,
            });
            let non_indexed_args_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("iletki-3d gpu cull non-indexed args"),
                size: u64::from(scene.total_non_indexed.max(1))
                    .saturating_mul(DRAW_NON_INDEXED_STRIDE),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
                mapped_at_creation: false,
            });
            let bucket_counts_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("iletki-3d gpu cull bucket counts"),
                size: (scene.buckets.len().max(1) as u64).saturating_mul(4),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let group0_bg = make_group0_bg(
                device,
                &pipelines.group0_layout,
                &uniform_buf,
                transforms_buf,
                &scene.items_buf,
                &vis_state_buf,
                &indexed_args_buf,
                &non_indexed_args_buf,
                &bucket_counts_buf,
            );
            *slot = Some(Self {
                scene_generation: scene.generation,
                transforms_generation: transforms.generation(),
                uniform_buf,
                vis_state_buf,
                indexed_args_buf,
                non_indexed_args_buf,
                bucket_counts_buf,
                group0_bg,
                cached_hiz: None,
            });
            return slot.as_ref();
        }

        let view = slot.as_mut()?;
        queue.write_buffer(&view.uniform_buf, 0, bytemuck::bytes_of(uniforms));
        if view.transforms_generation != transforms.generation() {
            view.group0_bg = make_group0_bg(
                device,
                &pipelines.group0_layout,
                &view.uniform_buf,
                transforms_buf,
                &scene.items_buf,
                &view.vis_state_buf,
                &view.indexed_args_buf,
                &view.non_indexed_args_buf,
                &view.bucket_counts_buf,
            );
            view.transforms_generation = transforms.generation();
        }
        Some(view)
    }

    pub(crate) fn encode_phase1(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipelines: &GpuCullPipelines,
        scene: &GpuCullScene,
    ) {
        encoder.clear_buffer(&self.bucket_counts_buf, 0, None);
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let gpu_timer = iletki_wgpu::profiling::pass("iletki gpu cull p1");
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let timestamp_writes = gpu_timer.as_ref().and_then(|t| t.compute_writes());
        #[cfg(not(all(feature = "profiling", not(target_arch = "wasm32"))))]
        let timestamp_writes = None;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("iletki-3d gpu cull phase1"),
            timestamp_writes,
        });
        pass.set_pipeline(&pipelines.phase1_pipeline);
        pass.set_bind_group(0, &self.group0_bg, &[]);
        pass.dispatch_workgroups(scene.item_count.div_ceil(64), 1, 1);
    }

    pub(crate) fn encode_phase2(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pipelines: &GpuCullPipelines,
        scene: &GpuCullScene,
        hiz_view: &wgpu::TextureView,
        hiz_generation: u64,
    ) {
        encoder.clear_buffer(&self.bucket_counts_buf, 0, None);
        let rebuild_hiz_bg = self
            .cached_hiz
            .as_ref()
            .is_none_or(|(cached_gen, _)| *cached_gen != hiz_generation);
        if rebuild_hiz_bg {
            let hiz_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("iletki-3d gpu cull hiz binding"),
                layout: &pipelines.hiz_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(hiz_view),
                }],
            });
            self.cached_hiz = Some((hiz_generation, hiz_bg));
        }
        let Some((_, hiz_bg)) = &self.cached_hiz else {
            return;
        };
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let gpu_timer = iletki_wgpu::profiling::pass("iletki gpu cull p2");
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let timestamp_writes = gpu_timer.as_ref().and_then(|t| t.compute_writes());
        #[cfg(not(all(feature = "profiling", not(target_arch = "wasm32"))))]
        let timestamp_writes = None;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("iletki-3d gpu cull phase2"),
            timestamp_writes,
        });
        pass.set_pipeline(&pipelines.phase2_pipeline);
        pass.set_bind_group(0, &self.group0_bg, &[]);
        pass.set_bind_group(1, hiz_bg, &[]);
        pass.dispatch_workgroups(scene.item_count.div_ceil(64), 1, 1);
    }
}

#[allow(clippy::too_many_arguments)]
fn make_group0_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform_buf: &wgpu::Buffer,
    transforms_buf: &wgpu::Buffer,
    items_buf: &wgpu::Buffer,
    vis_state_buf: &wgpu::Buffer,
    indexed_args_buf: &wgpu::Buffer,
    non_indexed_args_buf: &wgpu::Buffer,
    bucket_counts_buf: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("iletki-3d gpu cull group0 binding"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: transforms_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: items_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: vis_state_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: indexed_args_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: non_indexed_args_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: bucket_counts_buf.as_entire_binding(),
            },
        ],
    })
}
