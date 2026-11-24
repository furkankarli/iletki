//! Clean-room provoking-vertex and exact-record mesh indexing for `iletki-3d`.
//!
//! Flat-shaded chunks store face normal, color and 64-bit pick ID on each triangle's provoking
//! corner and deduplicate shared positions per chunk. Smooth chunks fall back to exact-record
//! indexing when vertex reuse is substantial, or remain unindexed otherwise.

use crate::Vertex3D;
use iletki_wgpu::wgpu;

/// Which corner of each triangle supplies `@interpolate(flat)` attributes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProvokingCorner {
    /// First triangle vertex (WGSL default on Vulkan, DX12, Metal and WebGPU core).
    First,
    /// Last triangle vertex (OpenGL / GLES / WebGL2 default with `@interpolate(flat, either)`).
    Last,
}

/// Verified backend capability for flat provoking-vertex delivery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlatSupport {
    pub(crate) corner: ProvokingCorner,
    pub(crate) use_either: bool,
}

/// Detects flat interpolation support and the active provoking corner for `device`.
///
/// On native OpenGL backends where `naga` requires `@interpolate(flat, either)`, this renders a
/// one-triangle self-test with distinct corner colors and falls back (`None`) if the observed
/// provoking corner cannot be verified.
pub(crate) fn detect_flat_support(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Option<FlatSupport> {
    match device.adapter_info().backend {
        wgpu::Backend::Vulkan
        | wgpu::Backend::Dx12
        | wgpu::Backend::Metal
        | wgpu::Backend::BrowserWebGpu => Some(FlatSupport {
            corner: ProvokingCorner::First,
            use_either: false,
        }),
        wgpu::Backend::Gl => {
            #[cfg(not(target_arch = "wasm32"))]
            {
                detect_gl_provoking_corner(device, queue).map(|corner| FlatSupport {
                    corner,
                    use_either: true,
                })
            }
            #[cfg(target_arch = "wasm32")]
            {
                let _ = queue;
                // OpenGL ES 3.0 / WebGL2 fixes the provoking vertex to the last corner.
                Some(FlatSupport {
                    corner: ProvokingCorner::Last,
                    use_either: true,
                })
            }
        }
        _ => None,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn detect_gl_provoking_corner(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Option<ProvokingCorner> {
    const SELF_TEST_WGSL: &str = r#"
struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat, either) col: vec4<f32>,
};
@vertex
fn vs(@builtin(vertex_index) idx: u32) -> Out {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var c = array<vec4<f32>, 3>(
        vec4<f32>(1.0, 0.0, 0.0, 1.0),
        vec4<f32>(0.0, 1.0, 0.0, 1.0),
        vec4<f32>(0.0, 0.0, 1.0, 1.0),
    );
    return Out(vec4<f32>(p[idx], 0.5, 1.0), c[idx]);
}
@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {
    return in.col;
}
"#;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("iletki-3d provoking self-test"),
        source: wgpu::ShaderSource::Wgsl(SELF_TEST_WGSL.into()),
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("iletki-3d provoking self-test pipeline"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("iletki-3d provoking self-test target"),
        size: wgpu::Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("iletki-3d provoking self-test readback"),
        size: u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("iletki-3d provoking self-test encoder"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("iletki-3d provoking self-test pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.draw(0..3, 0..1);
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT),
                rows_per_image: Some(4),
            },
        },
        wgpu::Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
    );
    let submission = queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .ok()?;
    rx.recv().ok()?.ok()?;
    let mapped = readback.slice(..).get_mapped_range().ok()?;
    let px = [
        *mapped.first()?,
        *mapped.get(1)?,
        *mapped.get(2)?,
        *mapped.get(3)?,
    ];
    drop(mapped);
    readback.unmap();
    if px[0] > 200 && px[1] < 20 && px[2] < 20 {
        Some(ProvokingCorner::First)
    } else if px[2] > 200 && px[0] < 20 && px[1] < 20 {
        Some(ProvokingCorner::Last)
    } else {
        None
    }
}

/// Adapts WGSL `@interpolate(flat)` attributes for backends requiring `either` (OpenGL/WebGL2).
pub(crate) fn adapt_flat_sampling(wgsl: &str, use_either: bool) -> String {
    if use_either {
        wgsl.replace("@interpolate(flat)", "@interpolate(flat, either)")
    } else {
        wgsl.to_owned()
    }
}

/// Transforms inter-stage color output structs in `mesh3d` WGSL to use flat normal and color
/// interpolation without touching vertex shader inputs (`VertIn`).
pub(crate) fn make_flat_color_wgsl(wgsl: &str, use_either: bool) -> String {
    let interp = if use_either {
        "@interpolate(flat, either)"
    } else {
        "@interpolate(flat)"
    };
    let adapted = adapt_flat_sampling(wgsl, use_either);
    adapted
        .replace(
            "@location(0) pos_rel: vec3<f32>,\n    @location(1) normal: vec3<f32>,\n    @location(2) color: vec4<f32>,",
            &format!(
                "@location(0) pos_rel: vec3<f32>,\n    @location(1) {interp} normal: vec3<f32>,\n    @location(2) {interp} color: vec4<f32>,"
            ),
        )
        .replace(
            "@location(0) @interpolate(perspective, sample) pos_rel: vec3<f32>,\n    @location(1) normal: vec3<f32>,\n    @location(2) color: vec4<f32>,",
            &format!(
                "@location(0) @interpolate(perspective, sample) pos_rel: vec3<f32>,\n    @location(1) {interp} normal: vec3<f32>,\n    @location(2) {interp} color: vec4<f32>,"
            ),
        )
}

type PosKey = [u32; 3];
type AttrKey = ([u32; 3], [u8; 4], [u32; 2]);
type ExactKey = (PosKey, [u32; 3], [u8; 4], [u32; 2]);

#[inline]
fn vec3_bits(v: [f32; 3]) -> [u32; 3] {
    [v[0].to_bits(), v[1].to_bits(), v[2].to_bits()]
}

#[inline]
fn hash_pos(k: PosKey) -> usize {
    let mut h = u64::from(k[0]).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    h = (h ^ u64::from(k[1])).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h = (h ^ u64::from(k[2])).wrapping_mul(0x94d0_49bb_1331_11eb);
    (h ^ (h >> 31)) as usize
}

#[inline]
fn hash_exact(k: ExactKey) -> usize {
    let (p, n, c, pid) = k;
    let mut h = u64::from(p[0]).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    h = (h ^ u64::from(p[1])).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h = (h ^ u64::from(p[2])).wrapping_mul(0x94d0_49bb_1331_11eb);
    h = (h ^ u64::from(n[0])).wrapping_mul(0x517c_c1b7_2722_0a95);
    h = (h ^ u64::from(n[1])).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    h = (h ^ u64::from(n[2])).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h = (h ^ u64::from(u32::from_ne_bytes(c))).wrapping_mul(0x94d0_49bb_1331_11eb);
    h = (h ^ u64::from(pid[0]) ^ (u64::from(pid[1]) << 32)).wrapping_mul(0x517c_c1b7_2722_0a95);
    (h ^ (h >> 31)) as usize
}

#[derive(Clone, Copy, Default)]
struct PosBucket {
    stamp: u32,
    key: PosKey,
    slot_idx: u32,
}

#[derive(Clone, Copy, Default)]
struct ExactBucket {
    stamp: u32,
    key: ExactKey,
    vid: u32,
}

#[derive(Clone, Copy)]
struct PosSlot {
    pk: PosKey,
    base_vid: u32,
    assigned: Option<AttrKey>,
    extra_head: u32,
}

const NONE_EXTRA: u32 = u32::MAX;
// Avoid hash probing on tiny meshes (boxes, plates, beams).
const LINEAR_DEDUP_MAX_VERTS: usize = 128;

/// Reusable scratch allocations for per-chunk indexing with O(1) generation-stamped clears.
#[derive(Default)]
pub(crate) struct DedupScratch {
    pos_table: Vec<PosBucket>,
    pos_mask: usize,
    pos_stamp: u32,
    exact_table: Vec<ExactBucket>,
    exact_mask: usize,
    exact_stamp: u32,
    slots_vec: Vec<PosSlot>,
    extras: Vec<(AttrKey, u32, u32)>,
    pub(crate) vertices: Vec<Vertex3D>,
    pub(crate) picks: Vec<[u32; 2]>,
    pub(crate) indices: Vec<u32>,
}

impl DedupScratch {
    fn reset_pos_table(&mut self, max_keys: usize) {
        // Keep active probes inside the smallest power-of-two prefix so medium chunks stay L1/L2-hot.
        let needed = max_keys.saturating_mul(2).max(64).next_power_of_two();
        if self.pos_table.len() < needed {
            self.pos_table.resize(needed, PosBucket::default());
        }
        self.pos_mask = needed.saturating_sub(1);
        self.pos_stamp = self.pos_stamp.wrapping_add(1);
        if self.pos_stamp == 0 {
            self.pos_table.fill(PosBucket::default());
            self.pos_stamp = 1;
        }
    }

    fn reset_exact_table(&mut self, max_keys: usize) {
        let needed = max_keys.saturating_mul(2).max(64).next_power_of_two();
        if self.exact_table.len() < needed {
            self.exact_table.resize(needed, ExactBucket::default());
        }
        self.exact_mask = needed.saturating_sub(1);
        self.exact_stamp = self.exact_stamp.wrapping_add(1);
        if self.exact_stamp == 0 {
            self.exact_table.fill(ExactBucket::default());
            self.exact_stamp = 1;
        }
    }

    #[inline]
    fn find_or_insert_pos(
        &mut self,
        pk: PosKey,
        make_slot: impl FnOnce(&mut Vec<Vertex3D>, &mut Vec<[u32; 2]>, &mut Vec<PosSlot>) -> usize,
    ) -> usize {
        let mask = self.pos_mask;
        let stamp = self.pos_stamp;
        let mut idx = hash_pos(pk) & mask;
        loop {
            let Some(bucket) = self.pos_table.get(idx) else {
                return 0;
            };
            if bucket.stamp != stamp {
                let si = make_slot(&mut self.vertices, &mut self.picks, &mut self.slots_vec);
                if let Some(slot) = self.pos_table.get_mut(idx) {
                    *slot = PosBucket {
                        stamp,
                        key: pk,
                        slot_idx: si as u32,
                    };
                }
                return si;
            }
            if bucket.key == pk {
                return bucket.slot_idx as usize;
            }
            idx = (idx + 1) & mask;
        }
    }

    #[inline]
    fn find_or_insert_exact(
        &mut self,
        key: ExactKey,
        make_vert: impl FnOnce(&mut Vec<Vertex3D>, &mut Vec<[u32; 2]>) -> u32,
    ) -> u32 {
        let mask = self.exact_mask;
        let stamp = self.exact_stamp;
        let mut idx = hash_exact(key) & mask;
        loop {
            let Some(bucket) = self.exact_table.get(idx) else {
                return 0;
            };
            if bucket.stamp != stamp {
                let vid = make_vert(&mut self.vertices, &mut self.picks);
                if let Some(slot) = self.exact_table.get_mut(idx) {
                    *slot = ExactBucket { stamp, key, vid };
                }
                return vid;
            }
            if bucket.key == key {
                return bucket.vid;
            }
            idx = (idx + 1) & mask;
        }
    }
}

/// Fast O(1) hint checking only the first triangle for upload-budget estimation.
pub(crate) fn first_triangle_is_flat(vertices: &[Vertex3D], picks: Option<&[[u32; 2]]>) -> bool {
    let Some([v0, v1, v2]) = vertices.first_chunk::<3>() else {
        return false;
    };
    let n0 = vec3_bits(v0.normal);
    if vec3_bits(v1.normal) != n0
        || vec3_bits(v2.normal) != n0
        || v1.color != v0.color
        || v2.color != v0.color
    {
        return false;
    }
    if let Some(picks) = picks {
        let Some([p0, p1, p2]) = picks.first_chunk::<3>() else {
            return false;
        };
        if p1 != p0 || p2 != p0 {
            return false;
        }
    }
    true
}

/// Returns `true` iff every triangle in `vertices` has bit-identical normal, color and pick ID
/// across all three corners.
pub(crate) fn is_flat_eligible(vertices: &[Vertex3D], picks: Option<&[[u32; 2]]>) -> bool {
    if vertices.is_empty() || !vertices.len().is_multiple_of(3) {
        return false;
    }
    if let Some(picks) = picks {
        if picks.len() != vertices.len() {
            return false;
        }
        let (v_tris, _) = vertices.as_chunks::<3>();
        let (p_tris, _) = picks.as_chunks::<3>();
        v_tris
            .iter()
            .zip(p_tris)
            .all(|([v0, v1, v2], [p0, p1, p2])| {
                let n0 = vec3_bits(v0.normal);
                vec3_bits(v1.normal) == n0
                    && vec3_bits(v2.normal) == n0
                    && v1.color == v0.color
                    && v2.color == v0.color
                    && p1 == p0
                    && p2 == p0
            })
    } else {
        let (v_tris, _) = vertices.as_chunks::<3>();
        v_tris.iter().all(|[v0, v1, v2]| {
            let n0 = vec3_bits(v0.normal);
            vec3_bits(v1.normal) == n0
                && vec3_bits(v2.normal) == n0
                && v1.color == v0.color
                && v2.color == v0.color
        })
    }
}

/// Builds provoking-vertex indexed geometry in `scratch` for a verified flat-eligible chunk.
///
/// Returns `true` iff the resulting indexed geometry is strictly smaller than unindexed storage.
pub(crate) fn build_provoking_indexed(
    vertices: &[Vertex3D],
    picks: Option<&[[u32; 2]]>,
    corner: ProvokingCorner,
    scratch: &mut DedupScratch,
) -> bool {
    let use_linear = vertices.len() <= LINEAR_DEDUP_MAX_VERTS;
    if !use_linear {
        // Upper bound on unique positions is vertices.len().
        scratch.reset_pos_table(vertices.len());
    }
    scratch.slots_vec.clear();
    scratch.extras.clear();
    scratch.vertices.clear();
    scratch.picks.clear();
    scratch.indices.clear();

    let has_picks = picks.is_some();
    scratch.indices.reserve(vertices.len());
    let (v_tris, _) = vertices.as_chunks::<3>();

    for (tri_idx, [v0, v1, v2]) in v_tris.iter().enumerate() {
        let pick_id = picks
            .and_then(|p| p.get(tri_idx * 3).copied())
            .unwrap_or([0, 0]);
        let normal = v0.normal;
        let color = v0.color;
        let attr: AttrKey = (vec3_bits(normal), color, pick_id);
        let positions = [v0.position, v1.position, v2.position];

        let mut slot_indices = [0usize; 3];
        for (dst, pos) in slot_indices.iter_mut().zip(positions) {
            let pk = vec3_bits(pos);
            *dst = if use_linear {
                match scratch.slots_vec.iter().position(|slot| slot.pk == pk) {
                    Some(si) => si,
                    None => {
                        let vid = scratch.vertices.len() as u32;
                        scratch
                            .vertices
                            .push(Vertex3D::new(pos, [0.0, 0.0, 1.0], [0, 0, 0, 255]));
                        if has_picks {
                            scratch.picks.push([0, 0]);
                        }
                        let si = scratch.slots_vec.len();
                        scratch.slots_vec.push(PosSlot {
                            pk,
                            base_vid: vid,
                            assigned: None,
                            extra_head: NONE_EXTRA,
                        });
                        si
                    }
                }
            } else {
                scratch.find_or_insert_pos(pk, |verts, pids, slots| {
                    let vid = verts.len() as u32;
                    verts.push(Vertex3D::new(pos, [0.0, 0.0, 1.0], [0, 0, 0, 255]));
                    if has_picks {
                        pids.push([0, 0]);
                    }
                    let si = slots.len();
                    slots.push(PosSlot {
                        pk,
                        base_vid: vid,
                        assigned: None,
                        extra_head: NONE_EXTRA,
                    });
                    si
                })
            };
        }

        // 1. Prefer an existing (pos, attr) match so unassigned corners remain free.
        let mut chosen: Option<(usize, u32)> = None;
        for (k, &si) in slot_indices.iter().enumerate() {
            let Some(&slot) = scratch.slots_vec.get(si) else {
                continue;
            };
            if slot.assigned == Some(attr) {
                chosen = Some((k, slot.base_vid));
                break;
            }
            let mut cur = slot.extra_head;
            while cur != NONE_EXTRA {
                let Some(&(ea, evid, next)) = scratch.extras.get(cur as usize) else {
                    break;
                };
                if ea == attr {
                    chosen = Some((k, evid));
                    break;
                }
                cur = next;
            }
            if chosen.is_some() {
                break;
            }
        }

        // 2. Otherwise claim an unassigned base vertex at one of the three corners.
        if chosen.is_none() {
            for (k, &si) in slot_indices.iter().enumerate() {
                if let Some(slot) = scratch.slots_vec.get_mut(si)
                    && slot.assigned.is_none()
                {
                    let vid = slot.base_vid;
                    if let Some(vert) = scratch.vertices.get_mut(vid as usize) {
                        vert.normal = normal;
                        vert.color = color;
                    }
                    if has_picks && let Some(pid) = scratch.picks.get_mut(vid as usize) {
                        *pid = pick_id;
                    }
                    slot.assigned = Some(attr);
                    chosen = Some((k, vid));
                    break;
                }
            }
        }

        // 3. Otherwise duplicate only corner 0 with this face's attributes.
        let (prov_k, prov_vid) = match chosen {
            Some(c) => c,
            None => {
                let vid = scratch.vertices.len() as u32;
                scratch
                    .vertices
                    .push(Vertex3D::new(positions[0], normal, color));
                if has_picks {
                    scratch.picks.push(pick_id);
                }
                if let Some(slot) = scratch.slots_vec.get_mut(slot_indices[0]) {
                    let new_extra = scratch.extras.len() as u32;
                    scratch.extras.push((attr, vid, slot.extra_head));
                    slot.extra_head = new_extra;
                }
                (0, vid)
            }
        };

        let next_idx = (prov_k + 1) % 3;
        let next2_idx = (prov_k + 2) % 3;
        let v_next = slot_indices
            .get(next_idx)
            .and_then(|&si| scratch.slots_vec.get(si))
            .map_or(prov_vid, |s| s.base_vid);
        let v_next2 = slot_indices
            .get(next2_idx)
            .and_then(|&si| scratch.slots_vec.get(si))
            .map_or(prov_vid, |s| s.base_vid);

        // Cyclic rotation strictly preserves CCW/CW triangle winding.
        match corner {
            ProvokingCorner::First => {
                scratch.indices.push(prov_vid);
                scratch.indices.push(v_next);
                scratch.indices.push(v_next2);
            }
            ProvokingCorner::Last => {
                scratch.indices.push(v_next);
                scratch.indices.push(v_next2);
                scratch.indices.push(prov_vid);
            }
        }
    }

    // Require strict total byte reduction (out_verts * 32 + in_verts * 2 < in_verts * 32).
    scratch.vertices.len().saturating_mul(16) < vertices.len().saturating_mul(15)
}

/// Builds bit-exact deduplicated indexed geometry for smooth-shaded chunks.
///
/// Returns `true` iff deduplication reduces vertex count by at least 25%.
pub(crate) fn build_exact_indexed(
    vertices: &[Vertex3D],
    picks: Option<&[[u32; 2]]>,
    scratch: &mut DedupScratch,
) -> bool {
    if vertices.is_empty() || !vertices.len().is_multiple_of(3) {
        return false;
    }
    scratch.reset_exact_table(vertices.len());
    scratch.vertices.clear();
    scratch.picks.clear();
    scratch.indices.clear();

    let has_picks = picks.is_some();
    scratch.indices.reserve(vertices.len());
    for (idx, v) in vertices.iter().enumerate() {
        let pick_id = picks.and_then(|p| p.get(idx).copied()).unwrap_or([0, 0]);
        let key: ExactKey = (vec3_bits(v.position), vec3_bits(v.normal), v.color, pick_id);
        let vid = scratch.find_or_insert_exact(key, |verts, pids| {
            let id = verts.len() as u32;
            verts.push(*v);
            if has_picks {
                pids.push(pick_id);
            }
            id
        });
        scratch.indices.push(vid);
    }

    scratch.vertices.len().saturating_mul(4) <= vertices.len().saturating_mul(3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Mesh3D;
    use iletki_core::{PickId, Vec3};

    #[test]
    fn provoking_box_reduces_36_vertices_to_8_without_duplication() {
        let mut mesh = Mesh3D::at_origin(Vec3::ZERO);
        mesh.push_box(
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, 1.0),
            [120, 170, 200, 255],
            PickId::new(1, 7, 0),
            true,
        );
        assert!(is_flat_eligible(&mesh.vertices, Some(&mesh.pick_ids)));
        let mut scratch = DedupScratch::default();
        for corner in [ProvokingCorner::First, ProvokingCorner::Last] {
            assert!(build_provoking_indexed(
                &mesh.vertices,
                Some(&mesh.pick_ids),
                corner,
                &mut scratch,
            ));
            assert_eq!(scratch.vertices.len(), 8);
            assert_eq!(scratch.picks.len(), 8);
            assert_eq!(scratch.indices.len(), 36);

            // Reconstruct every triangle and verify exact position, winding, normal, color, pick.
            for (t, orig) in mesh.vertices.as_chunks::<3>().0.iter().enumerate() {
                let i0 = scratch.indices[t * 3] as usize;
                let i1 = scratch.indices[t * 3 + 1] as usize;
                let i2 = scratch.indices[t * 3 + 2] as usize;
                let prov = match corner {
                    ProvokingCorner::First => i0,
                    ProvokingCorner::Last => i2,
                };
                assert_eq!(scratch.vertices[prov].normal, orig[0].normal);
                assert_eq!(scratch.vertices[prov].color, orig[0].color);
                assert_eq!(scratch.picks[prov], mesh.pick_ids[t * 3]);

                let p = [
                    scratch.vertices[i0].position,
                    scratch.vertices[i1].position,
                    scratch.vertices[i2].position,
                ];
                let o = [orig[0].position, orig[1].position, orig[2].position];
                // Must be a cyclic rotation of (o[0], o[1], o[2]) to preserve winding.
                assert!(
                    p == o || p == [o[1], o[2], o[0]] || p == [o[2], o[0], o[1]],
                    "winding changed on triangle {t}"
                );
            }
        }
    }

    #[test]
    fn smooth_triangles_are_rejected_from_flat_and_handled_by_exact_dedup() {
        let mut verts = Vec::new();
        // Smooth quad: 2 triangles sharing 2 vertices with differing corner normals.
        let v0 = Vertex3D::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [255, 255, 255, 255]);
        let v1 = Vertex3D::new([1.0, 0.0, 0.0], [0.1, 0.0, 0.99], [255, 255, 255, 255]);
        let v2 = Vertex3D::new([1.0, 1.0, 0.0], [0.1, 0.1, 0.98], [255, 255, 255, 255]);
        let v3 = Vertex3D::new([0.0, 1.0, 0.0], [0.0, 0.1, 0.99], [255, 255, 255, 255]);
        verts.extend_from_slice(&[v0, v1, v2, v0, v2, v3]);
        assert!(!is_flat_eligible(&verts, None));
        let mut scratch = DedupScratch::default();
        assert!(build_exact_indexed(&verts, None, &mut scratch));
        assert_eq!(scratch.vertices.len(), 4);
        assert_eq!(scratch.indices.len(), 6);
    }
}
