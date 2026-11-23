//! Per-viewport chunk draws and aligned object uniform pages.

use crate::scene::ObjectUniforms;
use crate::{Vertex3D, face_batch::PackedDraws};
use iletki_core::{Bounds3, Mat4, Vec4, ViewportId};
use iletki_wgpu::{PrepareCtx, ViewportLifetime, wgpu};
use std::num::NonZeroU64;

const OBJECT_BYTES: usize = std::mem::size_of::<ObjectUniforms>();
const PAGE_BYTES: usize = 1024 * 1024;

pub(crate) struct PreparedObject {
    pub(crate) chunk: usize,
    pub(crate) uniforms: ObjectUniforms,
    pub(crate) direct_slot: Option<usize>,
}

struct UniformPage {
    capacity: usize,
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

pub(crate) struct ViewObjects {
    pub(crate) id: ViewportId,
    pub(crate) lifetime: ViewportLifetime,
    pub(crate) draws: Vec<PreparedObject>,
    pub(crate) has_edges: bool,
    pub(crate) faces: PackedDraws,
    pub(crate) edges: PackedDraws,
    pub(crate) direct_uniforms: Vec<ObjectUniforms>,
    pages: Vec<UniformPage>,
    staging: Vec<u8>,
    stride: usize,
    slots_per_page: usize,
}

impl ViewObjects {
    pub(crate) fn new<W>(ctx: &PrepareCtx<'_, W>, batch: bool) -> Self {
        let limits = ctx.device.limits();
        let alignment = limits.min_uniform_buffer_offset_alignment as usize;
        let stride = OBJECT_BYTES.div_ceil(alignment.max(1)) * alignment.max(1);
        let bytes =
            usize::try_from(limits.max_buffer_size.min(PAGE_BYTES as u64)).unwrap_or(OBJECT_BYTES);
        let slots_per_page = if batch {
            bytes.saturating_sub(OBJECT_BYTES) / stride + 1
        } else {
            1
        };
        Self {
            id: ctx.viewport_id,
            lifetime: ctx.viewport_lifetime(),
            draws: Vec::new(),
            has_edges: false,
            faces: PackedDraws::default(),
            edges: PackedDraws::default(),
            direct_uniforms: Vec::new(),
            pages: Vec::new(),
            staging: Vec::new(),
            stride,
            slots_per_page,
        }
    }

    pub(crate) fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
    ) {
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let _profile = iletki_wgpu::profiling::scope("scene.uniforms");
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let mut total_bytes: usize = 0;
        for (index, objects) in self.direct_uniforms.chunks(self.slots_per_page).enumerate() {
            let bytes_for = |slots: usize| (slots.saturating_sub(1)) * self.stride + OBJECT_BYTES;
            let page = match self.pages.get(index) {
                Some(page) if page.capacity >= objects.len() => page,
                _ => {
                    // Page storage grows geometrically and is reused as the camera moves.
                    let capacity = objects.len().next_power_of_two().min(self.slots_per_page);
                    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("iletki-3d object uniform page"),
                        size: bytes_for(capacity) as u64,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("iletki-3d object uniform page binding"),
                        layout,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &buffer,
                                offset: 0,
                                size: NonZeroU64::new(OBJECT_BYTES as u64),
                            }),
                        }],
                    });
                    let page = UniformPage {
                        capacity,
                        buffer,
                        bind_group,
                    };
                    if let Some(slot) = self.pages.get_mut(index) {
                        *slot = page;
                        slot
                    } else {
                        self.pages.push(page);
                        let Some(page) = self.pages.last() else {
                            continue;
                        };
                        page
                    }
                }
            };
            let bytes = if let [only] = objects {
                bytemuck::bytes_of(only)
            } else {
                self.staging.resize(bytes_for(objects.len()), 0);
                for (slot, obj) in self.staging.chunks_mut(self.stride).zip(objects) {
                    if let Some(payload) = slot.get_mut(..OBJECT_BYTES) {
                        payload.copy_from_slice(bytemuck::bytes_of(obj));
                    }
                }
                &self.staging
            };
            queue.write_buffer(&page.buffer, 0, bytes);
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            {
                total_bytes = total_bytes.saturating_add(bytes.len());
            }
        }
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        {
            iletki_wgpu::profiling::count(
                "scene.uniform_writes",
                self.direct_uniforms.len().div_ceil(self.slots_per_page),
            );
            iletki_wgpu::profiling::count("scene.uniform_bytes", total_bytes);
        }
    }

    pub(crate) fn bind(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        index: usize,
        dynamic: bool,
    ) -> bool {
        let Some(index) = self.draws.get(index).and_then(|draw| draw.direct_slot) else {
            return false;
        };
        let Some(page) = self.pages.get(index / self.slots_per_page) else {
            return false;
        };
        let offset = (index % self.slots_per_page) * self.stride;
        if dynamic {
            // Page allocation limits every offset to PAGE_BYTES.
            pass.set_bind_group(1, &page.bind_group, &[offset as u32]);
        } else {
            pass.set_bind_group(1, &page.bind_group, &[]);
        }
        true
    }
}

pub(crate) fn pick_bounds_cover_vertices(bounds: Bounds3, vertices: &[Vertex3D]) -> bool {
    if bounds.is_empty() {
        return false;
    }
    // Both the vertices and the bounds are public. Direct mesh edits can leave stale bounds.
    let min = bounds.min.to_f32();
    let max = bounds.max.to_f32();
    vertices.iter().all(|vertex| {
        vertex
            .position
            .into_iter()
            .zip(min.into_iter().zip(max))
            .all(|(value, (lo, hi))| value >= lo && value <= hi)
    })
}

/// Four homogeneous side planes for the pick rectangle. Testing the camera-relative f32
/// bounds in f64 avoids rounding world-space positions a second, different way.
pub(crate) struct PickVolume {
    planes: [(Vec4, Vec4); 4],
}

impl PickVolume {
    pub(crate) fn new(projection: Mat4, [width, height]: [u32; 2], rect: [u32; 4]) -> Self {
        let [x, y, w, h] = rect;
        // A pixel guard covers fixed-point viewport rasterization and crop-edge rounding.
        let left = 2.0 * f64::from(x.saturating_sub(1)) / f64::from(width) - 1.0;
        let right = 2.0 * f64::from(x.saturating_add(w).saturating_add(1)) / f64::from(width) - 1.0;
        let top = 1.0 - 2.0 * f64::from(y.saturating_sub(1)) / f64::from(height);
        let bottom =
            1.0 - 2.0 * f64::from(y.saturating_add(h).saturating_add(1)) / f64::from(height);
        let rounded = |row| {
            let [x, y, z, w] = projection.row(row).to_f32().map(f64::from);
            Vec4::new(x, y, z, w)
        };
        let x_row = rounded(0);
        let y_row = rounded(1);
        let w_row = rounded(3);
        let side = |row: Vec4, sign: f64, edge: f64| {
            let combine = |a: f64, b: f64| a * sign + b * edge;
            let magnitude = |a: f64, b: f64| a.abs() + (b * edge).abs();
            (
                Vec4::new(
                    combine(row.x, w_row.x),
                    combine(row.y, w_row.y),
                    combine(row.z, w_row.z),
                    combine(row.w, w_row.w),
                ),
                Vec4::new(
                    magnitude(row.x, w_row.x),
                    magnitude(row.y, w_row.y),
                    magnitude(row.z, w_row.z),
                    magnitude(row.w, w_row.w),
                ),
            )
        };
        Self {
            planes: [
                side(x_row, 1.0, -left),
                side(x_row, -1.0, right),
                side(y_row, 1.0, -bottom),
                side(y_row, -1.0, top),
            ],
        }
    }

    pub(crate) fn intersects(&self, bounds: Bounds3, [ox, oy, oz]: [f32; 3]) -> bool {
        if bounds.is_empty() {
            return true;
        }
        let [min_x, min_y, min_z] = bounds.min.to_f32();
        let [max_x, max_y, max_z] = bounds.max.to_f32();
        let lo = [min_x + ox, min_y + oy, min_z + oz].map(f64::from);
        let hi = [max_x + ox, max_y + oy, max_z + oz].map(f64::from);
        if !lo.into_iter().chain(hi).all(f64::is_finite) {
            return true;
        }
        let [lx, ly, lz] = lo;
        let [hx, hy, hz] = hi;
        let [mx, my, mz] = [
            lx.abs().max(hx.abs()),
            ly.abs().max(hy.abs()),
            lz.abs().max(hz.abs()),
        ];
        let support = |a: f64, low, high| a * if a >= 0.0 { high } else { low };
        self.planes.iter().all(|(p, m)| {
            let distance = support(p.x, lx, hx) + support(p.y, ly, hy) + support(p.z, lz, hz) + p.w;
            let magnitude = m.x * mx + m.y * my + m.z * mz + m.w;
            let rounding =
                magnitude * (16.0 * f64::from(f32::EPSILON)) + f64::from(f32::MIN_POSITIVE);
            // Uncertain precision or overflow always keeps the object in the candidate set.
            !distance.is_finite() || magnitude >= f64::from(f32::MAX) * 0.5 || distance >= -rounding
        })
    }
}
