//! Per-viewport chunk draws and aligned object uniform pages.

use crate::gpu_cull::GpuCullView;
use crate::gtao::ViewGtaoState;
use crate::oit::ViewOitState;
use crate::section_cap::ViewSectionCaps;
use crate::style::ViewStyleUniform;
use crate::{
    Vertex3D,
    face_batch::PackedDraws,
    storage_batch::{StorageDraws, StorageViewBinding},
};
use crate::{outline_bounds::OutlineBounds, scene::ObjectUniforms};
use iletki_core::{Bounds3, Camera, Frustum, MAX_CLIP_PLANES, Mat4, Vec3, Vec4, ViewportId};
use iletki_wgpu::{PrepareCtx, ViewportLifetime, wgpu};
use std::num::NonZeroU64;
use std::ops::Range;

const OBJECT_BYTES: usize = std::mem::size_of::<ObjectUniforms>();
const PAGE_BYTES: usize = 1024 * 1024;
// Whole triangle lists, in submission order, so equal-depth color and pick ties are unchanged.
const FACE_RANGE_VERTICES: usize = 1536;

pub(crate) struct FaceRanges {
    bounds: Box<[Bounds3]>,
    vertex_count: u32,
}

impl FaceRanges {
    pub(crate) fn new(vertices: &[Vertex3D]) -> Option<Self> {
        if vertices.len() < 4 * FACE_RANGE_VERTICES {
            return None;
        }
        let bounds = vertices.chunks(FACE_RANGE_VERTICES).map(|vertices| {
            let mut bounds = Bounds3::EMPTY;
            for vertex in vertices {
                let [x, y, z] = vertex.position.map(f64::from);
                let position = Vec3::new(x, y, z);
                if !position.is_finite() {
                    // An uncertain range must always go through the rasterizer.
                    return Bounds3::EMPTY;
                }
                bounds.include(position);
            }
            bounds
        });
        Some(Self {
            bounds: bounds.collect(),
            vertex_count: u32::try_from(vertices.len()).ok()?,
        })
    }

    /// Emits adjacent visible blocks in a single draw and returns the number of omitted vertices.
    pub(crate) fn visit_visible(
        &self,
        mut visible: impl FnMut(Bounds3) -> bool,
        mut emit: impl FnMut(Range<u32>),
    ) -> u32 {
        let mut start = 0;
        let mut omitted = 0;
        for (block, &bounds) in self.bounds.iter().enumerate() {
            if !visible(bounds) {
                let end = (block * FACE_RANGE_VERTICES) as u32;
                if start < end {
                    emit(start..end);
                }
                start = end
                    .saturating_add(FACE_RANGE_VERTICES as u32)
                    .min(self.vertex_count);
                omitted += start - end;
            }
        }
        if start < self.vertex_count {
            emit(start..self.vertex_count);
        }
        omitted
    }

    pub(crate) fn iter_ranges(&self) -> impl Iterator<Item = (Range<u32>, Bounds3)> + '_ {
        self.bounds.iter().enumerate().map(|(block, &bounds)| {
            let start = (block * FACE_RANGE_VERTICES) as u32;
            let end = start
                .saturating_add(FACE_RANGE_VERTICES as u32)
                .min(self.vertex_count);
            (start..end, bounds)
        })
    }
}

pub(crate) struct PreparedObject {
    pub(crate) chunk: usize,
    pub(crate) uniforms: ObjectUniforms,
    pub(crate) section_culled: bool,
    pub(crate) bridge_section_gap: bool,
    pub(crate) direct_slot: Option<usize>,
}

struct UniformPage {
    capacity: usize,
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

pub(crate) struct ViewObjects {
    pub(crate) id: ViewportId,
    pub(crate) picking: bool,
    pub(crate) lifetime: ViewportLifetime,
    pub(crate) draws: Vec<PreparedObject>,
    pub(crate) has_edges: bool,
    pub(crate) faces: PackedDraws,
    pub(crate) edges: PackedDraws,
    pub(crate) mask: PackedDraws,
    pub(crate) storage_binding: Option<StorageViewBinding>,
    pub(crate) storage_faces: StorageDraws,
    pub(crate) storage_edges: StorageDraws,
    pub(crate) gpu_cull: Option<GpuCullView>,
    pub(crate) gpu_cull_active: bool,
    pub(crate) retained_geometry_rev: u64,
    pub(crate) retained_full_scene: bool,
    pub(crate) retained_has_edges: bool,
    pub(crate) outline_bounds: OutlineBounds,
    pub(crate) section_caps: Option<ViewSectionCaps>,
    pub(crate) oit: Option<ViewOitState>,
    pub(crate) gtao: Option<ViewGtaoState>,
    pub(crate) style_uniform: Option<ViewStyleUniform>,
    pub(crate) direct_uniforms: Vec<ObjectUniforms>,
    pages: Vec<UniformPage>,
    staging: Vec<u8>,
    stride: usize,
    slots_per_page: usize,
}

impl ViewObjects {
    pub(crate) fn new<W>(ctx: &PrepareCtx<'_, W>, batch: bool, picking: bool) -> Self {
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
            picking,
            lifetime: ctx.viewport_lifetime(),
            draws: Vec::new(),
            has_edges: false,
            faces: PackedDraws::default(),
            edges: PackedDraws::default(),
            mask: PackedDraws::default(),
            storage_binding: None,
            storage_faces: StorageDraws::default(),
            storage_edges: StorageDraws::default(),
            gpu_cull: None,
            gpu_cull_active: false,
            retained_geometry_rev: 0,
            retained_full_scene: false,
            retained_has_edges: false,
            outline_bounds: OutlineBounds::default(),
            section_caps: None,
            oit: None,
            gtao: None,
            style_uniform: None,
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

pub(crate) fn bounds_cover_vertices(bounds: Bounds3, vertices: &[Vertex3D]) -> bool {
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

/// Bounds rounded with the same camera-relative translation as the 3D vertex shaders.
fn rounded_bounds(bounds: Bounds3, offset: [f32; 3]) -> Option<([f64; 3], [f64; 3])> {
    if bounds.is_empty() {
        return None;
    }
    let lo = translated_corner(bounds.min, offset).map(f64::from);
    let hi = translated_corner(bounds.max, offset).map(f64::from);
    lo.into_iter()
        .chain(hi)
        .all(f64::is_finite)
        .then_some((lo, hi))
}

fn translated_corner(local: Vec3, [ox, oy, oz]: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = local.to_f32();
    [x + ox, y + oy, z + oz]
}

/// Conservative rejection for the color and picking section passes.
pub(crate) struct SectionVolume {
    planes: [[f64; 4]; MAX_CLIP_PLANES],
    count: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum SectionRelation {
    Outside,
    Inside,
    Crossing,
}

/// An envelope around the trusted members of a group of consecutive chunk slots.
#[derive(Default)]
pub(crate) struct SectionGroup {
    origins: Bounds3,
    local: Bounds3,
    world: Bounds3,
    invalid_origin: bool,
    has_untrusted: bool,
}

impl SectionGroup {
    pub(crate) fn include(&mut self, origin: Vec3, local: Bounds3) {
        // A nonfinite origin makes this group ineligible for either shortcut.
        if !origin.is_finite() {
            self.invalid_origin = true;
            return;
        }
        self.origins.min = self.origins.min.min(origin);
        self.origins.max = self.origins.max.max(origin);
        self.local.min = self.local.min.min(local.min);
        self.local.max = self.local.max.max(local.max);
        if !local.is_empty() {
            let wmin = origin + local.min;
            let wmax = origin + local.max;
            if wmin.is_finite() && wmax.is_finite() {
                self.world.include(wmin);
                self.world.include(wmax);
            } else {
                self.invalid_origin = true;
            }
        }
    }

    pub(crate) fn mark_untrusted(&mut self) {
        self.has_untrusted = true;
    }

    pub(crate) fn all_trusted(&self) -> bool {
        !self.has_untrusted && !self.invalid_origin
    }

    pub(crate) fn contains_chunk(&self, origin: Vec3, local: Bounds3) -> bool {
        if self.invalid_origin {
            return false;
        }
        let origins_ok = origin.min(self.origins.min) == self.origins.min
            && origin.max(self.origins.max) == self.origins.max;
        if !origins_ok {
            return false;
        }
        if local.is_empty() {
            return true;
        }
        let wmin = origin + local.min;
        let wmax = origin + local.max;
        wmin.is_finite()
            && wmax.is_finite()
            && wmin.min(self.world.min) == self.world.min
            && wmax.max(self.world.max) == self.world.max
    }

    pub(crate) fn section_relation(&self, volume: &SectionVolume, camera: Vec3) -> SectionRelation {
        if self.invalid_origin {
            return SectionRelation::Crossing;
        }
        // Bound the two f32 operations separately. Each chunk rounds its camera offset
        // before adding it to the local vertex, even when its world origin is far away.
        let min = translated_corner(self.local.min, (self.origins.min - camera).to_f32());
        let max = translated_corner(self.local.max, (self.origins.max - camera).to_f32());
        if min.into_iter().chain(max).all(f32::is_finite) {
            volume.classify(min.map(f64::from), max.map(f64::from), true)
        } else {
            SectionRelation::Crossing
        }
    }

    pub(crate) fn outside_frustum(&self, frustum: &Frustum, camera: Vec3) -> bool {
        if self.has_untrusted || self.invalid_origin || self.world.is_empty() {
            return false;
        }
        !frustum.intersects_bounds_world(self.world, camera)
    }

    pub(crate) fn inside_frustum(&self, frustum: &Frustum, camera: Vec3) -> bool {
        if self.invalid_origin || self.world.is_empty() {
            return false;
        }
        let lo = self.world.min - camera;
        let hi = self.world.max - camera;
        lo.is_finite()
            && hi.is_finite()
            && frustum.planes.iter().flatten().all(|plane| {
                let n = plane.normal;
                let near = Vec3::new(
                    if n.x >= 0.0 { lo.x } else { hi.x },
                    if n.y >= 0.0 { lo.y } else { hi.y },
                    if n.z >= 0.0 { lo.z } else { hi.z },
                );
                plane.signed_distance(near) >= 0.0
            })
    }
}

impl SectionVolume {
    pub(crate) fn new(camera: &Camera) -> Self {
        let mut planes = [[0.0; 4]; MAX_CLIP_PLANES];
        for (dst, plane) in planes.iter_mut().zip(camera.clip_planes()) {
            let coefficients = plane.to_camera_relative_f32(camera.position_3d());
            let [x, y, z, d] = coefficients;
            // Public plane fields can bypass normalization. Keep planes whose normals might
            // amplify subnormal inputs or be flushed to zero by the GPU out of this test.
            if d.is_finite()
                && [x, y, z]
                    .into_iter()
                    .all(|n| n == 0.0 || (n.is_normal() && n.abs() <= 1.0))
            {
                *dst = coefficients.map(f64::from);
            }
        }
        Self {
            planes,
            count: camera.clip_planes().len(),
        }
    }

    pub(crate) fn excludes(&self, bounds: Bounds3, offset: [f32; 3]) -> bool {
        let Some((lo, hi)) = rounded_bounds(bounds, offset) else {
            return false;
        };
        self.classify(lo, hi, false) == SectionRelation::Outside
    }

    fn classify(
        &self,
        [lx, ly, lz]: [f64; 3],
        [hx, hy, hz]: [f64; 3],
        check_inside: bool,
    ) -> SectionRelation {
        let [mx, my, mz] = [
            lx.abs().max(hx.abs()),
            ly.abs().max(hy.abs()),
            lz.abs().max(hz.abs()),
        ];
        let mut inside = check_inside;
        for &[x, y, z, d] in self.planes.iter().take(self.count) {
            let (near_x, far_x) = if x >= 0.0 { (lx, hx) } else { (hx, lx) };
            let (near_y, far_y) = if y >= 0.0 { (ly, hy) } else { (hy, ly) };
            let (near_z, far_z) = if z >= 0.0 { (lz, hz) } else { (hz, lz) };
            let magnitude = x.abs() * mx + y.abs() * my + z.abs() * mz + d.abs();
            // The shader adds the camera WGSL clip tolerance. Allow for f32 interpolation
            // and dot-product rounding before rejecting a whole primitive.
            let tolerance = f64::from(1e-4_f32);
            let error = (magnitude + tolerance) * (16.0 * f64::from(f32::EPSILON));
            if magnitude >= f64::from(f32::MAX) * 0.5 {
                inside = false;
                continue;
            }
            let support = x * far_x + y * far_y + z * far_z + d;
            if support + error < -tolerance {
                return SectionRelation::Outside;
            }
            if inside {
                let minimum = x * near_x + y * near_y + z * near_z + d;
                // Every member's own positive support plus its error margin is at least
                // this minimum. A subsequent per-member test could not reject it.
                inside = minimum - error >= -tolerance;
            }
        }
        if inside {
            SectionRelation::Inside
        } else {
            SectionRelation::Crossing
        }
    }
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

    pub(crate) fn intersects(
        &self,
        bounds: Bounds3,
        offset: [f32; 3],
        section: Option<&SectionVolume>,
    ) -> bool {
        let Some((lo @ [lx, ly, lz], hi @ [hx, hy, hz])) = rounded_bounds(bounds, offset) else {
            return true;
        };
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
        }) && section
            .is_none_or(|volume| volume.classify(lo, hi, false) != SectionRelation::Outside)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iletki_core::{Plane, Vec3};

    #[test]
    fn raster_section_keeps_crossing_and_uncertain_bounds_at_large_coordinates() {
        let origin = Vec3::new(1e9, 2e9, 3e9);
        let mut camera =
            Camera::perspective_3d(origin - Vec3::Y * 8.0, origin, Vec3::Z, 0.75, 0.05);
        camera.set_clip_planes(&[
            Plane::from_normal_point(Vec3::X, origin).unwrap(),
            Plane::from_normal_point(-Vec3::Z, origin + Vec3::Z).unwrap(),
        ]);
        let offset = (origin - camera.position_3d()).to_f32();
        let section = SectionVolume::new(&camera);
        let bounds =
            |lo, hi| Bounds3::from_corners(Vec3::new(lo, -1.0, 0.0), Vec3::new(hi, 1.0, 2.0));
        assert!(section.excludes(bounds(-3.0, -2.0), offset));
        assert!(section.excludes(bounds(0.0, 1.0), [0.0, 8.0, 2.0]));
        assert!(!section.excludes(bounds(-2.0, -0.00005), offset));
        assert!(!section.excludes(bounds(-1.0, 1.0), offset));
        assert!(!section.excludes(bounds(-3.0, -2.0), [f32::INFINITY, 8.0, 0.0]));
        assert!(!section.excludes(Bounds3::EMPTY, offset));
    }

    #[test]
    fn raster_section_leaves_subnormal_clip_inputs_to_the_shader() {
        let mut camera = Camera::perspective_3d(-Vec3::Y * 8.0, Vec3::ZERO, Vec3::Z, 0.75, 0.05);
        let bounds =
            |lo, hi| Bounds3::from_corners(Vec3::new(lo, -1.0, 0.0), Vec3::new(hi, 1.0, 2.0));
        let offset = [0.0, 8.0, 0.0];
        camera.set_clip_planes(&[Plane {
            normal: Vec3::X * f64::from(f32::MAX),
            d: 0.0,
        }]);
        assert!(!SectionVolume::new(&camera).excludes(bounds(-1e-38, -1e-39), offset));
        camera.set_clip_planes(&[Plane {
            normal: Vec3::X * (f64::from(f32::MIN_POSITIVE) * 0.5),
            d: 0.0,
        }]);
        assert!(!SectionVolume::new(&camera).excludes(bounds(-2e37, -1e37), offset));
    }
}
