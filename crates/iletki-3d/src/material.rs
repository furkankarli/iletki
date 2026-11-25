//! Persistent GPU material table for [`crate::Scene3DLayer`].
//!
//! Materials are stored in a persistent `MaterialGpu` storage buffer on backends with storage
//! buffer support (indexed from the upper 24 bits of each chunk's `ChunkTransformGpu.flags` with
//! zero extra vertex or transform bytes), with an aligned uniform mirror for direct draws and
//! OpenGL/WebGL2 fallback. When every chunk uses [`MaterialId::DEFAULT`], no GPU material buffers
//! or material pipelines are allocated.

use bytemuck::{Pod, Zeroable};
use iletki_wgpu::wgpu;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::num::NonZeroU64;

const MIN_MATERIAL_CAPACITY: usize = 16;
/// Maximum valid material slot index encodable inside the upper 24 bits of `ChunkTransformGpu.flags`.
pub(crate) const MAX_MATERIAL_ID: u32 = 0x00FF_FFFF;

/// Handle to a material stored inside [`crate::Scene3DLayer`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct MaterialId(u32);

impl MaterialId {
    /// Built-in default CAD/BIM material (slot `0`, identity tint and standard cutaway shading).
    pub const DEFAULT: Self = Self(0);

    /// Returns the raw zero-based slot index of this material.
    #[must_use]
    pub const fn index(self) -> u32 {
        self.0
    }
}

/// Bitflags controlling material shading behavior in [`Material3D`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct MaterialFlags(u32);

impl MaterialFlags {
    /// Standard directional CAD lighting with slightly darkened interior back-faces.
    pub const NONE: Self = Self(0);
    /// Skip directional lighting and output `base_color_factor * vertex_color + emissive`.
    pub const UNLIT: Self = Self(1 << 0);
    /// Shade back-faces with full front-face brightness instead of multiplying by `backface_tint`.
    pub const DOUBLE_SIDED_UNIFORM: Self = Self(1 << 1);
    /// Mark surface as CAD ghost / subdued translucent material.
    pub const GHOST: Self = Self(1 << 2);

    /// Combines two flag sets.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Returns `true` if all bits in `other` are set in `self`.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Returns the packed GPU flag bits.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }
}

/// Per-chunk CAD/BIM surface material parameters stored in a persistent GPU material table.
///
/// All floating-point fields are sanitized against `NaN`/`Inf` and clamped to safe ranges on
/// GPU upload. [`Material3D::DEFAULT`] matches the built-in vertex-color CAD shading path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material3D {
    /// Linear-space RGBA multiplier applied to the chunk's vertex color (default `[1, 1, 1, 1]`).
    pub base_color_factor: [f32; 4],
    /// Additive linear-space emissive RGB radiance (default `[0, 0, 0]`).
    pub emissive: [f32; 3],
    /// Surface roughness in `[0.04, 1.0]` controlling specular highlight width (default `0.55`).
    pub roughness: f32,
    /// Metallic factor in `[0.0, 1.0]` blending dielectric and metallic specular response (default `0.0`).
    pub metallic: f32,
    /// Opacity multiplier in `[0.0, 1.0]` combined with vertex alpha and `base_color_factor[3]`.
    pub opacity: f32,
    /// Brightness multiplier in `[0.0, 1.0]` for interior back-faces revealed by section planes
    /// (default `0.72`).
    pub backface_tint: f32,
    /// Material shading behavior flags.
    pub flags: MaterialFlags,
}

impl Default for Material3D {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl Material3D {
    /// Default built-in CAD surface material (identity vertex-color tint, matte dielectric).
    pub const DEFAULT: Self = Self {
        base_color_factor: [1.0, 1.0, 1.0, 1.0],
        emissive: [0.0, 0.0, 0.0],
        roughness: 0.55,
        metallic: 0.0,
        opacity: 1.0,
        backface_tint: 0.72,
        flags: MaterialFlags::NONE,
    };

    /// Creates a lit material that tints vertex colors by `rgba` (`[r, g, b, a]`).
    #[must_use]
    pub const fn tinted(rgba: [f32; 4]) -> Self {
        Self {
            base_color_factor: rgba,
            ..Self::DEFAULT
        }
    }

    /// Creates an unlit material with the given RGB factor and optional emissive glow.
    #[must_use]
    pub const fn unlit(rgb: [f32; 3]) -> Self {
        Self {
            base_color_factor: [rgb[0], rgb[1], rgb[2], 1.0],
            flags: MaterialFlags::UNLIT,
            ..Self::DEFAULT
        }
    }

    /// Creates a metallic-roughness CAD material (for brushed metals, anodized parts, glass).
    #[must_use]
    pub const fn surface(base_color_factor: [f32; 4], roughness: f32, metallic: f32) -> Self {
        Self {
            base_color_factor,
            roughness,
            metallic,
            ..Self::DEFAULT
        }
    }

    /// Returns whether this material produces partial transparency (`opacity * alpha < 1.0` or
    /// [`MaterialFlags::GHOST`]).
    #[must_use]
    pub fn is_translucent(&self) -> bool {
        let a = sanitize_unit(self.base_color_factor[3], 1.0);
        let op = sanitize_unit(self.opacity, 1.0);
        (a * op) < 0.999 || self.flags.contains(MaterialFlags::GHOST)
    }

    pub(crate) fn to_gpu(self) -> MaterialGpu {
        let base_color_factor = [
            sanitize_range(self.base_color_factor[0], 0.0, 4.0, 1.0),
            sanitize_range(self.base_color_factor[1], 0.0, 4.0, 1.0),
            sanitize_range(self.base_color_factor[2], 0.0, 4.0, 1.0),
            sanitize_unit(self.base_color_factor[3], 1.0),
        ];
        let emissive_roughness = [
            sanitize_range(self.emissive[0], 0.0, 16.0, 0.0),
            sanitize_range(self.emissive[1], 0.0, 16.0, 0.0),
            sanitize_range(self.emissive[2], 0.0, 16.0, 0.0),
            sanitize_range(self.roughness, 0.04, 1.0, 0.55),
        ];
        let opacity = if self.flags.contains(MaterialFlags::GHOST) {
            sanitize_unit(self.opacity, 1.0).min(0.35)
        } else {
            sanitize_unit(self.opacity, 1.0)
        };
        MaterialGpu {
            base_color_factor,
            emissive_roughness,
            metallic: sanitize_unit(self.metallic, 0.0),
            opacity,
            backface_tint: sanitize_unit(self.backface_tint, 0.72),
            flags: self.flags.bits(),
        }
    }
}

#[inline]
fn sanitize_range(value: f32, min: f32, max: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value.clamp(min, max)
    } else {
        fallback
    }
}

#[inline]
fn sanitize_unit(value: f32, fallback: f32) -> f32 {
    sanitize_range(value, 0.0, 1.0, fallback)
}

/// 48-byte WGSL-aligned GPU material record (`@group(1) @binding(3)` SSBO or `@group(2) @binding(0)`
/// uniform).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct MaterialGpu {
    pub(crate) base_color_factor: [f32; 4],
    pub(crate) emissive_roughness: [f32; 4],
    pub(crate) metallic: f32,
    pub(crate) opacity: f32,
    pub(crate) backface_tint: f32,
    pub(crate) flags: u32,
}

impl Default for MaterialGpu {
    fn default() -> Self {
        Material3D::DEFAULT.to_gpu()
    }
}

/// Bind group layout for direct/fallback per-draw material uniform (`@group(2) @binding(0)`).
pub(crate) fn direct_bind_layout(
    device: &wgpu::Device,
    dynamic_uniforms: bool,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("iletki-3d direct material layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: dynamic_uniforms,
                min_binding_size: NonZeroU64::new(size_of::<MaterialGpu>() as u64),
            },
            count: None,
        }],
    })
}

/// Persistent GPU material registry with lazy zero-tax allocation and dirty-span coalescing.
pub(crate) struct MaterialTable {
    descriptors: Vec<Option<Material3D>>,
    cpu: Vec<MaterialGpu>,
    free_slots: BinaryHeap<Reverse<u32>>,
    active_custom: usize,
    dirty: Vec<usize>,
    dirty_flags: Vec<bool>,
    ssbo: Option<wgpu::Buffer>,
    ssbo_capacity: usize,
    ssbo_generation: u64,
    direct_buf: Option<wgpu::Buffer>,
    direct_stride: u32,
    direct_capacity: usize,
    direct_dynamic: bool,
    direct_dynamic_bg: Option<wgpu::BindGroup>,
    direct_static_bgs: Vec<wgpu::BindGroup>,
    direct_dirty: bool,
}

impl Default for MaterialTable {
    fn default() -> Self {
        Self::new()
    }
}

impl MaterialTable {
    pub(crate) fn new() -> Self {
        Self {
            descriptors: vec![Some(Material3D::DEFAULT)],
            cpu: vec![MaterialGpu::default()],
            free_slots: BinaryHeap::new(),
            active_custom: 0,
            dirty: Vec::new(),
            dirty_flags: vec![false],
            ssbo: None,
            ssbo_capacity: 0,
            ssbo_generation: 0,
            direct_buf: None,
            direct_stride: 0,
            direct_capacity: 0,
            direct_dynamic: false,
            direct_dynamic_bg: None,
            direct_static_bgs: Vec::new(),
            direct_dirty: false,
        }
    }

    #[inline]
    pub(crate) fn has_custom_materials(&self) -> bool {
        self.active_custom > 0
    }

    pub(crate) fn contains(&self, id: MaterialId) -> bool {
        self.descriptors
            .get(id.0 as usize)
            .is_some_and(Option::is_some)
    }

    pub(crate) fn get(&self, id: MaterialId) -> Option<Material3D> {
        self.descriptors.get(id.0 as usize).copied().flatten()
    }

    pub(crate) fn is_translucent(&self, id: MaterialId) -> bool {
        if id == MaterialId::DEFAULT {
            return false;
        }
        self.get(id).is_some_and(|m| m.is_translucent())
    }

    pub(crate) fn insert(&mut self, material: Material3D) -> MaterialId {
        let gpu = material.to_gpu();
        if let Some(Reverse(slot)) = self.free_slots.pop() {
            let idx = slot as usize;
            if let Some(desc) = self.descriptors.get_mut(idx) {
                *desc = Some(material);
            }
            if let Some(raw) = self.cpu.get_mut(idx) {
                *raw = gpu;
            }
            self.mark_dirty(idx);
            self.active_custom = self.active_custom.saturating_add(1);
            return MaterialId(slot);
        }
        if self.cpu.len() > MAX_MATERIAL_ID as usize {
            return MaterialId::DEFAULT;
        }
        let slot = u32::try_from(self.cpu.len()).unwrap_or(0);
        if slot == 0 {
            return MaterialId::DEFAULT;
        }
        self.descriptors.push(Some(material));
        self.cpu.push(gpu);
        self.dirty_flags.push(false);
        self.mark_dirty(slot as usize);
        self.active_custom = self.active_custom.saturating_add(1);
        MaterialId(slot)
    }

    /// Updates an existing non-default material. Returns `Some(translucency_changed)` if valid.
    pub(crate) fn update(&mut self, id: MaterialId, material: Material3D) -> Option<bool> {
        if id == MaterialId::DEFAULT {
            return None;
        }
        let idx = id.0 as usize;
        let prev = self.descriptors.get(idx).copied().flatten()?;
        let prev_translucent = prev.is_translucent();
        let next_translucent = material.is_translucent();
        if let Some(desc) = self.descriptors.get_mut(idx) {
            *desc = Some(material);
        }
        if let Some(raw) = self.cpu.get_mut(idx) {
            *raw = material.to_gpu();
        }
        self.mark_dirty(idx);
        Some(prev_translucent != next_translucent)
    }

    /// Removes a non-default material slot. Returns `Some(was_translucent)` if removed.
    pub(crate) fn remove(&mut self, id: MaterialId) -> Option<bool> {
        if id == MaterialId::DEFAULT {
            return None;
        }
        let idx = id.0 as usize;
        let prev = self.descriptors.get_mut(idx)?.take()?;
        let was_translucent = prev.is_translucent();
        if let Some(raw) = self.cpu.get_mut(idx) {
            *raw = MaterialGpu::default();
        }
        self.mark_dirty(idx);
        self.free_slots.push(Reverse(id.0));
        self.active_custom = self.active_custom.saturating_sub(1);
        Some(was_translucent)
    }

    fn mark_dirty(&mut self, index: usize) {
        self.direct_dirty = true;
        if let Some(flag) = self.dirty_flags.get_mut(index)
            && !*flag
        {
            *flag = true;
            self.dirty.push(index);
        }
    }

    pub(crate) fn ssbo_generation(&self) -> u64 {
        self.ssbo_generation
    }

    pub(crate) fn ssbo(&self) -> Option<&wgpu::Buffer> {
        self.ssbo.as_ref()
    }

    /// Synchronizes only modified material entries into the persistent storage buffer.
    ///
    /// Does not allocate any GPU buffer while every chunk uses [`MaterialId::DEFAULT`].
    pub(crate) fn sync_ssbo(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if !self.has_custom_materials() {
            return;
        }
        let needed = self.cpu.len().max(1);
        let reallocate = self.ssbo.is_none() || self.ssbo_capacity < needed;
        if reallocate {
            let capacity = needed.max(MIN_MATERIAL_CAPACITY).next_power_of_two();
            let byte_size = (capacity as u64).saturating_mul(size_of::<MaterialGpu>() as u64);
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("iletki-3d materials storage"),
                size: byte_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            if !self.cpu.is_empty() {
                queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&self.cpu));
            }
            self.ssbo = Some(buffer);
            self.ssbo_capacity = capacity;
            self.ssbo_generation = self.ssbo_generation.wrapping_add(1);
            for idx in self.dirty.drain(..) {
                if let Some(flag) = self.dirty_flags.get_mut(idx) {
                    *flag = false;
                }
            }
            return;
        }

        if self.dirty.is_empty() {
            return;
        }
        let Some(buffer) = &self.ssbo else {
            return;
        };
        self.dirty.sort_unstable();
        let stride = size_of::<MaterialGpu>() as u64;
        let mut span_start: Option<usize> = None;
        let mut span_end = 0usize;
        let flush_span = |start: usize, end: usize| {
            if let Some(slice) = self.cpu.get(start..end) {
                queue.write_buffer(buffer, (start as u64) * stride, bytemuck::cast_slice(slice));
            }
        };
        for idx in self.dirty.drain(..) {
            if let Some(flag) = self.dirty_flags.get_mut(idx) {
                *flag = false;
            }
            match span_start {
                Some(_) if idx <= span_end.saturating_add(4) => {
                    span_end = idx + 1;
                }
                Some(start) => {
                    flush_span(start, span_end);
                    span_start = Some(idx);
                    span_end = idx + 1;
                }
                None => {
                    span_start = Some(idx);
                    span_end = idx + 1;
                }
            }
        }
        if let Some(start) = span_start {
            flush_span(start, span_end);
        }
    }

    /// Synchronizes an aligned uniform mirror for direct draws and OpenGL/WebGL2 fallback.
    pub(crate) fn sync_direct(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        dynamic_uniforms: bool,
    ) {
        if !self.has_custom_materials() {
            return;
        }
        let align = device.limits().min_uniform_buffer_offset_alignment.max(16) as usize;
        let stride = size_of::<MaterialGpu>().next_multiple_of(align);
        let needed = self.cpu.len().max(1);
        let reallocate = self.direct_buf.is_none()
            || self.direct_capacity < needed
            || self.direct_stride as usize != stride
            || self.direct_dynamic != dynamic_uniforms;

        if reallocate {
            let capacity = needed.max(MIN_MATERIAL_CAPACITY).next_power_of_two();
            let byte_size = (capacity as u64).saturating_mul(stride as u64);
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("iletki-3d direct materials uniform"),
                size: byte_size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let binding_size = NonZeroU64::new(size_of::<MaterialGpu>() as u64);
            if dynamic_uniforms {
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("iletki-3d direct materials dynamic bg"),
                    layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &buf,
                            offset: 0,
                            size: binding_size,
                        }),
                    }],
                });
                self.direct_dynamic_bg = Some(bg);
                self.direct_static_bgs.clear();
            } else {
                self.direct_dynamic_bg = None;
                self.direct_static_bgs = (0..capacity)
                    .map(|i| {
                        device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("iletki-3d direct material bg"),
                            layout,
                            entries: &[wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &buf,
                                    offset: (i as u64).saturating_mul(stride as u64),
                                    size: binding_size,
                                }),
                            }],
                        })
                    })
                    .collect();
            }
            self.direct_buf = Some(buf);
            self.direct_stride = u32::try_from(stride).unwrap_or(256);
            self.direct_capacity = capacity;
            self.direct_dynamic = dynamic_uniforms;
            self.direct_dirty = true;
        }

        if self.direct_dirty
            && let Some(buf) = &self.direct_buf
        {
            let total = self.cpu.len().saturating_mul(stride);
            let mut staging = vec![0u8; total];
            let entry_bytes = size_of::<MaterialGpu>();
            for (i, mat) in self.cpu.iter().enumerate() {
                let start = i.saturating_mul(stride);
                if let Some(dst) = staging.get_mut(start..start.saturating_add(entry_bytes)) {
                    dst.copy_from_slice(bytemuck::bytes_of(mat));
                }
            }
            if !staging.is_empty() {
                queue.write_buffer(buf, 0, &staging);
            }
            self.direct_dirty = false;
        }
    }

    /// Binds `@group(2)` for a direct-draw material slot, returning `false` if unavailable.
    pub(crate) fn bind_direct(&self, pass: &mut wgpu::RenderPass<'_>, id: MaterialId) -> bool {
        let slot = if self.contains(id) { id.0 as usize } else { 0 };
        if self.direct_dynamic {
            let Some(bg) = &self.direct_dynamic_bg else {
                return false;
            };
            let Some(offset) = u32::try_from(slot)
                .ok()
                .and_then(|s| s.checked_mul(self.direct_stride))
            else {
                return false;
            };
            pass.set_bind_group(2, bg, &[offset]);
            true
        } else {
            let Some(bg) = self.direct_static_bgs.get(slot) else {
                return false;
            };
            pass.set_bind_group(2, bg, &[]);
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_gpu_layout_is_48_bytes_and_sanitizes_hostile_inputs() {
        assert_eq!(size_of::<MaterialGpu>(), 48);
        let hostile = Material3D {
            base_color_factor: [f32::NAN, f32::INFINITY, -2.0, f32::NEG_INFINITY],
            emissive: [f32::NAN, 99.0, -5.0],
            roughness: f32::NAN,
            metallic: 5.0,
            opacity: -1.0,
            backface_tint: f32::INFINITY,
            flags: MaterialFlags::NONE,
        };
        let gpu = hostile.to_gpu();
        assert_eq!(gpu.base_color_factor, [1.0, 1.0, 0.0, 1.0]);
        assert_eq!(gpu.emissive_roughness, [0.0, 16.0, 0.0, 0.55]);
        assert_eq!(gpu.metallic, 1.0);
        assert_eq!(gpu.opacity, 0.0);
        assert_eq!(gpu.backface_tint, 0.72);
    }

    #[test]
    fn material_table_preserves_default_slot_and_reuses_freed_slots() {
        let mut table = MaterialTable::new();
        assert!(!table.has_custom_materials());
        assert!(table.contains(MaterialId::DEFAULT));
        assert!(table.remove(MaterialId::DEFAULT).is_none());
        assert!(
            table
                .update(MaterialId::DEFAULT, Material3D::DEFAULT)
                .is_none()
        );

        let m1 = table.insert(Material3D::tinted([1.0, 0.0, 0.0, 1.0]));
        let m2 = table.insert(Material3D::tinted([0.0, 1.0, 0.0, 0.5]));
        assert_eq!(m1.index(), 1);
        assert_eq!(m2.index(), 2);
        assert!(table.has_custom_materials());
        assert!(!table.is_translucent(m1));
        assert!(table.is_translucent(m2));

        assert_eq!(table.remove(m1), Some(false));
        assert!(!table.contains(m1));
        let m3 = table.insert(Material3D::unlit([0.2, 0.4, 0.8]));
        assert_eq!(m3.index(), 1);

        assert_eq!(
            table.update(m2, Material3D::tinted([0.0, 1.0, 0.0, 1.0])),
            Some(true)
        );
        assert!(!table.is_translucent(m2));
    }
}
