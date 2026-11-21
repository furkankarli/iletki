//! Viewport execution modes, dirty tracking, draw phases, upload budgets and picking IDs.

use crate::math::Vec3;

/// Unique identifier for a viewport instance, used by shared scenes to isolate per-viewport GPU
/// state (e.g. camera-relative transforms in 4-way quad views).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct ViewportId(pub u64);

/// Visual configuration for the Jump-Flooding (JFA) screen-space selection outline pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OutlineConfig {
    /// Outline RGBA color in `[0.0, 1.0]` (default CAD selection orange).
    pub color: [f32; 4],
    /// Outline thickness in physical pixels (clamped to `1.0..=15.0`, default `2.5`).
    pub width_px: f32,
}

impl Default for OutlineConfig {
    fn default() -> Self {
        Self {
            color: [1.0, 0.55, 0.08, 1.0],
            width_px: 2.5,
        }
    }
}

/// Scheduling mode of a viewport.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum ViewportMode {
    /// CAD / editor mode: caches the static scene offscreen and skips redrawing it when only
    /// overlays or gizmos change.
    #[default]
    OnDemand,
    /// Game / animation mode: renders every frame directly without maintaining a static cache
    /// copy.
    Continuous,
}

/// Bitmask tracking which parts of a viewport need work on the next frame.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct DirtyFlags(u32);

impl DirtyFlags {
    /// No work needed.
    pub const NONE: Self = Self(0);
    /// Static/base scene geometry or materials changed.
    pub const STATIC_SCENE: Self = Self(1 << 0);
    /// Selection / highlight set changed.
    pub const SELECTION: Self = Self(1 << 1);
    /// 3D gizmos or helpers changed.
    pub const GIZMO: Self = Self(1 << 2);
    /// 2D screen-space overlays (crosshair, snaps, marquee, rubberband) changed.
    pub const OVERLAY: Self = Self(1 << 3);
    /// Camera transform, zoom, projection or clip planes changed.
    pub const CAMERA: Self = Self(1 << 4);
    /// Viewport pixel size or DPI scale changed.
    pub const SIZE: Self = Self(1 << 5);
    /// Everything must be rebuilt and redrawn.
    pub const ALL: Self = Self(0x3F);

    /// Returns `true` if no flags are set.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns `true` if all bits in `other` are set in `self`.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Returns `true` if any bit in `other` is set in `self`.
    #[must_use]
    pub const fn intersects(self, other: Self) -> bool {
        (self.0 & other.0) != 0
    }

    /// Returns the union of two flag sets.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Inserts `other` bits into `self`.
    pub const fn insert(&mut self, other: Self) {
        self.0 |= other.0;
    }

    /// Removes `other` bits from `self`.
    pub const fn remove(&mut self, other: Self) {
        self.0 &= !other.0;
    }

    /// Returns `true` if the offscreen static scene cache must be re-rendered.
    #[must_use]
    pub const fn needs_static_rebuild(self) -> bool {
        self.intersects(Self::STATIC_SCENE.union(Self::CAMERA).union(Self::SIZE))
    }

    /// Returns `true` if any dynamic post-static layer must be drawn.
    #[must_use]
    pub const fn needs_dynamic_pass(self) -> bool {
        self.intersects(Self::SELECTION.union(Self::GIZMO).union(Self::OVERLAY))
    }
}

/// Ordered rendering phases inside a viewport frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DrawPhase {
    /// Solid 3D geometry and opaque 2D fills/strokes (depth write enabled).
    Opaque = 0,
    /// Alpha-blended surfaces and dense translucent hatches (depth test on, depth write off).
    Transparent = 1,
    /// Single-channel mask of selected/hovered items for Jump-Flood outline post-processing.
    OutlineMask = 2,
    /// 3D transform gizmos, infinite grids and ViewCubes (depth cleared before pass).
    Gizmo = 3,
    /// Crisp screen-space 2D overlays (crosshair, snap markers, grips, selection box).
    Overlay = 4,
    /// On-demand 64-bit ID + depth picking pass.
    Picking = 5,
}

impl DrawPhase {
    /// Returns the single-bit mask corresponding to this phase.
    #[must_use]
    pub const fn mask(self) -> DrawPhaseMask {
        DrawPhaseMask(1 << (self as u8))
    }
}

/// Bitmask of [`DrawPhase`]s that a layer participates in.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct DrawPhaseMask(u8);

impl DrawPhaseMask {
    /// Participates in no phases.
    pub const NONE: Self = Self(0);
    /// Opaque phase bit.
    pub const OPAQUE: Self = DrawPhase::Opaque.mask();
    /// Transparent phase bit.
    pub const TRANSPARENT: Self = DrawPhase::Transparent.mask();
    /// Outline mask phase bit.
    pub const OUTLINE_MASK: Self = DrawPhase::OutlineMask.mask();
    /// Gizmo phase bit.
    pub const GIZMO: Self = DrawPhase::Gizmo.mask();
    /// Overlay phase bit.
    pub const OVERLAY: Self = DrawPhase::Overlay.mask();
    /// Picking phase bit.
    pub const PICKING: Self = DrawPhase::Picking.mask();
    /// Every draw phase.
    pub const ALL: Self = Self(0x3F);

    /// Combines two phase masks.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Keeps only the phases present in both masks.
    #[must_use]
    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Returns `true` if `phase` is enabled in this mask.
    #[must_use]
    pub const fn contains(self, phase: DrawPhase) -> bool {
        (self.0 & phase.mask().0) != 0
    }
}

/// Per-frame GPU staging budget for progressive streaming of large scenes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UploadBudget {
    /// Maximum bytes uploaded to the GPU in a single frame.
    pub max_bytes_per_frame: usize,
    /// Soft CPU time cap (milliseconds) spent preparing uploads in a single frame.
    pub max_time_ms: f32,
}

impl Default for UploadBudget {
    fn default() -> Self {
        Self {
            max_bytes_per_frame: 8 * 1024 * 1024,
            max_time_ms: 4.0,
        }
    }
}

impl UploadBudget {
    /// Effectively unbounded upload budget (useful for headless batch rendering).
    pub const UNLIMITED: Self = Self {
        max_bytes_per_frame: usize::MAX,
        max_time_ms: f32::INFINITY,
    };
}

/// 64-bit sub-object picking identifier packed into an `Rg32Uint` render target.
///
/// A value of all zeros (`PickId::NONE`) represents empty background.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct PickId {
    /// Layer identifier (e.g. scene, gizmo, annotation).
    pub layer: u16,
    /// Entity or object identifier within the layer.
    pub object: u32,
    /// Sub-object identifier (face, edge, vertex, polyline segment, or gizmo handle).
    pub sub: u16,
}

impl PickId {
    /// Sentinel value representing no hit (background pixel).
    pub const NONE: Self = Self {
        layer: 0,
        object: 0,
        sub: 0,
    };

    /// Creates a new picking identifier.
    #[must_use]
    pub const fn new(layer: u16, object: u32, sub: u16) -> Self {
        Self { layer, object, sub }
    }

    /// Returns `true` if this ID represents empty background.
    #[must_use]
    pub const fn is_none(self) -> bool {
        self.layer == 0 && self.object == 0 && self.sub == 0
    }

    /// Packs into `[u32; 2]` for a `wgpu::TextureFormat::Rg32Uint` target.
    #[must_use]
    pub const fn to_rg32u(self) -> [u32; 2] {
        let meta = ((self.layer as u32) << 16) | (self.sub as u32);
        [self.object, meta]
    }

    /// Unpacks from `[u32; 2]` read back from an `Rg32Uint` target.
    #[must_use]
    pub const fn from_rg32u([object, meta]: [u32; 2]) -> Self {
        Self {
            layer: (meta >> 16) as u16,
            object,
            sub: (meta & 0xFFFF) as u16,
        }
    }

    /// Packs into a single `u64`.
    #[must_use]
    pub const fn to_u64(self) -> u64 {
        let [lo, hi] = self.to_rg32u();
        ((hi as u64) << 32) | (lo as u64)
    }

    /// Unpacks from a single `u64`.
    #[must_use]
    pub const fn from_u64(v: u64) -> Self {
        Self::from_rg32u([v as u32, (v >> 32) as u32])
    }
}

/// Result of a GPU or hybrid picking query at a pixel or region.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PickHit {
    /// Identifier of the hit object and sub-element.
    pub id: PickId,
    /// Reversed-Z depth value in `[0.0, 1.0]` (`1.0` at near plane, `0.0` at far/background).
    pub depth: f32,
    /// Exact world-space position reconstructed from depth and inverse camera projection.
    pub world_pos: Option<Vec3>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pick_id_round_trips_rg32u_and_u64() {
        let id = PickId::new(0x1234, 0xDEAD_BEEF, 0x5678);
        assert!(!id.is_none());
        assert_eq!(PickId::from_rg32u(id.to_rg32u()), id);
        assert_eq!(PickId::from_u64(id.to_u64()), id);
        assert!(PickId::NONE.is_none());
    }

    #[test]
    fn dirty_flags_distinguish_static_and_overlay() {
        let mut flags = DirtyFlags::NONE;
        flags.insert(DirtyFlags::OVERLAY);
        assert!(!flags.needs_static_rebuild());
        assert!(flags.needs_dynamic_pass());

        flags.insert(DirtyFlags::CAMERA);
        assert!(flags.needs_static_rebuild());
    }
}
