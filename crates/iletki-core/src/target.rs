//! Over-allocated 256-pixel block sizing for stutter-free viewport resizing.
//!
//! Dragging a window or dock splitter changes pixel dimensions every frame. Reallocating MSAA
//! color and depth textures on every pixel change causes visible stutter. [`BlockSizing`] rounds
//! allocated texture extents up to multiples of [`BLOCK_ALIGN_PX`] (256 px), crops rendering and
//! compositing to the active sub-rectangle via `uv_max`, and waits for a shrink hysteresis window
//! before releasing excess memory.

/// Allocation granularity in physical pixels.
pub const BLOCK_ALIGN_PX: u32 = 256;
/// Safety cap on a single target dimension (protects against hostile or buggy resize events).
pub const MAX_TARGET_DIM: u32 = 16_384;
/// Idle/smaller frames to wait before shrinking an over-allocated texture.
const SHRINK_HYSTERESIS_FRAMES: u32 = 60;

/// Active and allocated pixel extents of a viewport render target.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TargetSize {
    /// Active viewport extent in physical pixels (always `>= 1`).
    pub active_px: [u32; 2],
    /// Backing GPU texture extent in physical pixels (block-aligned up to the configured limit).
    pub allocated_px: [u32; 2],
    /// Normalized texture coordinate ceiling `(active / allocated)` for zero-copy UV cropping.
    pub uv_max: [f32; 2],
}

/// Stateful block-aligned resize policy for a GPU viewport target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockSizing {
    active: [u32; 2],
    allocated: [u32; 2],
    max_dimension: u32,
    shrink_countdown: u32,
}

impl Default for BlockSizing {
    fn default() -> Self {
        Self::new([1, 1])
    }
}

impl BlockSizing {
    /// Creates a new sizing tracker initialized for `initial_px`.
    #[must_use]
    pub fn new(initial_px: [u32; 2]) -> Self {
        Self::with_max_dimension(initial_px, MAX_TARGET_DIM)
    }

    /// Creates a sizing tracker capped at a GPU's maximum texture dimension.
    ///
    /// A limit smaller than a block or between blocks takes precedence over alignment.
    #[must_use]
    pub fn with_max_dimension(initial_px: [u32; 2], max_dimension: u32) -> Self {
        let max_dimension = max_dimension.clamp(1, MAX_TARGET_DIM);
        let active = initial_px.map(|px| px.clamp(1, max_dimension));
        let allocated = active.map(|px| align_up(px, max_dimension));
        Self {
            active,
            allocated,
            max_dimension,
            shrink_countdown: 0,
        }
    }

    /// Returns the current [`TargetSize`] snapshot.
    #[must_use]
    pub fn current(&self) -> TargetSize {
        let [aw, ah] = self.active;
        let [bw, bh] = self.allocated;
        TargetSize {
            active_px: self.active,
            allocated_px: self.allocated,
            uv_max: [aw as f32 / bw as f32, ah as f32 / bh as f32],
        }
    }

    /// Updates the desired active physical size.
    ///
    /// Returns `true` if the backing GPU texture must be reallocated.
    pub fn request(&mut self, requested_px: [u32; 2]) -> bool {
        self.active = requested_px.map(|px| px.clamp(1, self.max_dimension));
        let [nw, nh] = self.active.map(|px| align_up(px, self.max_dimension));
        let [bw, bh] = self.allocated;

        if nw > bw || nh > bh {
            self.allocated = [nw.max(bw), nh.max(bh)];
            self.shrink_countdown = 0;
            return true;
        }

        if nw < bw || nh < bh {
            self.shrink_countdown = self.shrink_countdown.saturating_add(1);
            if self.shrink_countdown >= SHRINK_HYSTERESIS_FRAMES {
                self.allocated = [nw, nh];
                self.shrink_countdown = 0;
                return true;
            }
        } else {
            self.shrink_countdown = 0;
        }
        false
    }
}

fn align_up(px: u32, max_dimension: u32) -> u32 {
    (px.div_ceil(BLOCK_ALIGN_PX) * BLOCK_ALIGN_PX).min(max_dimension)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_resizes_inside_block_never_reallocate() {
        let mut sizing = BlockSizing::new([800, 600]);
        assert_eq!(sizing.current().allocated_px, [1024, 768]);

        for w in 801..=1024 {
            assert!(!sizing.request([w, 600]));
        }
        assert_eq!(sizing.current().allocated_px, [1024, 768]);

        // Crossing 1024 triggers a single grow reallocation.
        assert!(sizing.request([1025, 600]));
        assert_eq!(sizing.current().allocated_px, [1280, 768]);
    }

    #[test]
    fn shrink_waits_for_hysteresis_then_reclaims() {
        let mut sizing = BlockSizing::new([1200, 900]);
        assert_eq!(sizing.current().allocated_px, [1280, 1024]);

        for _ in 0..(SHRINK_HYSTERESIS_FRAMES - 1) {
            assert!(!sizing.request([500, 400]));
        }
        assert!(sizing.request([500, 400]));
        assert_eq!(sizing.current().allocated_px, [512, 512]);
        let uv = sizing.current().uv_max;
        assert!((uv[0] - 500.0 / 512.0).abs() < 1e-6);
        assert!((uv[1] - 400.0 / 512.0).abs() < 1e-6);
    }

    #[test]
    fn zero_and_huge_sizes_are_clamped_safely() {
        let mut sizing = BlockSizing::new([0, 0]);
        assert_eq!(sizing.current().active_px, [1, 1]);
        assert_eq!(sizing.current().allocated_px, [256, 256]);

        assert!(sizing.request([u32::MAX, u32::MAX]));
        assert_eq!(sizing.current().active_px, [MAX_TARGET_DIM, MAX_TARGET_DIM]);
        assert_eq!(
            sizing.current().allocated_px,
            [MAX_TARGET_DIM, MAX_TARGET_DIM]
        );
    }

    #[test]
    fn device_limit_takes_precedence_over_block_alignment() {
        let mut sizing = BlockSizing::with_max_dimension([1, 5], 500);
        assert!(sizing.request([501, 5]));
        assert_eq!(sizing.current().active_px, [500, 5]);
        assert_eq!(sizing.current().allocated_px, [500, 256]);
        assert!(!sizing.request([u32::MAX, 5]));
        assert_eq!(
            BlockSizing::with_max_dimension([u32::MAX, 1], 64)
                .current()
                .allocated_px,
            [64, 64],
        );
    }
}
