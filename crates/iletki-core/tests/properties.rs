//! Property-based tests for `iletki-core` numerical invariants and hostile inputs.

use iletki_core::{BLOCK_ALIGN_PX, BlockSizing, Camera, MAX_TARGET_DIM, PickId, Vec2, split};
use proptest::prelude::*;

proptest! {
    #[test]
    fn split_reconstructs_within_double_f32_precision(v in -1e9_f64..1e9_f64) {
        let (hi, lo) = split(v);
        let reconstructed = f64::from(hi) + f64::from(lo);
        let tol = (v.abs() * 1e-13).max(1e-12);
        prop_assert!((reconstructed - v).abs() <= tol);
    }

    #[test]
    fn screen_world_2d_round_trip(
        cx in -1e7_f64..1e7_f64,
        cy in -1e7_f64..1e7_f64,
        zoom in 1e-3_f64..1e4_f64,
        sx in 0.0_f64..4000.0_f64,
        sy in 0.0_f64..4000.0_f64,
    ) {
        let size = Vec2::new(1920.0, 1080.0);
        let cam = Camera::ortho_2d(Vec2::new(cx, cy), zoom);
        let screen = Vec2::new(sx, sy);
        let world = cam.screen_to_world_2d(screen, size).unwrap();
        let back = cam.world_to_screen_2d(world, size).unwrap();
        prop_assert!(back.approx_eq(screen, 1e-5));
    }

    #[test]
    fn pick_id_bit_packing_is_lossless(layer in any::<u16>(), object in any::<u32>(), sub in any::<u16>()) {
        let id = PickId::new(layer, object, sub);
        prop_assert_eq!(PickId::from_rg32u(id.to_rg32u()), id);
        prop_assert_eq!(PickId::from_u64(id.to_u64()), id);
    }

    #[test]
    fn block_sizing_always_aligned_and_bounded(w in any::<u32>(), h in any::<u32>()) {
        let sizing = BlockSizing::new([w, h]);
        let cur = sizing.current();
        prop_assert!(cur.active_px[0] >= 1 && cur.active_px[0] <= MAX_TARGET_DIM);
        prop_assert!(cur.active_px[1] >= 1 && cur.active_px[1] <= MAX_TARGET_DIM);
        prop_assert!(cur.allocated_px[0] >= cur.active_px[0]);
        prop_assert!(cur.allocated_px[1] >= cur.active_px[1]);
        prop_assert_eq!(cur.allocated_px[0] % BLOCK_ALIGN_PX, 0);
        prop_assert_eq!(cur.allocated_px[1] % BLOCK_ALIGN_PX, 0);
        prop_assert!(cur.uv_max[0] > 0.0 && cur.uv_max[0] <= 1.0);
        prop_assert!(cur.uv_max[1] > 0.0 && cur.uv_max[1] <= 1.0);
    }
}
