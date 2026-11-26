// Direct/fallback per-draw sampled-section MRT selection material fragment entry point.
@fragment
fn fs_lit_select_mat_sampled(
    in: SampledColorSelectOut,
    @builtin(front_facing) is_front: bool,
) -> LitSelectOut {
    if (iletki_is_clipped(in.sample_pos_rel)) {
        discard;
    }
    return LitSelectOut(
        shade_mesh_material(in.pos_rel, in.normal, in.color, is_front, direct_material),
        pack_slot_id(in.slot_id),
    );
}
