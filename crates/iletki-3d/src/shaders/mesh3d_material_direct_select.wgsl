// Direct/fallback per-draw MRT selection material fragment entry points.
@fragment
fn fs_lit_select_mat(in: ColorSelectOut, @builtin(front_facing) is_front: bool) -> LitSelectOut {
    if (iletki_is_clipped(in.pos_rel)) {
        discard;
    }
    return LitSelectOut(
        shade_mesh_material(in.pos_rel, in.normal, in.color, is_front, direct_material),
        pack_slot_id(in.slot_id),
    );
}

@fragment
fn fs_lit_select_mat_unclipped(
    in: ColorSelectOut,
    @builtin(front_facing) is_front: bool,
) -> LitSelectOut {
    return LitSelectOut(
        shade_mesh_material(in.pos_rel, in.normal, in.color, is_front, direct_material),
        pack_slot_id(in.slot_id),
    );
}
