// Storage-buffer MRT color-frame selection + persistent MaterialGpu SSBO shaders.
struct StorageMatSelectOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) pos_rel: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) @interpolate(flat) slot_id: u32,
    @location(4) @interpolate(flat) mat_id: u32,
};

@vertex
fn vs_mesh_storage_mat_select(v: VertIn, @location(3) slot: u32) -> StorageMatSelectOut {
    let st = load_storage_transform(slot);
    let base = mesh_vertex_with_object(v, st.placement);
    var out = StorageMatSelectOut(
        base.clip_pos,
        base.pos_rel,
        base.normal,
        base.color,
        select(0u, slot + 1u, st.pickable),
        load_material_id(slot),
    );
    if (!st.visible) {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    }
    return out;
}

@fragment
fn fs_lit_storage_mat_select(
    in: StorageMatSelectOut,
    @builtin(front_facing) is_front: bool,
) -> LitSelectOut {
    if (iletki_is_clipped(in.pos_rel)) {
        discard;
    }
    return LitSelectOut(
        shade_storage_material(in.pos_rel, in.normal, in.color, is_front, in.mat_id),
        pack_slot_id(in.slot_id),
    );
}

@fragment
fn fs_lit_storage_mat_select_unclipped(
    in: StorageMatSelectOut,
    @builtin(front_facing) is_front: bool,
) -> LitSelectOut {
    return LitSelectOut(
        shade_storage_material(in.pos_rel, in.normal, in.color, is_front, in.mat_id),
        pack_slot_id(in.slot_id),
    );
}
