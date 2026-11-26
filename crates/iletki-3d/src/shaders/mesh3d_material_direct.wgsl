// Direct/fallback per-draw material fragment entry points (@group(2) @binding(0)).
@group(2) @binding(0)
var<uniform> direct_material: MaterialGpu;

@fragment
fn fs_lit_mat(in: ColorOut, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    if (iletki_is_clipped(in.pos_rel)) {
        discard;
    }
    return shade_mesh_material(in.pos_rel, in.normal, in.color, is_front, direct_material);
}

@fragment
fn fs_lit_mat_unclipped(
    in: ColorOut,
    @builtin(front_facing) is_front: bool,
) -> @location(0) vec4<f32> {
    return shade_mesh_material(in.pos_rel, in.normal, in.color, is_front, direct_material);
}
