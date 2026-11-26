// Direct/fallback per-draw sampled-section material fragment entry point.
@fragment
fn fs_lit_mat_sampled(
    in: SampledColorOut,
    @builtin(front_facing) is_front: bool,
) -> @location(0) vec4<f32> {
    if (iletki_is_clipped(in.sample_pos_rel)) {
        discard;
    }
    return shade_mesh_material(in.pos_rel, in.normal, in.color, is_front, direct_material);
}
