// Sample-rate clipped storage-buffer material shaders.
struct SampledStorageMatOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) pos_rel: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) @interpolate(perspective, sample) sample_pos_rel: vec3<f32>,
    @location(4) @interpolate(flat) mat_id: u32,
};

@vertex
fn vs_mesh_storage_mat_sampled(v: VertIn, @location(3) slot: u32) -> SampledStorageMatOut {
    let st = load_storage_transform(slot);
    let base = mesh_vertex_with_object(v, st.placement);
    var out = SampledStorageMatOut(
        base.clip_pos,
        base.pos_rel,
        base.normal,
        base.color,
        base.pos_rel,
        load_material_id(slot),
    );
    if (!st.visible) {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    }
    return out;
}

@fragment
fn fs_lit_storage_mat_sampled(
    in: SampledStorageMatOut,
    @builtin(front_facing) is_front: bool,
) -> @location(0) vec4<f32> {
    if (iletki_is_clipped(in.sample_pos_rel)) {
        discard;
    }
    return shade_storage_material(in.pos_rel, in.normal, in.color, is_front, in.mat_id);
}
