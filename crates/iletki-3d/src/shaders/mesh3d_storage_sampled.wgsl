@vertex
fn vs_mesh_storage_sampled(v: VertIn, @location(3) slot: u32) -> SampledColorOut {
    let st = load_storage_transform(slot);
    var out = sampled_color(mesh_vertex_with_object(v, st.placement));
    if (!st.visible) {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    }
    return out;
}

@vertex
fn vs_edge_storage_sampled(v: VertIn, @location(3) slot: u32) -> SampledColorOut {
    let st = load_storage_transform(slot);
    var out = sampled_color(edge_vertex_with_object(v, st.placement));
    if (!st.visible) {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    }
    return out;
}

struct SampledMaskOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(3) @interpolate(perspective, sample) sample_pos_rel: vec3<f32>,
};

@vertex
fn vs_mask_storage_sampled(@location(0) position: vec3<f32>, @location(3) slot: u32) -> SampledMaskOut {
    let st = load_storage_transform(slot);
    let m = mask_vertex_pos(position, st.placement);
    var out = SampledMaskOut(m.clip_pos, m.pos_rel);
    if (!st.visible) {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    }
    return out;
}

@fragment
fn fs_mask_sampled_pos(in: SampledMaskOut) -> @location(0) f32 {
    if (iletki_is_clipped(in.sample_pos_rel)) {
        discard;
    }
    return 1.0;
}
