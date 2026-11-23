@vertex
fn vs_mesh_sampled(v: VertIn) -> SampledColorOut {
    return sampled_color(mesh_vertex(v));
}

@vertex
fn vs_edge_sampled(v: VertIn) -> SampledColorOut {
    return sampled_color(edge_vertex(v));
}

@fragment
fn fs_lit_sampled(in: SampledColorOut, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    if (iletki_is_clipped(in.sample_pos_rel)) {
        discard;
    }
    return shade_mesh(in.pos_rel, in.normal, in.color, is_front);
}

@fragment
fn fs_edge_sampled(in: SampledColorOut) -> @location(0) vec4<f32> {
    if (iletki_is_clipped(in.sample_pos_rel)) {
        discard;
    }
    return vec4<f32>(iletki_color_to_target(in.color.rgb), in.color.a);
}

@fragment
fn fs_mask_sampled(in: SampledColorOut) -> @location(0) f32 {
    if (iletki_is_clipped(in.sample_pos_rel)) {
        discard;
    }
    return 1.0;
}
