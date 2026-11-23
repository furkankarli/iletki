@vertex
fn vs_inst_sampled(v: VertIn, inst: InstIn) -> SampledColorOut {
    return sampled_color(instanced_vertex(v, inst));
}

@fragment
fn fs_inst_sampled(in: SampledColorOut, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    if (iletki_is_clipped(in.sample_pos_rel)) {
        discard;
    }
    return shade_instanced(in.pos_rel, in.normal, in.color, is_front);
}
