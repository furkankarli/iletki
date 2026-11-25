// Sample-rate clipped MRT color-frame selection shader extension.
struct SampledColorSelectOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) pos_rel: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) @interpolate(perspective, sample) sample_pos_rel: vec3<f32>,
    @location(4) @interpolate(flat) slot_id: u32,
};

fn sampled_color_select(out: ColorOut, slot_id: u32) -> SampledColorSelectOut {
    return SampledColorSelectOut(
        out.clip_pos,
        out.pos_rel,
        out.normal,
        out.color,
        out.pos_rel,
        slot_id,
    );
}

@vertex
fn vs_mesh_select_sampled(v: VertIn, @builtin(instance_index) slot_id: u32) -> SampledColorSelectOut {
    return sampled_color_select(mesh_vertex(v), slot_id);
}

@vertex
fn vs_edge_select_sampled(v: VertIn) -> SampledColorSelectOut {
    return sampled_color_select(edge_vertex(v), 0u);
}

@fragment
fn fs_lit_select_sampled(
    in: SampledColorSelectOut,
    @builtin(front_facing) is_front: bool,
) -> LitSelectOut {
    if (iletki_is_clipped(in.sample_pos_rel)) {
        discard;
    }
    return LitSelectOut(
        shade_mesh(in.pos_rel, in.normal, in.color, is_front),
        pack_slot_id(in.slot_id),
    );
}

@fragment
fn fs_edge_select_sampled(in: SampledColorSelectOut) -> LitSelectOut {
    if (iletki_is_clipped(in.sample_pos_rel)) {
        discard;
    }
    return LitSelectOut(
        vec4<f32>(iletki_color_to_target(in.color.rgb), in.color.a),
        vec2<u32>(0u, 0u),
    );
}
