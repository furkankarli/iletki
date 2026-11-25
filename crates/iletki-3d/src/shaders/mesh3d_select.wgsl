// MRT color-frame selection preview outputs (@location(0) color, @location(1) 1-based slot ID).
struct ColorSelectOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) pos_rel: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) @interpolate(flat) slot_id: u32,
};

struct LitSelectOut {
    @location(0) color: vec4<f32>,
    @location(1) slot_id: vec2<u32>,
};

fn pack_slot_id(slot_id: u32) -> vec2<u32> {
    return vec2<u32>(slot_id & 0xFFFFu, slot_id >> 16u);
}

fn color_with_slot(base: ColorOut, slot_id: u32) -> ColorSelectOut {
    return ColorSelectOut(base.clip_pos, base.pos_rel, base.normal, base.color, slot_id);
}

@vertex
fn vs_mesh_select(v: VertIn, @builtin(instance_index) slot_id: u32) -> ColorSelectOut {
    return color_with_slot(mesh_vertex(v), slot_id);
}

@vertex
fn vs_edge_select(v: VertIn) -> ColorSelectOut {
    return color_with_slot(edge_vertex(v), 0u);
}

@fragment
fn fs_lit_select(in: ColorSelectOut, @builtin(front_facing) is_front: bool) -> LitSelectOut {
    if (iletki_is_clipped(in.pos_rel)) {
        discard;
    }
    return LitSelectOut(
        shade_mesh(in.pos_rel, in.normal, in.color, is_front),
        pack_slot_id(in.slot_id),
    );
}

@fragment
fn fs_lit_select_unclipped(
    in: ColorSelectOut,
    @builtin(front_facing) is_front: bool,
) -> LitSelectOut {
    return LitSelectOut(
        shade_mesh(in.pos_rel, in.normal, in.color, is_front),
        pack_slot_id(in.slot_id),
    );
}

@fragment
fn fs_edge_select(in: ColorSelectOut) -> LitSelectOut {
    if (iletki_is_clipped(in.pos_rel)) {
        discard;
    }
    return LitSelectOut(
        vec4<f32>(iletki_color_to_target(in.color.rgb), in.color.a),
        vec2<u32>(0u, 0u),
    );
}

@fragment
fn fs_unlit_select(in: ColorSelectOut) -> LitSelectOut {
    return LitSelectOut(
        vec4<f32>(iletki_color_to_target(in.color.rgb), in.color.a),
        vec2<u32>(0u, 0u),
    );
}
