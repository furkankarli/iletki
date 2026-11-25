// Hardware raster-clipped MRT color-frame selection shader extension.
struct RasterClippedColorSelectOut {
    @builtin(position) clip_pos: vec4<f32>,
    @builtin(clip_distances) plane_distances: array<f32, 6>,
    @location(0) pos_rel: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) @interpolate(flat) slot_id: u32,
};

fn raster_clipped_color_select(out: ColorOut, slot_id: u32) -> RasterClippedColorSelectOut {
    let base = raster_clipped_color(out);
    return RasterClippedColorSelectOut(
        base.clip_pos,
        base.plane_distances,
        base.pos_rel,
        base.normal,
        base.color,
        slot_id,
    );
}

@vertex
fn vs_mesh_select_raster(v: VertIn, @builtin(instance_index) slot_id: u32) -> RasterClippedColorSelectOut {
    return raster_clipped_color_select(mesh_vertex(v), slot_id);
}

@vertex
fn vs_edge_select_raster(v: VertIn) -> RasterClippedColorSelectOut {
    return raster_clipped_color_select(edge_vertex(v), 0u);
}

@fragment
fn fs_lit_select_raster(
    in: ColorSelectOut,
    @builtin(front_facing) is_front: bool,
) -> LitSelectOut {
    return LitSelectOut(
        shade_mesh(in.pos_rel, in.normal, in.color, is_front),
        pack_slot_id(in.slot_id),
    );
}

@fragment
fn fs_unlit_select_raster(in: ColorSelectOut) -> LitSelectOut {
    return LitSelectOut(
        vec4<f32>(iletki_color_to_target(in.color.rgb), in.color.a),
        vec2<u32>(0u, 0u),
    );
}
