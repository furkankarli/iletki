// Antialiased section entry points for style pipelines on devices with `CLIP_DISTANCES`.
// Hardware clip distances remove clipped MSAA samples during rasterization, so the fragment
// shader neither discards nor runs per sample and early depth testing stays enabled.
struct StyleRasterOut {
    @builtin(position) clip_pos: vec4<f32>,
    @builtin(clip_distances) plane_distances: array<f32, 6>,
    @location(0) pos_rel: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) @interpolate(flat) chunk_id: u32,
};

alias StyleSectionOut = StyleRasterOut;

fn style_section(out: StyleOut) -> StyleRasterOut {
    return StyleRasterOut(out.clip_pos, raster_clip_distances(out.pos_rel), out.pos_rel, out.normal, out.chunk_id);
}

@fragment
fn fs_style_section(in: StyleOut, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    return style_color(in, is_front, cross(dpdy(in.pos_rel), dpdx(in.pos_rel)));
}

@fragment
fn fs_style_edge_section() -> @location(0) vec4<f32> {
    return style_edge_color();
}
