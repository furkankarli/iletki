// Antialiased section entry points for style pipelines on devices with multisampled shading.
// Shade at the pixel center and evaluate clip visibility at each MSAA sample position.
struct StyleSampledOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) pos_rel: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) @interpolate(flat) chunk_id: u32,
    @location(3) @interpolate(perspective, sample) sample_pos_rel: vec3<f32>,
};

alias StyleSectionOut = StyleSampledOut;

fn style_section(out: StyleOut) -> StyleSampledOut {
    return StyleSampledOut(out.clip_pos, out.pos_rel, out.normal, out.chunk_id, out.pos_rel);
}

@fragment
fn fs_style_section(in: StyleSampledOut, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    // Evaluate screen-space derivatives in uniform control flow prior to any clip discard.
    let facet_cross = cross(dpdy(in.pos_rel), dpdx(in.pos_rel));
    if (iletki_is_clipped(in.sample_pos_rel)) {
        discard;
    }
    return style_color(StyleOut(in.clip_pos, in.pos_rel, in.normal, in.chunk_id), is_front, facet_cross);
}

@fragment
fn fs_style_edge_section(in: StyleSampledOut) -> @location(0) vec4<f32> {
    if (iletki_is_clipped(in.sample_pos_rel)) {
        discard;
    }
    return style_edge_color();
}
