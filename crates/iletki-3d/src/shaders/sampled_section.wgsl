// Append only when the adapter supports sample-rate interpolation.
struct SampledColorOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) pos_rel: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    // Shade at the pixel center and evaluate visibility at the sample position.
    @location(3) @interpolate(perspective, sample) sample_pos_rel: vec3<f32>,
};

fn sampled_color(out: ColorOut) -> SampledColorOut {
    return SampledColorOut(out.clip_pos, out.pos_rel, out.normal, out.color, out.pos_rel);
}
