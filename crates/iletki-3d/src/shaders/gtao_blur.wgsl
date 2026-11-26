// Half-resolution 3x3 depth-aware bilateral spatial denoise pass (prepended with `gtao_common.wgsl`).

@group(0) @binding(0)
var raw_ao_tex: texture_2d<f32>;

@group(0) @binding(1)
var half_depth_tex: texture_2d<u32>;

@group(0) @binding(2)
var<uniform> gtao: GtaoUniform;

@fragment
fn fs_gtao_blur(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(frag_pos.xy);
    let max_c = vec2<i32>(max(i32(gtao.half_px.x) - 1, 0), max(i32(gtao.half_px.y) - 1, 0));
    let c0 = clamp(coord, vec2<i32>(0, 0), max_c);
    let z_c = bitcast<f32>(textureLoad(half_depth_tex, c0, 0).r);
    if (z_c <= 0.0) {
        return vec4<f32>(1.0, 0.0, 0.0, 1.0);
    }

    let d_c = gtao_linear_depth(gtao, z_c);
    let is_persp = (gtao.flags & 1u) != 0u;
    let depth_sigma = select(
        max(gtao.radius * 0.25, 1e-4),
        max(d_c * 0.025, 1e-4),
        is_persp,
    );
    let inv_sigma = 1.0 / depth_sigma;

    var ao_sum = 0.0;
    var w_sum = 0.0;

    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let nc = clamp(c0 + vec2<i32>(dx, dy), vec2<i32>(0, 0), max_c);
            let z_n = bitcast<f32>(textureLoad(half_depth_tex, nc, 0).r);
            if (z_n <= 0.0) {
                continue;
            }
            let d_n = gtao_linear_depth(gtao, z_n);
            let dz = abs(d_n - d_c) * inv_sigma;
            let spatial_w = select(
                select(0.5, 0.25, dx != 0 && dy != 0),
                1.0,
                dx == 0 && dy == 0,
            );
            let depth_w = exp(-dz * dz * 1.8);
            let w = spatial_w * depth_w;
            ao_sum = ao_sum + textureLoad(raw_ao_tex, nc, 0).r * w;
            w_sum = w_sum + w;
        }
    }

    let ao = select(1.0, clamp(ao_sum / w_sum, 0.0, 1.0), w_sum > 1e-5);
    return vec4<f32>(ao, 0.0, 0.0, 1.0);
}
