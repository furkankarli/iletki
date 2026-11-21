// Fullscreen single-triangle textured blit with UV sub-rectangle cropping for 256px over-allocated
// viewport targets.

struct BlitParams {
    uv_max: vec2<f32>,
    _pad: vec2<f32>,
};

@group(0) @binding(0) var blit_tex: texture_2d<f32>;
@group(0) @binding(1) var blit_sampler: sampler;
@group(0) @binding(2) var<uniform> blit_params: BlitParams;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_blit(@builtin(vertex_index) idx: u32) -> VsOut {
    // Oversized triangle covering [-1, 1] x [-1, 1]:
    // idx 0: (-1, -1) -> uv (0, 1)
    // idx 1: ( 3, -1) -> uv (2, 1)
    // idx 2: (-1,  3) -> uv (0, -1)
    let x = f32((idx & 1u) * 4u) - 1.0;
    let y = f32(((idx >> 1u) & 1u) * 4u) - 1.0;
    let u = (x + 1.0) * 0.5;
    let v = (1.0 - y) * 0.5;

    var out: VsOut;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(u, v) * blit_params.uv_max;
    return out;
}

@fragment
fn fs_blit(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(blit_tex, blit_sampler, in.uv);
}
