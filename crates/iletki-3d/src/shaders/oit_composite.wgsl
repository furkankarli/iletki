// Fullscreen Weighted Blended OIT resolve/composite pass.
// Reads the resolved 1x Rgba16Float accumulation and R8Unorm revealage textures via `textureLoad`
// (no sampler or filterable-float requirement; works identically at 1x and 4x MSAA on Vulkan,
// DX12, Metal, OpenGL and WebGPU).

@group(0) @binding(0)
var oit_accum: texture_2d<f32>;

@group(0) @binding(1)
var oit_reveal: texture_2d<f32>;

@vertex
fn vs_oit_composite(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(pos[vid], 0.0, 1.0);
}

@fragment
fn fs_oit_composite(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(frag_pos.xy);
    let reveal = textureLoad(oit_reveal, coord, 0).r;
    if (reveal >= 0.9999) {
        discard;
    }
    let accum = textureLoad(oit_accum, coord, 0);
    let avg_rgb = accum.rgb / max(accum.a, 1e-5);
    let out_alpha = clamp(1.0 - reveal, 0.0, 1.0);
    return vec4<f32>(avg_rgb, out_alpha);
}
