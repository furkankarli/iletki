// Copies a sub-rectangle of a `Depth32Float` texture into a compact `R32Float` color target so
// WebGPU/Vulkan/Metal/DX12 can read back arbitrary pick regions without copying a full depth buffer.

struct DepthCopyParams {
    origin: vec2<u32>,
    _pad: vec2<u32>,
};

@group(0) @binding(0) var src_depth: texture_depth_2d;
@group(0) @binding(1) var<uniform> params: DepthCopyParams;

@vertex
fn vs_depth_copy(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    let x = f32((idx & 1u) * 4u) - 1.0;
    let y = f32(((idx >> 1u) & 1u) * 4u) - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_depth_copy(@builtin(position) pos: vec4<f32>) -> @location(0) f32 {
    let coord = vec2<i32>(params.origin) + vec2<i32>(pos.xy);
    return textureLoad(src_depth, coord, 0);
}
