// Conservative Reversed-Z min-depth Hi-Z pyramid builder (4 mip levels per compute pass).
// Reversed-Z stores 1.0 at the near plane and 0.0 at the far plane; taking the minimum across
// MSAA samples and 2x2 texels guarantees a conservative farthest depth bound.

struct HizParams {
    // xy = source dimensions, zw = dst0 dimensions
    src_and_dst0: vec4<u32>,
    // xy = dst1 dimensions, zw = dst2 dimensions
    dst1_and_dst2: vec4<u32>,
    // xy = dst3 dimensions, z = active output mip count (1..=4), w = MSAA sample count
    dst3_and_meta: vec4<u32>,
};

@group(0) @binding(0)
var<uniform> params: HizParams;

@group(0) @binding(2)
var dst0_tex: texture_storage_2d<r32float, write>;

@group(0) @binding(3)
var dst1_tex: texture_storage_2d<r32float, write>;

@group(0) @binding(4)
var dst2_tex: texture_storage_2d<r32float, write>;

@group(0) @binding(5)
var dst3_tex: texture_storage_2d<r32float, write>;

var<workgroup> tile8: array<f32, 64>;
var<workgroup> tile4: array<f32, 16>;
var<workgroup> tile2: array<f32, 4>;

fn reduce_and_store(gid: vec2<u32>, lid: vec2<u32>, wgid: vec2<u32>, val0: f32) {
    let dst0 = params.src_and_dst0.zw;
    let in0 = gid.x < dst0.x && gid.y < dst0.y;
    if (in0) {
        textureStore(dst0_tex, vec2<i32>(gid), vec4<f32>(val0, 0.0, 0.0, 0.0));
    }
    // 1.0 is the neutral identity for min() in Reversed-Z [0.0, 1.0], keeping non-square and
    // padded edges mathematically exact across every reduction level.
    tile8[lid.y * 8u + lid.x] = select(1.0, val0, in0);
    workgroupBarrier();

    let write_count = params.dst3_and_meta.z;
    if (lid.x < 4u && lid.y < 4u) {
        let sx = lid.x * 2u;
        let sy = lid.y * 2u;
        let v1 = min(
            min(tile8[sy * 8u + sx], tile8[sy * 8u + sx + 1u]),
            min(tile8[(sy + 1u) * 8u + sx], tile8[(sy + 1u) * 8u + sx + 1u]),
        );
        let out1 = wgid * 4u + lid;
        let dst1 = params.dst1_and_dst2.xy;
        let in1 = out1.x < dst1.x && out1.y < dst1.y;
        if (write_count >= 2u && in1) {
            textureStore(dst1_tex, vec2<i32>(out1), vec4<f32>(v1, 0.0, 0.0, 0.0));
        }
        tile4[lid.y * 4u + lid.x] = select(1.0, v1, in1);
    }
    workgroupBarrier();

    if (lid.x < 2u && lid.y < 2u) {
        let sx = lid.x * 2u;
        let sy = lid.y * 2u;
        let v2 = min(
            min(tile4[sy * 4u + sx], tile4[sy * 4u + sx + 1u]),
            min(tile4[(sy + 1u) * 4u + sx], tile4[(sy + 1u) * 4u + sx + 1u]),
        );
        let out2 = wgid * 2u + lid;
        let dst2 = params.dst1_and_dst2.zw;
        let in2 = out2.x < dst2.x && out2.y < dst2.y;
        if (write_count >= 3u && in2) {
            textureStore(dst2_tex, vec2<i32>(out2), vec4<f32>(v2, 0.0, 0.0, 0.0));
        }
        tile2[lid.y * 2u + lid.x] = select(1.0, v2, in2);
    }
    workgroupBarrier();

    if (lid.x == 0u && lid.y == 0u && write_count >= 4u) {
        let v3 = min(min(tile2[0], tile2[1]), min(tile2[2], tile2[3]));
        let dst3 = params.dst3_and_meta.xy;
        if (wgid.x < dst3.x && wgid.y < dst3.y) {
            textureStore(dst3_tex, vec2<i32>(wgid), vec4<f32>(v3, 0.0, 0.0, 0.0));
        }
    }
}
