// Jump-Flooding Algorithm (JFA) screen-space selection outline shaders.
// Produces smooth, constant-pixel-width silhouettes around any 2D or 3D geometry drawn in
// `DrawPhase::OutlineMask` in width-dependent fullscreen passes.

struct JfaStepParams {
    step_px: f32,
    width_px: f32,
    coverage_aa: u32,
    _pad: u32,
    color: vec4<f32>,
    active_px: vec2<f32>,
    rect_min: vec2<i32>,
    rect_max: vec2<i32>,
    _pad2: vec2<i32>,
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var mask_tex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: JfaStepParams;

@vertex
fn vs_full(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    let x = f32((idx & 1u) * 4u) - 1.0;
    let y = f32(((idx >> 1u) & 1u) * 4u) - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

const NO_SEED: vec2<f32> = vec2<f32>(-128.0);

fn mask_coverage(coord: vec2<i32>) -> f32 {
    let last = vec2<i32>(params.active_px) - vec2<i32>(1);
    return textureLoad(mask_tex, clamp(coord, vec2<i32>(0), last), 0).r;
}

// Store seed offsets relative to each texel center. Half-floats then keep subpixel precision
// even near the right or bottom edge of a large viewport.
@fragment
fn fs_seed(@builtin(position) pos: vec4<f32>) -> @location(0) vec2<f32> {
    let coord = vec2<i32>(pos.xy);
    let coverage = mask_coverage(coord);
    if (params.coverage_aa == 0u) {
        return select(NO_SEED, vec2<f32>(0.0), coverage > 0.5);
    }
    if (coverage <= 0.0) {
        return NO_SEED;
    }
    let dx = vec2<i32>(1, 0);
    let dy = vec2<i32>(0, 1);
    let gradient = vec2<f32>(
        mask_coverage(coord + dx) - mask_coverage(coord - dx),
        mask_coverage(coord + dy) - mask_coverage(coord - dy),
    );
    let len2 = dot(gradient, gradient);
    if (len2 > 1e-6) {
        return gradient * inverseSqrt(len2) * (0.5 - coverage);
    }
    return vec2<f32>(0.0);
}

// Pass 2..5: Jump-Flood step checking the 3x3 neighborhood at +-step_px.
@fragment
fn fs_step(@builtin(position) pos: vec4<f32>) -> @location(0) vec2<f32> {
    let center = vec2<i32>(pos.xy);
    let step = i32(params.step_px);

    var best_seed = NO_SEED;
    var best_dist2 = 1e30;

    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let sc = clamp(center + vec2<i32>(dx * step, dy * step), params.rect_min, params.rect_max);
            let seed = textureLoad(src_tex, sc, 0).rg;
            if (seed.x > NO_SEED.x) {
                let d = seed + vec2<f32>(sc - center);
                let dist2 = dot(d, d);
                if (dist2 < best_dist2) {
                    best_dist2 = dist2;
                    best_seed = d;
                }
            }
        }
    }
    return best_seed;
}

// Pass 6: composite smooth anti-aliased outline halo onto the color target.
@fragment
fn fs_composite(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(pos.xy);
    let m = textureLoad(mask_tex, coord, 0).r;
    // Multisampled masks let the outline fill just the uncovered part of a silhouette pixel.
    let uncovered = select(select(0.0, 1.0, m <= 0.5), 1.0 - m, params.coverage_aa != 0u);
    if (uncovered <= 0.0) {
        discard;
    }
    let seed = textureLoad(src_tex, coord, 0).rg;
    if (seed.x <= NO_SEED.x) {
        discard;
    }
    let dist = length(seed);
    let w = clamp(params.width_px, 1.0, 15.0);
    let alpha = uncovered * (1.0 - smoothstep(w - 0.65, w + 0.65, dist));
    if (alpha <= 0.002) {
        discard;
    }
    return vec4<f32>(params.color.rgb, params.color.a * alpha);
}
