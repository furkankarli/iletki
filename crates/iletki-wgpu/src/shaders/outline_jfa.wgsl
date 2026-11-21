// Jump-Flooding Algorithm (JFA) screen-space selection outline shaders.
// Produces smooth, constant-pixel-width silhouettes around any 2D or 3D geometry drawn in
// `DrawPhase::OutlineMask` in O(log(radius)) fullscreen passes (steps 8 -> 4 -> 2 -> 1).

struct JfaStepParams {
    step_px: f32,
    width_px: f32,
    _pad: vec2<f32>,
    color: vec4<f32>,
    active_px: vec2<f32>,
    _pad2: vec2<f32>,
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

// Pass 1: convert binary mask texture into initial nearest-seed coordinates (pixel coords).
@fragment
fn fs_seed(@builtin(position) pos: vec4<f32>) -> @location(0) vec2<f32> {
    let coord = vec2<i32>(pos.xy);
    let m = textureLoad(mask_tex, coord, 0).r;
    if (m > 0.5) {
        return pos.xy;
    }
    return vec2<f32>(-1.0, -1.0);
}

// Pass 2..5: Jump-Flood step checking the 3x3 neighborhood at +-step_px.
@fragment
fn fs_step(@builtin(position) pos: vec4<f32>) -> @location(0) vec2<f32> {
    let center = vec2<i32>(pos.xy);
    let max_c = vec2<i32>(params.active_px) - vec2<i32>(1, 1);
    let step = i32(params.step_px);

    var best_seed = vec2<f32>(-1.0, -1.0);
    var best_dist2 = 1e30;

    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let sc = clamp(center + vec2<i32>(dx * step, dy * step), vec2<i32>(0, 0), max_c);
            let seed = textureLoad(src_tex, sc, 0).rg;
            if (seed.x >= 0.0) {
                let d = seed - pos.xy;
                let dist2 = dot(d, d);
                if (dist2 < best_dist2) {
                    best_dist2 = dist2;
                    best_seed = seed;
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
    // Do not paint over the interior of the selected object itself.
    let m = textureLoad(mask_tex, coord, 0).r;
    if (m > 0.5) {
        discard;
    }
    let seed = textureLoad(src_tex, coord, 0).rg;
    if (seed.x < 0.0) {
        discard;
    }
    let dist = length(seed - pos.xy);
    let w = clamp(params.width_px, 1.0, 15.0);
    let alpha = 1.0 - smoothstep(w - 0.65, w + 0.65, dist);
    if (alpha <= 0.002) {
        discard;
    }
    return vec4<f32>(params.color.rgb, params.color.a * alpha);
}
