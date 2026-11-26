// Full-resolution 2-pass A-Trous edge-chained spatial denoiser (prepended with `gtao_common.wgsl`).
//
// Pass 1 (stride 1, [1, 2, 1] x [1, 2, 1]) annihilates period-2 spatial noise (H(pi) = 0).
// Pass 2 (stride 2, [1, 2, 1] x [1, 2, 1] gated by chained 1-pixel symmetric edges) annihilates
// period-4 noise (H(pi/2) = 0). Their product (1 + 2z + z^2)(1 + 2z^2 + z^4) = (1 + z + z^2 + z^3)^2
// is an exact symmetric 7x7 B-spline that cancels all 15 non-zero harmonics of the 4x4 stratified
// tile to machine precision on smooth surfaces while strictly blocking bleed across any 1-pixel
// depth discontinuity, corner, or section-cut lip.

@group(0) @binding(0)
var src_ao_tex: texture_2d<f32>;

@group(0) @binding(1)
var edges_tex: texture_2d<f32>;

@group(0) @binding(2)
var<uniform> gtao: GtaoUniform;

const GTAO_OCCLUSION_SCALE: f32 = 1.5;

fn unpack_edges(packed_val: f32) -> vec4<f32> {
    let p = u32(packed_val * 255.5);
    return clamp(
        vec4<f32>(
            f32((p >> 6u) & 3u),
            f32((p >> 4u) & 3u),
            f32((p >> 2u) & 3u),
            f32(p & 3u),
        ) * (1.0 / 3.0),
        vec4<f32>(0.0),
        vec4<f32>(1.0),
    );
}

fn load_edges(c: vec2<i32>, max_c: vec2<i32>) -> vec4<f32> {
    return unpack_edges(textureLoad(edges_tex, clamp(c, vec2<i32>(0, 0), max_c), 0).r);
}

fn load_ao(c: vec2<i32>, max_c: vec2<i32>) -> f32 {
    return textureLoad(src_ao_tex, clamp(c, vec2<i32>(0, 0), max_c), 0).r;
}

@fragment
fn fs_gtao_denoise_pass1(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let max_c = vec2<i32>(
        max(i32(gtao.active_px.x) - 1, 0),
        max(i32(gtao.active_px.y) - 1, 0),
    );
    let c = clamp(vec2<i32>(frag_pos.xy), vec2<i32>(0, 0), max_c);
    let packed_c = textureLoad(edges_tex, c, 0).r;
    let ao_c = load_ao(c, max_c);
    // Sky pixels and 1-pixel isolated silhouettes have all 4 edge weights zero -> exact identity.
    if (packed_c <= 0.0) {
        return vec4<f32>(ao_c, 0.0, 0.0, 1.0);
    }

    let e_c_raw = unpack_edges(packed_c);
    let e_l = load_edges(c + vec2<i32>(-1, 0), max_c);
    let e_r = load_edges(c + vec2<i32>(1, 0), max_c);
    let e_t = load_edges(c + vec2<i32>(0, -1), max_c);
    let e_b = load_edges(c + vec2<i32>(0, 1), max_c);

    // Strictly symmetric 1-pixel edge weights (zero bleed across depth discontinuities).
    let w_l = select(0.0, e_c_raw.x * e_l.y, c.x > 0);
    let w_r = select(0.0, e_c_raw.y * e_r.x, c.x < max_c.x);
    let w_t = select(0.0, e_c_raw.z * e_t.w, c.y > 0);
    let w_b = select(0.0, e_c_raw.w * e_b.z, c.y < max_c.y);

    let e_tl = load_edges(c + vec2<i32>(-1, -1), max_c);
    let e_tr = load_edges(c + vec2<i32>(1, -1), max_c);
    let e_bl = load_edges(c + vec2<i32>(-1, 1), max_c);
    let e_br = load_edges(c + vec2<i32>(1, 1), max_c);

    // Corner-safe diagonal weights: 1.0 on continuous planes, 0.0 across sharp corners/silhouettes.
    let w_tl = select(0.0, 0.5 * (w_l * e_l.z * e_tl.w + w_t * e_t.x * e_tl.y), c.x > 0 && c.y > 0);
    let w_tr = select(0.0, 0.5 * (w_r * e_r.z * e_tr.w + w_t * e_t.y * e_tr.x), c.x < max_c.x && c.y > 0);
    let w_bl = select(0.0, 0.5 * (w_l * e_l.w * e_bl.z + w_b * e_b.x * e_bl.y), c.x > 0 && c.y < max_c.y);
    let w_br = select(0.0, 0.5 * (w_r * e_r.w * e_br.z + w_b * e_b.y * e_br.x), c.x < max_c.x && c.y < max_c.y);

    // Exact [1, 2, 1] x [1, 2, 1] weights: center = 4, cardinals = 2, diagonals = 1.
    let sum = ao_c * 4.0
        + load_ao(c + vec2<i32>(-1, 0), max_c) * (2.0 * w_l)
        + load_ao(c + vec2<i32>(1, 0), max_c) * (2.0 * w_r)
        + load_ao(c + vec2<i32>(0, -1), max_c) * (2.0 * w_t)
        + load_ao(c + vec2<i32>(0, 1), max_c) * (2.0 * w_b)
        + load_ao(c + vec2<i32>(-1, -1), max_c) * w_tl
        + load_ao(c + vec2<i32>(1, -1), max_c) * w_tr
        + load_ao(c + vec2<i32>(-1, 1), max_c) * w_bl
        + load_ao(c + vec2<i32>(1, 1), max_c) * w_br;
    let w_sum = 4.0 + 2.0 * (w_l + w_r + w_t + w_b) + (w_tl + w_tr + w_bl + w_br);
    return vec4<f32>(sum / max(w_sum, 1e-5), 0.0, 0.0, 1.0);
}

@fragment
fn fs_gtao_denoise_pass2(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let max_c = vec2<i32>(
        max(i32(gtao.active_px.x) - 1, 0),
        max(i32(gtao.active_px.y) - 1, 0),
    );
    let c = clamp(vec2<i32>(frag_pos.xy), vec2<i32>(0, 0), max_c);
    let packed_c = textureLoad(edges_tex, c, 0).r;
    let ao_c = load_ao(c, max_c);
    if (packed_c <= 0.0) {
        return vec4<f32>(clamp(ao_c * GTAO_OCCLUSION_SCALE, 0.0, 1.0), 0.0, 0.0, 1.0);
    }

    let ec = unpack_edges(packed_c);
    let e_l1 = load_edges(c + vec2<i32>(-1, 0), max_c);
    let e_r1 = load_edges(c + vec2<i32>(1, 0), max_c);
    let e_t1 = load_edges(c + vec2<i32>(0, -1), max_c);
    let e_b1 = load_edges(c + vec2<i32>(0, 1), max_c);

    let e_l2 = load_edges(c + vec2<i32>(-2, 0), max_c);
    let e_r2 = load_edges(c + vec2<i32>(2, 0), max_c);
    let e_t2 = load_edges(c + vec2<i32>(0, -2), max_c);
    let e_b2 = load_edges(c + vec2<i32>(0, 2), max_c);

    // Chain two 1-pixel symmetric edges along each stride-2 step so thin walls & cut lips never leak.
    let w_l2 = select(0.0, (ec.x * e_l1.y) * (e_l1.x * e_l2.y), c.x >= 2);
    let w_r2 = select(0.0, (ec.y * e_r1.x) * (e_r1.y * e_r2.x), c.x + 2 <= max_c.x);
    let w_t2 = select(0.0, (ec.z * e_t1.w) * (e_t1.z * e_t2.w), c.y >= 2);
    let w_b2 = select(0.0, (ec.w * e_b1.z) * (e_b1.w * e_b2.z), c.y + 2 <= max_c.y);

    // Deduplicated perimeter edge loads for the 4 stride-2 diagonal chains.
    let e_l2_t1 = load_edges(c + vec2<i32>(-2, -1), max_c);
    let e_l2_b1 = load_edges(c + vec2<i32>(-2, 1), max_c);
    let e_r2_t1 = load_edges(c + vec2<i32>(2, -1), max_c);
    let e_r2_b1 = load_edges(c + vec2<i32>(2, 1), max_c);
    let e_l1_t2 = load_edges(c + vec2<i32>(-1, -2), max_c);
    let e_r1_t2 = load_edges(c + vec2<i32>(1, -2), max_c);
    let e_l1_b2 = load_edges(c + vec2<i32>(-1, 2), max_c);
    let e_r1_b2 = load_edges(c + vec2<i32>(1, 2), max_c);
    let e_tl2 = load_edges(c + vec2<i32>(-2, -2), max_c);
    let e_tr2 = load_edges(c + vec2<i32>(2, -2), max_c);
    let e_bl2 = load_edges(c + vec2<i32>(-2, 2), max_c);
    let e_br2 = load_edges(c + vec2<i32>(2, 2), max_c);

    let w_tl2 = select(
        0.0,
        0.5 * (
            w_l2 * (e_l2.z * e_l2_t1.w) * (e_l2_t1.z * e_tl2.w)
            + w_t2 * (e_t2.x * e_l1_t2.y) * (e_l1_t2.x * e_tl2.y)
        ),
        c.x >= 2 && c.y >= 2,
    );
    let w_tr2 = select(
        0.0,
        0.5 * (
            w_r2 * (e_r2.z * e_r2_t1.w) * (e_r2_t1.z * e_tr2.w)
            + w_t2 * (e_t2.y * e_r1_t2.x) * (e_r1_t2.y * e_tr2.x)
        ),
        c.x + 2 <= max_c.x && c.y >= 2,
    );
    let w_bl2 = select(
        0.0,
        0.5 * (
            w_l2 * (e_l2.w * e_l2_b1.z) * (e_l2_b1.w * e_bl2.z)
            + w_b2 * (e_b2.x * e_l1_b2.y) * (e_l1_b2.x * e_bl2.y)
        ),
        c.x >= 2 && c.y + 2 <= max_c.y,
    );
    let w_br2 = select(
        0.0,
        0.5 * (
            w_r2 * (e_r2.w * e_r2_b1.z) * (e_r2_b1.w * e_br2.z)
            + w_b2 * (e_b2.y * e_r1_b2.x) * (e_r1_b2.y * e_br2.x)
        ),
        c.x + 2 <= max_c.x && c.y + 2 <= max_c.y,
    );

    let sum = ao_c * 4.0
        + load_ao(c + vec2<i32>(-2, 0), max_c) * (2.0 * w_l2)
        + load_ao(c + vec2<i32>(2, 0), max_c) * (2.0 * w_r2)
        + load_ao(c + vec2<i32>(0, -2), max_c) * (2.0 * w_t2)
        + load_ao(c + vec2<i32>(0, 2), max_c) * (2.0 * w_b2)
        + load_ao(c + vec2<i32>(-2, -2), max_c) * w_tl2
        + load_ao(c + vec2<i32>(2, -2), max_c) * w_tr2
        + load_ao(c + vec2<i32>(-2, 2), max_c) * w_bl2
        + load_ao(c + vec2<i32>(2, 2), max_c) * w_br2;
    let w_sum = 4.0 + 2.0 * (w_l2 + w_r2 + w_t2 + w_b2) + (w_tl2 + w_tr2 + w_bl2 + w_br2);

    let out_val = clamp((sum / max(w_sum, 1e-5)) * GTAO_OCCLUSION_SCALE, 0.0, 1.0);
    return vec4<f32>(out_val, 0.0, 0.0, 1.0);
}
