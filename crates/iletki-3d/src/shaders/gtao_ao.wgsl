// Full-resolution XeGTAO horizon integration + packed 4-neighbor edge pass (prepended with
// `gtao_common.wgsl`). Clean-room WGSL architecture based on Jimenez et al. (2016) and Intel
// XeGTAO (MIT):
// - 5-level weighted linear-depth MIP pyramid sampling (L1-cache friendly even at 200 px radius).
// - Slope-adjusted 4-neighbor edge weights packed into R8Unorm (2 bits/side) + edge-weighted
//   4-quadrant normal reconstruction.
// - Zero-correlation 4x4 Bayer-Latin stratified spatial noise + per-(slice, step) R1 golden-ratio
//   radial sequence: every 4x4 block covers all 16 azimuthal and radial strata uniformly with zero
//   low-frequency blotches on static (non-TAA) CAD frames, and is annihilated cleanly by the
//   paired 2-pass A-Trous edge-chained denoiser.
// - Packed with 1.5x headroom into R8Unorm so denoiser averaging never clips bright slopes dark.

@group(0) @binding(0)
var depth_pyramid: texture_2d<f32>;

@group(0) @binding(1)
var<uniform> gtao: GtaoUniform;

const GTAO_PI: f32 = 3.141592653589793;
const GTAO_HALF_PI: f32 = 1.5707963267948966;
const GTAO_OCCLUSION_SCALE: f32 = 1.5;
const GTAO_RADIUS_MULTIPLIER: f32 = 1.457;
const GTAO_FALLOFF_RANGE: f32 = 0.615;
const GTAO_THIN_OCCLUDER_COMP: f32 = 0.18;

// Exact 4x4 Bayer stratification for azimuth (every 2x2 hits all 4 quarters; every 4x4 hits 0..15).
const PHI_LUT_4X4 = array<f32, 16>(
    0.03125, 0.53125, 0.15625, 0.65625,
    0.78125, 0.28125, 0.90625, 0.40625,
    0.21875, 0.71875, 0.09375, 0.59375,
    0.96875, 0.46875, 0.84375, 0.34375,
);

// Orthogonal zero-correlation (corr = 0.0000) 4x4 Latin-quarter permutation for radial jitter.
const RAD_LUT_4X4 = array<f32, 16>(
    0.78125, 0.71875, 0.09375, 0.40625,
    0.46875, 0.03125, 0.65625, 0.84375,
    0.21875, 0.28125, 0.90625, 0.59375,
    0.53125, 0.96875, 0.34375, 0.15625,
);

struct MainOut {
    @location(0) ao_packed: vec4<f32>,
    @location(1) edges_packed: vec4<f32>,
};

fn load_depth_mip0(coord: vec2<i32>, max_c: vec2<i32>) -> f32 {
    return textureLoad(depth_pyramid, clamp(coord, vec2<i32>(0, 0), max_c), 0).r;
}

fn spatial_noise(coord: vec2<i32>) -> vec2<f32> {
    let idx = (u32(coord.x) & 3u) | ((u32(coord.y) & 3u) << 2u);
    return vec2<f32>(PHI_LUT_4X4[idx], RAD_LUT_4X4[idx]);
}

// Slope-adjusted 4-neighbor depth continuity weights (L, R, T, B) in [0, 1] (XeGTAO eq.).
fn calculate_edges(
    center_d: f32,
    left_d: f32,
    right_d: f32,
    top_d: f32,
    bottom_d: f32,
) -> vec4<f32> {
    let is_persp = (gtao.flags & 1u) != 0u;
    let scale = select(
        max(gtao.radius * 0.05, 1e-4),
        max(center_d * 0.011, 1e-4),
        is_persp,
    );
    let dl = select(1e6, left_d - center_d, left_d > 0.0);
    let dr = select(1e6, right_d - center_d, right_d > 0.0);
    let dt = select(1e6, top_d - center_d, top_d > 0.0);
    let db = select(1e6, bottom_d - center_d, bottom_d > 0.0);
    let raw = vec4<f32>(dl, dr, dt, db);
    let slope_lr = (dr - dl) * 0.5;
    let slope_tb = (db - dt) * 0.5;
    let slope_adj = raw + vec4<f32>(slope_lr, -slope_lr, slope_tb, -slope_tb);
    let min_err = min(abs(raw), abs(slope_adj));
    return clamp(vec4<f32>(1.25) - min_err / scale, vec4<f32>(0.0), vec4<f32>(1.0));
}

// Pack 4 edge weights into R8Unorm (2 bits per side -> 0, 0.33, 0.67, 1.0).
fn pack_edges(e: vec4<f32>) -> f32 {
    let q = round(clamp(e, vec4<f32>(0.0), vec4<f32>(1.0)) * 2.9);
    return dot(q, vec4<f32>(64.0 / 255.0, 16.0 / 255.0, 4.0 / 255.0, 1.0 / 255.0));
}

// Edge-weighted 4-quadrant view-space normal reconstruction (XeGTAO eq.): smooth on curved
// surfaces, razor-sharp at 90-degree corners and silhouettes.
fn calculate_normal(
    edges: vec4<f32>,
    c_pos: vec3<f32>,
    l_pos: vec3<f32>,
    r_pos: vec3<f32>,
    t_pos: vec3<f32>,
    b_pos: vec3<f32>,
    view_dir: vec3<f32>,
) -> vec3<f32> {
    // edges = (L, R, T, B)
    let accepted = clamp(
        vec4<f32>(
            edges.x * edges.z,
            edges.z * edges.y,
            edges.y * edges.w,
            edges.w * edges.x,
        ) + vec4<f32>(0.01),
        vec4<f32>(0.0),
        vec4<f32>(1.0),
    );

    let dl = l_pos - c_pos;
    let dr = r_pos - c_pos;
    let dt = t_pos - c_pos;
    let db = b_pos - c_pos;
    let nl = select(vec3<f32>(-1.0, 0.0, 0.0), normalize(dl), dot(dl, dl) > 1e-12);
    let nr = select(vec3<f32>(1.0, 0.0, 0.0), normalize(dr), dot(dr, dr) > 1e-12);
    let nt = select(vec3<f32>(0.0, 1.0, 0.0), normalize(dt), dot(dt, dt) > 1e-12);
    let nb = select(vec3<f32>(0.0, -1.0, 0.0), normalize(db), dot(db, db) > 1e-12);

    // Right-handed view space (+X right, +Y up, +Z toward camera):
    // cross(nt, nl) = cross(+Y, -X) = +Z; cross(nr, nt) = +Z; cross(nb, nr) = +Z; cross(nl, nb) = +Z.
    let raw_n = accepted.x * cross(nt, nl)
        + accepted.y * cross(nr, nt)
        + accepted.z * cross(nb, nr)
        + accepted.w * cross(nl, nb);
    let nlen = length(raw_n);
    if (nlen <= 1e-7) {
        return view_dir;
    }
    var n = raw_n / nlen;
    if (dot(n, view_dir) < 0.0) {
        n = -n;
    }
    return n;
}

fn integrate_arc(h: f32, n: f32, cos_n: f32, sin_n: f32) -> f32 {
    return 0.25 * (-cos(2.0 * h - n) + cos_n + 2.0 * h * sin_n);
}

fn sample_horizon_mip(
    sample_px: vec2<f32>,
    mip_u: u32,
    max_full: vec2<i32>,
    pos_biased: vec3<f32>,
    view_dir: vec3<f32>,
    falloff_mul: f32,
    falloff_add: f32,
    cos_flat: f32,
    cur_max: f32,
) -> f32 {
    if (sample_px.x < 0.0
        || sample_px.y < 0.0
        || sample_px.x > f32(max_full.x)
        || sample_px.y > f32(max_full.y))
    {
        return cur_max;
    }
    let mip_dims = vec2<i32>(textureDimensions(depth_pyramid, i32(mip_u)));
    let max_mip = max(mip_dims - vec2<i32>(1, 1), vec2<i32>(0, 0));
    let scale = f32(1u << mip_u);
    let mcoord = clamp(vec2<i32>(floor(sample_px / scale)), vec2<i32>(0, 0), max_mip);
    let ds = textureLoad(depth_pyramid, mcoord, i32(mip_u)).r;
    if (ds <= 0.0) {
        return cur_max;
    }
    // Reconstruct at the exact center of the sampled mip texel so slanted planes keep exact slope.
    let exact_px = (vec2<f32>(mcoord) + vec2<f32>(0.5)) * scale;
    let delta = gtao_view_pos_linear(gtao, exact_px, ds) - pos_biased;
    let len2 = dot(delta, delta);
    if (len2 <= 1e-10) {
        return cur_max;
    }
    let dist = sqrt(len2);
    // XeGTAO thin-occluder compensation penalizes depth separation slightly more than XY separation
    // so floating shelves and hollow cut lips cast clean contact shadows without muddy halos.
    let falloff_base = length(vec3<f32>(
        delta.x,
        delta.y,
        delta.z * (1.0 + GTAO_THIN_OCCLUDER_COMP),
    ));
    let weight = clamp(falloff_base * falloff_mul + falloff_add, 0.0, 1.0);
    if (weight <= 0.0) {
        return cur_max;
    }
    let cos_h = dot(delta, view_dir) / dist;
    if (cos_h <= cos_flat) {
        return cur_max;
    }
    let eff = mix(cos_flat, cos_h, weight);
    return max(cur_max, eff);
}

@fragment
fn fs_gtao_ao(@builtin(position) frag_pos: vec4<f32>) -> MainOut {
    let max_c = vec2<i32>(
        max(i32(gtao.active_px.x) - 1, 0),
        max(i32(gtao.active_px.y) - 1, 0),
    );
    let coord = clamp(vec2<i32>(frag_pos.xy), vec2<i32>(0, 0), max_c);
    let d_c = load_depth_mip0(coord, max_c);
    if (d_c <= 0.0) {
        return MainOut(
            vec4<f32>(1.0 / GTAO_OCCLUSION_SCALE, 0.0, 0.0, 1.0),
            vec4<f32>(0.0, 0.0, 0.0, 1.0),
        );
    }

    let d_l = select(0.0, load_depth_mip0(coord + vec2<i32>(-1, 0), max_c), coord.x > 0);
    let d_r = select(0.0, load_depth_mip0(coord + vec2<i32>(1, 0), max_c), coord.x < max_c.x);
    let d_t = select(0.0, load_depth_mip0(coord + vec2<i32>(0, -1), max_c), coord.y > 0);
    let d_b = select(0.0, load_depth_mip0(coord + vec2<i32>(0, 1), max_c), coord.y < max_c.y);

    let edges = calculate_edges(d_c, d_l, d_r, d_t, d_b);
    let packed_edge = pack_edges(edges);

    let px_c = vec2<f32>(coord) + vec2<f32>(0.5);
    let pos_c = gtao_view_pos_linear(gtao, px_c, d_c);
    let view_dir = gtao_view_dir(gtao, pos_c);
    let pos_l = select(pos_c, gtao_view_pos_linear(gtao, px_c + vec2<f32>(-1.0, 0.0), d_l), d_l > 0.0);
    let pos_r = select(pos_c, gtao_view_pos_linear(gtao, px_c + vec2<f32>(1.0, 0.0), d_r), d_r > 0.0);
    let pos_t = select(pos_c, gtao_view_pos_linear(gtao, px_c + vec2<f32>(0.0, -1.0), d_t), d_t > 0.0);
    let pos_b = select(pos_c, gtao_view_pos_linear(gtao, px_c + vec2<f32>(0.0, 1.0), d_b), d_b > 0.0);
    let normal = calculate_normal(edges, pos_c, pos_l, pos_r, pos_t, pos_b, view_dir);

    let effect_radius = gtao.radius * GTAO_RADIUS_MULTIPLIER;
    let ah = max(f32(gtao.active_px.y), 1.0);
    let is_persp = (gtao.flags & 1u) != 0u;
    let world_h = select(2.0 * gtao.proj_xy.y, 2.0 * gtao.proj_xy.y * d_c, is_persp);
    let raw_radius_px = (effect_radius * ah) / max(world_h, 1e-5);
    if (raw_radius_px < 1.5) {
        return MainOut(
            vec4<f32>(1.0 / GTAO_OCCLUSION_SCALE, 0.0, 0.0, 1.0),
            vec4<f32>(packed_edge, 0.0, 0.0, 1.0),
        );
    }

    let radius_px = min(raw_radius_px, gtao.max_screen_radius_px);
    let eff_radius = max(effect_radius * (radius_px / raw_radius_px), 1e-4);
    let falloff_range = GTAO_FALLOFF_RANGE * eff_radius;
    let falloff_from = eff_radius * (1.0 - GTAO_FALLOFF_RANGE);
    let falloff_mul = -1.0 / max(falloff_range, 1e-5);
    let falloff_add = falloff_from / max(falloff_range, 1e-5) + 1.0;

    // Slight view-space nudge toward normal eliminates fp32 self-occlusion on flat planes.
    let pos_biased = pos_c + normal * max(eff_radius * 0.002, d_c * 0.00025);

    let noise = spatial_noise(coord);
    let slices = clamp(gtao.slices, 1u, 6u);
    let steps = clamp(gtao.steps_per_side, 2u, 8u);
    let max_mip = max(gtao.mip_count, 1u) - 1u;
    let min_s = 1.3 / max(radius_px, 1.5);

    var vis_sum = 0.0;
    var weight_sum = 0.0;

    for (var s: u32 = 0u; s < slices; s = s + 1u) {
        let phi = (f32(s) + noise.x) * (GTAO_PI / f32(slices));
        let omega = vec2<f32>(cos(phi), sin(phi));
        // Screen +Y points downward -> view-space slice direction flips Y.
        let dir_v = vec3<f32>(omega.x, -omega.y, 0.0);
        let ortho_dir = dir_v - view_dir * dot(dir_v, view_dir);
        let axis_raw = cross(dir_v, view_dir);
        let axis_len = length(axis_raw);
        if (axis_len <= 1e-5) {
            continue;
        }
        let axis_v = axis_raw / axis_len;
        let proj_n = normal - axis_v * dot(normal, axis_v);
        let proj_n_len = length(proj_n);
        if (proj_n_len <= 1e-4) {
            continue;
        }

        let cos_n = clamp(dot(proj_n, view_dir) / proj_n_len, -1.0, 1.0);
        let sign_n = select(-1.0, 1.0, dot(ortho_dir, proj_n) >= 0.0);
        let n_angle = sign_n * acos(cos_n);
        let sin_n = sin(n_angle);

        let cos_flat_pos = -sin_n;
        let cos_flat_neg = sin_n;
        var max_cos_pos = cos_flat_pos;
        var max_cos_neg = cos_flat_neg;

        for (var step_i: u32 = 0u; step_i < steps; step_i = step_i + 1u) {
            // Per-(slice, step) R1 golden-ratio sequence breaks radial phase locking across slices.
            let step_base = f32(s + step_i * steps) * 0.6180339887;
            let step_noise = fract(noise.y + step_base);
            let s_norm = (f32(step_i) + step_noise) / f32(steps);
            let s_dist = pow(s_norm, 1.65) + min_s;
            let offset_px = omega * (s_dist * radius_px);
            let offset_len = length(offset_px);

            // Conservative MIP bias (-3.3) keeps near/mid occluders sharp while far taps stay cache-hot.
            let mip_f = clamp(log2(max(offset_len, 1.0)) - 3.3, 0.0, f32(max_mip));
            let mip_u = u32(floor(mip_f));
            let snapped = round(offset_px);

            max_cos_pos = sample_horizon_mip(
                px_c + snapped,
                mip_u,
                max_c,
                pos_biased,
                view_dir,
                falloff_mul,
                falloff_add,
                cos_flat_pos,
                max_cos_pos,
            );
            max_cos_neg = sample_horizon_mip(
                px_c - snapped,
                mip_u,
                max_c,
                pos_biased,
                view_dir,
                falloff_mul,
                falloff_add,
                cos_flat_neg,
                max_cos_neg,
            );
        }

        let h1_raw = -acos(clamp(max_cos_neg, -1.0, 1.0));
        let h2_raw = acos(clamp(max_cos_pos, -1.0, 1.0));
        let h1 = n_angle + max(h1_raw - n_angle, -GTAO_HALF_PI);
        let h2 = n_angle + min(h2_raw - n_angle, GTAO_HALF_PI);
        let arc_sum = integrate_arc(h1, n_angle, cos_n, sin_n)
            + integrate_arc(h2, n_angle, cos_n, sin_n);
        let unoccluded = max(cos_n + n_angle * sin_n, 1e-4);
        let slice_vis = clamp(arc_sum / unoccluded, 0.0, 1.25);

        let p_w = mix(proj_n_len, 1.0, 0.05);
        vis_sum = vis_sum + p_w * slice_vis;
        weight_sum = weight_sum + p_w;
    }

    var visibility = select(1.0, vis_sum / weight_sum, weight_sum > 1e-5);
    // Gentle fade-out for tiny sub-10px screen radii avoids single-pixel shimmering in wide views.
    visibility = visibility + clamp((10.0 - raw_radius_px) / 100.0, 0.0, 1.0) * 0.5;
    visibility = clamp(visibility, 0.03, GTAO_OCCLUSION_SCALE);

    return MainOut(
        vec4<f32>(visibility / GTAO_OCCLUSION_SCALE, 0.0, 0.0, 1.0),
        vec4<f32>(packed_edge, 0.0, 0.0, 1.0),
    );
}
