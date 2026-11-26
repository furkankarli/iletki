// Half-resolution depth-only Ground-Truth Ambient Occlusion (GTAO) horizon integration pass
// (prepended with `gtao_common.wgsl`).

@group(0) @binding(0)
var half_depth_tex: texture_2d<u32>;

@group(0) @binding(1)
var<uniform> gtao: GtaoUniform;

const GTAO_PI: f32 = 3.141592653589793;
const GTAO_HALF_PI: f32 = 1.5707963267948966;

fn load_half_depth(coord: vec2<i32>) -> f32 {
    let max_c = vec2<i32>(max(i32(gtao.half_px.x) - 1, 0), max(i32(gtao.half_px.y) - 1, 0));
    let clamped = clamp(coord, vec2<i32>(0, 0), max_c);
    return bitcast<f32>(textureLoad(half_depth_tex, clamped, 0).r);
}

// Minimum-depth-discontinuity 4-neighbor normal reconstruction (eliminates silhouette halos on
// sharp CAD corners and thin walls).
fn reconstruct_normal(coord: vec2<i32>, pos_c: vec3<f32>, view_dir: vec3<f32>) -> vec3<f32> {
    let z_l = load_half_depth(coord + vec2<i32>(-1, 0));
    let z_r = load_half_depth(coord + vec2<i32>(1, 0));
    let z_u = load_half_depth(coord + vec2<i32>(0, -1));
    let z_d = load_half_depth(coord + vec2<i32>(0, 1));

    var dx = vec3<f32>(0.0);
    var has_dx = false;
    let has_l = z_l > 0.0 && coord.x > 0;
    let has_r = z_r > 0.0 && coord.x + 1 < i32(gtao.half_px.x);
    if (has_l && has_r) {
        let dl = pos_c - gtao_view_pos_half(gtao, coord + vec2<i32>(-1, 0), z_l);
        let dr = gtao_view_pos_half(gtao, coord + vec2<i32>(1, 0), z_r) - pos_c;
        dx = select(dr, dl, abs(dl.z) < abs(dr.z));
        has_dx = true;
    } else if (has_r) {
        dx = gtao_view_pos_half(gtao, coord + vec2<i32>(1, 0), z_r) - pos_c;
        has_dx = true;
    } else if (has_l) {
        dx = pos_c - gtao_view_pos_half(gtao, coord + vec2<i32>(-1, 0), z_l);
        has_dx = true;
    }

    var dy = vec3<f32>(0.0);
    var has_dy = false;
    let has_u = z_u > 0.0 && coord.y > 0;
    let has_d = z_d > 0.0 && coord.y + 1 < i32(gtao.half_px.y);
    if (has_u && has_d) {
        let du = gtao_view_pos_half(gtao, coord + vec2<i32>(0, -1), z_u) - pos_c;
        let dd = pos_c - gtao_view_pos_half(gtao, coord + vec2<i32>(0, 1), z_d);
        dy = select(dd, du, abs(du.z) < abs(dd.z));
        has_dy = true;
    } else if (has_u) {
        dy = gtao_view_pos_half(gtao, coord + vec2<i32>(0, -1), z_u) - pos_c;
        has_dy = true;
    } else if (has_d) {
        dy = pos_c - gtao_view_pos_half(gtao, coord + vec2<i32>(0, 1), z_d);
        has_dy = true;
    }

    if (!has_dx || !has_dy) {
        return view_dir;
    }
    let c = cross(dx, dy);
    let clen = length(c);
    if (clen <= 1e-8) {
        return view_dir;
    }
    var n = c / clen;
    if (dot(n, view_dir) < 0.0) {
        n = -n;
    }
    return n;
}

// Deterministic Interleaved Gradient Noise (Jimenez) for static-frame stability in OnDemand CAD.
fn ign(coord: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(coord, vec2<f32>(0.06711056, 0.00583715))));
}

fn integrate_arc(h: f32, n: f32, cos_n: f32, sin_n: f32) -> f32 {
    return 0.25 * (-cos(2.0 * h - n) + cos_n + 2.0 * h * sin_n);
}

@fragment
fn fs_gtao_ao(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(frag_pos.xy);
    let z_c = load_half_depth(coord);
    if (z_c <= 0.0) {
        return vec4<f32>(1.0, 0.0, 0.0, 1.0);
    }

    let pos_c = gtao_view_pos_half(gtao, coord, z_c);
    let view_dir = gtao_view_dir(gtao, pos_c);
    let normal = reconstruct_normal(coord, pos_c, view_dir);
    let dist_c = max(-pos_c.z, 1e-4);

    // Compute sampling radius in half-res pixels from world-space radius.
    let half_h = max(f32(gtao.half_px.y), 1.0);
    let world_h = select(
        2.0 * gtao.proj_xy.y,
        2.0 * gtao.proj_xy.y * dist_c,
        (gtao.flags & 1u) != 0u,
    );
    let raw_radius_px = (gtao.radius * half_h) / max(world_h, 1e-5);
    if (raw_radius_px < 1.25) {
        return vec4<f32>(1.0, 0.0, 0.0, 1.0);
    }
    let radius_px = min(raw_radius_px, gtao.max_half_radius_px);

    // Clamp effective world falloff radius when screen-space clamping shrinks the search kernel
    // at very close zoom distances, keeping contact shading crisp.
    let eff_world_radius = max(gtao.radius * (radius_px / raw_radius_px), 1e-4);
    let inv_r2 = 1.0 / (eff_world_radius * eff_world_radius);
    // Small depth bias along normal prevents flat-surface self-occlusion from fp32 quantization.
    let pos_biased = pos_c + normal * max(eff_world_radius * 0.008, dist_c * 0.0005);

    let noise_phi = ign(vec2<f32>(coord));
    let noise_step = ign(vec2<f32>(coord) + vec2<f32>(7.0, 13.0));
    let slices = clamp(gtao.slices, 1u, 4u);
    let steps = clamp(gtao.steps_per_side, 2u, 6u);
    let max_coord = vec2<i32>(max(i32(gtao.half_px.x) - 1, 0), max(i32(gtao.half_px.y) - 1, 0));

    var vis_sum = 0.0;
    var weight_sum = 0.0;

    for (var s: u32 = 0u; s < slices; s = s + 1u) {
        let phi = (f32(s) + noise_phi) * (GTAO_PI / f32(slices));
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

        var max_cos_neg = -1.0;
        var max_cos_pos = -1.0;

        for (var step_i: u32 = 0u; step_i < steps; step_i = step_i + 1u) {
            let t = (f32(step_i) + 0.35 + 0.55 * noise_step) / f32(steps);
            // Quadratic step distribution concentrates samples near fine CAD crevices.
            let sample_dist_px = max(1.0, t * t * radius_px);
            let offset_px = vec2<i32>(round(omega * sample_dist_px));
            if (offset_px.x == 0 && offset_px.y == 0) {
                continue;
            }

            // Positive side (+omega)
            let cp = coord + offset_px;
            if (cp.x >= 0 && cp.y >= 0 && cp.x <= max_coord.x && cp.y <= max_coord.y) {
                let zp = load_half_depth(cp);
                if (zp > 0.0) {
                    let delta = gtao_view_pos_half(gtao, cp, zp) - pos_biased;
                    let len2 = dot(delta, delta);
                    if (len2 > 1e-10) {
                        let falloff = clamp(len2 * inv_r2, 0.0, 1.0);
                        let cos_h = dot(delta, view_dir) * inverseSqrt(len2);
                        let eff = mix(cos_h, -1.0, falloff * falloff);
                        max_cos_pos = max(max_cos_pos, eff);
                    }
                }
            }

            // Negative side (-omega)
            let cn = coord - offset_px;
            if (cn.x >= 0 && cn.y >= 0 && cn.x <= max_coord.x && cn.y <= max_coord.y) {
                let zn = load_half_depth(cn);
                if (zn > 0.0) {
                    let delta = gtao_view_pos_half(gtao, cn, zn) - pos_biased;
                    let len2 = dot(delta, delta);
                    if (len2 > 1e-10) {
                        let falloff = clamp(len2 * inv_r2, 0.0, 1.0);
                        let cos_h = dot(delta, view_dir) * inverseSqrt(len2);
                        let eff = mix(cos_h, -1.0, falloff * falloff);
                        max_cos_neg = max(max_cos_neg, eff);
                    }
                }
            }
        }

        let h1_raw = -acos(clamp(max_cos_neg, -1.0, 1.0));
        let h2_raw = acos(clamp(max_cos_pos, -1.0, 1.0));
        let h1 = n_angle + max(h1_raw - n_angle, -GTAO_HALF_PI);
        let h2 = n_angle + min(h2_raw - n_angle, GTAO_HALF_PI);
        let arc_sum = integrate_arc(h1, n_angle, cos_n, sin_n)
            + integrate_arc(h2, n_angle, cos_n, sin_n);
        // Normalize by exact unoccluded slice integral so tilted flat planes evaluate to 1.000
        // with zero rotational banding even at 2 slices.
        let unoccluded = max(cos_n + n_angle * sin_n, 1e-4);
        let slice_vis = clamp(arc_sum / unoccluded, 0.0, 1.0);

        vis_sum = vis_sum + proj_n_len * slice_vis;
        weight_sum = weight_sum + proj_n_len;
    }

    let visibility = select(1.0, clamp(vis_sum / weight_sum, 0.0, 1.0), weight_sum > 1e-5);
    return vec4<f32>(visibility, 0.0, 0.0, 1.0);
}
