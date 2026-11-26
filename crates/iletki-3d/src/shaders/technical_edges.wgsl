// Fullscreen screen-space silhouette & geometric crease edge overlay for `VisualStyle::Technical`.
// Reads the existing Reversed-Z scene depth target directly (zero extra textures or prepasses) and
// evaluates edges per MSAA subsample when multisampling is active, producing true 4x MSAA
// anti-aliased silhouettes, section-cut lips and geometric creases on edge-less IFC/STL/CAD meshes.
//
// Line ownership: authored mesh edges are drawn by the edge pipelines. Crease detection only runs
// while the scene contains faces without authored edges (flag bit 2), and this overlay composites
// with `Min` blending so ink never accumulates where it overlaps an authored edge line. `Min`
// assumes dark ink on lighter paper, which is how `VisualStyle::Technical` is defined.

struct TechEdgeUniform {
    // Perspective: (tan_x, tan_y); Ortho3D: (half_w_world, half_h_world).
    proj_xy: vec2<f32>,
    // Perspective: (near, 0.0); Ortho3D: (far, far - near).
    depth_params: vec2<f32>,
    // Active physical viewport dimensions [aw, ah].
    active_px: vec2<u32>,
    // Bit 0: is_perspective; bit 1: is_ortho_3d; bit 2: detect creases.
    flags: u32,
    // Scene MSAA sample count (1 or 4).
    msaa_samples: u32,
    // Ink RGBA in the color target's space (linear for sRGB targets, sRGB for unorm targets).
    ink_color: vec4<f32>,
    // Brightest paper color in the same space; partial ink coverage fades toward it.
    paper_color: vec4<f32>,
};

@group(0) @binding(0)
__DEPTH_DECL__

@group(0) @binding(1)
var<uniform> tech_u: TechEdgeUniform;

@vertex
fn vs_tech_edge(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(pos[vid], 0.0, 1.0);
}

fn load_depth_sample(c: vec2<i32>, max_c: vec2<i32>, sample_idx: i32) -> f32 {
    let clamped = clamp(c, vec2<i32>(0, 0), max_c);
    __DEPTH_SAMPLE_LOAD__
}

fn tech_linear_depth(z_rev: f32) -> f32 {
    if ((tech_u.flags & 1u) != 0u) {
        return tech_u.depth_params.x / max(z_rev, 1e-6);
    }
    return tech_u.depth_params.x - z_rev * tech_u.depth_params.y;
}

fn tech_view_pos(c: vec2<i32>, z_rev: f32) -> vec3<f32> {
    let d = tech_linear_depth(z_rev);
    let aw = max(f32(tech_u.active_px.x), 1.0);
    let ah = max(f32(tech_u.active_px.y), 1.0);
    let uv = (vec2<f32>(c) + vec2<f32>(0.5)) / vec2<f32>(aw, ah);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    if ((tech_u.flags & 1u) != 0u) {
        return vec3<f32>(ndc.x * tech_u.proj_xy.x * d, ndc.y * tech_u.proj_xy.y * d, -d);
    }
    return vec3<f32>(ndc.x * tech_u.proj_xy.x, ndc.y * tech_u.proj_xy.y, -d);
}

fn quad_normal(dx: vec3<f32>, dy: vec3<f32>, view_dir: vec3<f32>) -> vec4<f32> {
    let c = cross(dx, dy);
    let len = length(c);
    if (len <= 1e-8) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    var n = c / len;
    if (dot(n, view_dir) < 0.0) {
        n = -n;
    }
    return vec4<f32>(n, 1.0);
}

// Evaluates silhouette + geometric crease alpha for a single hardware MSAA subsample index `s`.
// Comparing matching subsample indices across 4-neighbors preserves exact 1-pixel finite-difference
// spacing while giving true 4x subpixel anti-aliased edge coverage when averaged across `s = 0..4`.
fn eval_sample_edge(c: vec2<i32>, max_c: vec2<i32>, sample_idx: i32) -> f32 {
    let z_c = load_depth_sample(c, max_c, sample_idx);
    if (z_c <= 0.0) {
        return 0.0;
    }

    let pos_c = tech_view_pos(c, z_c);
    let d_c = max(-pos_c.z, 1e-4);
    let is_persp = (tech_u.flags & 1u) != 0u;
    let view_dir = select(
        vec3<f32>(0.0, 0.0, 1.0),
        normalize(-pos_c),
        is_persp && dot(pos_c, pos_c) > 1e-8,
    );

    let ah = max(f32(tech_u.active_px.y), 1.0);
    let world_h = select(2.0 * tech_u.proj_xy.y, 2.0 * tech_u.proj_xy.y * d_c, is_persp);
    let px_world = max(world_h / ah, 1e-5);

    let c_l = c + vec2<i32>(-1, 0);
    let c_r = c + vec2<i32>(1, 0);
    let c_u = c + vec2<i32>(0, -1);
    let c_d = c + vec2<i32>(0, 1);

    let in_l = c.x > 0;
    let in_r = c.x < max_c.x;
    let in_u = c.y > 0;
    let in_d = c.y < max_c.y;

    let z_l = select(z_c, load_depth_sample(c_l, max_c, sample_idx), in_l);
    let z_r = select(z_c, load_depth_sample(c_r, max_c, sample_idx), in_r);
    let z_u = select(z_c, load_depth_sample(c_u, max_c, sample_idx), in_u);
    let z_d = select(z_c, load_depth_sample(c_d, max_c, sample_idx), in_d);

    // 1. Outer silhouette against background (foreground subsample next to empty sky).
    var edge_alpha = 0.0;
    if ((in_l && z_l <= 0.0) || (in_r && z_r <= 0.0) || (in_u && z_u <= 0.0) || (in_d && z_d <= 0.0)) {
        edge_alpha = 0.92;
    }

    let pos_l = select(pos_c, tech_view_pos(c_l, z_l), in_l && z_l > 0.0);
    let pos_r = select(pos_c, tech_view_pos(c_r, z_r), in_r && z_r > 0.0);
    let pos_u = select(pos_c, tech_view_pos(c_u, z_u), in_u && z_u > 0.0);
    let pos_d = select(pos_c, tech_view_pos(c_d, z_d), in_d && z_d > 0.0);

    let dl = pos_c - pos_l;
    let dr = pos_r - pos_c;
    let du = pos_u - pos_c;
    let dd = pos_c - pos_d;

    let has_l = in_l && z_l > 0.0;
    let has_r = in_r && z_r > 0.0;
    let has_u = in_u && z_u > 0.0;
    let has_d = in_d && z_d > 0.0;

    var min_dx = vec3<f32>(0.0);
    if (has_l && has_r) {
        min_dx = select(dr, dl, abs(dl.z) < abs(dr.z));
    } else if (has_r) {
        min_dx = dr;
    } else if (has_l) {
        min_dx = dl;
    }
    var min_dy = vec3<f32>(0.0);
    if (has_u && has_d) {
        min_dy = select(dd, du, abs(du.z) < abs(dd.z));
    } else if (has_u) {
        min_dy = du;
    } else if (has_d) {
        min_dy = dd;
    }
    let n_center_v = quad_normal(min_dx, min_dy, view_dir);
    let n_c = select(view_dir, n_center_v.xyz, n_center_v.w > 0.5);

    // 2. Occlusion depth-jump silhouette (foreground side only -> crisp 1 px stroke, zero halo).
    let sil_lo = max(px_world * 3.0, d_c * 0.006);
    let sil_hi = max(px_world * 7.5, d_c * 0.018);
    let jump_cap = max(px_world * 5.0, d_c * 0.015);

    let pl_l = abs(dot(pos_l - pos_c, n_c));
    let pl_r = abs(dot(pos_r - pos_c, n_c));
    let pl_u = abs(dot(pos_u - pos_c, n_c));
    let pl_d = abs(dot(pos_d - pos_c, n_c));

    var max_fg_jump = 0.0;
    if (has_l && z_c > z_l) {
        max_fg_jump = max(max_fg_jump, pl_l);
    }
    if (has_r && z_c > z_r) {
        max_fg_jump = max(max_fg_jump, pl_r);
    }
    if (has_u && z_c > z_u) {
        max_fg_jump = max(max_fg_jump, pl_u);
    }
    if (has_d && z_c > z_d) {
        max_fg_jump = max(max_fg_jump, pl_d);
    }
    let sil_strength = smoothstep(sil_lo, sil_hi, max_fg_jump) * 0.88;
    edge_alpha = max(edge_alpha, sil_strength);

    // Authored mesh edges own the creases of every chunk that has them.
    if ((tech_u.flags & 4u) == 0u) {
        return edge_alpha;
    }

    // 3. Geometric crease detection via 4 quadrant normals around `c`.
    let cont_l = has_l && pl_l < jump_cap && abs((-pos_l.z) - d_c) < max(jump_cap * 3.0, d_c * 0.06);
    let cont_r = has_r && pl_r < jump_cap && abs((-pos_r.z) - d_c) < max(jump_cap * 3.0, d_c * 0.06);
    let cont_u = has_u && pl_u < jump_cap && abs((-pos_u.z) - d_c) < max(jump_cap * 3.0, d_c * 0.06);
    let cont_d = has_d && pl_d < jump_cap && abs((-pos_d.z) - d_c) < max(jump_cap * 3.0, d_c * 0.06);

    let n_lu = select(vec4<f32>(0.0), quad_normal(dl, du, view_dir), cont_l && cont_u);
    let n_ru = select(vec4<f32>(0.0), quad_normal(dr, du, view_dir), cont_r && cont_u);
    let n_ld = select(vec4<f32>(0.0), quad_normal(dl, dd, view_dir), cont_l && cont_d);
    let n_rd = select(vec4<f32>(0.0), quad_normal(dr, dd, view_dir), cont_r && cont_d);

    var min_dot = 1.0;
    if (n_lu.w > 0.5 && n_ru.w > 0.5) {
        min_dot = min(min_dot, dot(n_lu.xyz, n_ru.xyz));
    }
    if (n_ld.w > 0.5 && n_rd.w > 0.5) {
        min_dot = min(min_dot, dot(n_ld.xyz, n_rd.xyz));
    }
    if (n_lu.w > 0.5 && n_ld.w > 0.5) {
        min_dot = min(min_dot, dot(n_lu.xyz, n_ld.xyz));
    }
    if (n_ru.w > 0.5 && n_rd.w > 0.5) {
        min_dot = min(min_dot, dot(n_ru.xyz, n_rd.xyz));
    }
    if (n_lu.w > 0.5 && n_rd.w > 0.5) {
        min_dot = min(min_dot, dot(n_lu.xyz, n_rd.xyz));
    }
    if (n_ru.w > 0.5 && n_ld.w > 0.5) {
        min_dot = min(min_dot, dot(n_ru.xyz, n_ld.xyz));
    }

    let crease_strength = smoothstep(0.78, 0.42, min_dot) * 0.78;
    return max(edge_alpha, crease_strength);
}

@fragment
fn fs_tech_edge(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let max_c = vec2<i32>(
        max(i32(tech_u.active_px.x) - 1, 0),
        max(i32(tech_u.active_px.y) - 1, 0),
    );
    let c = clamp(vec2<i32>(frag_pos.xy), vec2<i32>(0, 0), max_c);
    __RESOLVE_EDGE_ALPHA__
    if (edge_alpha <= 0.01) {
        discard;
    }
    let coverage = clamp(edge_alpha * tech_u.ink_color.a, 0.0, 1.0);
    // Composited with `Min`: the darker of this ink and the existing pixel wins.
    return vec4<f32>(mix(tech_u.paper_color.rgb, tech_u.ink_color.rgb, coverage), coverage);
}
