// Weighted Blended Order-Independent Transparency (McGuire & Bavoil) accumulation helpers
// adapted for infinite Reversed-Z Perspective3D and linear Reversed-Z Ortho3D.
// Prepended with `CAMERA_WGSL` + `mesh3d.wgsl` + `material3d.wgsl`.

struct OitFragOut {
    @location(0) accum: vec4<f32>,
    @location(1) reveal: vec4<f32>,
};

fn oit_default_material() -> MaterialGpu {
    return MaterialGpu(
        vec4<f32>(1.0, 1.0, 1.0, 1.0),
        vec4<f32>(0.0, 0.0, 0.0, 0.55),
        0.0,
        1.0,
        0.72,
        0u,
    );
}

fn unpack_chunk_opacity(flags: u32) -> f32 {
    let code = (flags >> 3u) & 31u;
    if (code == 0u) {
        return 1.0;
    }
    return f32(code - 1u) / 30.0;
}

// Reversed-Z depth weighting tuned to keep fp16 accumulation overflow-free across 16+ layers
// while strongly prioritizing front translucent surfaces over distant ones.
fn oit_weight(z_rev: f32, alpha: f32) -> f32 {
    let z = clamp(z_rev, 1e-4, 1.0);
    let is_persp = (iletki_camera.flags & 1u) != 0u;
    let depth_w = select(
        exp2(z * 12.0 - 6.0),
        clamp(1e4 * z * z, 1e-4, 2e3),
        is_persp,
    );
    return clamp(max(alpha, 1e-3) * depth_w, 1e-4, 2e3);
}

fn evaluate_oit(
    clip_z: f32,
    pos_rel: vec3<f32>,
    normal: vec3<f32>,
    color: vec4<f32>,
    is_front: bool,
    mat: MaterialGpu,
    chunk_opacity: f32,
    chunk_ghost: bool,
    xray_opacity: f32,
    global_flags: u32,
) -> OitFragOut {
    if ((global_flags & 2u) != 0u && iletki_is_clipped(pos_rel)) {
        discard;
    }
    let lit = shade_mesh_material(pos_rel, normal, color, is_front, mat);
    let n_len = length(normal);
    let n_raw = select(vec3<f32>(0.0, 0.0, 1.0), normal / n_len, n_len > 1e-6);
    let n = select(-n_raw, n_raw, is_front);
    let view_dir = normalize(-pos_rel);
    let fresnel = pow(1.0 - clamp(abs(dot(n, view_dir)), 0.0, 1.0), 2.2);

    let is_xray = (global_flags & 1u) != 0u;
    let is_ghost = chunk_ghost || ((mat.flags & 4u) != 0u);
    var alpha = clamp(lit.a * chunk_opacity, 0.0, 1.0);
    var rgb = lit.rgb;
    if (is_xray) {
        let x_op = clamp(xray_opacity, 0.04, 0.95);
        alpha = clamp(x_op * (0.65 + 1.35 * fresnel), 0.04, 0.88);
        let lum = dot(rgb, vec3<f32>(0.299, 0.587, 0.114));
        let cool_tint = iletki_color_to_target(vec3<f32>(0.42, 0.72, 0.95));
        rgb = mix(rgb, mix(vec3<f32>(lum), cool_tint, 0.55), 0.35) + cool_tint * (fresnel * 0.28);
    } else if (is_ghost) {
        let use_default_ghost = chunk_opacity >= 0.999 && (mat.flags & 4u) == 0u;
        let g_op = select(min(alpha, 0.35), min(alpha * 0.25, 0.35), use_default_ghost);
        alpha = clamp(g_op + fresnel * 0.16, 0.04, 0.85);
        let lum = dot(rgb, vec3<f32>(0.299, 0.587, 0.114));
        let ghost_tint = iletki_color_to_target(vec3<f32>(0.68, 0.78, 0.88));
        rgb = mix(rgb, mix(vec3<f32>(lum), ghost_tint, 0.45), 0.45) + ghost_tint * (fresnel * 0.18);
    }
    if (alpha <= 1e-4) {
        discard;
    }
    let w = oit_weight(clip_z, alpha);
    return OitFragOut(
        vec4<f32>(rgb * alpha * w, alpha * w),
        vec4<f32>(alpha, alpha, alpha, alpha),
    );
}
