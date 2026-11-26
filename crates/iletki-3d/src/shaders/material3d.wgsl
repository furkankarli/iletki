// Persistent per-chunk CAD/BIM material evaluation (48-byte aligned MaterialGpu).
struct MaterialGpu {
    base_color_factor: vec4<f32>,
    emissive_roughness: vec4<f32>,
    metallic: f32,
    opacity: f32,
    backface_tint: f32,
    flags: u32,
};

fn shade_mesh_material(
    pos_rel: vec3<f32>,
    normal: vec3<f32>,
    color: vec4<f32>,
    is_front: bool,
    mat: MaterialGpu,
) -> vec4<f32> {
    let base_rgb = iletki_color_to_target(color.rgb) * mat.base_color_factor.rgb;
    let alpha = clamp(color.a * mat.base_color_factor.a * mat.opacity, 0.0, 1.0);
    // Bit 0: UNLIT.
    if ((mat.flags & 1u) != 0u) {
        return vec4<f32>(base_rgb + mat.emissive_roughness.rgb, alpha);
    }
    let n_raw = normalize(normal);
    let n = select(-n_raw, n_raw, is_front);
    let view_dir = normalize(-pos_rel);
    let key_dir = normalize(vec3<f32>(0.45, 0.35, 0.82));
    let fill_dir = normalize(vec3<f32>(-0.6, -0.5, 0.3));
    let hemi = mix(0.22, 0.36, n.z * 0.5 + 0.5);
    let key = max(dot(n, key_dir), 0.0) * 0.48;
    let fill = max(dot(n, fill_dir), 0.0) * 0.16;
    let head = max(dot(n, view_dir), 0.0) * 0.20;
    let rough = clamp(mat.emissive_roughness.w, 0.04, 1.0);
    let shininess = exp2(mix(8.0, 2.0, rough));
    let half_v = normalize(key_dir + view_dir);
    let spec_term = pow(max(dot(n, half_v), 0.0), shininess);
    let metallic = clamp(mat.metallic, 0.0, 1.0);
    let spec_strength = mix(0.07, 0.48, metallic) * mix(1.35, 0.25, rough);
    let spec_color = mix(vec3<f32>(1.0), base_rgb, metallic);
    // Bit 1: DOUBLE_SIDED_UNIFORM disables interior back-face darkening.
    let back_tint = select(mat.backface_tint, 1.0, is_front || (mat.flags & 2u) != 0u);
    let diffuse_scale = 1.0 - metallic * 0.65;
    let shade = min(hemi + key + fill + head, 1.05) * back_tint * diffuse_scale;
    let lit = base_rgb * shade + spec_color * (spec_term * spec_strength * back_tint) + mat.emissive_roughness.rgb;
    return vec4<f32>(lit, alpha);
}
