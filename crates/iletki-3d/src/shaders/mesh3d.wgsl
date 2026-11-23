// Camera-relative Reversed-Z 3D mesh, depth-offset edge, outline-mask and picking shaders
// (prepended with `iletki_wgpu::CAMERA_WGSL`).
// Supports up to 6 world-space section clip planes (`iletki_is_clipped`) and 64-bit GPU picking.

struct ObjectUniforms {
    // (chunk_origin - camera_pos) computed in f64 on CPU, plus uniform scale for gizmos.
    offset_rel: vec3<f32>,
    scale: f32,
};

@group(1) @binding(0)
var<uniform> object: ObjectUniforms;

struct VertIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

struct PickIn {
    @location(3) pick_id: vec2<u32>,
};

struct ColorOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) pos_rel: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

struct MaskOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) pos_rel: vec3<f32>,
};

struct PickOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) pos_rel: vec3<f32>,
    @location(1) @interpolate(flat) pick_id: vec2<u32>,
};

fn mesh_vertex(v: VertIn) -> ColorOut {
    let pos_rel = v.position * object.scale + object.offset_rel;
    var out: ColorOut;
    out.clip_pos = iletki_project_3d(pos_rel);
    out.pos_rel = pos_rel;
    out.normal = v.normal;
    out.color = v.color;
    return out;
}

@vertex
fn vs_mesh(v: VertIn) -> ColorOut {
    return mesh_vertex(v);
}

// Reversed-Z depth-offset vertex shader for crisp 3D CAD edges over solid faces without z-fighting.
fn edge_vertex(v: VertIn) -> ColorOut {
    var out = mesh_vertex(v);
    var clip = out.clip_pos;
    // In infinite Reversed-Z (near = 1.0, far = 0.0), clip.z scales with `near` (Perspective) or
    // linear depth (Ortho). Scaling multiplicatively pulls the edge ~0.08% closer across every
    // distance without ever punching through solid geometry behind it.
    clip.z = min(clip.z * 1.0008 + 1e-5, clip.w);
    out.clip_pos = clip;
    return out;
}

@vertex
fn vs_edge(v: VertIn) -> ColorOut {
    return edge_vertex(v);
}

@vertex
fn vs_mask(v: VertIn) -> MaskOut {
    let pos_rel = v.position * object.scale + object.offset_rel;
    var out: MaskOut;
    out.clip_pos = iletki_project_3d(pos_rel);
    out.pos_rel = pos_rel;
    return out;
}

@vertex
fn vs_mesh_pick(v: VertIn, p: PickIn) -> PickOut {
    let pos_rel = v.position * object.scale + object.offset_rel;
    var out: PickOut;
    out.clip_pos = iletki_project_3d(pos_rel);
    out.pos_rel = pos_rel;
    out.pick_id = p.pick_id;
    return out;
}

fn shade_mesh(pos_rel: vec3<f32>, normal: vec3<f32>, color: vec4<f32>, is_front: bool) -> vec4<f32> {
    let n_raw = normalize(normal);
    // Flip normals on interior back-faces revealed by section clip planes and shade slightly
    // darker so cutaway cavities read cleanly like professional CAD/BIM viewports.
    let n = select(-n_raw, n_raw, is_front);
    let view_dir = normalize(-pos_rel);
    // Combine key and fill lights, hemispherical ambient light, and a view-aligned light
    // to keep faces readable as the camera orbits.
    let key_dir = normalize(vec3<f32>(0.45, 0.35, 0.82));
    let fill_dir = normalize(vec3<f32>(-0.6, -0.5, 0.3));
    let hemi = mix(0.22, 0.36, n.z * 0.5 + 0.5);
    let key = max(dot(n, key_dir), 0.0) * 0.48;
    let fill = max(dot(n, fill_dir), 0.0) * 0.16;
    let head = max(dot(n, view_dir), 0.0) * 0.20;
    let half_v = normalize(key_dir + view_dir);
    let spec = pow(max(dot(n, half_v), 0.0), 32.0) * 0.07;
    let back_tint = select(0.72, 1.0, is_front);
    let shade = min(hemi + key + fill + head, 1.05) * back_tint;
    let base_rgb = iletki_color_to_target(color.rgb);
    let lit = base_rgb * shade + vec3<f32>(spec * back_tint);
    return vec4<f32>(lit, color.a);
}

@fragment
fn fs_lit(in: ColorOut, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    if (iletki_is_clipped(in.pos_rel)) {
        discard;
    }
    return shade_mesh(in.pos_rel, in.normal, in.color, is_front);
}

@fragment
fn fs_edge(in: ColorOut) -> @location(0) vec4<f32> {
    if (iletki_is_clipped(in.pos_rel)) {
        discard;
    }
    return vec4<f32>(iletki_color_to_target(in.color.rgb), in.color.a);
}

@fragment
fn fs_unlit(in: ColorOut) -> @location(0) vec4<f32> {
    return vec4<f32>(iletki_color_to_target(in.color.rgb), in.color.a);
}

@fragment
fn fs_mask(in: MaskOut) -> @location(0) f32 {
    if (iletki_is_clipped(in.pos_rel)) {
        discard;
    }
    return 1.0;
}

@fragment
fn fs_pick(in: PickOut) -> @location(0) vec2<u32> {
    if (iletki_is_clipped(in.pos_rel)) {
        discard;
    }
    return in.pick_id;
}

@fragment
fn fs_pick_unclipped(in: PickOut) -> @location(0) vec2<u32> {
    return in.pick_id;
}
