// GPU-instanced camera-relative Reversed-Z 3D mesh shader (prepended with `iletki_wgpu::CAMERA_WGSL`).
// Draws 100,000+ repeated CAD/BIM/game objects in 1 draw call with 6 section clip planes and
// per-instance 64-bit GPU picking.

struct BatchUniforms {
    // (batch_origin - camera_pos) computed in f64 on CPU every frame per ViewportId.
    offset_rel: vec3<f32>,
    _pad: f32,
};

@group(1) @binding(0)
var<uniform> batch: BatchUniforms;

struct VertIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct InstIn {
    @location(2) inst_offset: vec3<f32>,
    @location(3) inst_scale: vec3<f32>,
    @location(4) inst_color: vec4<f32>,
    @location(5) pick_id: vec2<u32>,
};

struct ColorOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) pos_rel: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

struct PickOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) pos_rel: vec3<f32>,
    @location(1) @interpolate(flat) pick_id: vec2<u32>,
};

@vertex
fn vs_inst_color(v: VertIn, inst: InstIn) -> ColorOut {
    let pos_rel = v.position * inst.inst_scale + inst.inst_offset + batch.offset_rel;
    var out: ColorOut;
    out.clip_pos = iletki_project_3d(pos_rel);
    out.pos_rel = pos_rel;
    out.normal = v.normal;
    out.color = inst.inst_color;
    return out;
}

@vertex
fn vs_inst_pick(v: VertIn, inst: InstIn) -> PickOut {
    let pos_rel = v.position * inst.inst_scale + inst.inst_offset + batch.offset_rel;
    var out: PickOut;
    out.clip_pos = iletki_project_3d(pos_rel);
    out.pos_rel = pos_rel;
    out.pick_id = inst.pick_id;
    return out;
}

@fragment
fn fs_inst_lit(in: ColorOut, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    if (iletki_is_clipped(in.pos_rel)) {
        discard;
    }
    let n_raw = normalize(in.normal);
    let n = select(-n_raw, n_raw, is_front);
    let view_dir = normalize(-in.pos_rel);
    let key_dir = normalize(vec3<f32>(0.45, 0.35, 0.82));
    let hemi = mix(0.22, 0.36, n.z * 0.5 + 0.5);
    let key = max(dot(n, key_dir), 0.0) * 0.48;
    let fill = max(dot(n, normalize(vec3<f32>(-0.6, -0.5, 0.3))), 0.0) * 0.16;
    let head = max(dot(n, view_dir), 0.0) * 0.20;
    let back_tint = select(0.72, 1.0, is_front);
    let shade = min(hemi + key + fill + head, 1.05) * back_tint;
    let base_rgb = iletki_color_to_target(in.color.rgb);
    return vec4<f32>(base_rgb * shade, in.color.a);
}

@fragment
fn fs_inst_pick(in: PickOut) -> @location(0) vec2<u32> {
    if (iletki_is_clipped(in.pos_rel)) {
        discard;
    }
    return in.pick_id;
}
