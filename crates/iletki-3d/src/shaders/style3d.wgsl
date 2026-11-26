// CAD visual style (`Technical`) and engineering diagnostic (`Normals`, `Facets`, `Backfaces`,
// `Overdraw`, `Chunks`, `Depth`) shader routines (prepended with `CAMERA_WGSL` + `mesh3d.wgsl` +
// optional batched/storage transform helpers).

struct StyleUniform {
    // x = mode (0=Technical, 1=Normals, 2=Facets, 3=Backfaces, 4=Overdraw, 5=Chunks, 6=Depth)
    // y = flags (bit 0 = clip_active)
    // z, w = reserved
    params: vec4<u32>,
    // Primary tint/paper color in sRGB [0..1].
    tint: vec4<f32>,
};

@group(2) @binding(0)
var<uniform> style_u: StyleUniform;

struct StyleOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) pos_rel: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) @interpolate(flat) chunk_id: u32,
};

fn style_vertex(position: vec3<f32>, normal: vec3<f32>, placement: ObjectUniforms, chunk_id: u32) -> StyleOut {
    let pos_rel = position * placement.scale + placement.offset_rel;
    return StyleOut(iletki_project_3d(pos_rel), pos_rel, normal, chunk_id);
}

fn style_edge_vertex(position: vec3<f32>, normal: vec3<f32>, placement: ObjectUniforms, chunk_id: u32) -> StyleOut {
    var out = style_vertex(position, normal, placement, chunk_id);
    var clip = out.clip_pos;
    clip.z = min(clip.z * 1.0008 + 1e-5, clip.w);
    out.clip_pos = clip;
    return out;
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let k = vec3<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0);
    let p = abs(fract(vec3<f32>(h) + k) * 6.0 - vec3<f32>(3.0));
    return v * mix(vec3<f32>(1.0), clamp(p - vec3<f32>(1.0), vec3<f32>(0.0), vec3<f32>(1.0)), s);
}

fn chunk_palette_srgb(chunk_id: u32) -> vec3<f32> {
    var h = (chunk_id + 1u) * 2654435761u;
    h = (h ^ (h >> 16u)) * 2246822519u;
    h = h ^ (h >> 13u);
    let hue = f32(h & 255u) / 255.0;
    let sat = 0.62 + 0.28 * (f32((h >> 8u) & 15u) / 15.0);
    let val = 0.72 + 0.24 * (f32((h >> 12u) & 15u) / 15.0);
    return hsv_to_rgb(hue, sat, val);
}

@fragment
fn fs_style(in: StyleOut, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    // Evaluate screen-space derivatives in uniform control flow prior to any clip discard.
    let facet_cross = cross(dpdy(in.pos_rel), dpdx(in.pos_rel));

    if ((style_u.params.y & 1u) != 0u && iletki_is_clipped(in.pos_rel)) {
        discard;
    }
    return style_color(in, is_front, facet_cross);
}

// Shared by the pixel-rate entry point above and the antialiased section entry points.
// `facet_cross` must be evaluated by the caller in uniform control flow.
fn style_color(in: StyleOut, is_front: bool, facet_cross: vec3<f32>) -> vec4<f32> {
    let mode = style_u.params.x;
    let view_dir = normalize(-in.pos_rel);
    let n_len = length(in.normal);
    let n_raw = select(vec3<f32>(0.0, 0.0, 1.0), in.normal / n_len, n_len > 1e-6);
    let n = select(-n_raw, n_raw, is_front);
    let key_dir = normalize(vec3<f32>(0.45, 0.35, 0.82));
    let ndotl = max(dot(n, key_dir), 0.0);
    let head = max(dot(n, view_dir), 0.0);

    // 0: Technical (Hidden-Line paper fill with subtle directional modeling)
    if (mode == 0u) {
        let paper = iletki_color_to_target(style_u.tint.rgb);
        let shade = select(0.82, 0.88 + 0.12 * (ndotl * 0.6 + head * 0.4), is_front);
        return vec4<f32>(paper * shade, 1.0);
    }

    // 1: Normals (world-space normal mapped [-1, 1] -> [0, 1])
    if (mode == 1u) {
        let srgb = clamp(n * 0.5 + vec3<f32>(0.5), vec3<f32>(0.0), vec3<f32>(1.0));
        return vec4<f32>(iletki_color_to_target(srgb), 1.0);
    }

    // 2: Facets (geometric triangle flat normal via dpdx/dpdy)
    if (mode == 2u) {
        let flen = length(facet_cross);
        var fn_vec = select(n, facet_cross / flen, flen > 1e-8);
        if (dot(fn_vec, view_dir) < 0.0) {
            fn_vec = -fn_vec;
        }
        let fkey = max(dot(fn_vec, key_dir), 0.0) * 0.52;
        let fhead = max(dot(fn_vec, view_dir), 0.0) * 0.28;
        let fhemi = mix(0.24, 0.38, fn_vec.z * 0.5 + 0.5);
        let facet_tint = mix(vec3<f32>(0.84, 0.88, 0.94), fn_vec * 0.5 + vec3<f32>(0.5), 0.22);
        let rgb = iletki_color_to_target(facet_tint) * min(fhemi + fkey + fhead, 1.05);
        return vec4<f32>(rgb, 1.0);
    }

    // 3: Backfaces (front = cool blue-teal, back = vivid orange-red)
    if (mode == 3u) {
        let base_srgb = select(vec3<f32>(0.95, 0.20, 0.12), vec3<f32>(0.18, 0.56, 0.88), is_front);
        let shade = 0.38 + 0.42 * ndotl + 0.22 * head;
        return vec4<f32>(iletki_color_to_target(base_srgb) * shade, 1.0);
    }

    // 4: Overdraw (additive pixel accumulation ramp)
    if (mode == 4u) {
        return vec4<f32>(0.14, 0.055, 0.018, 1.0);
    }

    // 5: Chunks (deterministic per-chunk palette)
    if (mode == 5u) {
        let base_srgb = chunk_palette_srgb(in.chunk_id);
        let shade = (0.42 + 0.38 * ndotl + 0.22 * head) * select(0.75, 1.0, is_front);
        return vec4<f32>(iletki_color_to_target(base_srgb) * shade, 1.0);
    }

    // 6: Depth (Reversed-Z depth ramp: near = bright, far = dark)
    let z_rev = clamp(in.clip_pos.z, 0.0, 1.0);
    let is_persp = (iletki_camera.flags & 1u) != 0u;
    let vis = select(z_rev, pow(z_rev, 0.35), is_persp);
    let depth_srgb = mix(vec3<f32>(0.04, 0.08, 0.16), vec3<f32>(0.95, 0.98, 1.0), vis);
    return vec4<f32>(iletki_color_to_target(depth_srgb), 1.0);
}

@fragment
fn fs_style_edge(in: StyleOut) -> @location(0) vec4<f32> {
    if ((style_u.params.y & 1u) != 0u && iletki_is_clipped(in.pos_rel)) {
        discard;
    }
    return style_edge_color();
}

fn style_edge_color() -> vec4<f32> {
    return vec4<f32>(iletki_color_to_target(style_u.tint.rgb), style_u.tint.a);
}
