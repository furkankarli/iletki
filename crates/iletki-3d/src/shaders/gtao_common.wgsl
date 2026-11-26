// Shared uniform block, fullscreen triangle vertex shader, and Reversed-Z view-space
// reconstruction helpers for full-resolution XeGTAO (Perspective3D & Ortho3D).

struct GtaoUniform {
    // Perspective: (tan_x, tan_y); Ortho3D: (half_w_world, half_h_world).
    proj_xy: vec2<f32>,
    // Perspective: (near, 0.0); Ortho3D: (far, far - near).
    depth_params: vec2<f32>,
    // Active physical viewport dimensions [aw, ah].
    active_px: vec2<u32>,
    // Number of levels in the depth pyramid [1..=5].
    mip_count: u32,
    // Primary scene MSAA sample count (1 or 4).
    msaa_samples: u32,
    // World-space AO sampling radius.
    radius: f32,
    // Occlusion strength multiplier [0..3].
    intensity: f32,
    // Occlusion contrast exponent [0.25..4].
    power: f32,
    // Maximum screen-space sampling radius in full-resolution pixels.
    max_screen_radius_px: f32,
    // Azimuthal horizon slices per pixel [1..6].
    slices: u32,
    // Horizon march steps per side of each slice [2..8].
    steps_per_side: u32,
    // Bit 0: is_perspective; bit 1: is_ortho_3d.
    flags: u32,
    _pad: u32,
};

@vertex
fn vs_gtao(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(pos[vid], 0.0, 1.0);
}

// Positive distance from camera plane (-z_view) for Reversed-Z depth z_rev in (0, 1].
fn gtao_linear_depth(u: GtaoUniform, z_rev: f32) -> f32 {
    if ((u.flags & 1u) != 0u) {
        return u.depth_params.x / max(z_rev, 1e-6);
    }
    return u.depth_params.x - z_rev * u.depth_params.y;
}

// Reconstructs right-handed camera-space position (camera at origin looking down -Z) from a
// full-resolution pixel position `px_center` and positive linear depth `d`.
fn gtao_view_pos_linear(u: GtaoUniform, px_center: vec2<f32>, d: f32) -> vec3<f32> {
    let aw = max(f32(u.active_px.x), 1.0);
    let ah = max(f32(u.active_px.y), 1.0);
    let uv = px_center / vec2<f32>(aw, ah);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    if ((u.flags & 1u) != 0u) {
        return vec3<f32>(ndc.x * u.proj_xy.x * d, ndc.y * u.proj_xy.y * d, -d);
    }
    return vec3<f32>(ndc.x * u.proj_xy.x, ndc.y * u.proj_xy.y, -d);
}

fn gtao_view_dir(u: GtaoUniform, pos_v: vec3<f32>) -> vec3<f32> {
    if ((u.flags & 1u) != 0u) {
        let len = length(pos_v);
        return select(vec3<f32>(0.0, 0.0, 1.0), -pos_v / len, len > 1e-6);
    }
    return vec3<f32>(0.0, 0.0, 1.0);
}
