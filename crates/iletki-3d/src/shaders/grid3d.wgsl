// Infinite analytic anti-aliased 3D ground grid (prepended with `iletki_wgpu::CAMERA_WGSL`).
// Writes exact Reversed-Z fragment depth so 3D geometry occludes the grid cleanly.

struct GridUniforms {
    // Camera position in world f64 reduced modulo major_spacing on CPU for jitter-free lines.
    cam_pos_rel_plane: vec3<f32>,
    minor_spacing: f32,
    major_spacing: f32,
    fade_distance: f32,
    // 0 = XY plane (Z-up CAD/BIM), 1 = XZ plane (Y-up game engines).
    plane_mode: u32,
    _pad: u32,
    minor_color: vec4<f32>,
    major_color: vec4<f32>,
    axis_u_color: vec4<f32>,
    axis_v_color: vec4<f32>,
};

@group(1) @binding(0)
var<uniform> grid: GridUniforms;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

struct FsOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

struct FsSelectOut {
    @location(0) color: vec4<f32>,
    @location(1) slot_id: vec2<u32>,
    @builtin(frag_depth) depth: f32,
};

@vertex
fn vs_grid(@builtin(vertex_index) idx: u32) -> VsOut {
    let x = f32((idx & 1u) * 4u) - 1.0;
    let y = f32(((idx >> 1u) & 1u) * 4u) - 1.0;
    var out: VsOut;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.ndc = vec2<f32>(x, y);
    return out;
}

fn unproject_rel(ndc_x: f32, ndc_y: f32, z: f32) -> vec3<f32> {
    let p = iletki_camera.inv_view_proj_rel * vec4<f32>(ndc_x, ndc_y, z, 1.0);
    return p.xyz / p.w;
}

fn grid_alpha(coord: vec2<f32>, spacing: f32) -> f32 {
    let scaled = coord / spacing;
    let deriv = fwidth(scaled);
    let g = abs(fract(scaled - 0.5) - 0.5) / max(deriv, vec2<f32>(1e-5, 1e-5));
    let line = min(g.x, g.y);
    return 1.0 - min(line, 1.0);
}

fn eval_grid(in: VsOut) -> FsOut {
    // Reversed-Z: z = 1.0 is near, z = 0.5 is mid-depth.
    let near_p = unproject_rel(in.ndc.x, in.ndc.y, 1.0);
    let mid_p = unproject_rel(in.ndc.x, in.ndc.y, 0.5);
    let ray_dir = mid_p - near_p;

    var denom: f32;
    var plane_offset: f32;
    if (grid.plane_mode == 0u) {
        // XY plane at world z = 0 => rel_z = -cam_z
        denom = ray_dir.z;
        plane_offset = -grid.cam_pos_rel_plane.z - near_p.z;
    } else {
        // XZ plane at world y = 0 => rel_y = -cam_y
        denom = ray_dir.y;
        plane_offset = -grid.cam_pos_rel_plane.y - near_p.y;
    }

    if (abs(denom) < 1e-7) {
        discard;
    }
    let t = plane_offset / denom;
    if (t <= 0.0) {
        discard;
    }

    let hit_rel = near_p + ray_dir * t;
    let clip = iletki_project_3d(hit_rel);
    let depth = clip.z / clip.w;
    if (depth <= 0.0 || depth > 1.0) {
        discard;
    }

    let dist = length(hit_rel);
    let fade = clamp(1.0 - dist / max(grid.fade_distance, 1.0), 0.0, 1.0);
    if (fade <= 0.001) {
        discard;
    }

    var uv: vec2<f32>;
    if (grid.plane_mode == 0u) {
        uv = hit_rel.xy + grid.cam_pos_rel_plane.xy;
    } else {
        uv = hit_rel.xz + grid.cam_pos_rel_plane.xz;
    }

    let minor = grid_alpha(uv, max(grid.minor_spacing, 1e-4));
    let major = grid_alpha(uv, max(grid.major_spacing, 1e-4));

    var col = grid.minor_color;
    var a = minor * grid.minor_color.a;
    if (major > 0.01) {
        let ma = major * grid.major_color.a;
        if (ma >= a) {
            col = grid.major_color;
            a = ma;
        }
    }

    // Highlight world origin axes (u = 0 and v = 0).
    let deriv_uv = max(fwidth(uv), vec2<f32>(1e-5, 1e-5));
    let on_u_axis = 1.0 - min(abs(uv.y) / (deriv_uv.y * 1.5), 1.0);
    let on_v_axis = 1.0 - min(abs(uv.x) / (deriv_uv.x * 1.5), 1.0);
    if (on_u_axis > 0.05) {
        col = grid.axis_u_color;
        a = max(a, on_u_axis * grid.axis_u_color.a);
    }
    if (on_v_axis > 0.05) {
        col = grid.axis_v_color;
        a = max(a, on_v_axis * grid.axis_v_color.a);
    }

    let final_a = a * fade * fade;
    if (final_a <= 0.002) {
        discard;
    }

    var out: FsOut;
    out.color = vec4<f32>(iletki_color_to_target(col.rgb), final_a);
    out.depth = depth;
    return out;
}

@fragment
fn fs_grid(in: VsOut) -> FsOut {
    return eval_grid(in);
}

@fragment
fn fs_grid_select(in: VsOut) -> FsSelectOut {
    let base = eval_grid(in);
    return FsSelectOut(base.color, vec2<u32>(0u, 0u), base.depth);
}
