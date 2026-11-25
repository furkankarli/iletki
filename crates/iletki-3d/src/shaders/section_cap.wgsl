// Plane-aligned section cap quad shader (prepended with `iletki_wgpu::CAMERA_WGSL`).
// Renders solid, hatched or cross-hatched CAD section fills on pixels marked by the two-sided
// stencil pass, clipped against all other active section planes and writing exact Reversed-Z depth.

struct CapQuadUniform {
    // xyz = camera-relative quad center lying on the clip plane; w = quad half-extent.
    origin_extent: vec4<f32>,
    // xyz = rotated unit tangent U on the plane; w = 1.0 / hatch_spacing.
    axis_u_inv_spacing: vec4<f32>,
    // xyz = rotated unit bitangent V on the plane; w = hatch line width in physical pixels.
    axis_v_width_px: vec4<f32>,
    // sRGB base cap fill RGBA.
    color: vec4<f32>,
    // sRGB hatch line RGBA.
    hatch_color: vec4<f32>,
    // xy = exact f64-reduced [0, 1) world-locked hatch phase at quad origin; zw = unused.
    hatch_phase: vec4<f32>,
    // x = clip plane index (0..6); y = pattern (0=Solid, 1=Hatch, 2=CrossHatch); zw = unused.
    meta: vec4<u32>,
};

@group(1) @binding(0)
var<uniform> cap_quad: CapQuadUniform;

struct CapQuadOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) pos_rel: vec3<f32>,
    @location(1) hatch_cycles: vec2<f32>,
};

@vertex
fn vs_cap_quad(@builtin(vertex_index) vid: u32) -> CapQuadOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
    );
    let corner = corners[vid];
    let offset_plane = corner * cap_quad.origin_extent.w;
    let pos_rel = cap_quad.origin_extent.xyz
        + cap_quad.axis_u_inv_spacing.xyz * offset_plane.x
        + cap_quad.axis_v_width_px.xyz * offset_plane.y;
    var out: CapQuadOut;
    out.clip_pos = iletki_project_3d(pos_rel);
    out.pos_rel = pos_rel;
    out.hatch_cycles = cap_quad.hatch_phase.xy + offset_plane * cap_quad.axis_u_inv_spacing.w;
    return out;
}

@fragment
fn fs_cap_quad(in: CapQuadOut) -> @location(0) vec4<f32> {
    // Evaluate screen-space derivatives in uniform control flow prior to any clip discard.
    let fw = max(fwidth(in.hatch_cycles), vec2<f32>(1e-5));

    // Clip against every active clip plane other than this cap's own plane so multi-plane cuts
    // (corner cutaways, section boxes, opposing parallel slabs) meet cleanly without overhang.
    let self_plane = cap_quad.meta.x;
    let count = min(iletki_camera.clip_plane_count, 6u);
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        if (i == self_plane) {
            continue;
        }
        let plane = iletki_camera.clip_planes[i];
        if (dot(plane.xyz, in.pos_rel) + plane.w < -ILETKI_CLIP_TOLERANCE) {
            discard;
        }
    }

    let base_rgb = iletki_color_to_target(cap_quad.color.rgb);
    let base_a = cap_quad.color.a;
    let pattern = cap_quad.meta.y;
    if (pattern == 0u) {
        return vec4<f32>(base_rgb, base_a);
    }

    let half_w = max(cap_quad.axis_v_width_px.w, 0.5) * 0.5;
    let dist_cycles = abs(fract(in.hatch_cycles + vec2<f32>(0.5)) - vec2<f32>(0.5));
    let dist_px = dist_cycles / fw;
    // Fade hatch smoothly before Nyquist aliasing when zoomed far out.
    let fade = vec2<f32>(1.0) - smoothstep(vec2<f32>(0.25), vec2<f32>(0.5), fw);
    let line_u = clamp(half_w + 0.5 - dist_px.x, 0.0, 1.0) * fade.x;
    var coverage = line_u;
    if (pattern == 2u) {
        let line_v = clamp(half_w + 0.5 - dist_px.y, 0.0, 1.0) * fade.y;
        coverage = max(line_u, line_v);
    }
    let line_mix = coverage * cap_quad.hatch_color.a;
    let hatch_rgb = iletki_color_to_target(cap_quad.hatch_color.rgb);
    let rgb = mix(base_rgb, hatch_rgb, line_mix);
    let alpha = max(base_a, line_mix);
    return vec4<f32>(rgb, alpha);
}
