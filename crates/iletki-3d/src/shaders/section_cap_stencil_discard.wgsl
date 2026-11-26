// Portable discard-based stencil marking helpers for devices without `CLIP_DISTANCES`.
struct CapPlaneUniform {
    plane: vec4<f32>,
};

@group(2) @binding(0)
var<uniform> cap_plane: CapPlaneUniform;

struct CapStencilOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) plane_dist: f32,
};

fn cap_stencil_vertex(position: vec3<f32>, placement: ObjectUniforms) -> CapStencilOut {
    let pos_rel = position * placement.scale + placement.offset_rel;
    let dist = dot(cap_plane.plane.xyz, pos_rel) + cap_plane.plane.w + ILETKI_CLIP_TOLERANCE;
    return CapStencilOut(iletki_project_3d(pos_rel), dist);
}

fn cap_stencil_hidden() -> CapStencilOut {
    return CapStencilOut(vec4<f32>(2.0, 2.0, 2.0, 1.0), -1.0);
}

@fragment
fn fs_cap_stencil(in: CapStencilOut) {
    if (in.plane_dist < 0.0) {
        discard;
    }
}
