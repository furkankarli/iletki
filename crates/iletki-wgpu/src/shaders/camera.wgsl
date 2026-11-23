// Standard camera and section-clip uniform block bound at @group(0) @binding(0).
// Layers can prepend `iletki_wgpu::CAMERA_WGSL` directly to their WGSL shaders.

struct CameraUniforms {
    view_proj_rel: mat4x4<f32>,
    inv_view_proj_rel: mat4x4<f32>,
    camera_hi: vec2<f32>,
    camera_lo: vec2<f32>,
    scale_2d: vec2<f32>,
    point_size: vec2<f32>,
    viewport_px: vec2<f32>,
    dpi_scale: f32,
    flags: u32,
    clip_planes: array<vec4<f32>, 6>,
    clip_plane_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0)
var<uniform> iletki_camera: CameraUniforms;

// Converts a split hi/lo 2D world coordinate into NDC clip space.
fn iletki_to_clip_2d(hi: vec2<f32>, lo: vec2<f32>) -> vec2<f32> {
    return ((hi - iletki_camera.camera_hi) + (lo - iletki_camera.camera_lo)) * iletki_camera.scale_2d;
}

// Projects a camera-relative 3D point into Reversed-Z clip space.
fn iletki_project_3d(pos_rel: vec3<f32>) -> vec4<f32> {
    return iletki_camera.view_proj_rel * vec4<f32>(pos_rel, 1.0);
}

// Returns true if a camera-relative 3D point is clipped away by any active section clip plane.
// Uses a small negative tolerance (-1e-4) so faces lying exactly coplanar with a clip plane
// (common at CAD grid axes like x=0, y=0) never suffer floating-point stipple/striping artifacts.
const ILETKI_CLIP_TOLERANCE: f32 = 1e-4;

fn iletki_is_clipped(pos_rel: vec3<f32>) -> bool {
    let count = min(iletki_camera.clip_plane_count, 6u);
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let plane = iletki_camera.clip_planes[i];
        if (dot(plane.xyz, pos_rel) + plane.w < -ILETKI_CLIP_TOLERANCE) {
            return true;
        }
    }
    return false;
}

fn iletki_srgb_to_linear_channel(c: f32) -> f32 {
    if (c <= 0.04045) {
        return c / 12.92;
    }
    return pow((c + 0.055) / 1.055, 2.4);
}

// Converts an sRGB [0..1] RGB color to linear space when rendering into an sRGB texture target
// (flag bit 3 = 8u), avoiding double-gamma washed-out colors on sRGB swapchains.
fn iletki_color_to_target(rgb: vec3<f32>) -> vec3<f32> {
    if ((iletki_camera.flags & 8u) != 0u) {
        return vec3<f32>(
            iletki_srgb_to_linear_channel(rgb.r),
            iletki_srgb_to_linear_channel(rgb.g),
            iletki_srgb_to_linear_channel(rgb.b),
        );
    }
    return rgb;
}
