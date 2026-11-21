// High-precision 2D CAD/vector shaders (prepended with `iletki_wgpu::CAMERA_WGSL`).
// Positions are split into high and low f32 components so subtraction relative to the camera
// preserves ~48 bits of precision even at national-grid coordinates (10^6..10^9).
//
// Color passes bind only slot 0 (compact 24-byte PlainIn / 52-byte WideIn) for maximum bandwidth;
// picking passes additionally bind slot 1 (8-byte PickId) only when GPU picking is invoked.

struct PlainIn {
    @location(0) hi: vec2<f32>,
    @location(1) lo: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) depth: f32,
};

struct WideIn {
    @location(0) a_hi: vec2<f32>,
    @location(1) a_lo: vec2<f32>,
    @location(2) b_hi: vec2<f32>,
    @location(3) b_lo: vec2<f32>,
    @location(4) corner: vec2<f32>,
    @location(5) width: f32,
    @location(6) color: vec4<f32>,
    @location(7) depth: f32,
};

struct PickIn {
    @location(8) pick_id: vec2<u32>,
};

struct ColorOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

struct PickOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) pick_id: vec2<u32>,
};

fn wide_clip(v: WideIn) -> vec4<f32> {
    let a = iletki_to_clip_2d(v.a_hi, v.a_lo);
    let b = iletki_to_clip_2d(v.b_hi, v.b_lo);
    let pt = max(iletki_camera.point_size, vec2<f32>(1e-7, 1e-7));
    let along_pt = (b - a) / pt;
    let len = length(along_pt);
    var dir = vec2<f32>(1.0, 0.0);
    if (len > 0.0) {
        dir = along_pt / len;
    }
    let normal = vec2<f32>(-dir.y, dir.x);
    let half_w = 0.5 * v.width;
    let end = mix(a, b, v.corner.x);
    let offset = (normal * v.corner.y + dir * (2.0 * v.corner.x - 1.0)) * half_w * pt;
    return vec4<f32>(end + offset, v.depth, 1.0);
}

@vertex
fn vs_plain(v: PlainIn) -> ColorOut {
    var out: ColorOut;
    out.pos = vec4<f32>(iletki_to_clip_2d(v.hi, v.lo), v.depth, 1.0);
    out.color = v.color;
    return out;
}

@vertex
fn vs_wide(v: WideIn) -> ColorOut {
    var out: ColorOut;
    out.pos = wide_clip(v);
    out.color = v.color;
    return out;
}

@vertex
fn vs_plain_pick(v: PlainIn, p: PickIn) -> PickOut {
    var out: PickOut;
    out.pos = vec4<f32>(iletki_to_clip_2d(v.hi, v.lo), v.depth, 1.0);
    out.pick_id = p.pick_id;
    return out;
}

@vertex
fn vs_wide_pick(v: WideIn, p: PickIn) -> PickOut {
    var out: PickOut;
    out.pos = wide_clip(v);
    out.pick_id = p.pick_id;
    return out;
}

@fragment
fn fs_color(in: ColorOut) -> @location(0) vec4<f32> {
    return in.color;
}

@fragment
fn fs_pick(in: PickOut) -> @location(0) vec2<u32> {
    return in.pick_id;
}
