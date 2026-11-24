// Persistent scene-wide storage-buffer transforms using exact Q32.32 fixed-point subtraction.
struct ViewCameraOrigin {
    cam_whole: vec3<i32>,
    _pad0: u32,
    cam_frac: vec3<u32>,
    _pad1: u32,
};

struct ChunkTransformGpu {
    whole: vec3<i32>,
    scale: f32,
    frac: vec3<u32>,
    flags: u32,
};

struct StorageTransform {
    placement: ObjectUniforms,
    visible: bool,
    pickable: bool,
};

@group(1) @binding(1)
var<uniform> view_origin: ViewCameraOrigin;

@group(1) @binding(2)
var<storage, read> chunk_transforms: array<ChunkTransformGpu>;

// Exact 64-bit Q32.32 fixed-point subtraction with single IEEE-754 rounding to f32.
// Immune to GPU floating-point fast-math reassociation across Vulkan/DX12/Metal/WebGPU.
fn sub_q32_32(
    a_hi: vec3<i32>,
    a_lo: vec3<u32>,
    b_hi: vec3<i32>,
    b_lo: vec3<u32>,
) -> vec3<f32> {
    let borrow = select(vec3<i32>(0), vec3<i32>(1), a_lo < b_lo);
    let d_lo = a_lo - b_lo;
    let d_hi = a_hi - b_hi - borrow;
    let neg = d_hi < vec3<i32>(0);
    let neg_lo = vec3<u32>(0u) - d_lo;
    let neg_carry = select(vec3<u32>(0u), vec3<u32>(1u), d_lo != vec3<u32>(0u));
    let neg_hi = (vec3<u32>(0u) - bitcast<vec3<u32>>(d_hi)) - neg_carry;
    let mag_hi = select(bitcast<vec3<u32>>(d_hi), neg_hi, neg);
    let mag_lo = select(d_lo, neg_lo, neg);

    let lz = countLeadingZeros(mag_hi);
    let lshift = lz & vec3<u32>(31u);
    let rshift = (vec3<u32>(32u) - lz) & vec3<u32>(31u);
    let top_raw = (mag_hi << lshift) | select(vec3<u32>(0u), mag_lo >> rshift, lz != vec3<u32>(0u));
    let rem = mag_lo << lshift;
    let top = top_raw | select(vec3<u32>(0u), vec3<u32>(1u), rem != vec3<u32>(0u));
    let big_val = ldexp(vec3<f32>(top), -vec3<i32>(lz));
    let small_val = ldexp(vec3<f32>(mag_lo), vec3<i32>(-32));
    let mag = select(big_val, small_val, mag_hi == vec3<u32>(0u));
    return select(mag, -mag, neg);
}

fn load_storage_transform(slot: u32) -> StorageTransform {
    let t = chunk_transforms[slot];
    let offset_rel = sub_q32_32(t.whole, t.frac, view_origin.cam_whole, view_origin.cam_frac);
    return StorageTransform(
        ObjectUniforms(offset_rel, t.scale),
        (t.flags & 1u) != 0u,
        (t.flags & 2u) != 0u,
    );
}

@vertex
fn vs_mesh_storage(v: VertIn, @location(3) slot: u32) -> ColorOut {
    let st = load_storage_transform(slot);
    var out = mesh_vertex_with_object(v, st.placement);
    if (!st.visible) {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    }
    return out;
}

@vertex
fn vs_edge_storage(v: VertIn, @location(3) slot: u32) -> ColorOut {
    let st = load_storage_transform(slot);
    var out = edge_vertex_with_object(v, st.placement);
    if (!st.visible) {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    }
    return out;
}

@vertex
fn vs_pick_storage(
    @location(0) position: vec3<f32>,
    @location(3) slot: u32,
    @location(4) id: vec2<u32>,
) -> PickOut {
    let st = load_storage_transform(slot);
    var out = pick_vertex(position, st.placement, id);
    if (!st.visible) {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    }
    return out;
}

@vertex
fn vs_mask_storage(v: VertIn, @location(3) slot: u32) -> MaskOut {
    let st = load_storage_transform(slot);
    var out = mask_vertex(v, st.placement);
    if (!st.visible) {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    }
    return out;
}
