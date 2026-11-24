// GPU-driven frustum, section-plane and conservative two-phase Reversed-Z Hi-Z culling.

const FLAG_TRUSTED: u32 = 1u;
const FLAG_TRANSLUCENT: u32 = 2u;
const FLAG_FACE_INDEXED: u32 = 4u;
const FLAG_HAS_FACES: u32 = 8u;
const FLAG_HAS_EDGES: u32 = 16u;
const FLAG_EDGE_INDEXED: u32 = 32u;

const VIS_PREV: u32 = 1u;
const VIS_DRAWN_P1: u32 = 2u;
const VIS_PASSED_FS: u32 = 4u;

// 16 * f32::EPSILON matches the conservative CPU rounding margin in `objects.rs`.
const F32_ERR_SCALE: f32 = 1.9073486e-6;
const SECTION_TOLERANCE: f32 = 1e-4;

struct CullViewUniforms {
    view_proj: mat4x4<f32>,
    frustum: array<vec4<f32>, 6>,
    clip_planes: array<vec4<f32>, 6>,
    cam_whole: vec3<i32>,
    frustum_count: u32,
    cam_frac: vec3<u32>,
    clip_plane_count: u32,
    item_count: u32,
    active_w: u32,
    active_h: u32,
    _pad0: u32,
};

struct ChunkTransformGpu {
    whole: vec3<i32>,
    scale: f32,
    frac: vec3<u32>,
    flags: u32,
};

struct ChunkCullItem {
    local_min: vec3<f32>,
    chunk_index: u32,
    local_max: vec3<f32>,
    flags: u32,
    face_bucket: u32,
    face_first: u32,
    face_count: u32,
    face_indirect_base: u32,
    edge_bucket: u32,
    edge_first: u32,
    edge_count: u32,
    edge_indirect_base: u32,
};

struct DrawIndexedArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
};

struct DrawArgs {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};

@group(0) @binding(0)
var<uniform> u: CullViewUniforms;

@group(0) @binding(1)
var<storage, read> chunk_transforms: array<ChunkTransformGpu>;

@group(0) @binding(2)
var<storage, read> items: array<ChunkCullItem>;

@group(0) @binding(3)
var<storage, read_write> vis_state: array<u32>;

@group(0) @binding(4)
var<storage, read_write> indexed_args: array<DrawIndexedArgs>;

@group(0) @binding(5)
var<storage, read_write> non_indexed_args: array<DrawArgs>;

@group(0) @binding(6)
var<storage, read_write> bucket_counts: array<atomic<u32>>;

// Exact 64-bit Q32.32 fixed-point subtraction matching `mesh3d_storage.wgsl`.
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

fn emit_draw(bucket: u32, indirect_base: u32, first: u32, count: u32, indexed: bool) {
    let slot = atomicAdd(&bucket_counts[bucket], 1u);
    let dst = indirect_base + slot;
    if (indexed) {
        indexed_args[dst] = DrawIndexedArgs(count, 1u, first, 0, 0u);
    } else {
        non_indexed_args[dst] = DrawArgs(count, 1u, first, 0u);
    }
}
