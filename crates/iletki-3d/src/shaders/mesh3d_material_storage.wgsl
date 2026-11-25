// Persistent storage-buffer material lookup (@group(1) @binding(3)) indexed from ChunkTransformGpu.flags >> 8.
@group(1) @binding(3)
var<storage, read> scene_materials: array<MaterialGpu>;

struct StorageMatColorOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) pos_rel: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) @interpolate(flat) mat_id: u32,
};

fn load_material_id(slot: u32) -> u32 {
    return chunk_transforms[slot].flags >> 8u;
}

fn shade_storage_material(
    pos_rel: vec3<f32>,
    normal: vec3<f32>,
    color: vec4<f32>,
    is_front: bool,
    mat_id: u32,
) -> vec4<f32> {
    if (mat_id == 0u) {
        return shade_mesh(pos_rel, normal, color, is_front);
    }
    return shade_mesh_material(pos_rel, normal, color, is_front, scene_materials[mat_id]);
}

@vertex
fn vs_mesh_storage_mat(v: VertIn, @location(3) slot: u32) -> StorageMatColorOut {
    let st = load_storage_transform(slot);
    let base = mesh_vertex_with_object(v, st.placement);
    var out = StorageMatColorOut(
        base.clip_pos,
        base.pos_rel,
        base.normal,
        base.color,
        load_material_id(slot),
    );
    if (!st.visible) {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    }
    return out;
}

@fragment
fn fs_lit_storage_mat(
    in: StorageMatColorOut,
    @builtin(front_facing) is_front: bool,
) -> @location(0) vec4<f32> {
    if (iletki_is_clipped(in.pos_rel)) {
        discard;
    }
    return shade_storage_material(in.pos_rel, in.normal, in.color, is_front, in.mat_id);
}

@fragment
fn fs_lit_storage_mat_unclipped(
    in: StorageMatColorOut,
    @builtin(front_facing) is_front: bool,
) -> @location(0) vec4<f32> {
    return shade_storage_material(in.pos_rel, in.normal, in.color, is_front, in.mat_id);
}
