// Hardware raster-clipped storage-buffer MRT selection + MaterialGpu SSBO vertex shader.
struct RasterClippedStorageMatSelectOut {
    @builtin(position) clip_pos: vec4<f32>,
    @builtin(clip_distances) plane_distances: array<f32, 6>,
    @location(0) pos_rel: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) @interpolate(flat) slot_id: u32,
    @location(4) @interpolate(flat) mat_id: u32,
};

@vertex
fn vs_mesh_storage_mat_select_raster(
    v: VertIn,
    @location(3) slot: u32,
) -> RasterClippedStorageMatSelectOut {
    let st = load_storage_transform(slot);
    let base = raster_clipped_color(mesh_vertex_with_object(v, st.placement));
    var out = RasterClippedStorageMatSelectOut(
        base.clip_pos,
        base.plane_distances,
        base.pos_rel,
        base.normal,
        base.color,
        select(0u, slot + 1u, st.pickable),
        load_material_id(slot),
    );
    if (!st.visible) {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    }
    return out;
}
