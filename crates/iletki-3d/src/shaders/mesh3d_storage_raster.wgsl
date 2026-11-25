@vertex
fn vs_mesh_storage_raster(v: VertIn, @location(3) slot: u32) -> RasterClippedColorOut {
    let st = load_storage_transform(slot);
    var out = raster_clipped_color(mesh_vertex_with_object(v, st.placement));
    if (!st.visible) {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    }
    return out;
}

@vertex
fn vs_edge_storage_raster(v: VertIn, @location(3) slot: u32) -> RasterClippedColorOut {
    let st = load_storage_transform(slot);
    var out = raster_clipped_color(edge_vertex_with_object(v, st.placement));
    if (!st.visible) {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    }
    return out;
}

struct RasterClippedMaskOut {
    @builtin(position) clip_pos: vec4<f32>,
    @builtin(clip_distances) plane_distances: array<f32, 6>,
};

fn raster_clipped_mask(out: MaskOut) -> RasterClippedMaskOut {
    var distances = array<f32, 6>(1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
    let count = min(iletki_camera.clip_plane_count, 6u);
    if (count >= 1u) { distances[0] = raster_plane_distance(iletki_camera.clip_planes[0], out.pos_rel); }
    if (count >= 2u) { distances[1] = raster_plane_distance(iletki_camera.clip_planes[1], out.pos_rel); }
    if (count >= 3u) { distances[2] = raster_plane_distance(iletki_camera.clip_planes[2], out.pos_rel); }
    if (count >= 4u) { distances[3] = raster_plane_distance(iletki_camera.clip_planes[3], out.pos_rel); }
    if (count >= 5u) { distances[4] = raster_plane_distance(iletki_camera.clip_planes[4], out.pos_rel); }
    if (count >= 6u) { distances[5] = raster_plane_distance(iletki_camera.clip_planes[5], out.pos_rel); }
    return RasterClippedMaskOut(out.clip_pos, distances);
}

@vertex
fn vs_mask_storage_raster(@location(0) position: vec3<f32>, @location(3) slot: u32) -> RasterClippedMaskOut {
    let st = load_storage_transform(slot);
    var out = raster_clipped_mask(mask_vertex_pos(position, st.placement));
    if (!st.visible) {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    }
    return out;
}
