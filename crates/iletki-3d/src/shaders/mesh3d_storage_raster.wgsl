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
