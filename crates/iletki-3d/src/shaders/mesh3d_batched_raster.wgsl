@vertex
fn vs_mesh_batched_raster(v: VertIn, @location(3) slot: u32) -> RasterClippedColorOut {
    return raster_clipped_color(mesh_vertex_with_object(v, page_objects[slot]));
}

@vertex
fn vs_edge_batched_raster(v: VertIn, @location(3) slot: u32) -> RasterClippedColorOut {
    return raster_clipped_color(edge_vertex_with_object(v, page_objects[slot]));
}
