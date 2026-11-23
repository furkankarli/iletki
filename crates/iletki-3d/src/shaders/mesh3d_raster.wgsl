@vertex
fn vs_mesh_raster(v: VertIn) -> RasterClippedColorOut {
    return raster_clipped_color(mesh_vertex(v));
}

@vertex
fn vs_edge_raster(v: VertIn) -> RasterClippedColorOut {
    return raster_clipped_color(edge_vertex(v));
}

@fragment
fn fs_lit_raster(in: ColorOut, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    return shade_mesh(in.pos_rel, in.normal, in.color, is_front);
}

@fragment
fn fs_mask_unclipped() -> @location(0) f32 {
    return 1.0;
}
