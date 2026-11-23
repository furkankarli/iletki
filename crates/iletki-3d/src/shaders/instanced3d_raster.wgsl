@vertex
fn vs_inst_raster(v: VertIn, inst: InstIn) -> RasterClippedColorOut {
    return raster_clipped_color(instanced_vertex(v, inst));
}

@fragment
fn fs_inst_raster(in: ColorOut, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    return shade_instanced(in.pos_rel, in.normal, in.color, is_front);
}
