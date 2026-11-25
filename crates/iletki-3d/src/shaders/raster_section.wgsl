struct RasterClippedColorOut {
    @builtin(position) clip_pos: vec4<f32>,
    @builtin(clip_distances) plane_distances: array<f32, 6>,
    @location(0) pos_rel: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

fn raster_clipped_color(out: ColorOut) -> RasterClippedColorOut {
    var distances = array<f32, 6>(1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
    let count = min(iletki_camera.clip_plane_count, 6u);
    if (count >= 1u) { distances[0] = raster_plane_distance(iletki_camera.clip_planes[0], out.pos_rel); }
    if (count >= 2u) { distances[1] = raster_plane_distance(iletki_camera.clip_planes[1], out.pos_rel); }
    if (count >= 3u) { distances[2] = raster_plane_distance(iletki_camera.clip_planes[2], out.pos_rel); }
    if (count >= 4u) { distances[3] = raster_plane_distance(iletki_camera.clip_planes[3], out.pos_rel); }
    if (count >= 5u) { distances[4] = raster_plane_distance(iletki_camera.clip_planes[4], out.pos_rel); }
    if (count >= 6u) { distances[5] = raster_plane_distance(iletki_camera.clip_planes[5], out.pos_rel); }
    return RasterClippedColorOut(out.clip_pos, distances, out.pos_rel, out.normal, out.color);
}

fn raster_plane_distance(plane: vec4<f32>, pos_rel: vec3<f32>) -> f32 {
    return dot(plane.xyz, pos_rel) + plane.w + ILETKI_CLIP_TOLERANCE;
}
