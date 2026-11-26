struct RasterClippedColorOut {
    @builtin(position) clip_pos: vec4<f32>,
    @builtin(clip_distances) plane_distances: array<f32, 6>,
    @location(0) pos_rel: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

fn raster_clipped_color(out: ColorOut) -> RasterClippedColorOut {
    return RasterClippedColorOut(out.clip_pos, raster_clip_distances(out.pos_rel), out.pos_rel, out.normal, out.color);
}

fn raster_clip_distances(pos_rel: vec3<f32>) -> array<f32, 6> {
    var distances = array<f32, 6>(1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
    let count = min(iletki_camera.clip_plane_count, 6u);
    if (count >= 1u) { distances[0] = raster_plane_distance(iletki_camera.clip_planes[0], pos_rel); }
    if (count >= 2u) { distances[1] = raster_plane_distance(iletki_camera.clip_planes[1], pos_rel); }
    if (count >= 3u) { distances[2] = raster_plane_distance(iletki_camera.clip_planes[2], pos_rel); }
    if (count >= 4u) { distances[3] = raster_plane_distance(iletki_camera.clip_planes[3], pos_rel); }
    if (count >= 5u) { distances[4] = raster_plane_distance(iletki_camera.clip_planes[4], pos_rel); }
    if (count >= 6u) { distances[5] = raster_plane_distance(iletki_camera.clip_planes[5], pos_rel); }
    return distances;
}

fn raster_plane_distance(plane: vec4<f32>, pos_rel: vec3<f32>) -> f32 {
    return dot(plane.xyz, pos_rel) + plane.w + ILETKI_CLIP_TOLERANCE;
}
