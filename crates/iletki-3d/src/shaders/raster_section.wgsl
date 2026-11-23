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
    for (var i: u32 = 0u; i < count; i += 1u) {
        let plane = iletki_camera.clip_planes[i];
        distances[i] = dot(plane.xyz, out.pos_rel) + plane.w + ILETKI_CLIP_TOLERANCE;
    }
    return RasterClippedColorOut(out.clip_pos, distances, out.pos_rel, out.normal, out.color);
}
