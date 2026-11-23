@vertex
fn vs_mesh_batched_sampled(v: VertIn, @location(3) slot: u32) -> SampledColorOut {
    return sampled_color(mesh_vertex_with_object(v, page_objects[slot]));
}

@vertex
fn vs_edge_batched_sampled(v: VertIn, @location(3) slot: u32) -> SampledColorOut {
    return sampled_color(edge_vertex_with_object(v, page_objects[slot]));
}
