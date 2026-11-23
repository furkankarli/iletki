// Each vertex stores the slot of its chunk within one immutable geometry page.
@group(1) @binding(1)
var<uniform> page_objects: array<ObjectUniforms, 256>;

@vertex
fn vs_mesh_batched(v: VertIn, @location(3) slot: u32) -> ColorOut {
    return mesh_vertex_with_object(v, page_objects[slot]);
}

@vertex
fn vs_edge_batched(v: VertIn, @location(3) slot: u32) -> ColorOut {
    return edge_vertex_with_object(v, page_objects[slot]);
}
