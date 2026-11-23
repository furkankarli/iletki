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

@vertex
fn vs_pick_batched(
    @location(0) position: vec3<f32>,
    @location(3) slot: u32,
    @location(4) id: vec2<u32>,
) -> PickOut {
    return pick_vertex(position, page_objects[slot], id);
}

@vertex
fn vs_mask_batched(v: VertIn, @location(3) slot: u32) -> MaskOut {
    return mask_vertex(v, page_objects[slot]);
}
