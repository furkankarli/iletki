@vertex
fn vs_mesh_storage_select_sampled(v: VertIn, @location(3) slot: u32) -> SampledColorSelectOut {
    let st = load_storage_transform(slot);
    var out = sampled_color_select(
        mesh_vertex_with_object(v, st.placement),
        select(0u, slot + 1u, st.pickable),
    );
    if (!st.visible) {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    }
    return out;
}

@vertex
fn vs_edge_storage_select_sampled(v: VertIn, @location(3) slot: u32) -> SampledColorSelectOut {
    let st = load_storage_transform(slot);
    var out = sampled_color_select(edge_vertex_with_object(v, st.placement), 0u);
    if (!st.visible) {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    }
    return out;
}
