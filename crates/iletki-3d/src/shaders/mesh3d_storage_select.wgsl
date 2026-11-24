// Storage-buffer color-frame selection vertex entry points.
@vertex
fn vs_mesh_storage_select(v: VertIn, @location(3) slot: u32) -> ColorSelectOut {
    let st = load_storage_transform(slot);
    var out = color_with_slot(
        mesh_vertex_with_object(v, st.placement),
        select(0u, slot + 1u, st.pickable),
    );
    if (!st.visible) {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    }
    return out;
}

@vertex
fn vs_edge_storage_select(v: VertIn, @location(3) slot: u32) -> ColorSelectOut {
    let st = load_storage_transform(slot);
    var out = color_with_slot(edge_vertex_with_object(v, st.placement), 0u);
    if (!st.visible) {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    }
    return out;
}
