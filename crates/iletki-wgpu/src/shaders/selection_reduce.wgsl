// Compute bitset reduction for color-frame selection preview (`Rg16Uint` slot IDs -> bitset).
struct ReduceParams {
    // [x0, y0, width, height] in physical pixels within the active target.
    rect: vec4<u32>,
    word_count: u32,
    sample_count: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0)
var<uniform> params: ReduceParams;

@group(0) @binding(2)
var<storage, read_write> bitset: array<atomic<u32>>;

fn mark_slot(id: u32) {
    if (id == 0u) {
        return;
    }
    let slot = id - 1u;
    let word = slot >> 5u;
    if (word >= params.word_count) {
        return;
    }
    let bit = 1u << (slot & 31u);
    if ((atomicLoad(&bitset[word]) & bit) == 0u) {
        atomicOr(&bitset[word], bit);
    }
}
