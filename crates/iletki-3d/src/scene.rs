//! Camera-relative Reversed-Z 3D scene layer with automatic 3D frustum culling, 6 section clip
//! planes, *Shaded with Edges* depth-offset wireframe overlays, Jump-Flood selection outline masks,
//! 64-bit sub-object GPU picking and simultaneous multi-viewport sharing.

use crate::face_batch::{
    self, GeometryPage, PageBuilder, PageKind, PaintPipelines, VertexBuffers, VertexRange,
};
use crate::gpu_cull::{
    self, CullChunkInput, CullViewUniforms, GpuCullPipelines, GpuCullScene, GpuCullView,
};
use crate::material::{Material3D, MaterialId, MaterialTable};
use crate::objects::{
    FaceRanges, PickVolume, PreparedObject, SectionGroup, SectionRelation, SectionVolume,
    ViewObjects, bounds_cover_vertices,
};
use crate::outline_bounds::OutlineUpdate;
use crate::provoking::{self, DedupScratch};
use crate::section::SectionAa;
use crate::shading::ScenePipelines;
use crate::storage_batch::{self, ChunkTransformTable, StorageDraws, StorageViewBinding};
use crate::vertex::Mesh3D;
use bytemuck::{Pod, Zeroable};
use iletki_core::{Bounds3, DrawPhase, DrawPhaseMask, Vec3, ViewportId};
use iletki_wgpu::wgpu;
use iletki_wgpu::wgpu::util::DeviceExt;
use iletki_wgpu::{GpuCullCtx, LayerRevision, PaintCtx, PrepareCtx, ViewportLayer};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

static NEXT_CHUNK_ID: AtomicU64 = AtomicU64::new(1);
const SECTION_GROUP_SIZE: usize = 32;
// Keep tiny packed objects together in groups that need raster clipping anyway.
const MAX_DEFERRED_SECTION_VERTICES: usize = 64;
// Above this chunk count, project one shared conservative world envelope instead of per-chunk AABBs.
const LARGE_SELECTION_ENVELOPE_THRESHOLD: usize = 128;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct ObjectUniforms {
    pub(crate) offset_rel: [f32; 3],
    pub(crate) scale: f32,
}

/// Identifier of a 3D mesh chunk stored inside [`Scene3DLayer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Chunk3DId {
    index: usize,
    generation: u64,
}

/// Compact L1-cache-resident bitset of selected chunk slot indices (O(1) toggle, sequential scan).
#[derive(Default)]
struct SelectionSet {
    words: Vec<u64>,
    len: usize,
}

impl SelectionSet {
    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn len(&self) -> usize {
        self.len
    }

    fn contains(&self, index: usize) -> bool {
        self.words
            .get(index >> 6)
            .is_some_and(|&word| (word & (1u64 << (index & 63))) != 0)
    }

    fn insert(&mut self, index: usize) -> bool {
        let word = index >> 6;
        let bit = 1u64 << (index & 63);
        if word >= self.words.len() {
            self.words.resize(word + 1, 0);
        }
        if let Some(slot) = self.words.get_mut(word)
            && (*slot & bit) == 0
        {
            *slot |= bit;
            self.len = self.len.saturating_add(1);
            return true;
        }
        false
    }

    fn remove(&mut self, index: usize) -> bool {
        let word = index >> 6;
        let bit = 1u64 << (index & 63);
        if let Some(slot) = self.words.get_mut(word)
            && (*slot & bit) != 0
        {
            *slot &= !bit;
            self.len = self.len.saturating_sub(1);
            return true;
        }
        false
    }

    fn clear(&mut self) {
        self.words.fill(0);
        self.len = 0;
    }

    fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.words.iter().enumerate().flat_map(|(w_idx, &word)| {
            let base = w_idx << 6;
            let mut rem = word;
            std::iter::from_fn(move || {
                if rem == 0 {
                    return None;
                }
                let tz = rem.trailing_zeros() as usize;
                rem &= rem - 1;
                Some(base + tz)
            })
        })
    }
}

const SELECT_META_VISIBLE: u32 = 1 << 0;
const SELECT_META_HAS_VERTS: u32 = 1 << 1;
const SELECT_META_BOUNDS_F32: u32 = 1 << 2;
const MAX_SELECT_BOUNDS_REL: f64 = 16_777_216.0;

/// Compact 32-byte L3-cache-resident per-chunk selection metadata (avoids striding `ChunkEntry`).
#[derive(Clone, Copy)]
struct ChunkSelectMeta {
    /// Dense index into `Scene3DLayer::repeated_face_pages`, or `u32::MAX` for non-repeated/direct.
    repeated_face_page: u32,
    flags: u32,
    /// Conservative anchor-relative world AABB `[min_x, min_y, min_z, max_x, max_y, max_z]`.
    bounds_rel: [f32; 6],
}

impl ChunkSelectMeta {
    const EMPTY: Self = Self {
        repeated_face_page: u32::MAX,
        flags: 0,
        bounds_rel: [0.0; 6],
    };
}

fn pack_select_bounds(origin: Vec3, local_bounds: Bounds3, anchor: Vec3) -> Option<[f32; 6]> {
    if !origin.is_finite() || local_bounds.is_empty() || !anchor.is_finite() {
        return None;
    }
    let rel = origin - anchor;
    let min = rel + local_bounds.min;
    let max = rel + local_bounds.max;
    let coords = [min.x, min.y, min.z, max.x, max.y, max.z];
    if !coords
        .iter()
        .all(|&v| v.is_finite() && v.abs() < MAX_SELECT_BOUNDS_REL)
    {
        return None;
    }
    let floor_f32 = |v: f64| -> f32 {
        let f = v as f32;
        if f64::from(f) > v {
            f - (f.abs() * 1e-6).max(1e-6)
        } else {
            f
        }
    };
    let ceil_f32 = |v: f64| -> f32 {
        let f = v as f32;
        if f64::from(f) < v {
            f + (f.abs() * 1e-6).max(1e-6)
        } else {
            f
        }
    };
    Some([
        floor_f32(min.x),
        floor_f32(min.y),
        floor_f32(min.z),
        ceil_f32(max.x),
        ceil_f32(max.y),
        ceil_f32(max.z),
    ])
}

struct ChunkEntry {
    generation: u64,
    origin: Vec3,
    local_bounds: Bounds3,
    visible: bool,
    pending_mesh: Option<Box<Mesh3D>>,
    vertex_buf: Option<VertexRange>,
    vertex_count: u32,
    face_ranges: Option<Box<FaceRanges>>,
    edge_buf: Option<VertexRange>,
    edge_count: u32,
    pick_buf: Option<Box<wgpu::Buffer>>,
    has_picks: bool,
    pick_bounds_trusted: bool,
    section_bounds_trusted: bool,
    small_packed: bool,
    has_translucent: bool,
    material_id: MaterialId,
}

struct SceneEnvelope {
    bounds: SectionGroup,
    all_trusted: bool,
    all_small_packed: bool,
    has_packed: bool,
}

/// Camera-relative Reversed-Z 3D scene layer implementing [`ViewportLayer`].
///
/// Each chunk keeps its own `f64` world origin. Preparation computes
/// `(chunk.origin - camera_pos).to_f32()` for visible chunks and uploads their uniforms in
/// per-[`iletki_core::ViewportId`] pages so 3D geometry
/// stays jitter-free even hundreds of kilometres from `(0, 0, 0)` and can be shared safely across
/// simultaneous multi-viewport quad views in a single `CommandEncoder`.
pub struct Scene3DLayer {
    layer_id: u16,
    revision: LayerRevision,
    geometry_revision: u64,
    pipelines: Option<ScenePipelines>,
    materials: MaterialTable,
    use_antialiased_section: bool,
    chunks: Vec<Option<ChunkEntry>>,
    /// Compact per-slot generation (`0` = vacant) for cache-hot `chunk_id_at_slot` / `set_selected`.
    chunk_generations: Vec<u64>,
    /// Compact 32-byte per-slot visibility, repeated-page index, and anchor-relative bounds.
    chunk_select_meta: Vec<ChunkSelectMeta>,
    repeated_face_pages: Vec<Arc<GeometryPage>>,
    repeated_face_page_map: HashMap<usize, u32>,
    repeated_mask_scratch: Vec<(u32, u32)>,
    free_chunks: BinaryHeap<Reverse<usize>>,
    selected_slots: SelectionSet,
    selected_revision: u64,
    selected_bounds_revision: u64,
    shared_mask_rev: u64,
    shared_mask_bounds_rev: u64,
    shared_selected_envelope: Option<(Bounds3, Vec3)>,
    shared_storage_mask: StorageDraws,
    pending_meshes: usize,
    hidden_chunks: usize,
    has_repeated_pages: bool,
    section_groups: Vec<Option<SectionGroup>>,
    section_relations: Vec<(SectionRelation, bool, bool)>,
    scene_envelope: Option<SceneEnvelope>,
    transform_table: ChunkTransformTable,
    gpu_cull_pipelines: Option<GpuCullPipelines>,
    gpu_cull_scene: Option<GpuCullScene>,
    gpu_cull_built_rev: u64,
    views: Vec<ViewObjects>,
    dedup_scratch: DedupScratch,
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    profile_batch_uniforms: bool,
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    profile_batch_geometry: bool,
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    profile_indexed_geometry: bool,
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    profile_storage_transforms: bool,
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    profile_gpu_culling: bool,
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    profile_batch_picking: bool,
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    profile_repeated_pages: bool,
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    profile_section_culling: bool,
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    profile_section_groups: bool,
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    profile_section_bridging: bool,
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    uploaded_chunks: usize,
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    uploaded_flat_chunks: usize,
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    uploaded_smooth_indexed_chunks: usize,
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    uploaded_input_vertices: usize,
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    uploaded_stored_vertices: usize,
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    uploaded_preprocess_us: usize,
}

impl std::fmt::Debug for Scene3DLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scene3DLayer")
            .field("layer_id", &self.layer_id)
            .field("chunks", &self.chunks.len())
            .finish()
    }
}

impl Scene3DLayer {
    /// Creates an empty 3D scene layer with identifier `layer_id`.
    #[must_use]
    pub fn new(layer_id: u16) -> Self {
        Self {
            layer_id,
            revision: LayerRevision::new(),
            geometry_revision: 1,
            pipelines: None,
            materials: MaterialTable::new(),
            use_antialiased_section: false,
            chunks: Vec::new(),
            chunk_generations: Vec::new(),
            chunk_select_meta: Vec::new(),
            repeated_face_pages: Vec::new(),
            repeated_face_page_map: HashMap::new(),
            repeated_mask_scratch: Vec::new(),
            free_chunks: BinaryHeap::new(),
            selected_slots: SelectionSet::default(),
            selected_revision: 1,
            selected_bounds_revision: 1,
            shared_mask_rev: 0,
            shared_mask_bounds_rev: 0,
            shared_selected_envelope: None,
            shared_storage_mask: StorageDraws::default(),
            pending_meshes: 0,
            hidden_chunks: 0,
            has_repeated_pages: false,
            section_groups: Vec::new(),
            section_relations: Vec::new(),
            scene_envelope: None,
            transform_table: ChunkTransformTable::default(),
            gpu_cull_pipelines: None,
            gpu_cull_scene: None,
            gpu_cull_built_rev: 0,
            views: Vec::new(),
            dedup_scratch: DedupScratch::default(),
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            profile_batch_uniforms: true,
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            profile_batch_geometry: true,
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            profile_indexed_geometry: true,
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            profile_storage_transforms: true,
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            profile_gpu_culling: true,
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            profile_batch_picking: true,
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            profile_repeated_pages: true,
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            profile_section_culling: true,
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            profile_section_groups: true,
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            profile_section_bridging: true,
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            uploaded_chunks: 0,
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            uploaded_flat_chunks: 0,
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            uploaded_smooth_indexed_chunks: 0,
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            uploaded_input_vertices: 0,
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            uploaded_stored_vertices: 0,
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            uploaded_preprocess_us: 0,
        }
    }

    /// Sets uniform upload batching for A/B profiling. Visibility filtering still applies.
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    pub fn set_profile_batch_uniforms(&mut self, enabled: bool) {
        if self.profile_batch_uniforms != enabled {
            self.profile_batch_uniforms = enabled;
            self.views.clear();
            self.revision.mark_static();
        }
    }

    /// Controls shared geometry pages for A/B profiling. Set before uploading the scene.
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    pub fn set_profile_batch_geometry(&mut self, enabled: bool) {
        if self.profile_batch_geometry != enabled {
            self.profile_batch_geometry = enabled;
            self.revision.mark_static();
        }
    }

    /// Controls provoking-vertex flat indexing and smooth exact-record indexing for A/B profiling.
    /// Set before uploading the scene.
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    pub fn set_profile_indexed_geometry(&mut self, enabled: bool) {
        if self.profile_indexed_geometry != enabled {
            self.profile_indexed_geometry = enabled;
            self.revision.mark_static();
        }
    }

    /// Controls persistent storage-buffer transforms and retained draw lists for A/B profiling.
    /// Set before uploading the scene.
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    pub fn set_profile_storage_transforms(&mut self, enabled: bool) {
        if self.profile_storage_transforms != enabled {
            self.profile_storage_transforms = enabled;
            self.views.clear();
            self.revision.mark_static();
        }
    }

    /// Controls GPU-driven frustum, section and two-phase Hi-Z occlusion culling for A/B profiling.
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    pub fn set_profile_gpu_culling(&mut self, enabled: bool) {
        if self.profile_gpu_culling != enabled {
            self.profile_gpu_culling = enabled;
            self.views.clear();
            self.revision.mark_static();
        }
    }

    /// Switches the picking pass between grouped and per-chunk draws for A/B profiling.
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    pub fn set_profile_batch_picking(&mut self, enabled: bool) {
        self.profile_batch_picking = enabled;
    }

    /// Sets whether newly uploaded identical meshes can use instanced color draws.
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    pub fn set_profile_repeated_pages(&mut self, enabled: bool) {
        self.profile_repeated_pages = enabled;
    }

    /// Enables conservative CPU rejection in color and picking sections for A/B profiling.
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    pub fn set_profile_section_culling(&mut self, enabled: bool) {
        if self.profile_section_culling != enabled {
            self.profile_section_culling = enabled;
            self.revision.mark_static();
        }
    }

    /// Enables grouped section and frustum tests for A/B profiling.
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    pub fn set_profile_section_groups(&mut self, enabled: bool) {
        if self.profile_section_groups != enabled {
            self.profile_section_groups = enabled;
            self.revision.mark_static();
        }
    }

    /// Enables short clipped gaps between packed raster draws for A/B profiling.
    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
    pub fn set_profile_section_bridging(&mut self, enabled: bool) {
        if self.profile_section_bridging != enabled {
            self.profile_section_bridging = enabled;
            self.revision.mark_static();
        }
    }

    /// Inserts a 3D mesh chunk anchored at `mesh.origin` and returns its [`Chunk3DId`].
    pub fn insert(&mut self, mesh: Mesh3D) -> Chunk3DId {
        self.revision.mark_static();
        self.geometry_revision = self.geometry_revision.wrapping_add(1);
        self.scene_envelope = None;
        self.pending_meshes += 1;
        let generation = NEXT_CHUNK_ID.fetch_add(1, Ordering::Relaxed);
        let origin = mesh.origin;
        let local_bounds = mesh.local_bounds;
        let has_picks = !mesh.pick_ids.is_empty();
        let entry = ChunkEntry {
            generation,
            origin,
            local_bounds,
            visible: true,
            pending_mesh: Some(Box::new(mesh)),
            vertex_buf: None,
            vertex_count: 0,
            face_ranges: None,
            edge_buf: None,
            edge_count: 0,
            pick_buf: None,
            has_picks,
            pick_bounds_trusted: false,
            section_bounds_trusted: false,
            small_packed: false,
            has_translucent: false,
            material_id: MaterialId::DEFAULT,
        };
        // Reuse the lowest vacant slot to preserve draw order, including equal-depth picks.
        let index = if let Some(Reverse(index)) = self.free_chunks.pop()
            && let Some(slot @ None) = self.chunks.get_mut(index)
        {
            *slot = Some(entry);
            index
        } else {
            let index = self.chunks.len();
            self.chunks.push(Some(entry));
            index
        };
        self.transform_table
            .mark_chunk(index, origin, true, has_picks, MaterialId::DEFAULT);
        if index >= self.chunk_generations.len() {
            self.chunk_generations.resize(index.saturating_add(1), 0);
            self.chunk_select_meta
                .resize(index.saturating_add(1), ChunkSelectMeta::EMPTY);
        }
        if let Some(gen_slot) = self.chunk_generations.get_mut(index) {
            *gen_slot = generation;
        }
        if let Some(meta) = self.chunk_select_meta.get_mut(index) {
            let (bounds_flag, bounds_rel) =
                match pack_select_bounds(origin, local_bounds, self.transform_table.anchor()) {
                    Some(bounds) => (SELECT_META_BOUNDS_F32, bounds),
                    None => (0, [0.0; 6]),
                };
            *meta = ChunkSelectMeta {
                repeated_face_page: u32::MAX,
                flags: SELECT_META_VISIBLE | bounds_flag,
                bounds_rel,
            };
        }
        Chunk3DId { index, generation }
    }

    /// Returns the live [`Chunk3DId`] stored at internal chunk slot `slot`, if any.
    ///
    /// Pair with [`iletki_wgpu::ViewportHost::take_color_frame_selection_preview`] to resolve
    /// compact 0-based slot indices from color-frame selection previews in O(1).
    #[must_use]
    pub fn chunk_id_at_slot(&self, slot: usize) -> Option<Chunk3DId> {
        let &generation = self.chunk_generations.get(slot)?;
        (generation != 0).then_some(Chunk3DId {
            index: slot,
            generation,
        })
    }

    /// Updates a chunk's world origin (for moving 3D objects without re-uploading vertices).
    pub fn set_origin(&mut self, id: Chunk3DId, origin: Vec3) {
        if origin.is_finite()
            && let Some(entry) = self.chunk_mut(id)
        {
            let visible = entry.visible;
            let has_picks = entry.has_picks;
            let material_id = entry.material_id;
            let local_bounds = entry.local_bounds;
            let trusted = entry.section_bounds_trusted;
            entry.origin = origin;
            if self.selected_slots.contains(id.index) {
                self.selected_bounds_revision = self.selected_bounds_revision.wrapping_add(1);
            }
            self.transform_table
                .mark_chunk(id.index, origin, visible, has_picks, material_id);
            if let Some(meta) = self.chunk_select_meta.get_mut(id.index) {
                match pack_select_bounds(origin, local_bounds, self.transform_table.anchor()) {
                    Some(bounds) => {
                        meta.bounds_rel = bounds;
                        meta.flags |= SELECT_META_BOUNDS_F32;
                    }
                    None => {
                        meta.flags &= !SELECT_META_BOUNDS_F32;
                    }
                }
            }
            // Conservatively expand the group and scene envelopes in O(1) if chunk moves outside.
            if trusted {
                if let Some(Some(group)) =
                    self.section_groups.get_mut(id.index / SECTION_GROUP_SIZE)
                    && !group.contains_chunk(origin, local_bounds)
                {
                    group.include(origin, local_bounds);
                }
                if let Some(env) = self.scene_envelope.as_mut()
                    && !env.bounds.contains_chunk(origin, local_bounds)
                {
                    env.bounds.include(origin, local_bounds);
                }
            }
            self.revision.mark_static();
        }
    }

    /// Toggles visibility of a chunk.
    pub fn set_visible(&mut self, id: Chunk3DId, visible: bool) {
        if let Some(entry) = self.chunk_mut(id) {
            let changed = entry.visible != visible;
            let origin = entry.origin;
            let has_picks = entry.has_picks;
            let material_id = entry.material_id;
            let is_direct = entry
                .vertex_buf
                .as_ref()
                .is_some_and(VertexRange::is_direct)
                || entry.edge_buf.as_ref().is_some_and(VertexRange::is_direct);
            entry.visible = visible;
            if changed {
                if visible {
                    self.hidden_chunks = self.hidden_chunks.saturating_sub(1);
                } else {
                    self.hidden_chunks = self.hidden_chunks.saturating_add(1);
                }
                if let Some(meta) = self.chunk_select_meta.get_mut(id.index) {
                    if visible {
                        meta.flags |= SELECT_META_VISIBLE;
                    } else {
                        meta.flags &= !SELECT_META_VISIBLE;
                    }
                }
                if self.selected_slots.contains(id.index) {
                    self.selected_revision = self.selected_revision.wrapping_add(1);
                    self.selected_bounds_revision = self.selected_bounds_revision.wrapping_add(1);
                }
                self.transform_table
                    .mark_chunk(id.index, origin, visible, has_picks, material_id);
                if is_direct {
                    self.geometry_revision = self.geometry_revision.wrapping_add(1);
                }
            }
            self.revision.mark_static();
        }
    }

    /// Toggles Jump-Flood selection outline highlighting for a chunk.
    ///
    /// When any visible chunk is selected, this layer automatically participates in
    /// [`DrawPhase::OutlineMask`] so [`iletki_wgpu::ViewportHost`] renders a crisp screen-space
    /// Jump-Flood silhouette outline around it.
    pub fn set_selected(&mut self, id: Chunk3DId, selected: bool) {
        if id.generation != 0
            && self
                .chunk_generations
                .get(id.index)
                .is_some_and(|&g| g == id.generation)
        {
            let changed = if selected {
                self.selected_slots.insert(id.index)
            } else {
                self.selected_slots.remove(id.index)
            };
            if changed {
                self.selected_revision = self.selected_revision.wrapping_add(1);
                self.selected_bounds_revision = self.selected_bounds_revision.wrapping_add(1);
            }
            self.revision.mark_dynamic();
        }
    }

    /// Returns whether `id` is currently marked as selected.
    #[must_use]
    pub fn is_selected(&self, id: Chunk3DId) -> bool {
        id.generation != 0
            && self
                .chunk_generations
                .get(id.index)
                .is_some_and(|&g| g == id.generation)
            && self.selected_slots.contains(id.index)
    }

    /// Clears selection on all chunks.
    pub fn clear_selection(&mut self) {
        self.revision.mark_dynamic();
        if !self.selected_slots.is_empty() {
            self.selected_revision = self.selected_revision.wrapping_add(1);
            self.selected_bounds_revision = self.selected_bounds_revision.wrapping_add(1);
            self.selected_slots.clear();
        }
    }

    /// Removes a chunk and releases its GPU resources. Other chunks can retain a shared page.
    pub fn remove(&mut self, id: Chunk3DId) {
        if let Some(slot) = self.chunks.get_mut(id.index)
            && slot
                .as_ref()
                .is_some_and(|entry| entry.generation == id.generation)
        {
            if let Some(removed) = slot.take() {
                if removed.pending_mesh.is_some() {
                    self.pending_meshes -= 1;
                }
                if !removed.visible {
                    self.hidden_chunks = self.hidden_chunks.saturating_sub(1);
                }
            }
            if let Some(gen_slot) = self.chunk_generations.get_mut(id.index) {
                *gen_slot = 0;
            }
            if let Some(meta) = self.chunk_select_meta.get_mut(id.index) {
                *meta = ChunkSelectMeta::EMPTY;
            }
            self.transform_table.clear_chunk(id.index);
            self.selected_slots.remove(id.index);
            self.free_chunks.push(Reverse(id.index));
            if let Some(group) = self.section_groups.get_mut(id.index / SECTION_GROUP_SIZE) {
                *group = None;
            }
            self.scene_envelope = None;
            self.geometry_revision = self.geometry_revision.wrapping_add(1);
            self.selected_revision = self.selected_revision.wrapping_add(1);
            self.selected_bounds_revision = self.selected_bounds_revision.wrapping_add(1);
            self.shared_storage_mask.clear();
            self.shared_mask_rev = 0;
            self.shared_mask_bounds_rev = 0;
            self.shared_selected_envelope = None;
            for view in &mut self.views {
                view.faces.clear();
                view.edges.clear();
                view.mask.clear();
                view.storage_faces.clear();
                view.storage_edges.clear();
                view.gpu_cull_active = false;
                view.retained_full_scene = false;
            }
            self.revision.mark_static();
        }
    }

    fn chunk_mut(&mut self, id: Chunk3DId) -> Option<&mut ChunkEntry> {
        self.chunks
            .get_mut(id.index)?
            .as_mut()
            .filter(|entry| entry.generation == id.generation)
    }

    /// Inserts a persistent [`Material3D`] into the scene's GPU material table and returns its
    /// [`MaterialId`].
    ///
    /// Material parameters live in a persistent GPU storage buffer (with a uniform mirror on
    /// direct/OpenGL paths) and are referenced by chunks via [`Self::set_chunk_material`] with
    /// zero extra vertex bytes.
    pub fn insert_material(&mut self, material: Material3D) -> MaterialId {
        let id = self.materials.insert(material);
        self.revision.mark_static();
        id
    }

    /// Updates an existing non-default material in place without rebuilding scene geometry.
    ///
    /// Returns `false` if `id` is [`MaterialId::DEFAULT`] or no longer exists.
    pub fn update_material(&mut self, id: MaterialId, material: Material3D) -> bool {
        let Some(translucency_changed) = self.materials.update(id, material) else {
            return false;
        };
        if translucency_changed {
            self.geometry_revision = self.geometry_revision.wrapping_add(1);
        }
        self.revision.mark_static();
        true
    }

    /// Returns the [`Material3D`] descriptor stored at `id`, if valid.
    #[must_use]
    pub fn material(&self, id: MaterialId) -> Option<Material3D> {
        self.materials.get(id)
    }

    /// Removes a non-default material and resets any chunks referencing it back to
    /// [`MaterialId::DEFAULT`].
    pub fn remove_material(&mut self, id: MaterialId) -> bool {
        let Some(was_translucent) = self.materials.remove(id) else {
            return false;
        };
        let mut any_reset = false;
        for (index, entry) in self.chunks.iter_mut().enumerate() {
            if let Some(chunk) = entry.as_mut()
                && chunk.material_id == id
            {
                chunk.material_id = MaterialId::DEFAULT;
                self.transform_table.mark_chunk(
                    index,
                    chunk.origin,
                    chunk.visible,
                    chunk.has_picks,
                    MaterialId::DEFAULT,
                );
                any_reset = true;
            }
        }
        if any_reset && was_translucent {
            self.geometry_revision = self.geometry_revision.wrapping_add(1);
        }
        self.revision.mark_static();
        true
    }

    /// Assigns a [`MaterialId`] from this scene to chunk `id` in O(1) without touching vertex
    /// buffers. Invalid chunk or material handles are ignored.
    pub fn set_chunk_material(&mut self, id: Chunk3DId, material: MaterialId) {
        if !self.materials.contains(material) {
            return;
        }
        let next_translucent = self.materials.is_translucent(material);
        let prev_translucent = self
            .chunks
            .get(id.index)
            .and_then(Option::as_ref)
            .filter(|entry| entry.generation == id.generation)
            .map(|entry| {
                (
                    entry.material_id,
                    entry.has_translucent || self.materials.is_translucent(entry.material_id),
                    entry.has_translucent || next_translucent,
                )
            });
        let Some((prev_mat, was_translucent, now_translucent)) = prev_translucent else {
            return;
        };
        if prev_mat == material {
            return;
        }
        if let Some(entry) = self.chunk_mut(id) {
            entry.material_id = material;
            let origin = entry.origin;
            let visible = entry.visible;
            let has_picks = entry.has_picks;
            self.transform_table
                .mark_chunk(id.index, origin, visible, has_picks, material);
            if was_translucent != now_translucent {
                self.geometry_revision = self.geometry_revision.wrapping_add(1);
            }
            self.revision.mark_static();
        }
    }

    /// Returns the [`MaterialId`] assigned to chunk `id`, if `id` is live.
    #[must_use]
    pub fn chunk_material(&self, id: Chunk3DId) -> Option<MaterialId> {
        self.chunks
            .get(id.index)?
            .as_ref()
            .filter(|entry| entry.generation == id.generation)
            .map(|entry| entry.material_id)
    }

    /// Releases per-viewport binding sets for closed viewports and returns the number released.
    ///
    /// The layer also prunes closed bindings before preparing a viewport.
    pub fn prune_closed_viewports(&mut self) -> usize {
        let previous = self.views.len();
        self.views.retain(|view| view.lifetime.is_alive());
        previous - self.views.len()
    }

    fn ensure_pipelines<W>(&mut self, ctx: &PrepareCtx<'_, W>) {
        let section_aa = SectionAa::for_viewport(ctx);
        let want_storage = storage_batch::supports(ctx.device);
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let want_storage = want_storage && self.profile_storage_transforms;
        if self
            .pipelines
            .as_ref()
            .is_some_and(|p| p.matches(ctx, section_aa, want_storage))
        {
            return;
        }
        self.views.clear();
        self.pipelines = Some(ScenePipelines::build(ctx, want_storage));
    }

    fn ensure_select_pipelines<W>(&mut self, ctx: &PrepareCtx<'_, W>) {
        if let Some(p) = &mut self.pipelines {
            p.ensure_select(ctx);
        }
    }

    fn sync_shared_selection(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        use_storage: bool,
    ) {
        if !use_storage && self.shared_storage_mask.has_packed() {
            self.shared_storage_mask.clear();
            self.shared_mask_rev = 0;
        }
        if self.selected_slots.is_empty() {
            if self.shared_mask_rev != self.selected_revision {
                self.shared_storage_mask.clear();
                self.shared_selected_envelope = None;
                self.shared_mask_rev = self.selected_revision;
                self.shared_mask_bounds_rev = self.selected_bounds_revision;
            }
            return;
        }
        let large_sel = self.selected_slots.len() > LARGE_SELECTION_ENVELOPE_THRESHOLD;
        let need_mask = use_storage && self.shared_mask_rev != self.selected_revision;
        let need_env = large_sel && self.shared_mask_bounds_rev != self.selected_bounds_revision;
        if !need_mask && !need_env {
            return;
        }
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let _scope = iletki_wgpu::profiling::scope("scene.outline_mask_prepare");

        let mut env_bounds = Bounds3::EMPTY;
        let mut env_valid = true;
        let mut env_rel_min = [f32::INFINITY; 3];
        let mut env_rel_max = [f32::NEG_INFINITY; 3];
        let mut used_rel_env = false;
        let mut used_world_env = false;

        if need_mask {
            let mut fast_repeated = !self.repeated_face_pages.is_empty();
            if fast_repeated {
                self.repeated_mask_scratch.clear();
                let required = SELECT_META_VISIBLE | SELECT_META_HAS_VERTS;
                for slot in self.selected_slots.iter() {
                    let Some(&meta) = self.chunk_select_meta.get(slot) else {
                        continue;
                    };
                    if (meta.flags & required) != required {
                        continue;
                    }
                    if meta.repeated_face_page == u32::MAX {
                        fast_repeated = false;
                        break;
                    }
                    if let Ok(slot_u32) = u32::try_from(slot) {
                        self.repeated_mask_scratch
                            .push((meta.repeated_face_page, slot_u32));
                    }
                    if need_env && env_valid {
                        if (meta.flags & SELECT_META_BOUNDS_F32) != 0 {
                            env_rel_min[0] = env_rel_min[0].min(meta.bounds_rel[0]);
                            env_rel_min[1] = env_rel_min[1].min(meta.bounds_rel[1]);
                            env_rel_min[2] = env_rel_min[2].min(meta.bounds_rel[2]);
                            env_rel_max[0] = env_rel_max[0].max(meta.bounds_rel[3]);
                            env_rel_max[1] = env_rel_max[1].max(meta.bounds_rel[4]);
                            env_rel_max[2] = env_rel_max[2].max(meta.bounds_rel[5]);
                            used_rel_env = true;
                        } else if let Some(Some(chunk)) = self.chunks.get(slot) {
                            if chunk.local_bounds.is_empty() || !chunk.origin.is_finite() {
                                env_valid = false;
                            } else {
                                env_bounds.include(chunk.origin + chunk.local_bounds.min);
                                env_bounds.include(chunk.origin + chunk.local_bounds.max);
                                used_world_env = true;
                            }
                        }
                    }
                }
            }
            if fast_repeated {
                self.shared_storage_mask.rebuild_repeated_only(
                    device,
                    queue,
                    &self.repeated_face_pages,
                    self.repeated_mask_scratch.iter().copied(),
                );
            } else {
                env_bounds = Bounds3::EMPTY;
                env_valid = true;
                used_rel_env = false;
                used_world_env = false;
                let chunks = &self.chunks;
                let selected = &self.selected_slots;
                let iter = selected.iter().filter_map(|slot| {
                    let chunk = chunks.get(slot)?.as_ref()?;
                    if !chunk.visible || chunk.vertex_count == 0 {
                        return None;
                    }
                    if need_env && env_valid {
                        if chunk.local_bounds.is_empty() || !chunk.origin.is_finite() {
                            env_valid = false;
                        } else {
                            env_bounds.include(chunk.origin + chunk.local_bounds.min);
                            env_bounds.include(chunk.origin + chunk.local_bounds.max);
                            used_world_env = true;
                        }
                    }
                    let geometry = chunk.vertex_buf.as_ref()?;
                    Some((slot, geometry))
                });
                self.shared_storage_mask
                    .rebuild_unordered(device, queue, iter);
            }
            self.shared_mask_rev = self.selected_revision;
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            iletki_wgpu::profiling::count(
                "scene.outline_mask_draws",
                self.shared_storage_mask.count(),
            );
        } else if need_env {
            let required = SELECT_META_VISIBLE | SELECT_META_HAS_VERTS;
            for slot in self.selected_slots.iter() {
                let Some(&meta) = self.chunk_select_meta.get(slot) else {
                    continue;
                };
                if (meta.flags & required) != required {
                    continue;
                }
                if (meta.flags & SELECT_META_BOUNDS_F32) != 0 {
                    env_rel_min[0] = env_rel_min[0].min(meta.bounds_rel[0]);
                    env_rel_min[1] = env_rel_min[1].min(meta.bounds_rel[1]);
                    env_rel_min[2] = env_rel_min[2].min(meta.bounds_rel[2]);
                    env_rel_max[0] = env_rel_max[0].max(meta.bounds_rel[3]);
                    env_rel_max[1] = env_rel_max[1].max(meta.bounds_rel[4]);
                    env_rel_max[2] = env_rel_max[2].max(meta.bounds_rel[5]);
                    used_rel_env = true;
                } else if let Some(Some(chunk)) = self.chunks.get(slot) {
                    if chunk.local_bounds.is_empty() || !chunk.origin.is_finite() {
                        env_valid = false;
                        break;
                    }
                    env_bounds.include(chunk.origin + chunk.local_bounds.min);
                    env_bounds.include(chunk.origin + chunk.local_bounds.max);
                    used_world_env = true;
                }
            }
        }

        if need_env {
            let anchor = self.transform_table.anchor();
            self.shared_selected_envelope = if !env_valid {
                None
            } else if used_rel_env && !used_world_env {
                let local = Bounds3::from_corners(
                    Vec3::new(
                        f64::from(env_rel_min[0]),
                        f64::from(env_rel_min[1]),
                        f64::from(env_rel_min[2]),
                    ),
                    Vec3::new(
                        f64::from(env_rel_max[0]),
                        f64::from(env_rel_max[1]),
                        f64::from(env_rel_max[2]),
                    ),
                );
                (!local.is_empty()).then_some((local, anchor))
            } else {
                if used_rel_env {
                    env_bounds.include(
                        anchor
                            + Vec3::new(
                                f64::from(env_rel_min[0]),
                                f64::from(env_rel_min[1]),
                                f64::from(env_rel_min[2]),
                            ),
                    );
                    env_bounds.include(
                        anchor
                            + Vec3::new(
                                f64::from(env_rel_max[0]),
                                f64::from(env_rel_max[1]),
                                f64::from(env_rel_max[2]),
                            ),
                    );
                }
                env_bounds.center().map(|center| {
                    let local =
                        Bounds3::from_corners(env_bounds.min - center, env_bounds.max - center);
                    (local, center)
                })
            };
            self.shared_mask_bounds_rev = self.selected_bounds_revision;
        }
    }

    fn prepare_packed_mask<W>(
        ctx: &PrepareCtx<'_, W>,
        layout: &wgpu::BindGroupLayout,
        chunks: &[Option<ChunkEntry>],
        selected_slots: &SelectionSet,
        view: &mut ViewObjects,
    ) -> bool {
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let _scope = iletki_wgpu::profiling::scope("scene.outline_mask_prepare");
        let selected = view.draws.iter().enumerate().filter_map(|(index, object)| {
            if !selected_slots.contains(object.chunk) {
                return None;
            }
            let chunk = chunks.get(object.chunk)?.as_ref()?;
            Some((index, object.uniforms, chunk.vertex_buf.as_ref()?))
        });
        if selected.clone().take(128).count() < 128 {
            return false;
        }
        view.mask.clear();
        for (index, uniforms, geometry) in selected {
            view.mask.push(geometry, uniforms, index);
        }
        view.mask.upload(ctx.device, layout, ctx.queue);
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        iletki_wgpu::profiling::count("scene.outline_mask_draws", view.mask.counts().0);
        true
    }

    fn paint_outline_mask<W>(
        &self,
        pipeline: &wgpu::RenderPipeline,
        packed: Option<&wgpu::RenderPipeline>,
        storage: Option<&wgpu::RenderPipeline>,
        repeated_storage: Option<&wgpu::RenderPipeline>,
        pass: &mut wgpu::RenderPass<'_>,
        ctx: &PaintCtx<'_, W>,
    ) {
        let Some(p) = &self.pipelines else {
            return;
        };
        let Some(view) = self
            .views
            .iter()
            .find(|view| view.id == ctx.viewport_id && !view.picking)
        else {
            return;
        };
        let draw_direct = |pass: &mut wgpu::RenderPass<'_>, index: usize| {
            if let Some(object) = view.draws.get(index)
                && self.selected_slots.contains(object.chunk)
                && let Some(Some(chunk)) = self.chunks.get(object.chunk)
                && chunk.visible
                && let Some(vertices) = &chunk.vertex_buf
                && view.bind(pass, index, p.dynamic_uniforms)
            {
                vertices.draw_direct(pass, chunk.vertex_count);
            }
        };
        if let (Some(storage_pl), Some(storage_bg)) = (
            storage,
            view.storage_binding
                .as_ref()
                .and_then(StorageViewBinding::bind_group),
        ) && self.shared_storage_mask.has_packed()
        {
            self.shared_storage_mask.paint_position_only(
                pass,
                storage_bg,
                storage_pl,
                repeated_storage,
                |pass, chunk_slot| {
                    if let Some(draw_index) = view.draws.iter().position(|d| d.chunk == chunk_slot)
                    {
                        pass.set_pipeline(pipeline);
                        draw_direct(pass, draw_index);
                    }
                },
            );
        } else if let Some(packed) = packed
            && view.mask.has_packed()
        {
            let pipelines = PaintPipelines {
                smooth: packed,
                flat: None,
                repeated_smooth: None,
                repeated_flat: None,
                repeated_slots: None,
            };
            view.mask.paint(pass, &pipelines, false, |pass, index| {
                pass.set_pipeline(pipeline);
                draw_direct(pass, index);
            });
        } else {
            pass.set_pipeline(pipeline);
            for index in 0..view.draws.len() {
                draw_direct(pass, index);
            }
        }
    }
}

impl<W> ViewportLayer<W> for Scene3DLayer {
    fn id(&self) -> u16 {
        self.layer_id
    }

    fn phases(&self) -> DrawPhaseMask {
        let mut mask = DrawPhaseMask::OPAQUE;
        let any_picks = self.chunks.iter().flatten().any(|c| {
            c.visible
                && (c.has_picks
                    || c.pending_mesh
                        .as_ref()
                        .is_some_and(|m| !m.pick_ids.is_empty()))
        });
        if any_picks {
            mask = mask.union(DrawPhaseMask::PICKING);
        }
        let any_selected = self.selected_slots.iter().any(|index| {
            self.chunk_select_meta
                .get(index)
                .is_some_and(|meta| (meta.flags & SELECT_META_VISIBLE) != 0)
        });
        if any_selected {
            mask = mask.union(DrawPhaseMask::OUTLINE_MASK);
        }
        mask
    }

    fn revision(&self) -> LayerRevision {
        self.revision
    }

    fn prepare(&mut self, ctx: &mut PrepareCtx<'_, W>) -> bool {
        self.prune_closed_viewports();
        self.ensure_pipelines(ctx);
        let section = SectionAa::for_viewport(ctx).filter(|_| !ctx.camera.clip_planes().is_empty());
        self.use_antialiased_section = section.is_some();
        let Some(p) = &self.pipelines else {
            return false;
        };
        let cam_pos = ctx.camera.position_3d();
        let mut pending = false;
        let pack_geometry = p.packed.is_some() || p.storage.is_some();
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let pack_geometry = pack_geometry && self.profile_batch_geometry;
        let use_storage = pack_geometry
            && p.storage.is_some()
            && self.transform_table.in_range_for_camera(cam_pos);
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let use_indexed = self.profile_indexed_geometry;
        #[cfg(not(all(feature = "profiling", not(target_arch = "wasm32"))))]
        let use_indexed = true;
        let flat_corner = p.flat_support.filter(|_| use_indexed).map(|s| s.corner);
        let repeat_pages = pack_geometry;
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let repeat_pages = repeat_pages && self.profile_repeated_pages;
        let vertex_layout = p.vertex_layout;
        let mut flat_builder = PageBuilder::new(
            vertex_layout,
            PageKind::IndexedFlat,
            repeat_pages,
            use_storage,
        );
        let mut smooth_idx_builder = PageBuilder::new(
            vertex_layout,
            PageKind::IndexedSmooth,
            repeat_pages,
            use_storage,
        );
        let mut builder = PageBuilder::new(
            vertex_layout,
            PageKind::Unindexed,
            repeat_pages,
            use_storage,
        );
        let mut uploaded_faces = Vec::new();
        let mut edge_builder = PageBuilder::new(
            vertex_layout,
            PageKind::Unindexed,
            repeat_pages,
            use_storage,
        );
        let mut uploaded_edges = Vec::new();
        let mut uploaded_any = false;

        let upload_slots = if self.pending_meshes == 0 {
            0
        } else {
            self.chunks.len()
        };
        for (chunk_index, entry) in self.chunks.iter_mut().take(upload_slots).enumerate() {
            let Some(chunk) = entry else { continue };
            let Some(mesh) = &chunk.pending_mesh else {
                continue;
            };
            let pack = pack_geometry && PageBuilder::can_pack(&mesh.vertices, 3);
            let pack_edges = pack_geometry && PageBuilder::can_pack(&mesh.edges, 2);
            {
                let picks_slice = (mesh.pick_ids.len() == mesh.vertices.len())
                    .then_some(mesh.pick_ids.as_slice());
                let likely_indexed = use_indexed
                    && mesh.vertices.len().is_multiple_of(3)
                    && (flat_corner.is_some()
                        && provoking::first_triangle_is_flat(&mesh.vertices, picks_slice));
                let face_bytes = if likely_indexed {
                    // Provoking-indexed pages store ~35% as many vertices plus u16 indices.
                    mesh.vertices.len().saturating_mul(14)
                } else {
                    let mut b = size_of_val(mesh.vertices.as_slice());
                    if pack {
                        b = b.saturating_add(mesh.vertices.len().saturating_mul(size_of::<u32>()));
                    }
                    if pack || picks_slice.is_some() {
                        b = b.saturating_add(
                            mesh.vertices.len().saturating_mul(size_of::<[u32; 2]>()),
                        );
                    }
                    b
                };
                let mut bytes = face_bytes.saturating_add(size_of_val(mesh.edges.as_slice()));
                if pack_edges {
                    bytes = bytes.saturating_add(mesh.edges.len().saturating_mul(size_of::<u32>()));
                }
                if !ctx.uploads.reserve(bytes) {
                    pending = true;
                    continue;
                }
            }
            if let Some(mesh) = chunk.pending_mesh.take() {
                uploaded_any = true;
                self.pending_meshes -= 1;
                self.revision.mark_static();
                if let Some(group) = self
                    .section_groups
                    .get_mut(chunk_index / SECTION_GROUP_SIZE)
                {
                    *group = None;
                }
                let face_bounds = bounds_cover_vertices(chunk.local_bounds, &mesh.vertices);
                chunk.pick_bounds_trusted = face_bounds;
                chunk.section_bounds_trusted =
                    face_bounds && bounds_cover_vertices(chunk.local_bounds, &mesh.edges);
                chunk.small_packed = (pack || mesh.vertices.is_empty())
                    && (pack_edges || mesh.edges.is_empty())
                    && mesh.vertices.len().saturating_add(mesh.edges.len())
                        <= MAX_DEFERRED_SECTION_VERTICES;
                chunk.has_translucent = mesh.vertices.iter().any(|v| v.color[3] < 255);
                if mesh.vertices.is_empty() {
                    chunk.vertex_buf = None;
                    chunk.vertex_count = 0;
                    chunk.pick_buf = None;
                    chunk.has_picks = false;
                } else {
                    chunk.vertex_count = u32::try_from(mesh.vertices.len()).unwrap_or(0);
                    let picks = (mesh.pick_ids.len() == mesh.vertices.len())
                        .then_some(mesh.pick_ids.as_slice());
                    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
                    let prep_start = std::time::Instant::now();
                    let mut indexed_mode: Option<bool> = None; // Some(true)=flat, Some(false)=smooth
                    if use_indexed && mesh.vertices.len().is_multiple_of(3) {
                        if let Some(corner) = flat_corner
                            && provoking::is_flat_eligible(&mesh.vertices, picks)
                        {
                            if provoking::build_provoking_indexed(
                                &mesh.vertices,
                                picks,
                                corner,
                                &mut self.dedup_scratch,
                            ) {
                                indexed_mode = Some(true);
                            }
                        } else if provoking::build_exact_indexed(
                            &mesh.vertices,
                            picks,
                            &mut self.dedup_scratch,
                        ) {
                            indexed_mode = Some(false);
                        }
                    }
                    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
                    {
                        self.uploaded_chunks = self.uploaded_chunks.saturating_add(1);
                        self.uploaded_input_vertices = self
                            .uploaded_input_vertices
                            .saturating_add(mesh.vertices.len());
                        self.uploaded_preprocess_us = self
                            .uploaded_preprocess_us
                            .saturating_add(prep_start.elapsed().as_micros() as usize);
                        match indexed_mode {
                            Some(true) => {
                                self.uploaded_flat_chunks =
                                    self.uploaded_flat_chunks.saturating_add(1);
                                self.uploaded_stored_vertices = self
                                    .uploaded_stored_vertices
                                    .saturating_add(self.dedup_scratch.vertices.len());
                            }
                            Some(false) => {
                                self.uploaded_smooth_indexed_chunks =
                                    self.uploaded_smooth_indexed_chunks.saturating_add(1);
                                self.uploaded_stored_vertices = self
                                    .uploaded_stored_vertices
                                    .saturating_add(self.dedup_scratch.vertices.len());
                            }
                            None => {
                                self.uploaded_stored_vertices = self
                                    .uploaded_stored_vertices
                                    .saturating_add(mesh.vertices.len());
                            }
                        }
                    }

                    if let Some(is_flat) = indexed_mode {
                        let idx_verts = self.dedup_scratch.vertices.as_slice();
                        let idx_indices = self.dedup_scratch.indices.as_slice();
                        let idx_picks = (!self.dedup_scratch.picks.is_empty())
                            .then_some(self.dedup_scratch.picks.as_slice());
                        if pack_geometry && idx_verts.len() <= face_batch::MAX_VERTICES {
                            chunk.face_ranges = FaceRanges::new(&mesh.vertices).map(Box::new);
                            let target_builder = if is_flat {
                                &mut flat_builder
                            } else {
                                &mut smooth_idx_builder
                            };
                            target_builder.push_indexed(
                                chunk_index,
                                idx_verts,
                                idx_indices,
                                idx_picks,
                                ctx.device,
                                &mut uploaded_faces,
                            );
                            chunk.pick_buf = None;
                        } else {
                            let index_count = u32::try_from(idx_indices.len()).unwrap_or(0);
                            chunk.vertex_buf = Some(VertexRange::DirectIndexed {
                                vertices: Box::new(VertexBuffers::upload(
                                    ctx.device,
                                    vertex_layout,
                                    idx_verts,
                                )),
                                indices: Box::new(face_batch::upload_index_buffer_u32(
                                    ctx.device,
                                    idx_indices,
                                )),
                                index_format: wgpu::IndexFormat::Uint32,
                                index_count,
                                flat: is_flat,
                            });
                            chunk.pick_buf = idx_picks.map(|ids| {
                                Box::new(ctx.device.create_buffer_init(
                                    &wgpu::util::BufferInitDescriptor {
                                        label: Some("iletki-3d indexed pick ids"),
                                        contents: bytemuck::cast_slice(ids),
                                        usage: wgpu::BufferUsages::VERTEX,
                                    },
                                ))
                            });
                        }
                    } else if pack {
                        chunk.face_ranges = FaceRanges::new(&mesh.vertices).map(Box::new);
                        builder.push(
                            chunk_index,
                            &mesh.vertices,
                            picks,
                            ctx.device,
                            &mut uploaded_faces,
                        );
                        chunk.pick_buf = None;
                    } else {
                        chunk.vertex_buf = Some(VertexRange::Direct(Box::new(
                            VertexBuffers::upload(ctx.device, vertex_layout, &mesh.vertices),
                        )));
                        chunk.pick_buf = picks.map(|ids| {
                            Box::new(ctx.device.create_buffer_init(
                                &wgpu::util::BufferInitDescriptor {
                                    label: Some("iletki-3d pick ids"),
                                    contents: bytemuck::cast_slice(ids),
                                    usage: wgpu::BufferUsages::VERTEX,
                                },
                            ))
                        });
                    }
                    chunk.has_picks = picks.is_some();
                }
                self.transform_table.mark_chunk(
                    chunk_index,
                    chunk.origin,
                    chunk.visible,
                    chunk.has_picks,
                    chunk.material_id,
                );
                if let Some(meta) = self.chunk_select_meta.get_mut(chunk_index) {
                    meta.repeated_face_page = u32::MAX;
                    if chunk.vertex_count != 0 {
                        meta.flags |= SELECT_META_HAS_VERTS;
                    } else {
                        meta.flags &= !SELECT_META_HAS_VERTS;
                    }
                }
                if mesh.edges.is_empty() {
                    chunk.edge_buf = None;
                    chunk.edge_count = 0;
                } else {
                    chunk.edge_count = u32::try_from(mesh.edges.len()).unwrap_or(0);
                    if pack_edges {
                        edge_builder.push(
                            chunk_index,
                            &mesh.edges,
                            None,
                            ctx.device,
                            &mut uploaded_edges,
                        );
                    } else {
                        chunk.edge_buf = Some(VertexRange::Direct(Box::new(
                            VertexBuffers::upload(ctx.device, vertex_layout, &mesh.edges),
                        )));
                    }
                }
                ctx.uploads.finish_upload();
            }
        }
        flat_builder.flush(ctx.device, &mut uploaded_faces);
        smooth_idx_builder.flush(ctx.device, &mut uploaded_faces);
        builder.flush(ctx.device, &mut uploaded_faces);
        for (index, faces) in uploaded_faces {
            self.has_repeated_pages |= faces.repeats();
            if let VertexRange::Packed { page, .. } = &faces
                && page.repeated_elements() != 0
                && page.repeated_chunk_ids().is_some()
            {
                let ptr = Arc::as_ptr(page) as usize;
                let page_idx = if let Some(&existing) = self.repeated_face_page_map.get(&ptr) {
                    existing
                } else if let Ok(idx) = u32::try_from(self.repeated_face_pages.len()) {
                    self.repeated_face_pages.push(Arc::clone(page));
                    self.repeated_face_page_map.insert(ptr, idx);
                    idx
                } else {
                    u32::MAX
                };
                if let Some(meta) = self.chunk_select_meta.get_mut(index) {
                    meta.repeated_face_page = page_idx;
                }
            }
            if let Some(Some(chunk)) = self.chunks.get_mut(index) {
                chunk.vertex_buf = Some(faces);
            }
        }

        edge_builder.flush(ctx.device, &mut uploaded_edges);
        for (index, edges) in uploaded_edges {
            self.has_repeated_pages |= edges.repeats();
            if let Some(Some(chunk)) = self.chunks.get_mut(index) {
                chunk.edge_buf = Some(edges);
            }
        }
        if uploaded_any {
            self.geometry_revision = self.geometry_revision.wrapping_add(1);
            self.scene_envelope = None;
            if !self.selected_slots.is_empty() {
                self.selected_revision = self.selected_revision.wrapping_add(1);
                self.selected_bounds_revision = self.selected_bounds_revision.wrapping_add(1);
            }
        }

        let has_custom_materials = self.materials.has_custom_materials();
        if ctx.selection_id_attachment() {
            self.ensure_select_pipelines(ctx);
        }
        if has_custom_materials
            && ctx.picking_rect_px().is_none()
            && let Some(pipelines) = self.pipelines.as_mut()
        {
            pipelines.ensure_materials(ctx, ctx.selection_id_attachment());
        }
        if self.has_repeated_pages
            && let Some(pipelines) = self.pipelines.as_mut()
        {
            if let Some(packed) = pipelines.packed.as_mut() {
                packed.ensure_repeated(ctx, vertex_layout);
            }
            if let Some(storage) = pipelines.storage.as_mut() {
                storage.ensure_repeated(ctx, vertex_layout);
                if let Some(sel_storage) = pipelines
                    .select
                    .as_mut()
                    .and_then(|sel| sel.storage.as_mut())
                {
                    sel_storage.ensure_repeated(ctx, vertex_layout, &storage.pipeline_layout);
                }
            }
            if let Some(mat_storage) = pipelines
                .materials
                .as_mut()
                .and_then(|mats| mats.storage.as_mut())
            {
                mat_storage.ensure_repeated(ctx, vertex_layout);
            }
        }
        if use_storage {
            self.transform_table.sync(ctx.device, ctx.queue);
            if has_custom_materials && ctx.picking_rect_px().is_none() {
                self.materials.sync_ssbo(ctx.device, ctx.queue);
            }
        }
        if has_custom_materials
            && ctx.picking_rect_px().is_none()
            && let Some(p) = self.pipelines.as_ref()
            && let Some(mats) = p.materials.as_ref()
        {
            self.materials.sync_direct(
                ctx.device,
                ctx.queue,
                &mats.direct_material_layout,
                p.dynamic_uniforms,
            );
        }
        let want_gpu_cull = use_storage
            && ctx.picking_rect_px().is_none()
            && ctx.supports_gpu_culling()
            && gpu_cull::supports(ctx.device);
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let want_gpu_cull = want_gpu_cull && self.profile_gpu_culling;
        if want_gpu_cull {
            if self.gpu_cull_pipelines.is_none() {
                self.gpu_cull_pipelines = Some(GpuCullPipelines::new(ctx.device));
            }
            if self.gpu_cull_built_rev != self.geometry_revision {
                let materials = &self.materials;
                self.gpu_cull_scene = GpuCullScene::build(
                    ctx.device,
                    self.geometry_revision,
                    self.chunks
                        .iter()
                        .enumerate()
                        .filter_map(|(chunk_index, entry)| {
                            let chunk = entry.as_ref()?;
                            if chunk.vertex_count == 0 && chunk.edge_count == 0 {
                                return None;
                            }
                            Some(CullChunkInput {
                                chunk_index,
                                origin: chunk.origin,
                                local_bounds: chunk.local_bounds,
                                trusted: chunk.section_bounds_trusted,
                                translucent: chunk.has_translucent
                                    || materials.is_translucent(chunk.material_id),
                                vertex_buf: chunk.vertex_buf.as_ref(),
                                face_ranges: chunk.face_ranges.as_deref(),
                                edge_buf: chunk.edge_buf.as_ref(),
                            })
                        }),
                );
                self.gpu_cull_built_rev = self.geometry_revision;
            }
        }
        let picking = ctx.picking_rect_px();
        if picking.is_none() {
            self.sync_shared_selection(ctx.device, ctx.queue, use_storage);
        }
        let Some(p) = self.pipelines.as_ref() else {
            return pending;
        };
        let batch = p.dynamic_uniforms;
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let batch = batch && self.profile_batch_uniforms;
        let index = self
            .views
            .iter()
            .position(|view| view.id == ctx.viewport_id && view.picking == picking.is_some())
            .unwrap_or_else(|| {
                let index = self.views.len();
                self.views
                    .push(ViewObjects::new(ctx, batch, picking.is_some()));
                index
            });
        let Some(view) = self.views.get_mut(index) else {
            return pending;
        };
        if use_storage && let Some(storage_pl) = &p.storage {
            let binding = view
                .storage_binding
                .get_or_insert_with(|| StorageViewBinding::new(ctx.device));
            binding.update(
                ctx.device,
                ctx.queue,
                &storage_pl.layout,
                &self.transform_table,
                cam_pos,
            );
            if let Some(mat_storage) = p.materials.as_ref().and_then(|m| m.storage.as_ref()) {
                binding.update_materials(
                    ctx.device,
                    &mat_storage.layout,
                    &self.transform_table,
                    &self.materials,
                );
            }
        }
        if picking.is_none() {
            let update = {
                #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
                let _scope = iletki_wgpu::profiling::scope("scene.outline_bounds_prepare");
                if self.selected_slots.len() > LARGE_SELECTION_ENVELOPE_THRESHOLD {
                    view.outline_bounds.update_envelope(
                        ctx,
                        self.revision,
                        self.shared_selected_envelope,
                    )
                } else {
                    view.outline_bounds.update(
                        ctx,
                        self.revision,
                        self.chunks.len(),
                        self.selected_slots.iter().filter_map(|index| {
                            let chunk = self.chunks.get(index)?.as_ref()?;
                            (chunk.visible && chunk.vertex_count != 0).then_some((
                                index,
                                chunk.local_bounds,
                                chunk.origin,
                            ))
                        }),
                    )
                }
            };
            match update {
                OutlineUpdate::Unchanged => return pending,
                OutlineUpdate::Selection if self.selected_slots.is_empty() => {
                    view.mask.clear();
                    return pending;
                }
                OutlineUpdate::Selection
                    if use_storage
                        && (view.storage_faces.has_packed()
                            || view.retained_full_scene
                            || view.gpu_cull_active) =>
                {
                    return pending;
                }
                OutlineUpdate::Selection
                    if pack_geometry && !use_storage && view.faces.has_packed() =>
                {
                    if let Some(packed) = &p.packed
                        && Self::prepare_packed_mask(
                            ctx,
                            &packed.layout,
                            &self.chunks,
                            &self.selected_slots,
                            view,
                        )
                    {
                        // A dense mask uses the cached face ranges and its own uniform pages.
                        // Direct face draws already have their own transform slots.
                        return pending;
                    }
                }
                OutlineUpdate::Selection | OutlineUpdate::View => {}
            }
        }

        let projection = ctx.camera.view_proj_relative(ctx.logical_size);
        let frustum = iletki_core::Frustum::from_view_proj_reverse_z(projection);
        let pick_volume = picking.map(|rect| PickVolume::new(projection, ctx.active_px, rect));
        let cull_sections = !ctx.camera.clip_planes().is_empty()
            && (picking.is_some() || section == Some(SectionAa::Raster));
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let cull_sections = cull_sections && self.profile_section_culling;
        let bridge_sections_enabled = picking.is_none() && cull_sections && pack_geometry;
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let bridge_sections_enabled = bridge_sections_enabled && self.profile_section_bridging;
        let section_volume = cull_sections.then(|| SectionVolume::new(ctx.camera));
        let group_visibility = self.chunks.len() >= SECTION_GROUP_SIZE * 4;
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let group_visibility = group_visibility && self.profile_section_groups;
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let profiling = iletki_wgpu::profiling::active();

        if use_storage {
            view.faces.clear();
            view.edges.clear();
            view.mask.clear();
            if picking.is_none() {
                if want_gpu_cull
                    && let (Some(gpu_pipelines), Some(gpu_scene)) =
                        (&self.gpu_cull_pipelines, &self.gpu_cull_scene)
                {
                    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
                    let _visibility = iletki_wgpu::profiling::scope("scene.visibility");
                    let uniforms = CullViewUniforms::build(
                        ctx.camera,
                        ctx.logical_size,
                        ctx.dpi_scale,
                        ctx.active_px,
                        self.transform_table.anchor(),
                        cull_sections,
                        gpu_scene,
                    );
                    if GpuCullView::ensure_and_update(
                        &mut view.gpu_cull,
                        ctx.device,
                        ctx.queue,
                        gpu_pipelines,
                        gpu_scene,
                        &self.transform_table,
                        &uniforms,
                        cam_pos,
                    )
                    .is_some()
                    {
                        view.retained_full_scene = false;
                        view.gpu_cull_active = true;
                        view.draws.clear();
                        view.direct_uniforms.clear();
                        view.storage_faces.clear();
                        view.storage_edges.clear();
                        view.has_edges = gpu_scene.has_edges;

                        for &chunk_index in &gpu_scene.extra_chunks {
                            let Some(Some(chunk)) = self.chunks.get(chunk_index) else {
                                continue;
                            };
                            if !chunk.visible || (chunk.vertex_count == 0 && chunk.edge_count == 0)
                            {
                                continue;
                            }
                            if chunk.section_bounds_trusted {
                                let bounds = Bounds3::from_corners(
                                    chunk.origin + chunk.local_bounds.min,
                                    chunk.origin + chunk.local_bounds.max,
                                );
                                if !bounds.is_empty()
                                    && !frustum.intersects_bounds_world(bounds, cam_pos)
                                {
                                    continue;
                                }
                            }
                            let offset_rel = (chunk.origin - cam_pos).to_f32();
                            if chunk.section_bounds_trusted
                                && section_volume
                                    .as_ref()
                                    .is_some_and(|v| v.excludes(chunk.local_bounds, offset_rel))
                            {
                                continue;
                            }
                            let face_extra = chunk.vertex_buf.as_ref().filter(|v| {
                                !gpu_cull::face_in_gpu_cull(v, chunk.face_ranges.is_some())
                            });
                            let edge_extra = chunk
                                .edge_buf
                                .as_ref()
                                .filter(|e| !gpu_cull::edge_in_gpu_cull(e));
                            let needs_direct = face_extra.is_some_and(VertexRange::is_direct)
                                || edge_extra.is_some_and(VertexRange::is_direct);
                            let draw_index = if needs_direct {
                                let idx = view.draws.len();
                                let uniforms = ObjectUniforms {
                                    offset_rel,
                                    scale: 1.0,
                                };
                                view.draws.push(PreparedObject {
                                    chunk: chunk_index,
                                    section_culled: false,
                                    bridge_section_gap: false,
                                    direct_slot: Some(view.direct_uniforms.len()),
                                    uniforms,
                                });
                                view.direct_uniforms.push(uniforms);
                                idx
                            } else {
                                0
                            };
                            if let Some(geometry) = face_extra {
                                view.storage_faces
                                    .push_color(geometry, draw_index, false, false, true);
                            }
                            if let Some(geometry) = edge_extra {
                                view.has_edges = true;
                                view.storage_edges
                                    .push_color(geometry, draw_index, false, false, true);
                            }
                        }
                        if !view.direct_uniforms.is_empty() {
                            view.upload(ctx.device, ctx.queue, &p.object_layout);
                        }
                        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
                        if profiling {
                            iletki_wgpu::profiling::count("scene.gpu_cull_active", 1);
                            iletki_wgpu::profiling::count(
                                "scene.gpu_cull_items",
                                gpu_scene.item_count as usize,
                            );
                            iletki_wgpu::profiling::count("scene.retained_hit", 1);
                            iletki_wgpu::profiling::count(
                                "scene.scanned_slots",
                                gpu_scene.extra_chunks.len(),
                            );
                            iletki_wgpu::profiling::count(
                                "scene.opaque_draws",
                                gpu_scene
                                    .buckets
                                    .len()
                                    .saturating_add(view.storage_faces.count())
                                    .saturating_add(view.storage_edges.count()),
                            );
                            iletki_wgpu::profiling::count("scene.face_page_writes", 0);
                            iletki_wgpu::profiling::count("scene.edge_page_writes", 0);
                            iletki_wgpu::profiling::count(
                                "scene.uploaded_chunks",
                                self.uploaded_chunks,
                            );
                            iletki_wgpu::profiling::count(
                                "scene.uploaded_flat_chunks",
                                self.uploaded_flat_chunks,
                            );
                            iletki_wgpu::profiling::count(
                                "scene.uploaded_smooth_indexed_chunks",
                                self.uploaded_smooth_indexed_chunks,
                            );
                            iletki_wgpu::profiling::count(
                                "scene.uploaded_input_vertices",
                                self.uploaded_input_vertices,
                            );
                            iletki_wgpu::profiling::count(
                                "scene.uploaded_stored_vertices",
                                self.uploaded_stored_vertices,
                            );
                            iletki_wgpu::profiling::count(
                                "scene.uploaded_preprocess_us",
                                self.uploaded_preprocess_us,
                            );
                        }
                        return pending;
                    }
                }
                view.gpu_cull_active = false;
                #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
                let visibility = iletki_wgpu::profiling::scope("scene.visibility");
                let env = self.scene_envelope.get_or_insert_with(|| {
                    let mut bounds = SectionGroup::default();
                    let mut all_trusted = true;
                    let mut all_small_packed = true;
                    let mut has_packed = false;
                    for chunk in self.chunks.iter().flatten() {
                        if chunk.vertex_count == 0 && chunk.edge_count == 0 {
                            continue;
                        }
                        if chunk.section_bounds_trusted {
                            bounds.include(chunk.origin, chunk.local_bounds);
                        } else {
                            bounds.mark_untrusted();
                            all_trusted = false;
                        }
                        all_small_packed &= chunk.small_packed;
                        has_packed |= chunk.vertex_buf.as_ref().is_some_and(|v| !v.is_direct())
                            || chunk.edge_buf.as_ref().is_some_and(|e| !e.is_direct());
                    }
                    SceneEnvelope {
                        bounds,
                        all_trusted,
                        all_small_packed,
                        has_packed,
                    }
                });
                let retain_eligible = env.has_packed
                    && env.all_trusted
                    && self.hidden_chunks.saturating_mul(16) <= self.chunks.len();
                let frustum_all_inside =
                    retain_eligible && env.bounds.inside_frustum(&frustum, cam_pos);
                let fast_retain = frustum_all_inside
                    && (!cull_sections
                        || section_volume.as_ref().is_some_and(|volume| {
                            env.bounds.section_relation(volume, cam_pos) == SectionRelation::Inside
                        }));
                let all_small_packed = env.all_small_packed;

                self.section_relations.clear();
                let mut crossing_groups = 0usize;
                let mut outside_groups = 0usize;
                let mut crossing_frustum_groups = 0usize;
                if !fast_retain && group_visibility {
                    self.section_groups
                        .resize_with(self.chunks.len().div_ceil(SECTION_GROUP_SIZE), || None);
                    for (cached, chunks) in self
                        .section_groups
                        .iter_mut()
                        .zip(self.chunks.chunks(SECTION_GROUP_SIZE))
                    {
                        let group = cached.get_or_insert_with(|| {
                            let mut bounds = SectionGroup::default();
                            for chunk in chunks.iter().flatten() {
                                if chunk.section_bounds_trusted {
                                    bounds.include(chunk.origin, chunk.local_bounds);
                                } else if chunk.vertex_count != 0 || chunk.edge_count != 0 {
                                    bounds.mark_untrusted();
                                }
                            }
                            bounds
                        });
                        let relation = section_volume
                            .as_ref()
                            .map_or(SectionRelation::Crossing, |volume| {
                                group.section_relation(volume, cam_pos)
                            });
                        let outside_frustum = group.outside_frustum(&frustum, cam_pos);
                        let outside = outside_frustum
                            || (relation == SectionRelation::Outside && group.all_trusted());
                        let inside_frustum = !outside && group.inside_frustum(&frustum, cam_pos);
                        outside_groups += usize::from(outside);
                        crossing_frustum_groups += usize::from(!outside && !inside_frustum);
                        crossing_groups +=
                            usize::from(relation == SectionRelation::Crossing && !outside);
                        self.section_relations
                            .push((relation, inside_frustum, outside));
                    }
                }
                let group_count = self.section_relations.len();
                let bridge_sections = bridge_sections_enabled
                    && group_count != 0
                    && crossing_groups >= group_count - group_count / 4;
                // Keep the retained full-scene draw list when:
                // 1. All geometry fits inside the frustum and (if clipped) inside or bridged; or
                // 2. Every chunk is tiny and >=75% of groups are visible (avoiding thousands of
                //    fragmented instanced draw splits to frustum-cull a few percent of boxes); or
                // 3. Unclipped view where 0 groups are outside and >=94% are wholly inside.
                let retain_small = retain_eligible
                    && all_small_packed
                    && group_count != 0
                    && outside_groups.saturating_mul(4) < group_count
                    && (!cull_sections || bridge_sections);
                let retain_near_full = retain_eligible
                    && !cull_sections
                    && group_count != 0
                    && outside_groups == 0
                    && crossing_frustum_groups.saturating_mul(16) <= group_count;
                let can_retain_full = fast_retain || retain_small || retain_near_full;

                if can_retain_full {
                    let rebuilt = !view.retained_full_scene
                        || view.retained_geometry_rev != self.geometry_revision;
                    if rebuilt {
                        view.draws.clear();
                        view.direct_uniforms.clear();
                        view.storage_faces.clear();
                        view.storage_edges.clear();
                        let mut has_edges = false;
                        for (chunk_index, entry) in self.chunks.iter().enumerate() {
                            let Some(chunk) = entry else { continue };
                            if chunk.vertex_count == 0 && chunk.edge_count == 0 {
                                continue;
                            }
                            let face_direct = chunk
                                .vertex_buf
                                .as_ref()
                                .is_some_and(VertexRange::is_direct);
                            let edge_direct =
                                chunk.edge_buf.as_ref().is_some_and(VertexRange::is_direct);
                            if face_direct || edge_direct {
                                if !chunk.visible {
                                    continue;
                                }
                                let draw_index = view.draws.len();
                                let uniforms = ObjectUniforms {
                                    offset_rel: (chunk.origin - cam_pos).to_f32(),
                                    scale: 1.0,
                                };
                                view.draws.push(PreparedObject {
                                    chunk: chunk_index,
                                    section_culled: false,
                                    bridge_section_gap: false,
                                    direct_slot: Some(view.direct_uniforms.len()),
                                    uniforms,
                                });
                                view.direct_uniforms.push(uniforms);
                                if let Some(geometry) = &chunk.vertex_buf {
                                    view.storage_faces
                                        .push_color(geometry, draw_index, false, false, true);
                                }
                                if let Some(geometry) = &chunk.edge_buf {
                                    has_edges = true;
                                    view.storage_edges
                                        .push_color(geometry, draw_index, false, false, true);
                                }
                            } else {
                                if let Some(geometry) = &chunk.vertex_buf {
                                    view.storage_faces
                                        .push_color(geometry, 0, false, false, true);
                                }
                                if let Some(geometry) = &chunk.edge_buf {
                                    has_edges = true;
                                    view.storage_edges
                                        .push_color(geometry, 0, false, false, true);
                                }
                            }
                        }
                        view.retained_geometry_rev = self.geometry_revision;
                        view.retained_full_scene = true;
                        view.retained_has_edges = has_edges;
                        view.has_edges = has_edges;
                        if !view.direct_uniforms.is_empty() {
                            view.upload(ctx.device, ctx.queue, &p.object_layout);
                        }
                    } else {
                        view.has_edges = view.retained_has_edges;
                        if !view.draws.is_empty() {
                            view.direct_uniforms.clear();
                            for object in &mut view.draws {
                                if let Some(Some(chunk)) = self.chunks.get(object.chunk) {
                                    object.uniforms.offset_rel = (chunk.origin - cam_pos).to_f32();
                                    object.direct_slot = Some(view.direct_uniforms.len());
                                    view.direct_uniforms.push(object.uniforms);
                                }
                            }
                            if !view.direct_uniforms.is_empty() {
                                view.upload(ctx.device, ctx.queue, &p.object_layout);
                            }
                        }
                    }
                    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
                    drop(visibility);
                    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
                    if profiling {
                        let draws = view
                            .storage_faces
                            .count()
                            .saturating_add(view.storage_edges.count());
                        iletki_wgpu::profiling::count("scene.retained_hit", usize::from(!rebuilt));
                        iletki_wgpu::profiling::count(
                            "scene.scanned_slots",
                            if rebuilt {
                                self.chunks.len()
                            } else {
                                group_count
                            },
                        );
                        iletki_wgpu::profiling::count("scene.opaque_draws", draws);
                        iletki_wgpu::profiling::count("scene.face_page_writes", 0);
                        iletki_wgpu::profiling::count("scene.edge_page_writes", 0);
                        iletki_wgpu::profiling::count(
                            "scene.uploaded_chunks",
                            self.uploaded_chunks,
                        );
                        iletki_wgpu::profiling::count(
                            "scene.uploaded_flat_chunks",
                            self.uploaded_flat_chunks,
                        );
                        iletki_wgpu::profiling::count(
                            "scene.uploaded_smooth_indexed_chunks",
                            self.uploaded_smooth_indexed_chunks,
                        );
                        iletki_wgpu::profiling::count(
                            "scene.uploaded_input_vertices",
                            self.uploaded_input_vertices,
                        );
                        iletki_wgpu::profiling::count(
                            "scene.uploaded_stored_vertices",
                            self.uploaded_stored_vertices,
                        );
                        iletki_wgpu::profiling::count(
                            "scene.uploaded_preprocess_us",
                            self.uploaded_preprocess_us,
                        );
                    }
                    return pending;
                }

                // Culled storage path: single pass over visible groups directly into StorageDraws.
                view.retained_full_scene = false;
                view.draws.clear();
                view.direct_uniforms.clear();
                view.storage_faces.clear();
                view.storage_edges.clear();
                view.has_edges = false;
                let [width, height] = ctx.active_px;
                let viewport_volume =
                    PickVolume::new(projection, ctx.active_px, [0, 0, width, height]);
                #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
                let (
                    mut culled_chunks,
                    mut culled_triangles,
                    mut section_tests,
                    mut frustum_tests,
                    mut opaque_triangles,
                    mut deferred_chunks,
                    mut omitted_face_vertices,
                    mut scanned_slots,
                ) = (
                    0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize,
                );

                for (group_index, chunks) in self.chunks.chunks(SECTION_GROUP_SIZE).enumerate() {
                    let (relation, inside_frustum, skip_group) = self
                        .section_relations
                        .get(group_index)
                        .copied()
                        .unwrap_or((SectionRelation::Crossing, false, false));
                    if skip_group {
                        continue;
                    }
                    let bridge_group = bridge_sections && relation == SectionRelation::Crossing;
                    let defer_small = bridge_group && inside_frustum;
                    for (slot, chunk) in chunks.iter().enumerate() {
                        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
                        {
                            scanned_slots += 1;
                        }
                        let Some(chunk) = chunk else { continue };
                        if !chunk.visible || (chunk.vertex_count == 0 && chunk.edge_count == 0) {
                            continue;
                        }
                        if !inside_frustum || !chunk.section_bounds_trusted {
                            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
                            if profiling {
                                frustum_tests += 1;
                            }
                            let bounds = Bounds3::from_corners(
                                chunk.origin + chunk.local_bounds.min,
                                chunk.origin + chunk.local_bounds.max,
                            );
                            if !bounds.is_empty()
                                && !frustum.intersects_bounds_world(bounds, cam_pos)
                            {
                                continue;
                            }
                        }
                        let mut cached_offset: Option<[f32; 3]> = None;
                        let mut offset_rel = || {
                            *cached_offset.get_or_insert_with(|| (chunk.origin - cam_pos).to_f32())
                        };
                        let section_culled = chunk.section_bounds_trusted
                            && match relation {
                                SectionRelation::Outside => true,
                                SectionRelation::Inside => false,
                                SectionRelation::Crossing if defer_small && chunk.small_packed => {
                                    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
                                    if profiling {
                                        deferred_chunks += 1;
                                    }
                                    false
                                }
                                SectionRelation::Crossing => {
                                    section_volume.as_ref().is_some_and(|volume| {
                                        #[cfg(all(
                                            feature = "profiling",
                                            not(target_arch = "wasm32")
                                        ))]
                                        if profiling {
                                            section_tests += 1;
                                        }
                                        volume.excludes(chunk.local_bounds, offset_rel())
                                    })
                                }
                            };
                        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
                        if profiling {
                            let triangles = chunk.vertex_count as usize / 3;
                            if section_culled {
                                culled_chunks += 1;
                                culled_triangles += triangles;
                            } else {
                                opaque_triangles += triangles;
                            }
                        }
                        view.has_edges |= chunk.edge_count != 0 && !section_culled;
                        let face_direct = chunk
                            .vertex_buf
                            .as_ref()
                            .is_some_and(VertexRange::is_direct);
                        let edge_direct =
                            chunk.edge_buf.as_ref().is_some_and(VertexRange::is_direct);
                        let chunk_slot = group_index * SECTION_GROUP_SIZE + slot;
                        let draw_index = if (face_direct
                            && (!section_culled || self.selected_slots.contains(chunk_slot)))
                            || (edge_direct && !section_culled)
                        {
                            let idx = view.draws.len();
                            let uniforms = ObjectUniforms {
                                offset_rel: offset_rel(),
                                scale: 1.0,
                            };
                            view.draws.push(PreparedObject {
                                chunk: chunk_slot,
                                section_culled,
                                bridge_section_gap: bridge_group,
                                direct_slot: Some(view.direct_uniforms.len()),
                                uniforms,
                            });
                            view.direct_uniforms.push(uniforms);
                            idx
                        } else {
                            0
                        };
                        if let Some(geometry) = &chunk.vertex_buf {
                            if !section_culled
                                && !face_direct
                                && let Some(ranges) = &chunk.face_ranges
                            {
                                let section_ref = section_volume.as_ref().filter(|_| {
                                    relation != SectionRelation::Inside
                                        || !chunk.section_bounds_trusted
                                });
                                let volume = (!inside_frustum || !chunk.section_bounds_trusted)
                                    .then_some(&viewport_volume);
                                if volume.is_some() || section_ref.is_some() {
                                    let offset = offset_rel();
                                    let _omitted = ranges.visit_visible(
                                        |bounds| {
                                            volume.map_or_else(
                                                || {
                                                    section_ref
                                                        .is_none_or(|s| !s.excludes(bounds, offset))
                                                },
                                                |volume| {
                                                    volume.intersects(bounds, offset, section_ref)
                                                },
                                            )
                                        },
                                        |range| {
                                            if let Some(sub) = geometry.packed_subrange(range) {
                                                view.storage_faces.push(&sub, 0);
                                            }
                                        },
                                    );
                                    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
                                    {
                                        omitted_face_vertices += _omitted as usize;
                                    }
                                } else {
                                    view.storage_faces.push_color(
                                        geometry,
                                        draw_index,
                                        false,
                                        bridge_group,
                                        true,
                                    );
                                }
                            } else {
                                view.storage_faces.push_color(
                                    geometry,
                                    draw_index,
                                    section_culled,
                                    bridge_group,
                                    true,
                                );
                            }
                        }
                        if let Some(geometry) = &chunk.edge_buf {
                            view.storage_edges.push_color(
                                geometry,
                                draw_index,
                                section_culled,
                                bridge_group,
                                true,
                            );
                        }
                    }
                }
                #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
                drop(visibility);
                if !view.direct_uniforms.is_empty() {
                    view.upload(ctx.device, ctx.queue, &p.object_layout);
                }
                #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
                if profiling {
                    let bridged_triangles = view.storage_faces.section_gap_vertices() / 3;
                    let omitted_ranges = omitted_face_vertices / 3;
                    let draws = view
                        .storage_faces
                        .count()
                        .saturating_add(view.storage_edges.count());
                    iletki_wgpu::profiling::count("scene.retained_hit", 0);
                    iletki_wgpu::profiling::count("scene.range_culled_triangles", omitted_ranges);
                    iletki_wgpu::profiling::count(
                        "scene.section_bridged_triangles",
                        bridged_triangles,
                    );
                    iletki_wgpu::profiling::count(
                        "scene.section_bridged_edge_vertices",
                        view.storage_edges.section_gap_vertices(),
                    );
                    iletki_wgpu::profiling::count(
                        "scene.section_crossing_groups",
                        if cull_sections { crossing_groups } else { 0 },
                    );
                    iletki_wgpu::profiling::count("scene.section_chunk_tests", section_tests);
                    iletki_wgpu::profiling::count("scene.section_deferred_chunks", deferred_chunks);
                    iletki_wgpu::profiling::count("scene.frustum_chunk_tests", frustum_tests);
                    iletki_wgpu::profiling::count(
                        "scene.opaque_triangles",
                        (opaque_triangles + bridged_triangles).saturating_sub(omitted_ranges),
                    );
                    iletki_wgpu::profiling::count("scene.scanned_slots", scanned_slots);
                    iletki_wgpu::profiling::count("scene.section_culled_chunks", culled_chunks);
                    iletki_wgpu::profiling::count(
                        "scene.section_culled_triangles",
                        culled_triangles.saturating_sub(bridged_triangles),
                    );
                    iletki_wgpu::profiling::count("scene.face_page_writes", 0);
                    iletki_wgpu::profiling::count("scene.edge_page_writes", 0);
                    iletki_wgpu::profiling::count("scene.opaque_draws", draws);
                    iletki_wgpu::profiling::count("scene.uploaded_chunks", self.uploaded_chunks);
                    iletki_wgpu::profiling::count(
                        "scene.uploaded_flat_chunks",
                        self.uploaded_flat_chunks,
                    );
                    iletki_wgpu::profiling::count(
                        "scene.uploaded_smooth_indexed_chunks",
                        self.uploaded_smooth_indexed_chunks,
                    );
                    iletki_wgpu::profiling::count(
                        "scene.uploaded_input_vertices",
                        self.uploaded_input_vertices,
                    );
                    iletki_wgpu::profiling::count(
                        "scene.uploaded_stored_vertices",
                        self.uploaded_stored_vertices,
                    );
                    iletki_wgpu::profiling::count(
                        "scene.uploaded_preprocess_us",
                        self.uploaded_preprocess_us,
                    );
                }
                return pending;
            }

            // Picking path with storage transforms.
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            let visibility = iletki_wgpu::profiling::scope("scene.visibility");
            view.retained_full_scene = false;
            view.gpu_cull_active = false;
            view.draws.clear();
            view.direct_uniforms.clear();
            view.storage_faces.clear();
            view.storage_edges.clear();
            view.has_edges = false;
            let storage_pick = p.storage.as_ref().is_some_and(|s| s.pick.is_some());
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            let storage_pick = storage_pick && self.profile_batch_picking;
            self.section_relations.clear();
            if group_visibility && section_volume.is_some() {
                self.section_groups
                    .resize_with(self.chunks.len().div_ceil(SECTION_GROUP_SIZE), || None);
                for (cached, chunks) in self
                    .section_groups
                    .iter_mut()
                    .zip(self.chunks.chunks(SECTION_GROUP_SIZE))
                {
                    let group = cached.get_or_insert_with(|| {
                        let mut bounds = SectionGroup::default();
                        for chunk in chunks.iter().flatten() {
                            if chunk.section_bounds_trusted {
                                bounds.include(chunk.origin, chunk.local_bounds);
                            } else if chunk.vertex_count != 0 || chunk.edge_count != 0 {
                                bounds.mark_untrusted();
                            }
                        }
                        bounds
                    });
                    let relation = section_volume
                        .as_ref()
                        .map_or(SectionRelation::Crossing, |volume| {
                            group.section_relation(volume, cam_pos)
                        });
                    let skip = relation == SectionRelation::Outside && group.all_trusted();
                    self.section_relations.push((relation, false, skip));
                }
            }
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            let mut omitted_face_vertices = 0usize;
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            let mut candidates = 0usize;
            for (group_index, chunks) in self.chunks.chunks(SECTION_GROUP_SIZE).enumerate() {
                let (relation, _, skip_group) = self
                    .section_relations
                    .get(group_index)
                    .copied()
                    .unwrap_or((SectionRelation::Crossing, false, false));
                if skip_group {
                    continue;
                }
                let pick_section = section_volume
                    .as_ref()
                    .filter(|_| relation != SectionRelation::Inside);
                for (slot, chunk) in chunks.iter().enumerate() {
                    let Some(chunk) = chunk else { continue };
                    if !chunk.visible || !chunk.has_picks || chunk.vertex_count == 0 {
                        continue;
                    }
                    let offset_rel = (chunk.origin - cam_pos).to_f32();
                    if let Some(volume) = &pick_volume
                        && ((relation == SectionRelation::Outside && chunk.section_bounds_trusted)
                            || (chunk.pick_bounds_trusted
                                && !volume.intersects(
                                    chunk.local_bounds,
                                    offset_rel,
                                    pick_section,
                                )))
                    {
                        continue;
                    }
                    let Some(geometry) = &chunk.vertex_buf else {
                        continue;
                    };
                    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
                    {
                        candidates += 1;
                    }
                    if storage_pick && !geometry.is_direct() {
                        if let Some(ranges) = &chunk.face_ranges {
                            let section_ref = section_volume.as_ref().filter(|_| {
                                relation != SectionRelation::Inside || !chunk.section_bounds_trusted
                            });
                            if pick_volume.is_some() || section_ref.is_some() {
                                let _omitted = ranges.visit_visible(
                                    |bounds| {
                                        pick_volume.as_ref().map_or_else(
                                            || {
                                                section_ref
                                                    .is_none_or(|s| !s.excludes(bounds, offset_rel))
                                            },
                                            |volume| {
                                                volume.intersects(bounds, offset_rel, section_ref)
                                            },
                                        )
                                    },
                                    |range| {
                                        if let Some(sub) = geometry.packed_subrange(range) {
                                            view.storage_faces.push(&sub, 0);
                                        }
                                    },
                                );
                                #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
                                {
                                    omitted_face_vertices += _omitted as usize;
                                }
                            } else {
                                view.storage_faces.push(geometry, 0);
                            }
                        } else {
                            view.storage_faces.push(geometry, 0);
                        }
                    } else {
                        let draw_index = view.draws.len();
                        let uniforms = ObjectUniforms {
                            offset_rel,
                            scale: 1.0,
                        };
                        view.draws.push(PreparedObject {
                            chunk: group_index * SECTION_GROUP_SIZE + slot,
                            section_culled: false,
                            bridge_section_gap: false,
                            direct_slot: Some(view.direct_uniforms.len()),
                            uniforms,
                        });
                        view.direct_uniforms.push(uniforms);
                        if storage_pick {
                            view.storage_faces.push(geometry, draw_index);
                        }
                    }
                }
            }
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            drop(visibility);
            if !view.direct_uniforms.is_empty() {
                view.upload(ctx.device, ctx.queue, &p.object_layout);
            }
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            if profiling {
                iletki_wgpu::profiling::count(
                    "scene.range_culled_triangles",
                    omitted_face_vertices / 3,
                );
                iletki_wgpu::profiling::count("scene.candidates", candidates);
                let draws = if storage_pick && view.storage_faces.has_packed() {
                    view.storage_faces.count()
                } else {
                    view.draws.len()
                };
                iletki_wgpu::profiling::count("scene.pick_draws", draws);
            }
            return pending;
        }

        // Fallback path (OpenGL / WebGL2 / storage transforms disabled).
        view.retained_full_scene = false;
        view.gpu_cull_active = false;
        view.storage_faces.clear();
        view.storage_edges.clear();
        view.draws.clear();
        view.has_edges = false;
        view.faces.clear();
        view.edges.clear();
        view.mask.clear();
        view.direct_uniforms.clear();

        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let visibility = iletki_wgpu::profiling::scope("scene.visibility");
        let bridge_sections = bridge_sections_enabled;
        self.section_relations.clear();
        let mut crossing_groups = 0;
        if group_visibility && (picking.is_none() || section_volume.is_some()) {
            self.section_groups
                .resize_with(self.chunks.len().div_ceil(SECTION_GROUP_SIZE), || None);
            for (cached, chunks) in self
                .section_groups
                .iter_mut()
                .zip(self.chunks.chunks(SECTION_GROUP_SIZE))
            {
                let group = cached.get_or_insert_with(|| {
                    let mut bounds = SectionGroup::default();
                    for chunk in chunks.iter().flatten() {
                        if chunk.section_bounds_trusted {
                            bounds.include(chunk.origin, chunk.local_bounds);
                        } else if chunk.vertex_count != 0 || chunk.edge_count != 0 {
                            bounds.mark_untrusted();
                        }
                    }
                    bounds
                });
                let relation = section_volume
                    .as_ref()
                    .map_or(SectionRelation::Crossing, |volume| {
                        group.section_relation(volume, cam_pos)
                    });
                let outside = (picking.is_none() && group.outside_frustum(&frustum, cam_pos))
                    || (relation == SectionRelation::Outside && group.all_trusted());
                let inside_frustum =
                    picking.is_none() && !outside && group.inside_frustum(&frustum, cam_pos);
                crossing_groups +=
                    usize::from(relation == SectionRelation::Crossing && inside_frustum);
                self.section_relations
                    .push((relation, inside_frustum, outside));
            }
        }
        // Use batching across gaps when at least three quarters of the groups intersect a
        // section inside the frustum. Localized scenes usually expose long runs of clipped
        // groups, where the existing CPU rejection avoids more useful vertex work.
        let group_count = self.section_relations.len();
        let bridge_sections =
            bridge_sections && group_count != 0 && crossing_groups >= group_count - group_count / 4;
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let (mut culled_chunks, mut culled_triangles, mut section_tests, mut frustum_tests) =
            (0, 0, 0, 0);
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let mut opaque_triangles = 0;
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let mut deferred_chunks = 0;
        for (group_index, chunks) in self.chunks.chunks(SECTION_GROUP_SIZE).enumerate() {
            let (relation, inside_frustum, skip_group) = self
                .section_relations
                .get(group_index)
                .copied()
                .unwrap_or((SectionRelation::Crossing, false, false));
            if skip_group {
                continue;
            }
            let bridge_group = bridge_sections && relation == SectionRelation::Crossing;
            let defer_small = bridge_group && inside_frustum;
            let pick_section = section_volume
                .as_ref()
                .filter(|_| relation != SectionRelation::Inside);
            for (slot, chunk) in chunks.iter().enumerate() {
                let Some(chunk) = chunk else { continue };
                if !chunk.visible || (chunk.vertex_count == 0 && chunk.edge_count == 0) {
                    continue;
                }
                let offset_rel = (chunk.origin - cam_pos).to_f32();
                if let Some(volume) = &pick_volume {
                    if !chunk.has_picks
                        || (relation == SectionRelation::Outside && chunk.section_bounds_trusted)
                        || (chunk.pick_bounds_trusted
                            && !volume.intersects(chunk.local_bounds, offset_rel, pick_section))
                    {
                        continue;
                    }
                } else if !inside_frustum || !chunk.section_bounds_trusted {
                    #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
                    if profiling {
                        frustum_tests += 1;
                    }
                    let bounds = Bounds3::from_corners(
                        chunk.origin + chunk.local_bounds.min,
                        chunk.origin + chunk.local_bounds.max,
                    );
                    if !bounds.is_empty() && !frustum.intersects_bounds_world(bounds, cam_pos) {
                        continue;
                    }
                }
                let section_culled = picking.is_none()
                    && chunk.section_bounds_trusted
                    && match relation {
                        SectionRelation::Outside => true,
                        SectionRelation::Inside => false,
                        SectionRelation::Crossing if defer_small && chunk.small_packed => {
                            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
                            if profiling {
                                deferred_chunks += 1;
                            }
                            false
                        }
                        SectionRelation::Crossing => {
                            section_volume.as_ref().is_some_and(|volume| {
                                #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
                                if profiling {
                                    section_tests += 1;
                                }
                                volume.excludes(chunk.local_bounds, offset_rel)
                            })
                        }
                    };
                #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
                if profiling && picking.is_none() {
                    let triangles = chunk.vertex_count as usize / 3;
                    if section_culled {
                        culled_chunks += 1;
                        culled_triangles += triangles;
                    } else {
                        opaque_triangles += triangles;
                    }
                }
                view.has_edges |= chunk.edge_count != 0 && !section_culled;
                view.draws.push(PreparedObject {
                    chunk: group_index * SECTION_GROUP_SIZE + slot,
                    section_culled,
                    bridge_section_gap: bridge_group,
                    direct_slot: None,
                    uniforms: ObjectUniforms {
                        offset_rel,
                        scale: 1.0,
                    },
                });
            }
        }
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        drop(visibility);

        // Small visible subsets use direct draws. Each larger pass preserves chunk order
        // and stages aligned uniforms only for direct draws or a selected mask.
        let use_packed = pack_geometry && !ctx.selection_id_attachment() && view.draws.len() >= 128;
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let use_packed = use_packed && (picking.is_none() || self.profile_batch_picking);
        let packed_layout = p
            .packed
            .as_ref()
            .filter(|packed| use_packed && (picking.is_none() || packed.pick.is_some()))
            .map(|packed| &packed.layout);
        let packed_mask = picking.is_none()
            && self.selected_slots.len() >= 128
            && packed_layout.is_some_and(|layout| {
                Self::prepare_packed_mask(ctx, layout, &self.chunks, &self.selected_slots, view)
            });
        let [width, height] = ctx.active_px;
        let viewport_volume = PickVolume::new(projection, ctx.active_px, [0, 0, width, height]);
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        let mut omitted_face_vertices = 0usize;
        {
            #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
            let _profile = iletki_wgpu::profiling::scope("scene.geometry_batch");
            for (index, object) in view.draws.iter_mut().enumerate() {
                let Some(Some(chunk)) = self.chunks.get(object.chunk) else {
                    continue;
                };
                let mut direct = packed_layout.is_none()
                    || (picking.is_none()
                        && self.selected_slots.contains(object.chunk)
                        && !packed_mask);
                if packed_layout.is_some() {
                    if let Some(geometry) = &chunk.vertex_buf {
                        let custom_mat =
                            picking.is_none() && chunk.material_id != MaterialId::DEFAULT;
                        // A direct face can still be selected even when its opaque draw is culled.
                        direct |= geometry.is_direct() || custom_mat;
                        if custom_mat {
                            if !object.section_culled {
                                view.faces.push_direct(index);
                            }
                        } else if !object.section_culled
                            && let Some(ranges) = &chunk.face_ranges
                        {
                            let (relation, inside_frustum, _) = self
                                .section_relations
                                .get(object.chunk / SECTION_GROUP_SIZE)
                                .copied()
                                .unwrap_or((SectionRelation::Crossing, false, false));
                            let section = section_volume.as_ref().filter(|_| {
                                relation != SectionRelation::Inside || !chunk.section_bounds_trusted
                            });
                            let volume = pick_volume.as_ref().or_else(|| {
                                (!inside_frustum || !chunk.section_bounds_trusted)
                                    .then_some(&viewport_volume)
                            });
                            let offset = object.uniforms.offset_rel;
                            if volume.is_some() || section.is_some() {
                                let _omitted = ranges.visit_visible(
                                    |bounds| {
                                        volume.map_or_else(
                                            || section.is_none_or(|s| !s.excludes(bounds, offset)),
                                            |volume| volume.intersects(bounds, offset, section),
                                        )
                                    },
                                    |range| {
                                        if let Some(range) = geometry.packed_subrange(range) {
                                            view.faces.push(&range, object.uniforms, index);
                                        }
                                    },
                                );
                                #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
                                {
                                    omitted_face_vertices += _omitted as usize;
                                }
                            } else {
                                view.faces.push(geometry, object.uniforms, index);
                            }
                        } else {
                            view.faces.push_color(
                                geometry,
                                object.uniforms,
                                index,
                                object.section_culled,
                                object.bridge_section_gap,
                                picking.is_none(),
                            );
                        }
                    }
                    if picking.is_none()
                        && let Some(geometry) = &chunk.edge_buf
                    {
                        direct |= !object.section_culled && geometry.is_direct();
                        view.edges.push_color(
                            geometry,
                            object.uniforms,
                            index,
                            object.section_culled,
                            object.bridge_section_gap,
                            true,
                        );
                    }
                }
                if direct {
                    object.direct_slot = Some(view.direct_uniforms.len());
                    view.direct_uniforms.push(object.uniforms);
                }
            }
            if let Some(layout) = packed_layout {
                view.faces.upload(ctx.device, layout, ctx.queue);
                view.edges.upload(ctx.device, layout, ctx.queue);
            }
        }
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        if profiling {
            let bridged_triangles = view.faces.section_gap_vertices() / 3;
            let omitted_ranges = omitted_face_vertices / 3;
            iletki_wgpu::profiling::count("scene.range_culled_triangles", omitted_ranges);
            iletki_wgpu::profiling::count("scene.section_bridged_triangles", bridged_triangles);
            iletki_wgpu::profiling::count(
                "scene.section_bridged_edge_vertices",
                view.edges.section_gap_vertices(),
            );
            iletki_wgpu::profiling::count(
                "scene.section_crossing_groups",
                if cull_sections { crossing_groups } else { 0 },
            );
            iletki_wgpu::profiling::count("scene.section_chunk_tests", section_tests);
            iletki_wgpu::profiling::count("scene.section_deferred_chunks", deferred_chunks);
            iletki_wgpu::profiling::count("scene.frustum_chunk_tests", frustum_tests);
            iletki_wgpu::profiling::count(
                "scene.opaque_triangles",
                (opaque_triangles + bridged_triangles).saturating_sub(omitted_ranges),
            );
            iletki_wgpu::profiling::count("scene.scanned_slots", self.chunks.len());
            iletki_wgpu::profiling::count("scene.candidates", view.draws.len());
            iletki_wgpu::profiling::count("scene.section_culled_chunks", culled_chunks);
            iletki_wgpu::profiling::count(
                "scene.section_culled_triangles",
                culled_triangles.saturating_sub(bridged_triangles),
            );
            let pass = if picking.is_some() {
                "scene.pick_draws"
            } else {
                "scene.opaque_draws"
            };
            let draws = if packed_layout.is_some() {
                let (faces, face_pages) = view.faces.counts();
                let (edges, edge_pages) = view.edges.counts();
                iletki_wgpu::profiling::count("scene.face_page_writes", face_pages);
                iletki_wgpu::profiling::count("scene.edge_page_writes", edge_pages);
                faces.saturating_add(edges)
            } else {
                view.draws
                    .iter()
                    .filter(|object| !object.section_culled)
                    .filter_map(|object| self.chunks.get(object.chunk)?.as_ref())
                    .map(|chunk| {
                        usize::from(chunk.vertex_buf.is_some())
                            + usize::from(picking.is_none() && chunk.edge_buf.is_some())
                    })
                    .sum()
            };
            iletki_wgpu::profiling::count("scene.uploaded_chunks", self.uploaded_chunks);
            iletki_wgpu::profiling::count("scene.uploaded_flat_chunks", self.uploaded_flat_chunks);
            iletki_wgpu::profiling::count(
                "scene.uploaded_smooth_indexed_chunks",
                self.uploaded_smooth_indexed_chunks,
            );
            iletki_wgpu::profiling::count(
                "scene.uploaded_input_vertices",
                self.uploaded_input_vertices,
            );
            iletki_wgpu::profiling::count(
                "scene.uploaded_stored_vertices",
                self.uploaded_stored_vertices,
            );
            iletki_wgpu::profiling::count(
                "scene.uploaded_preprocess_us",
                self.uploaded_preprocess_us,
            );
            iletki_wgpu::profiling::count(pass, draws);
        }
        view.upload(ctx.device, ctx.queue, &p.object_layout);
        pending
    }

    fn supports_multisampled_outline_mask(&self) -> bool {
        self.pipelines
            .as_ref()
            .is_some_and(|p| p.multisampled_mask.is_some())
    }

    fn paint_multisampled_outline_mask(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        ctx: &PaintCtx<'_, W>,
    ) {
        let Some(p) = &self.pipelines else {
            return;
        };
        let mask = p
            .section
            .as_ref()
            .filter(|_| self.use_antialiased_section)
            .map(|section| &section.multisampled_mask)
            .or(p.multisampled_mask.as_ref());
        if let Some(mask) = mask {
            let packed = p.packed.as_ref().and_then(|packed| {
                packed
                    .section_mask
                    .as_ref()
                    .filter(|_| self.use_antialiased_section)
                    .or(packed.multisampled_mask.as_ref())
            });
            let storage = p.storage.as_ref().and_then(|storage| {
                storage
                    .section_mask
                    .as_ref()
                    .filter(|_| self.use_antialiased_section)
                    .or(storage.multisampled_mask.as_ref())
            });
            let repeated_storage = p.storage.as_ref().and_then(|storage| {
                let rep = storage.repeated.as_ref()?;
                rep.section_mask
                    .as_ref()
                    .filter(|_| self.use_antialiased_section)
                    .or(rep.multisampled_mask.as_ref())
            });
            self.paint_outline_mask(mask, packed, storage, repeated_storage, pass, ctx);
        }
    }

    fn outline_mask_rect(&self, ctx: &PaintCtx<'_, W>) -> Option<[u32; 4]> {
        self.views
            .iter()
            .find(|view| view.id == ctx.viewport_id && !view.picking)?
            .outline_bounds
            .rect()
    }

    fn supports_color_frame_selection(&self) -> bool {
        #[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
        {
            self.profile_storage_transforms
        }
        #[cfg(not(all(feature = "profiling", not(target_arch = "wasm32"))))]
        {
            true
        }
    }

    fn selection_slot_count(&self) -> u32 {
        u32::try_from(self.chunks.len()).unwrap_or(u32::MAX)
    }

    fn uses_gpu_culling(&self, viewport_id: ViewportId) -> bool {
        self.gpu_cull_pipelines.is_some()
            && self.gpu_cull_scene.is_some()
            && self
                .views
                .iter()
                .any(|view| view.id == viewport_id && !view.picking && view.gpu_cull_active)
    }

    fn encode_gpu_cull_phase1(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &GpuCullCtx<'_, W>,
    ) {
        if let (Some(pipelines), Some(scene)) = (&self.gpu_cull_pipelines, &self.gpu_cull_scene)
            && let Some(view) = self
                .views
                .iter()
                .find(|v| v.id == ctx.viewport_id && !v.picking && v.gpu_cull_active)
            && let Some(gpu_view) = &view.gpu_cull
        {
            gpu_view.encode_phase1(encoder, pipelines, scene);
        }
    }

    fn encode_gpu_cull_phase2(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        hiz_view: &wgpu::TextureView,
        _hiz_size: [u32; 2],
        _hiz_mip_count: u32,
        hiz_generation: u64,
        ctx: &GpuCullCtx<'_, W>,
    ) {
        if let (Some(pipelines), Some(scene)) = (&self.gpu_cull_pipelines, &self.gpu_cull_scene)
            && let Some(view) = self
                .views
                .iter_mut()
                .find(|v| v.id == ctx.viewport_id && !v.picking && v.gpu_cull_active)
            && let Some(gpu_view) = view.gpu_cull.as_mut()
        {
            gpu_view.encode_phase2(
                ctx.device,
                encoder,
                pipelines,
                scene,
                hiz_view,
                hiz_generation,
            );
        }
    }

    fn paint_phase2(
        &self,
        phase: DrawPhase,
        pass: &mut wgpu::RenderPass<'_>,
        ctx: &PaintCtx<'_, W>,
    ) {
        if phase != DrawPhase::Opaque {
            return;
        }
        let Some(p) = &self.pipelines else {
            return;
        };
        let Some(gpu_scene) = &self.gpu_cull_scene else {
            return;
        };
        let Some(view) = self
            .views
            .iter()
            .find(|v| v.id == ctx.viewport_id && !v.picking && v.gpu_cull_active)
        else {
            return;
        };
        let Some(gpu_view) = &view.gpu_cull else {
            return;
        };
        let has_clip_planes = !ctx.camera.clip_planes().is_empty();
        let aa = self.use_antialiased_section;
        let use_select = ctx.selection_id_attachment();
        let use_materials = self.materials.has_custom_materials();
        let storage_set = if use_materials {
            let mat_bg = view
                .storage_binding
                .as_ref()
                .and_then(StorageViewBinding::material_bind_group);
            if use_select {
                p.materials
                    .as_ref()
                    .and_then(|m| m.storage.as_ref())
                    .and_then(|st| st.select.as_ref())
                    .zip(mat_bg)
                    .map(|(pl, bg)| {
                        (
                            pl.select_color(has_clip_planes, aa),
                            pl.select_repeated(has_clip_planes, aa),
                            bg,
                        )
                    })
            } else {
                p.materials
                    .as_ref()
                    .and_then(|m| m.storage.as_ref())
                    .zip(mat_bg)
                    .map(|(pl, bg)| {
                        (
                            pl.select_color(has_clip_planes, aa),
                            pl.select_repeated(has_clip_planes, aa),
                            bg,
                        )
                    })
            }
        } else {
            let storage_bg = view
                .storage_binding
                .as_ref()
                .and_then(StorageViewBinding::bind_group);
            if use_select {
                p.select
                    .as_ref()
                    .and_then(|sel| sel.storage.as_ref())
                    .zip(storage_bg)
                    .map(|(pl, bg)| {
                        (
                            pl.select_color(has_clip_planes, aa),
                            pl.select_repeated(has_clip_planes, aa),
                            bg,
                        )
                    })
            } else {
                p.storage.as_ref().zip(storage_bg).map(|(pl, bg)| {
                    (
                        pl.select_color(has_clip_planes, aa),
                        pl.select_repeated(has_clip_planes, aa)
                            .map(|(c, f, _)| (c, f)),
                        bg,
                    )
                })
            }
        };
        let Some(((color, flat), rep, storage_bg)) = storage_set else {
            return;
        };
        let face_pipelines = PaintPipelines {
            smooth: &color.faces,
            flat,
            repeated_smooth: rep.map(|(c, _)| &c.faces),
            repeated_flat: rep.and_then(|(_, f)| f),
            repeated_slots: None,
        };
        let edge_pipelines = PaintPipelines {
            smooth: &color.edges,
            flat: None,
            repeated_smooth: rep.map(|(c, _)| &c.edges),
            repeated_flat: None,
            repeated_slots: None,
        };
        gpu_scene.paint_phase2(pass, gpu_view, storage_bg, &face_pipelines, &edge_pipelines);

        // Draw any extra direct/repeated edges (hybrid scenes).
        if view.has_edges && (view.storage_edges.has_packed() || !view.draws.is_empty()) {
            let edge_direct = if use_select {
                p.select.as_ref().map(|sel| {
                    if !has_clip_planes {
                        &sel.edge_unclipped
                    } else {
                        sel.section
                            .as_ref()
                            .filter(|_| aa)
                            .map_or(&sel.edge, |s| &s.edge)
                    }
                })
            } else if !has_clip_planes {
                Some(&p.edge_unclipped)
            } else {
                Some(
                    p.section
                        .as_ref()
                        .filter(|_| aa)
                        .map_or(&p.edge, |s| &s.edge),
                )
            };
            let draw_direct_edge = |pass: &mut wgpu::RenderPass<'_>, index: usize| {
                let Some(direct_pl) = edge_direct else {
                    return;
                };
                if let Some(object) = view.draws.get(index)
                    && let Some(Some(chunk)) = self.chunks.get(object.chunk)
                    && chunk.visible
                    && let Some(buffer) = &chunk.edge_buf
                    && buffer.is_direct()
                    && view.bind(pass, index, p.dynamic_uniforms)
                {
                    pass.set_pipeline(direct_pl);
                    buffer.draw_direct(pass, chunk.edge_count);
                }
            };
            if view.storage_edges.has_packed() {
                let edge_pipes = PaintPipelines {
                    smooth: &color.edges,
                    flat: None,
                    repeated_smooth: rep.map(|(c, _)| &c.edges),
                    repeated_flat: None,
                    repeated_slots: None,
                };
                view.storage_edges
                    .paint(pass, storage_bg, &edge_pipes, false, draw_direct_edge);
            } else {
                for index in 0..view.draws.len() {
                    draw_direct_edge(pass, index);
                }
            }
        }
    }

    fn paint(&self, phase: DrawPhase, pass: &mut wgpu::RenderPass<'_>, ctx: &PaintCtx<'_, W>) {
        let Some(p) = &self.pipelines else {
            return;
        };
        let Some(view) = self.views.iter().find(|view| {
            view.id == ctx.viewport_id && view.picking == (phase == DrawPhase::Picking)
        }) else {
            return;
        };

        let has_clip_planes = !ctx.camera.clip_planes().is_empty();
        let use_select = phase == DrawPhase::Opaque && ctx.selection_id_attachment();
        let (lit, lit_flat, edge) = if use_select {
            let Some(sel) = &p.select else {
                return;
            };
            if !has_clip_planes {
                (
                    &sel.lit_unclipped,
                    sel.lit_flat_unclipped.as_ref(),
                    &sel.edge_unclipped,
                )
            } else {
                sel.section
                    .as_ref()
                    .filter(|_| self.use_antialiased_section)
                    .map_or((&sel.lit, sel.lit_flat.as_ref(), &sel.edge), |section| {
                        (&section.lit, section.lit_flat.as_ref(), &section.edge)
                    })
            }
        } else if !has_clip_planes {
            (
                &p.lit_unclipped,
                p.lit_flat_unclipped.as_ref(),
                &p.edge_unclipped,
            )
        } else {
            p.section
                .as_ref()
                .filter(|_| self.use_antialiased_section)
                .map_or((&p.lit, p.lit_flat.as_ref(), &p.edge), |section| {
                    (&section.lit, section.lit_flat.as_ref(), &section.edge)
                })
        };

        match phase {
            DrawPhase::Opaque => {
                let aa = self.use_antialiased_section;
                let use_materials = self.materials.has_custom_materials();
                let direct_mat = use_materials
                    .then(|| {
                        let mats = p.materials.as_ref()?;
                        let set = if use_select {
                            mats.direct_select.as_ref()?
                        } else {
                            &mats.direct
                        };
                        Some(set.select(has_clip_planes, aa))
                    })
                    .flatten();
                let storage_set = if use_materials {
                    let mat_bg = view
                        .storage_binding
                        .as_ref()
                        .and_then(StorageViewBinding::material_bind_group);
                    if use_select {
                        p.materials
                            .as_ref()
                            .and_then(|m| m.storage.as_ref())
                            .and_then(|st| st.select.as_ref())
                            .zip(mat_bg)
                            .map(|(pl, bg)| {
                                (
                                    pl.select_color(has_clip_planes, aa),
                                    pl.select_repeated(has_clip_planes, aa),
                                    bg,
                                )
                            })
                    } else {
                        p.materials
                            .as_ref()
                            .and_then(|m| m.storage.as_ref())
                            .zip(mat_bg)
                            .map(|(pl, bg)| {
                                (
                                    pl.select_color(has_clip_planes, aa),
                                    pl.select_repeated(has_clip_planes, aa),
                                    bg,
                                )
                            })
                    }
                } else {
                    let storage_bg = view
                        .storage_binding
                        .as_ref()
                        .and_then(StorageViewBinding::bind_group);
                    if use_select {
                        p.select
                            .as_ref()
                            .and_then(|sel| sel.storage.as_ref())
                            .zip(storage_bg)
                            .map(|(pl, bg)| {
                                (
                                    pl.select_color(has_clip_planes, aa),
                                    pl.select_repeated(has_clip_planes, aa),
                                    bg,
                                )
                            })
                    } else {
                        p.storage.as_ref().zip(storage_bg).map(|(pl, bg)| {
                            (
                                pl.select_color(has_clip_planes, aa),
                                pl.select_repeated(has_clip_planes, aa)
                                    .map(|(c, f, _)| (c, f)),
                                bg,
                            )
                        })
                    }
                };
                if view.gpu_cull_active
                    && let (Some(gpu_scene), Some(gpu_view), Some(((color, flat), rep, storage_bg))) =
                        (&self.gpu_cull_scene, &view.gpu_cull, storage_set)
                {
                    let face_pipelines = PaintPipelines {
                        smooth: &color.faces,
                        flat,
                        repeated_smooth: rep.map(|(c, _)| &c.faces),
                        repeated_flat: rep.and_then(|(_, f)| f),
                        repeated_slots: None,
                    };
                    gpu_scene.paint_phase1(pass, gpu_view, storage_bg, &face_pipelines);
                    // Draw any direct/repeated face extras in Phase 1 (hybrid scenes).
                    if view.storage_faces.has_packed() || !view.draws.is_empty() {
                        let draw_direct_face = |pass: &mut wgpu::RenderPass<'_>, index: usize| {
                            if let Some(object) = view.draws.get(index)
                                && let Some(Some(chunk)) = self.chunks.get(object.chunk)
                                && chunk.visible
                                && let Some(buffer) = &chunk.vertex_buf
                                && buffer.is_direct()
                                && view.bind(pass, index, p.dynamic_uniforms)
                            {
                                let custom_mat = chunk.material_id != MaterialId::DEFAULT;
                                let (smooth_pl, flat_pl) = if custom_mat
                                    && let Some((ms, mf)) = direct_mat
                                    && self.materials.bind_direct(pass, chunk.material_id)
                                {
                                    (ms, mf)
                                } else {
                                    (lit, lit_flat)
                                };
                                let use_flat = buffer.is_flat() && flat_pl.is_some();
                                let pl = if use_flat {
                                    flat_pl.unwrap_or(smooth_pl)
                                } else {
                                    smooth_pl
                                };
                                pass.set_pipeline(pl);
                                if use_select {
                                    let slot_id = if chunk.has_picks {
                                        u32::try_from(object.chunk)
                                            .unwrap_or(u32::MAX - 1)
                                            .saturating_add(1)
                                    } else {
                                        0
                                    };
                                    buffer.draw_direct_instance(pass, chunk.vertex_count, slot_id);
                                } else {
                                    buffer.draw_direct(pass, chunk.vertex_count);
                                }
                            }
                        };
                        if view.storage_faces.has_packed() {
                            view.storage_faces.paint(
                                pass,
                                storage_bg,
                                &face_pipelines,
                                false,
                                draw_direct_face,
                            );
                        } else {
                            for index in 0..view.draws.len() {
                                draw_direct_face(pass, index);
                            }
                        }
                    }
                    return;
                }
                let packed = (!use_select)
                    .then(|| {
                        p.packed
                            .as_ref()
                            .map(|pl| pl.select_color(has_clip_planes, aa))
                    })
                    .flatten();
                let repeated = (!use_select)
                    .then(|| {
                        p.packed
                            .as_ref()
                            .and_then(|pl| pl.select_repeated(has_clip_planes, aa))
                    })
                    .flatten();
                let passes = [
                    (
                        lit,
                        lit_flat,
                        packed.map(|(c, _)| &c.faces),
                        packed.and_then(|(_, f)| f),
                        &view.faces,
                        &view.storage_faces,
                        false,
                    ),
                    (
                        edge,
                        None,
                        packed.map(|(c, _)| &c.edges),
                        None,
                        &view.edges,
                        &view.storage_edges,
                        true,
                    ),
                ];
                for (
                    direct_pipeline,
                    direct_flat,
                    packed_pipeline,
                    packed_flat,
                    draws,
                    storage_draws,
                    edges,
                ) in passes
                {
                    if edges && !view.has_edges {
                        continue;
                    }
                    let draw_direct =
                        |pass: &mut wgpu::RenderPass<'_>,
                         index: usize,
                         active_direct: &mut Option<(bool, bool)>| {
                            let Some(object) = view.draws.get(index) else {
                                return;
                            };
                            if object.section_culled {
                                return;
                            }
                            let Some(Some(chunk)) = self.chunks.get(object.chunk) else {
                                return;
                            };
                            if !chunk.visible {
                                return;
                            }
                            let (geometry, count) = if edges {
                                (&chunk.edge_buf, chunk.edge_count)
                            } else {
                                (&chunk.vertex_buf, chunk.vertex_count)
                            };
                            if let Some(buffer) = geometry
                                && view.bind(pass, index, p.dynamic_uniforms)
                            {
                                let use_mat = !edges
                                    && chunk.material_id != MaterialId::DEFAULT
                                    && direct_mat.is_some()
                                    && self.materials.bind_direct(pass, chunk.material_id);
                                let (base_pl, flat_pl) =
                                    if use_mat && let Some((ms, mf)) = direct_mat {
                                        (ms, mf)
                                    } else {
                                        (direct_pipeline, direct_flat)
                                    };
                                let use_flat = !edges && buffer.is_flat() && flat_pl.is_some();
                                if *active_direct != Some((use_flat, use_mat)) {
                                    let pl = if use_flat {
                                        flat_pl.unwrap_or(base_pl)
                                    } else {
                                        base_pl
                                    };
                                    pass.set_pipeline(pl);
                                    *active_direct = Some((use_flat, use_mat));
                                }
                                if use_select && !edges {
                                    let slot_id = if chunk.has_picks {
                                        u32::try_from(object.chunk)
                                            .unwrap_or(u32::MAX - 1)
                                            .saturating_add(1)
                                    } else {
                                        0
                                    };
                                    buffer.draw_direct_instance(pass, count, slot_id);
                                } else {
                                    buffer.draw_direct(pass, count);
                                }
                            }
                        };
                    if let Some(((color, flat), rep, storage_bg)) = storage_set
                        && storage_draws.has_packed()
                    {
                        let paint_pipelines = PaintPipelines {
                            smooth: if edges { &color.edges } else { &color.faces },
                            flat: if edges { None } else { flat },
                            repeated_smooth: rep.map(
                                |(c, _)| {
                                    if edges { &c.edges } else { &c.faces }
                                },
                            ),
                            repeated_flat: if edges {
                                None
                            } else {
                                rep.and_then(|(_, f)| f)
                            },
                            repeated_slots: None,
                        };
                        storage_draws.paint(
                            pass,
                            storage_bg,
                            &paint_pipelines,
                            false,
                            |pass, index| {
                                let mut active_direct = None;
                                draw_direct(pass, index, &mut active_direct);
                            },
                        );
                    } else if let Some(pipeline) = packed_pipeline
                        && draws.has_packed()
                    {
                        let paint_pipelines = PaintPipelines {
                            smooth: pipeline,
                            flat: packed_flat,
                            repeated_smooth: repeated
                                .map(|(c, _, _)| if edges { &c.edges } else { &c.faces }),
                            repeated_flat: if edges {
                                None
                            } else {
                                repeated.and_then(|(_, f, _)| f)
                            },
                            repeated_slots: repeated.map(|(_, _, slots)| slots),
                        };
                        draws.paint(pass, &paint_pipelines, false, |pass, index| {
                            let mut active_direct = None;
                            draw_direct(pass, index, &mut active_direct);
                        });
                    } else {
                        let mut active_direct = None;
                        for index in 0..view.draws.len() {
                            draw_direct(pass, index, &mut active_direct);
                        }
                    }
                }
            }
            DrawPhase::OutlineMask => {
                self.paint_outline_mask(
                    &p.mask,
                    p.packed.as_ref().map(|packed| &packed.mask),
                    p.storage.as_ref().map(|storage| &storage.mask),
                    p.storage
                        .as_ref()
                        .and_then(|storage| storage.repeated.as_ref())
                        .and_then(|rep| rep.mask.as_ref()),
                    pass,
                    ctx,
                );
            }
            DrawPhase::Picking => {
                let direct_pick = if has_clip_planes {
                    &p.pick
                } else {
                    &p.pick_unclipped
                };
                let draw_direct = |pass: &mut wgpu::RenderPass<'_>, index: usize| {
                    if let Some(object) = view.draws.get(index)
                        && let Some(Some(chunk)) = self.chunks.get(object.chunk)
                        && chunk.visible
                        && let Some(vertices) = &chunk.vertex_buf
                        && view.bind(pass, index, p.dynamic_uniforms)
                    {
                        vertices.draw_direct_pick(
                            pass,
                            chunk.pick_buf.as_deref(),
                            chunk.vertex_count,
                        );
                    }
                };
                let storage_pick = p.storage.as_ref().and_then(|storage| {
                    if has_clip_planes {
                        storage.pick.as_ref()
                    } else {
                        storage.pick_unclipped.as_ref()
                    }
                });
                let storage_bg = view
                    .storage_binding
                    .as_ref()
                    .and_then(StorageViewBinding::bind_group);
                let packed_pick = p.packed.as_ref().and_then(|packed| {
                    if has_clip_planes {
                        packed.pick.as_ref()
                    } else {
                        packed.pick_unclipped.as_ref()
                    }
                });
                if let (Some(pipeline), Some(storage_bg)) = (storage_pick, storage_bg)
                    && view.storage_faces.has_packed()
                {
                    let paint_pipelines = PaintPipelines {
                        smooth: pipeline,
                        flat: None,
                        repeated_smooth: None,
                        repeated_flat: None,
                        repeated_slots: None,
                    };
                    view.storage_faces.paint(
                        pass,
                        storage_bg,
                        &paint_pipelines,
                        true,
                        |pass, index| {
                            pass.set_pipeline(direct_pick);
                            draw_direct(pass, index);
                        },
                    );
                } else if let Some(pipeline) = packed_pick
                    && view.faces.has_packed()
                {
                    let paint_pipelines = PaintPipelines {
                        smooth: pipeline,
                        flat: None,
                        repeated_smooth: None,
                        repeated_flat: None,
                        repeated_slots: None,
                    };
                    view.faces
                        .paint(pass, &paint_pipelines, true, |pass, index| {
                            pass.set_pipeline(direct_pick);
                            draw_direct(pass, index);
                        });
                } else {
                    pass.set_pipeline(direct_pick);
                    for index in 0..view.draws.len() {
                        draw_direct(pass, index);
                    }
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insertion_reuses_the_lowest_vacant_slot_and_rejects_stale_handles() {
        let mut scene = Scene3DLayer::new(1);
        let insert = |scene: &mut Scene3DLayer| scene.insert(Mesh3D::at_origin(Vec3::ZERO));
        let first = insert(&mut scene);
        let second = insert(&mut scene);
        let third = insert(&mut scene);
        scene.remove(third);
        scene.remove(first);
        scene.remove(first);

        let replacement_first = insert(&mut scene);
        scene.remove(first);
        assert_eq!(replacement_first.index, first.index);
        assert!(scene.chunk_mut(replacement_first).is_some());
        assert!(scene.chunk_mut(first).is_none());
        assert_eq!(insert(&mut scene).index, third.index);
        assert_eq!(insert(&mut scene).index, 3);
        assert!(scene.chunk_mut(second).is_some());
    }

    #[test]
    fn pending_uploads_and_selection_follow_chunk_lifetimes() {
        let mut scene = Scene3DLayer::new(1);
        let first = scene.insert(Mesh3D::at_origin(Vec3::ZERO));
        let second = scene.insert(Mesh3D::at_origin(Vec3::ZERO));
        scene.set_selected(first, true);
        scene.set_selected(first, true);
        scene.set_selected(second, true);
        scene.set_visible(second, false);
        assert!(ViewportLayer::<()>::phases(&scene).contains(DrawPhase::OutlineMask));

        scene.remove(first);
        assert_eq!(scene.pending_meshes, 1);
        assert!(!ViewportLayer::<()>::phases(&scene).contains(DrawPhase::OutlineMask));
        let replacement = scene.insert(Mesh3D::at_origin(Vec3::ZERO));
        scene.set_selected(first, true);
        scene.remove(first);
        assert_eq!(scene.pending_meshes, 2);
        assert!(!scene.is_selected(replacement));

        scene.set_visible(second, true);
        assert!(ViewportLayer::<()>::phases(&scene).contains(DrawPhase::OutlineMask));
        scene.clear_selection();
        assert!(!scene.is_selected(second));
        assert!(scene.selected_slots.is_empty());
    }
}
