//! Paged, incremental GPU vertex buffer pool (`PagedBuffer<V>`).
//!
//! Supports both:
//! - **Zero-copy bulk uploads** ([`PagedBuffer::upload_direct`]) using `create_buffer_init` across
//!   64 MiB pages for large drawing loads with minimum draw calls and zero intermediate CPU copies.
//! - **Incremental slot updates** ([`PagedBuffer::insert`], [`PagedBuffer::update`],
//!   [`PagedBuffer::remove`]) under a per-frame [`UploadBudget`].

use crate::spatial::TileRange;
use bytemuck::{Pod, Zeroable};
use iletki_core::{Bounds2, UploadBudget};
use iletki_wgpu::wgpu;
use iletki_wgpu::wgpu::util::DeviceExt;
use std::marker::PhantomData;

/// Default maximum bytes per GPU vertex page (64 MiB, clamped further by device limits).
const DEFAULT_PAGE_BYTES: u64 = 64 * 1024 * 1024;

/// Handle to a vertex range allocated inside a [`PagedBuffer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SlotHandle {
    page: usize,
    offset: usize,
    capacity: usize,
    len: usize,
}

impl SlotHandle {
    /// Number of active vertices currently stored in this slot.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if this slot currently stores zero vertices.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[derive(Debug, Clone, Copy)]
struct FreeSpan {
    offset: usize,
    capacity: usize,
}

struct Page {
    capacity_verts: usize,
    high_water_verts: usize,
    free: Vec<FreeSpan>,
    tiles: Vec<TileRange>,
    gpu: Option<wgpu::Buffer>,
}

struct PendingWrite {
    page: usize,
    offset_verts: usize,
    bytes: Vec<u8>,
}

/// Paged GPU vertex buffer supporting both zero-copy bulk loads and incremental slot updates under
/// a per-frame [`UploadBudget`].
pub struct PagedBuffer<V: Pod + Zeroable> {
    label: &'static str,
    group_size: usize,
    page_bytes: u64,
    pages: Vec<Page>,
    pending: Vec<PendingWrite>,
    _marker: PhantomData<V>,
}

impl<V: Pod + Zeroable> std::fmt::Debug for PagedBuffer<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PagedBuffer")
            .field("label", &self.label)
            .field("group_size", &self.group_size)
            .field("pages", &self.pages.len())
            .field("pending_writes", &self.pending.len())
            .finish()
    }
}

impl<V: Pod + Zeroable> PagedBuffer<V> {
    /// Creates an empty paged vertex pool for primitives of `group_size` vertices
    /// (`2` for lines, `3` for triangles, `6` for wide quads).
    #[must_use]
    pub fn new(label: &'static str, group_size: usize) -> Self {
        Self {
            label,
            group_size: group_size.max(1),
            page_bytes: DEFAULT_PAGE_BYTES,
            pages: Vec::new(),
            pending: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// Returns the number of allocated pages.
    #[must_use]
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Returns `true` if this pool holds zero vertices across all pages.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pages.iter().all(|p| p.high_water_verts == 0)
    }

    /// Returns `true` if there are queued GPU writes waiting to be flushed.
    #[must_use]
    pub fn has_pending_uploads(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Clears all pages and queued writes.
    pub fn clear(&mut self) {
        self.pages.clear();
        self.pending.clear();
    }

    fn vertices_per_page(&self, min_verts: usize, max_buffer_bytes: u64) -> usize {
        let stride = std::mem::size_of::<V>().max(1) as u64;
        let cap_bytes = self.page_bytes.min(max_buffer_bytes).max(stride);
        let default_verts = usize::try_from(cap_bytes / stride).unwrap_or(usize::MAX);
        let max_verts = usize::try_from(max_buffer_bytes / stride).unwrap_or(usize::MAX);
        let target = default_verts.max(min_verts).min(max_verts);
        let g = self.group_size;
        (target / g * g).max(g)
    }

    /// Zero-copy bulk upload directly into GPU vertex buffers, reusing existing page GPU buffers
    /// when capacity fits (avoiding Vulkan/DX12 memory alloc stalls on rebuilds) and attaching
    /// optional spatial [`TileRange`]s for 2D frustum culling.
    pub fn upload_direct_tiled(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[V],
        tiles: &[TileRange],
    ) {
        self.pending.clear();
        if data.is_empty() {
            for page in &mut self.pages {
                page.high_water_verts = 0;
                page.free.clear();
                page.tiles.clear();
            }
            return;
        }
        let stride = std::mem::size_of::<V>().max(1) as u64;
        let per_page = self.vertices_per_page(0, device.limits().max_buffer_size);
        let old_pages = std::mem::take(&mut self.pages);
        let mut reusable = old_pages.into_iter();
        let mut page_base = 0usize;

        for chunk in data.chunks(per_page) {
            let chunk_len = chunk.len();
            let page_end = page_base.saturating_add(chunk_len);
            let bytes: &[u8] = bytemuck::cast_slice(chunk);

            // Slice any global tile ranges that overlap this page into page-local indices.
            let mut page_tiles = Vec::new();
            for t in tiles {
                let t_end = t.start.saturating_add(t.count);
                let lo = t.start.max(page_base);
                let hi = t_end.min(page_end);
                if hi > lo {
                    page_tiles.push(TileRange {
                        start: lo - page_base,
                        count: hi - lo,
                        bounds: t.bounds,
                    });
                }
            }

            // Reuse an existing GPU buffer of sufficient size (and not more than 4x oversized) to
            // avoid synchronous driver memory allocation stalls during interactive zoom rebuilds.
            let reused_gpu = reusable.next().and_then(|p| p.gpu).filter(|buf| {
                let need = bytes.len() as u64;
                buf.size() >= need && buf.size() <= need.saturating_mul(4).max(4 * 1024 * 1024)
            });

            let (gpu, capacity_verts) = if let Some(buf) = reused_gpu {
                queue.write_buffer(&buf, 0, bytes);
                let cap = usize::try_from(buf.size() / stride).unwrap_or(chunk_len);
                (buf, cap)
            } else {
                // Allocate with a small 25% headroom (capped at page limit) so subsequent zoom
                // rebuilds with slightly more tessellated curves reuse the same GPU allocation.
                let alloc_verts = chunk_len
                    .saturating_add(chunk_len / 4)
                    .min(per_page)
                    .max(chunk_len);
                let alloc_bytes = (alloc_verts as u64).saturating_mul(stride);
                if alloc_verts == chunk_len {
                    let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(self.label),
                        contents: bytes,
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    });
                    (buf, chunk_len)
                } else {
                    let buf = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(self.label),
                        size: alloc_bytes,
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    queue.write_buffer(&buf, 0, bytes);
                    (buf, alloc_verts)
                }
            };

            self.pages.push(Page {
                capacity_verts,
                high_water_verts: chunk_len,
                free: Vec::new(),
                tiles: page_tiles,
                gpu: Some(gpu),
            });
            page_base = page_end;
        }
    }

    /// Zero-copy bulk upload directly into GPU vertex buffers (`create_buffer_init`), avoiding any
    /// intermediate CPU byte copy.
    pub fn upload_direct(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[V]) {
        self.upload_direct_tiled(device, queue, data, &[]);
    }

    /// Queues replacement of the entire buffer contents with `data` for deferred upload during
    /// [`PagedBuffer::flush`] (respects [`UploadBudget`]).
    pub fn replace_all(&mut self, data: &[V]) {
        self.clear();
        if data.is_empty() {
            return;
        }
        let per_page = self.vertices_per_page(0, DEFAULT_PAGE_BYTES);
        for chunk in data.chunks(per_page) {
            let page_idx = self.pages.len();
            self.pages.push(Page {
                capacity_verts: chunk.len(),
                high_water_verts: chunk.len(),
                free: Vec::new(),
                tiles: Vec::new(),
                gpu: None,
            });
            self.pending.push(PendingWrite {
                page: page_idx,
                offset_verts: 0,
                bytes: bytemuck::cast_slice(chunk).to_vec(),
            });
        }
    }

    /// Allocates a slot and queues `data` for incremental upload.
    ///
    /// Returns `None` if `data` is empty or cannot fit in a single page.
    pub fn insert(&mut self, data: &[V]) -> Option<SlotHandle> {
        let len = data.len();
        if len == 0 {
            return None;
        }
        let g = self.group_size;
        let needed = len.div_ceil(g).saturating_mul(g);

        for (page_idx, page) in self.pages.iter_mut().enumerate() {
            if let Some(pos) = page.free.iter().position(|s| s.capacity >= needed) {
                let span = page.free.swap_remove(pos);
                let leftover = span.capacity.saturating_sub(needed);
                if leftover >= g {
                    page.free.push(FreeSpan {
                        offset: span.offset.saturating_add(needed),
                        capacity: leftover,
                    });
                }
                let slot = SlotHandle {
                    page: page_idx,
                    offset: span.offset,
                    capacity: needed,
                    len,
                };
                self.queue_slot_bytes(slot, data);
                return Some(slot);
            }
            let avail = page.capacity_verts.saturating_sub(page.high_water_verts);
            if avail >= needed {
                let offset = page.high_water_verts;
                page.high_water_verts = offset.saturating_add(needed);
                let slot = SlotHandle {
                    page: page_idx,
                    offset,
                    capacity: needed,
                    len,
                };
                self.queue_slot_bytes(slot, data);
                return Some(slot);
            }
        }

        // Open a new page sized cleanly for incremental edits (8 MiB default or `needed`).
        let page_cap = self.vertices_per_page(needed, 8 * 1024 * 1024);
        if page_cap < needed {
            return None;
        }
        let page_idx = self.pages.len();
        self.pages.push(Page {
            capacity_verts: page_cap,
            high_water_verts: needed,
            free: Vec::new(),
            tiles: Vec::new(),
            gpu: None,
        });
        let slot = SlotHandle {
            page: page_idx,
            offset: 0,
            capacity: needed,
            len,
        };
        self.queue_slot_bytes(slot, data);
        Some(slot)
    }

    /// Updates an existing slot in place when `data` fits, or reallocates if it grew.
    pub fn update(&mut self, slot: SlotHandle, data: &[V]) -> Option<SlotHandle> {
        if data.is_empty() {
            self.remove(slot);
            return None;
        }
        if data.len() <= slot.capacity && slot.page < self.pages.len() {
            let updated = SlotHandle {
                len: data.len(),
                ..slot
            };
            self.queue_slot_bytes(updated, data);
            return Some(updated);
        }
        self.remove(slot);
        self.insert(data)
    }

    /// Frees a slot by zeroing its vertices (turning primitives degenerate so they rasterize 0
    /// pixels) and returning its span to the page's free list.
    pub fn remove(&mut self, slot: SlotHandle) {
        let Some(page) = self.pages.get_mut(slot.page) else {
            return;
        };
        let zero_bytes = slot.capacity.saturating_mul(std::mem::size_of::<V>());
        self.pending.push(PendingWrite {
            page: slot.page,
            offset_verts: slot.offset,
            bytes: vec![0u8; zero_bytes],
        });
        page.free.push(FreeSpan {
            offset: slot.offset,
            capacity: slot.capacity,
        });
    }

    fn queue_slot_bytes(&mut self, slot: SlotHandle, data: &[V]) {
        let stride = std::mem::size_of::<V>();
        let mut bytes = Vec::with_capacity(slot.capacity.saturating_mul(stride));
        bytes.extend_from_slice(bytemuck::cast_slice(data));
        let pad_verts = slot.capacity.saturating_sub(data.len());
        if pad_verts > 0 {
            bytes.resize(bytes.len().saturating_add(pad_verts * stride), 0);
        }
        self.pending.push(PendingWrite {
            page: slot.page,
            offset_verts: slot.offset,
            bytes,
        });
    }

    /// Ensures GPU page buffers exist and flushes pending writes up to `budget.max_bytes_per_frame`.
    ///
    /// Returns `true` if more pending writes remain for subsequent frames.
    pub fn flush(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        budget: UploadBudget,
    ) -> bool {
        let stride = std::mem::size_of::<V>().max(1) as u64;
        for page in &mut self.pages {
            if page.gpu.is_none() && page.capacity_verts > 0 {
                let size = (page.capacity_verts as u64)
                    .saturating_mul(stride)
                    .min(device.limits().max_buffer_size);
                page.gpu = Some(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(self.label),
                    size,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
        }

        let mut uploaded = 0usize;
        let mut remaining = Vec::new();
        for write in self.pending.drain(..) {
            if uploaded > 0
                && uploaded.saturating_add(write.bytes.len()) > budget.max_bytes_per_frame
            {
                remaining.push(write);
                continue;
            }
            if let Some(page) = self.pages.get(write.page)
                && let Some(buf) = &page.gpu
            {
                let byte_offset = (write.offset_verts as u64).saturating_mul(stride);
                let end = byte_offset.saturating_add(write.bytes.len() as u64);
                if end <= buf.size() {
                    queue.write_buffer(buf, byte_offset, &write.bytes);
                    uploaded = uploaded.saturating_add(write.bytes.len());
                }
            }
        }
        self.pending = remaining;
        !self.pending.is_empty()
    }

    /// Issues draw calls for all allocated pages into `pass` (binding slot 0).
    pub fn draw_all(&self, pass: &mut wgpu::RenderPass<'_>) {
        self.draw_culled(pass, None);
    }

    /// Issues frustum-culled draw calls for all allocated pages into `pass`, skipping spatial
    /// tiles whose conservative bounds lie completely outside `visible` and coalescing adjacent
    /// visible ranges into minimal draw calls.
    pub fn draw_culled(&self, pass: &mut wgpu::RenderPass<'_>, visible: Option<Bounds2>) {
        for page in &self.pages {
            let Some(buf) = &page.gpu else {
                continue;
            };
            let Ok(count) = u32::try_from(page.high_water_verts) else {
                continue;
            };
            if count == 0 {
                continue;
            }
            pass.set_vertex_buffer(0, buf.slice(..));
            let Some(view) = visible.filter(|_| !page.tiles.is_empty()) else {
                pass.draw(0..count, 0..1);
                continue;
            };
            // Coalesce contiguous visible tile ranges so zoomed-out views emit 1 draw call while
            // zoomed-in views skip off-screen geometry entirely.
            let mut run_start: Option<u32> = None;
            let mut run_end: u32 = 0;
            for tile in &page.tiles {
                if !tile.bounds.intersects(view) {
                    if let Some(s) = run_start.take() {
                        pass.draw(s..run_end, 0..1);
                    }
                    continue;
                }
                let Ok(s) = u32::try_from(tile.start) else {
                    continue;
                };
                let Ok(c) = u32::try_from(tile.count) else {
                    continue;
                };
                let e = s.saturating_add(c).min(count);
                if e <= s {
                    continue;
                }
                match run_start {
                    Some(rs) if s == run_end => {
                        run_end = e;
                        run_start = Some(rs);
                    }
                    Some(rs) => {
                        pass.draw(rs..run_end, 0..1);
                        run_start = Some(s);
                        run_end = e;
                    }
                    None => {
                        run_start = Some(s);
                        run_end = e;
                    }
                }
            }
            if let Some(s) = run_start {
                pass.draw(s..run_end, 0..1);
            }
        }
    }

    /// Issues paired draw calls for picking, binding `self` at slot 0 and parallel `pick_ids` at
    /// slot 1.
    pub fn draw_all_with_pick(
        &self,
        pick_ids: &PagedBuffer<[u32; 2]>,
        pass: &mut wgpu::RenderPass<'_>,
    ) {
        for (page, pick_page) in self.pages.iter().zip(pick_ids.pages.iter()) {
            let (Some(vbuf), Some(pbuf)) = (&page.gpu, &pick_page.gpu) else {
                continue;
            };
            let count_usize = page.high_water_verts.min(pick_page.high_water_verts);
            let Ok(count) = u32::try_from(count_usize) else {
                continue;
            };
            if count == 0 {
                continue;
            }
            pass.set_vertex_buffer(0, vbuf.slice(..));
            pass.set_vertex_buffer(1, pbuf.slice(..));
            pass.draw(0..count, 0..1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vertex::Vertex2D;

    #[test]
    fn insert_update_remove_reuses_slots_without_growing_pages() {
        let mut pool = PagedBuffer::<Vertex2D>::new("test", 2);
        let seg = [Vertex2D::zeroed(); 2];
        let a = pool.insert(&seg).unwrap();
        let b = pool.insert(&seg).unwrap();
        assert_eq!(pool.page_count(), 1);
        assert_ne!(a.offset, b.offset);

        let a2 = pool.update(a, &seg).unwrap();
        assert_eq!(a2.offset, a.offset);

        pool.remove(a2);
        let c = pool.insert(&seg).unwrap();
        assert_eq!(c.offset, a.offset);
        assert_eq!(pool.page_count(), 1);
    }
}
