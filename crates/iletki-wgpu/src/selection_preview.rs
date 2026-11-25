//! Color-frame selection preview: compute bitset reduction and lifecycle tracking.
//!
//! During an active marquee gesture on compute-capable backends, participating layers write a
//! compact 1-based slot ID (`Rg16Uint`) as a second color attachment alongside shading (zero extra
//! vertex passes). A compute shader reduces the marquee rectangle into a compact per-slot bitset
//! (~2 KB for 17k objects), avoiding a 786 KB full ID+depth readback on every drag frame.

use crate::error::ViewportError;
use crate::layer::LayerRevision;
use crate::target::SELECTION_ID_FORMAT;
use bytemuck::{Pod, Zeroable};
use iletki_core::{Camera, ViewportId};
use std::future::Future;
use std::num::NonZeroU64;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

const REDUCE_1X_SUFFIX: &str = r#"
@group(0) @binding(1)
var id_tex: texture_2d<u32>;

@compute @workgroup_size(16, 16, 1)
fn reduce_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.rect.z || gid.y >= params.rect.w) {
        return;
    }
    let coord = vec2<i32>(i32(params.rect.x + gid.x), i32(params.rect.y + gid.y));
    let raw = textureLoad(id_tex, coord, 0);
    let id = raw.x | (raw.y << 16u);
    mark_slot(id);
}
"#;

const REDUCE_MS_SUFFIX: &str = r#"
@group(0) @binding(1)
var id_tex: texture_multisampled_2d<u32>;

@compute @workgroup_size(16, 16, 1)
fn reduce_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.rect.z || gid.y >= params.rect.w) {
        return;
    }
    let coord = vec2<i32>(i32(params.rect.x + gid.x), i32(params.rect.y + gid.y));
    var last = 0u;
    for (var s: i32 = 0; s < i32(params.sample_count); s = s + 1) {
        let raw = textureLoad(id_tex, coord, s);
        let id = raw.x | (raw.y << 16u);
        if (id != 0u && id != last) {
            mark_slot(id);
            last = id;
        }
    }
}
"#;

// Cap bitset readback to 1 MiB (up to 8,388,608 distinct slots).
const MAX_BITSET_WORDS: u32 = 262_144;
const MIN_BITSET_WORDS: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ReduceParams {
    rect: [u32; 4],
    word_count: u32,
    sample_count: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Snapshot tying an in-flight color-frame selection preview to its exact frame state.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionPreviewSnapshot {
    /// Viewport that produced this preview.
    pub viewport_id: ViewportId,
    /// Backing target allocation generation (changes on resize/reallocation).
    pub target_generation: u64,
    /// Camera state rendered into the color frame.
    pub camera: Camera,
    /// Clipped physical-pixel rectangle `[x, y, width, height]` that was reduced.
    pub rect_px: [u32; 4],
    /// Static scene revisions of participating layers at the time of capture.
    pub scene_revisions: Vec<(u16, LayerRevision)>,
}

/// Lifecycle counters for color-frame selection preview requests on a [`crate::ViewportHost`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SelectionPreviewStats {
    /// Number of color-frame preview reductions submitted to the GPU.
    pub submitted: u64,
    /// Number of previews replaced before being consumed or displayed.
    pub superseded: u64,
    /// Number of previews discarded because camera, target or scene state changed.
    pub stale: u64,
    /// Number of previews whose bitset readback completed and decoded.
    pub ready: u64,
    /// Number of previews applied and displayed by the host.
    pub displayed: u64,
}

/// Returns whether `device` can run MRT color-frame selection + compute bitset reduction.
pub(crate) fn supports_compute_preview(
    device: &wgpu::Device,
    color_format: wgpu::TextureFormat,
    msaa_samples: u32,
) -> bool {
    let info = device.adapter_info();
    if info.backend == wgpu::Backend::Gl {
        return false;
    }
    let limits = device.limits();
    if limits.max_compute_workgroups_per_dimension == 0
        || limits.max_compute_invocations_per_workgroup < 256
        || limits.max_storage_buffers_per_shader_stage < 1
        || limits.max_color_attachments < 2
    {
        return false;
    }
    let Some(color_cost) = color_format.target_pixel_byte_cost() else {
        return false;
    };
    let Some(id_cost) = SELECTION_ID_FORMAT.target_pixel_byte_cost() else {
        return false;
    };
    if color_cost.saturating_add(id_cost) > limits.max_color_attachment_bytes_per_sample {
        return false;
    }
    let features = SELECTION_ID_FORMAT.guaranteed_format_features(device.features());
    features
        .allowed_usages
        .contains(wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING)
        && features.flags.sample_count_supported(msaa_samples)
}

pub(crate) type SlotStagingCache = Arc<Mutex<Option<wgpu::Buffer>>>;

/// Compute pipeline and persistent GPU bitset buffer for reducing `Rg16Uint` slot IDs.
pub(crate) struct SelectionPreviewReducer {
    msaa_samples: u32,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    params_buf: wgpu::Buffer,
    bitset_buf: wgpu::Buffer,
    bitset_words: u32,
}

impl SelectionPreviewReducer {
    pub(crate) fn ensure<'a>(
        slot: &'a mut Option<Self>,
        device: &wgpu::Device,
        msaa_samples: u32,
    ) -> &'a mut Self {
        let rebuild = slot.as_ref().is_none_or(|r| r.msaa_samples != msaa_samples);
        if rebuild {
            *slot = Some(Self::new(device, msaa_samples));
        }
        // Safe unwrap alternative: re-insert only if None (unreachable here).
        slot.get_or_insert_with(|| Self::new(device, msaa_samples))
    }

    fn new(device: &wgpu::Device, msaa_samples: u32) -> Self {
        let multisampled = msaa_samples > 1;
        let wgsl = format!(
            "{}\n{}",
            include_str!("shaders/selection_reduce.wgsl"),
            if multisampled {
                REDUCE_MS_SUFFIX
            } else {
                REDUCE_1X_SUFFIX
            }
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("iletki selection bitset reduce"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("iletki selection reduce layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(size_of::<ReduceParams>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(4),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("iletki selection reduce pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("iletki selection reduce pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("reduce_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("iletki selection reduce params"),
            size: size_of::<ReduceParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bitset_words = MIN_BITSET_WORDS;
        let bitset_buf = create_bitset_buffer(device, bitset_words);
        Self {
            msaa_samples,
            layout,
            pipeline,
            params_buf,
            bitset_buf,
            bitset_words,
        }
    }

    /// Encodes bitset clear, compute reduction over `rect`, and copy to a compact staging buffer.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        id_view: &wgpu::TextureView,
        rect: [u32; 4],
        slot_count: u32,
        cache: &SlotStagingCache,
    ) -> Option<(wgpu::Buffer, u32)> {
        let [_, _, w, h] = rect;
        if w == 0 || h == 0 || slot_count == 0 {
            return None;
        }
        let word_count = slot_count.div_ceil(32).min(MAX_BITSET_WORDS);
        if word_count > self.bitset_words {
            let cap = word_count
                .next_power_of_two()
                .clamp(MIN_BITSET_WORDS, MAX_BITSET_WORDS);
            self.bitset_buf = create_bitset_buffer(device, cap);
            self.bitset_words = cap;
        }
        let byte_size = u64::from(word_count) * 4;
        let params = ReduceParams {
            rect,
            word_count,
            sample_count: self.msaa_samples,
            _pad0: 0,
            _pad1: 0,
        };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        encoder.clear_buffer(&self.bitset_buf, 0, Some(byte_size));

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("iletki selection reduce bind group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(id_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.bitset_buf,
                        offset: 0,
                        size: NonZeroU64::new(byte_size),
                    }),
                },
            ],
        });
        {
            let gpu_timer = crate::trace::pass("iletki selection reduce");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("iletki selection reduce"),
                timestamp_writes: gpu_timer.as_ref().and_then(|timer| timer.compute_writes()),
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(w.div_ceil(16), h.div_ceil(16), 1);
        }

        let staging = cache
            .lock()
            .ok()
            .and_then(|mut available| {
                if available
                    .as_ref()
                    .is_some_and(|buf| buf.size() >= byte_size)
                {
                    available.take()
                } else {
                    None
                }
            })
            .unwrap_or_else(|| {
                let alloc = byte_size.next_power_of_two().max(256);
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("iletki selection bitset staging"),
                    size: alloc,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                })
            });
        encoder.copy_buffer_to_buffer(&self.bitset_buf, 0, &staging, 0, byte_size);
        Some((staging, word_count))
    }
}

fn create_bitset_buffer(device: &wgpu::Device, words: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("iletki selection bitset"),
        size: u64::from(words) * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

pub(crate) struct StagedSlotPreview {
    pub(crate) staging: wgpu::Buffer,
    pub(crate) word_count: u32,
    pub(crate) slot_count: u32,
    pub(crate) cache: SlotStagingCache,
    pub(crate) snapshot: SelectionPreviewSnapshot,
    pub(crate) stats_ready: Arc<Mutex<SelectionPreviewStats>>,
}

struct MapState {
    result: Option<Result<(), wgpu::BufferAsyncError>>,
    waker: Option<Waker>,
}

/// An in-flight color-frame selection preview readback returning hit 0-based slot indices.
pub struct PendingSlotPreview {
    #[cfg(not(target_arch = "wasm32"))]
    device: wgpu::Device,
    map: Arc<Mutex<MapState>>,
    staged: Option<StagedSlotPreview>,
    snapshot: SelectionPreviewSnapshot,
}

impl std::fmt::Debug for PendingSlotPreview {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingSlotPreview")
            .field("rect_px", &self.snapshot.rect_px)
            .field("target_generation", &self.snapshot.target_generation)
            .finish()
    }
}

impl PendingSlotPreview {
    pub(crate) fn from_staged(device: &wgpu::Device, staged: StagedSlotPreview) -> Self {
        #[cfg(target_arch = "wasm32")]
        let _ = device;
        let byte_size = u64::from(staged.word_count) * 4;
        let state = Arc::new(Mutex::new(MapState {
            result: None,
            waker: None,
        }));
        let cb_state = Arc::clone(&state);
        staged
            .staging
            .slice(..byte_size)
            .map_async(wgpu::MapMode::Read, move |result| {
                let wake = cb_state.lock().ok().and_then(|mut guard| {
                    guard.result = Some(result);
                    guard.waker.take()
                });
                if let Some(waker) = wake {
                    waker.wake();
                }
            });
        let snapshot = staged.snapshot.clone();
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            device: device.clone(),
            map: state,
            staged: Some(staged),
            snapshot,
        }
    }

    /// Returns the viewport, target generation, camera, rectangle and scene snapshot for this preview.
    #[must_use]
    pub fn snapshot(&self) -> &SelectionPreviewSnapshot {
        &self.snapshot
    }

    /// Checks whether the compact bitset readback is complete, driving one nonblocking GPU poll on native.
    pub fn try_read(&mut self) -> Poll<Result<Vec<u32>, ViewportError>> {
        self.poll_completion(None)
    }

    fn poll_completion(&mut self, wake: Option<&Waker>) -> Poll<Result<Vec<u32>, ViewportError>> {
        let result = self.poll_staged(wake);
        if result.is_ready() {
            self.staged = None;
        }
        result
    }

    fn poll_staged(&mut self, wake: Option<&Waker>) -> Poll<Result<Vec<u32>, ViewportError>> {
        if self.staged.is_none() {
            return Poll::Ready(Err(ViewportError::PickAlreadyRead));
        }
        {
            let _profile = crate::trace::scope("selection_preview.poll");
            #[cfg(not(target_arch = "wasm32"))]
            self.device
                .poll(wgpu::PollType::Poll)
                .map_err(|_| ViewportError::BufferMapFailed)?;

            let mut guard = self
                .map
                .lock()
                .map_err(|_| ViewportError::BufferMapFailed)?;
            match &guard.result {
                Some(Ok(())) => {}
                Some(Err(_)) => return Poll::Ready(Err(ViewportError::BufferMapFailed)),
                None => {
                    if let Some(waker) = wake {
                        guard.waker = Some(waker.clone());
                    }
                    return Poll::Pending;
                }
            }
        }
        let staged = self.staged.take().ok_or(ViewportError::PickAlreadyRead)?;
        let slots = decode_staged_bitset(staged)?;
        Poll::Ready(Ok(slots))
    }
}

impl Future for PendingSlotPreview {
    type Output = Result<Vec<u32>, ViewportError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.poll_completion(Some(cx.waker()))
    }
}

fn decode_staged_bitset(staged: StagedSlotPreview) -> Result<Vec<u32>, ViewportError> {
    let _profile = crate::trace::scope("selection_preview.decode");
    let byte_size = u64::from(staged.word_count) * 4;
    let slots = {
        let slice = staged.staging.slice(..byte_size);
        let mapped = slice
            .get_mapped_range()
            .map_err(|_| ViewportError::BufferMapFailed)?;
        decode_bitset_bytes(&mapped, staged.slot_count)
    };
    staged.staging.unmap();
    if let Ok(mut stats) = staged.stats_ready.lock() {
        stats.ready = stats.ready.saturating_add(1);
    }
    if staged.staging.size() <= 1024 * 1024
        && let Ok(mut available) = staged.cache.lock()
        && available
            .as_ref()
            .is_none_or(|buf| staged.staging.size() >= buf.size())
    {
        *available = Some(staged.staging);
    }
    Ok(slots)
}

fn decode_bitset_bytes(bytes: &[u8], slot_count: u32) -> Vec<u32> {
    let mut slots = Vec::new();
    let (words, _) = bytes.as_chunks::<4>();
    for (word_idx, raw) in words.iter().enumerate() {
        let mut mask = u32::from_le_bytes(*raw);
        while mask != 0 {
            let tz = mask.trailing_zeros();
            let slot = (word_idx as u32).saturating_mul(32).saturating_add(tz);
            if slot < slot_count {
                slots.push(slot);
            }
            mask &= mask - 1;
        }
    }
    slots
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_sparse_bitset_and_respects_slot_count() {
        let words: [u32; 3] = [
            (1u32 << 0) | (1u32 << 5) | (1u32 << 31),
            0,
            (1u32 << 2) | (1u32 << 10),
        ];
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(decode_bitset_bytes(&bytes, 70), vec![0, 5, 31, 66]);
        assert_eq!(decode_bitset_bytes(&bytes, 100), vec![0, 5, 31, 66, 74]);
    }
}
