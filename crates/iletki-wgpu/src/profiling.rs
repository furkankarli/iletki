//! Optional native benchmark capture. Keep captures off in latency measurements that compare
//! uninstrumented frames.

use crate::ViewportError;
use std::cell::RefCell;
use std::time::{Duration, Instant};

const MAX_QUERIES: u32 = 256;
const MAX_STAT_PASSES: u32 = 64;

thread_local! {
    static ACTIVE: RefCell<Option<Recorder>> = const { RefCell::new(None) };
}

struct Recorder {
    capture: Capture,
    queries: Option<wgpu::QuerySet>,
    stat_queries: Option<wgpu::QuerySet>,
    passes: Vec<&'static str>,
    stat_passes: Vec<&'static str>,
}

/// CPU timings, GPU timings and work counters for one capture.
#[derive(Debug, Default)]
pub struct Capture {
    /// CPU scope durations.
    pub cpu: Vec<(&'static str, Duration)>,
    /// GPU pass durations, present when the device enabled timestamp queries.
    pub gpu: Vec<(&'static str, Duration)>,
    /// Counts emitted by layers during preparation.
    pub counters: Vec<(&'static str, usize)>,
}

/// Starts recording on the current thread. Enable `TIMESTAMP_QUERY` on the device for GPU timings
/// and `PIPELINE_STATISTICS_QUERY` for vertex/clipper/fragment invocation counters.
///
/// Any earlier capture on this thread is replaced.
pub fn begin(device: &wgpu::Device) {
    let features = device.features();
    let queries = features.contains(wgpu::Features::TIMESTAMP_QUERY).then(|| {
        device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("iletki benchmark timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: MAX_QUERIES,
        })
    });
    let stat_queries = features
        .contains(wgpu::Features::PIPELINE_STATISTICS_QUERY)
        .then(|| {
            let types = wgpu::PipelineStatisticsTypes::VERTEX_SHADER_INVOCATIONS
                | wgpu::PipelineStatisticsTypes::CLIPPER_INVOCATIONS
                | wgpu::PipelineStatisticsTypes::CLIPPER_PRIMITIVES_OUT
                | wgpu::PipelineStatisticsTypes::FRAGMENT_SHADER_INVOCATIONS;
            device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("iletki benchmark pipeline stats"),
                ty: wgpu::QueryType::PipelineStatistics(types),
                count: MAX_STAT_PASSES,
            })
        });
    ACTIVE.with(|state| {
        *state.borrow_mut() = Some(Recorder {
            capture: Capture::default(),
            queries,
            stat_queries,
            passes: Vec::new(),
            stat_passes: Vec::new(),
        });
    });
}

/// Ends recording and resolves the GPU results after all captured work has been submitted.
///
/// This can wait for the device. Call it outside the frame timer.
pub fn finish(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Capture, ViewportError> {
    let Some(mut record) = ACTIVE.with(|state| state.borrow_mut().take()) else {
        return Ok(Capture::default());
    };
    let has_ts = record.queries.is_some() && !record.passes.is_empty();
    let has_ps = record.stat_queries.is_some() && !record.stat_passes.is_empty();
    if !has_ts && !has_ps {
        return Ok(record.capture);
    }

    let make_buffer = |label, size, usage| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    };

    let mut encoder = device.create_command_encoder(&Default::default());
    let ts_staging = if let (true, Some(queries)) = (has_ts, &record.queries) {
        let count = (record.passes.len() * 2) as u32;
        let size = u64::from(count) * 8;
        let resolve = make_buffer(
            "iletki benchmark timestamp resolve",
            size,
            wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        );
        let staging = make_buffer(
            "iletki benchmark timestamp readback",
            size,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        encoder.resolve_query_set(queries, 0..count, &resolve, 0);
        encoder.copy_buffer_to_buffer(&resolve, 0, &staging, 0, size);
        Some(staging)
    } else {
        None
    };

    let ps_staging = if let (true, Some(stat_queries)) = (has_ps, &record.stat_queries) {
        let count = record.stat_passes.len() as u32;
        let size = u64::from(count) * 32;
        let resolve = make_buffer(
            "iletki benchmark pstats resolve",
            size,
            wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        );
        let staging = make_buffer(
            "iletki benchmark pstats readback",
            size,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        encoder.resolve_query_set(stat_queries, 0..count, &resolve, 0);
        encoder.copy_buffer_to_buffer(&resolve, 0, &staging, 0, size);
        Some(staging)
    } else {
        None
    };

    let submission = queue.submit([encoder.finish()]);

    let map_and_read = |staging: &wgpu::Buffer| -> Result<Vec<u8>, ViewportError> {
        let (tx, rx) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission.clone()),
                timeout: None,
            })
            .map_err(|_| ViewportError::BufferMapFailed)?;
        rx.recv()
            .map_err(|_| ViewportError::BufferMapFailed)?
            .map_err(|_| ViewportError::BufferMapFailed)?;
        let bytes = {
            let data = staging
                .get_mapped_range(..)
                .map_err(|_| ViewportError::BufferMapFailed)?;
            data.to_vec()
        };
        staging.unmap();
        Ok(bytes)
    };

    if let Some(staging) = ts_staging {
        let data = map_and_read(&staging)?;
        let period = f64::from(queue.get_timestamp_period());
        for (&label, bytes) in record.passes.iter().zip(data.as_chunks::<16>().0) {
            let [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p] = *bytes;
            let first = u64::from_le_bytes([a, b, c, d, e, f, g, h]);
            let second = u64::from_le_bytes([i, j, k, l, m, n, o, p]);
            if let Some(ticks) = second.checked_sub(first) {
                record
                    .capture
                    .gpu
                    .push((label, Duration::from_nanos((ticks as f64 * period) as u64)));
            }
        }
    }

    if let Some(staging) = ps_staging {
        let data = map_and_read(&staging)?;
        for (&label, bytes) in record.stat_passes.iter().zip(data.as_chunks::<32>().0) {
            let read_u64 = |offset: usize| -> usize {
                bytes
                    .get(offset..offset + 8)
                    .and_then(|s| s.try_into().ok())
                    .map_or(0, |arr| u64::from_le_bytes(arr) as usize)
            };
            let (vs_name, ci_name, co_name, fs_name) = match label {
                "iletki static scene" | "iletki continuous scene" => (
                    "gpu.scene.vs_invocations",
                    "gpu.scene.clipper_in",
                    "gpu.scene.clipper_out",
                    "gpu.scene.fs_invocations",
                ),
                "iletki static scene phase2" | "iletki continuous scene phase2" => (
                    "gpu.scene_p2.vs_invocations",
                    "gpu.scene_p2.clipper_in",
                    "gpu.scene_p2.clipper_out",
                    "gpu.scene_p2.fs_invocations",
                ),
                "iletki pick pass" => (
                    "gpu.pick.vs_invocations",
                    "gpu.pick.clipper_in",
                    "gpu.pick.clipper_out",
                    "gpu.pick.fs_invocations",
                ),
                _ => (
                    "gpu.other.vs_invocations",
                    "gpu.other.clipper_in",
                    "gpu.other.clipper_out",
                    "gpu.other.fs_invocations",
                ),
            };
            record.capture.counters.push((vs_name, read_u64(0)));
            record.capture.counters.push((ci_name, read_u64(8)));
            record.capture.counters.push((co_name, read_u64(16)));
            record.capture.counters.push((fs_name, read_u64(24)));
        }
    }

    Ok(record.capture)
}

/// Records a count when a capture is active on this thread.
pub fn count(label: &'static str, value: usize) {
    ACTIVE.with(|state| {
        if let Some(record) = state.borrow_mut().as_mut() {
            record.capture.counters.push((label, value));
        }
    });
}

/// Returns whether the current thread is recording a capture.
pub fn active() -> bool {
    ACTIVE.with(|state| state.borrow().is_some())
}

/// Records CPU time when dropped. It does no timing work outside an active capture.
#[derive(Debug)]
pub struct Scope {
    label: &'static str,
    start: Option<Instant>,
}

impl Drop for Scope {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            ACTIVE.with(|state| {
                if let Some(record) = state.borrow_mut().as_mut() {
                    record.capture.cpu.push((self.label, start.elapsed()));
                }
            });
        }
    }
}

/// Opens a CPU profiling scope on the current thread.
pub fn scope(label: &'static str) -> Scope {
    Scope {
        label,
        start: ACTIVE.with(|state| state.borrow().is_some().then(Instant::now)),
    }
}

/// Active GPU pass timer handle returned by [`pass`].
#[derive(Debug)]
pub struct Pass {
    queries: Option<wgpu::QuerySet>,
    first: u32,
    stat_queries: Option<wgpu::QuerySet>,
    stat_index: u32,
}

impl Pass {
    pub(crate) fn writes(&self) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        self.queries
            .as_ref()
            .map(|query_set| wgpu::RenderPassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(self.first),
                end_of_pass_write_index: Some(self.first + 1),
            })
    }

    /// Returns timestamp writes for a compute pass when timestamp queries are active.
    #[must_use]
    pub fn compute_writes(&self) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        self.queries
            .as_ref()
            .map(|query_set| wgpu::ComputePassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(self.first),
                end_of_pass_write_index: Some(self.first + 1),
            })
    }

    pub(crate) fn begin_stats(&self, pass: &mut wgpu::RenderPass<'_>) {
        if let Some(query_set) = &self.stat_queries {
            pass.begin_pipeline_statistics_query(query_set, self.stat_index);
        }
    }

    pub(crate) fn end_stats(&self, pass: &mut wgpu::RenderPass<'_>) {
        if self.stat_queries.is_some() {
            pass.end_pipeline_statistics_query();
        }
    }
}

/// Allocates timestamp and pipeline-statistics query slots for a GPU pass during an active capture.
#[must_use]
pub fn pass(label: &'static str) -> Option<Pass> {
    ACTIVE.with(|state| {
        let mut state = state.borrow_mut();
        let record = state.as_mut()?;
        let (queries, first) = match record.queries.as_ref() {
            Some(q) if ((record.passes.len() * 2) as u32) < MAX_QUERIES => {
                let first = (record.passes.len() * 2) as u32;
                record.passes.push(label);
                (Some(q.clone()), first)
            }
            _ => (None, 0),
        };
        let want_stats = matches!(
            label,
            "iletki static scene"
                | "iletki continuous scene"
                | "iletki static scene phase2"
                | "iletki continuous scene phase2"
                | "iletki pick pass"
        );
        let (stat_queries, stat_index) = match record.stat_queries.as_ref() {
            Some(q) if want_stats && (record.stat_passes.len() as u32) < MAX_STAT_PASSES => {
                let idx = record.stat_passes.len() as u32;
                record.stat_passes.push(label);
                (Some(q.clone()), idx)
            }
            _ => (None, 0),
        };
        if queries.is_none() && stat_queries.is_none() {
            return None;
        }
        Some(Pass {
            queries,
            first,
            stat_queries,
            stat_index,
        })
    })
}
