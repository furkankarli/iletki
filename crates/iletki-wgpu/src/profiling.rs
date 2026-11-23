//! Optional native benchmark capture. Keep captures off in latency measurements that compare
//! uninstrumented frames.

use crate::ViewportError;
use std::cell::RefCell;
use std::time::{Duration, Instant};

const MAX_QUERIES: u32 = 256;

thread_local! {
    static ACTIVE: RefCell<Option<Recorder>> = const { RefCell::new(None) };
}

struct Recorder {
    capture: Capture,
    queries: Option<wgpu::QuerySet>,
    passes: Vec<&'static str>,
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

/// Starts recording on the current thread. Enable `TIMESTAMP_QUERY` on the device for GPU timings.
///
/// Any earlier capture on this thread is replaced.
pub fn begin(device: &wgpu::Device) {
    let queries = device
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY)
        .then(|| {
            device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("iletki benchmark timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: MAX_QUERIES,
            })
        });
    ACTIVE.with(|state| {
        *state.borrow_mut() = Some(Recorder {
            capture: Capture::default(),
            queries,
            passes: Vec::new(),
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
    let Some(queries) = record.queries.filter(|_| !record.passes.is_empty()) else {
        return Ok(record.capture);
    };
    let count = (record.passes.len() * 2) as u32;
    let size = u64::from(count) * 8;
    let make_buffer = |usage| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("iletki benchmark timestamp readback"),
            size,
            usage,
            mapped_at_creation: false,
        })
    };
    let resolve = make_buffer(wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC);
    let staging = make_buffer(wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ);
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.resolve_query_set(&queries, 0..count, &resolve, 0);
    encoder.copy_buffer_to_buffer(&resolve, 0, &staging, 0, size);
    let submission = queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .map_err(|_| ViewportError::BufferMapFailed)?;
    rx.recv()
        .map_err(|_| ViewportError::BufferMapFailed)?
        .map_err(|_| ViewportError::BufferMapFailed)?;
    {
        let data = staging
            .get_mapped_range(..)
            .map_err(|_| ViewportError::BufferMapFailed)?;
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
    staging.unmap();
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

pub(crate) struct Pass {
    queries: wgpu::QuerySet,
    first: u32,
}

impl Pass {
    pub(crate) fn writes(&self) -> wgpu::RenderPassTimestampWrites<'_> {
        wgpu::RenderPassTimestampWrites {
            query_set: &self.queries,
            beginning_of_pass_write_index: Some(self.first),
            end_of_pass_write_index: Some(self.first + 1),
        }
    }
}

pub(crate) fn pass(label: &'static str) -> Option<Pass> {
    ACTIVE.with(|state| {
        let mut state = state.borrow_mut();
        let record = state.as_mut()?;
        let queries = record.queries.as_ref()?;
        let first = (record.passes.len() * 2) as u32;
        if first >= MAX_QUERIES {
            return None;
        }
        record.passes.push(label);
        Some(Pass {
            queries: queries.clone(),
            first,
        })
    })
}
