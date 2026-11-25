//! Schedules marquee previews against the window's nominal refresh interval.

use std::time::{Duration, Instant};

pub(crate) struct PreviewPacing {
    period: Duration,
    last_input: Option<Instant>,
    last_submit: Option<Instant>,
    burst: bool,
    redraw_deadline: Option<Instant>,
    gpu_backlogged_until: Option<Instant>,
}

impl PreviewPacing {
    pub(crate) fn new(refresh_millihertz: Option<u32>) -> Self {
        let rate = refresh_millihertz
            .filter(|rate| (10_000..=1_000_000).contains(rate))
            .unwrap_or(60_000);
        Self {
            period: Duration::from_nanos(1_000_000_000_000 / u64::from(rate)),
            last_input: None,
            last_submit: None,
            burst: false,
            redraw_deadline: None,
            gpu_backlogged_until: None,
        }
    }

    pub(crate) fn input(&mut self, now: Instant) {
        let limit = if self.burst {
            self.period
        } else {
            self.period * 3 / 4
        };
        self.burst = self
            .last_input
            .is_some_and(|last| now.saturating_duration_since(last) < limit);
        self.last_input = Some(now);
    }

    pub(crate) fn submitted(&mut self, now: Instant) {
        self.last_submit = Some(now);
    }

    pub(crate) fn request_frame(&mut self, now: Instant) {
        // Give a ready result a chance to reach RedrawRequested before encoding a newer preview.
        // The deadline keeps selection moving when redraw events are throttled or suppressed.
        self.redraw_deadline = Some(now + self.period);
    }

    pub(crate) fn frame_started(&mut self) {
        self.redraw_deadline = None;
    }

    pub(crate) fn pick_ready(&mut self, submitted: Instant, now: Instant) {
        if now.saturating_duration_since(submitted) >= self.pick_budget() {
            // Keep room for picks while the render and readback queue drains.
            self.gpu_backlogged_until = Some(now + self.period * 4);
        }
    }

    pub(crate) fn gpu_backlogged(&self, now: Instant, pending_since: Option<Instant>) -> bool {
        self.gpu_backlogged_until.is_some_and(|until| now < until)
            || pending_since.is_some_and(|submitted| {
                now.saturating_duration_since(submitted) >= self.pick_budget()
            })
    }

    fn pick_budget(&self) -> Duration {
        // Leave time to encode and submit the highlighted frame in the same interval.
        self.period * 3 / 4
    }

    pub(crate) fn deadline(&self) -> Option<Instant> {
        let rate_limit = self
            .last_submit
            .filter(|_| self.burst)
            .map(|last| last + self.period);
        rate_limit.into_iter().chain(self.redraw_deadline).max()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_rate_input_runs_without_added_wait() {
        let now = Instant::now();
        let mut pacing = PreviewPacing::new(Some(60_000));
        pacing.input(now);
        pacing.submitted(now);
        pacing.request_frame(now + Duration::from_millis(6));
        pacing.frame_started();
        pacing.input(now + Duration::from_millis(16));
        assert_eq!(pacing.deadline(), None);
    }

    #[test]
    fn burst_waits_for_the_next_slot_and_a_pending_frame() {
        let now = Instant::now();
        let mut pacing = PreviewPacing::new(Some(120_000));
        pacing.input(now);
        pacing.submitted(now);
        pacing.input(now + Duration::from_millis(2));
        assert_eq!(
            pacing.deadline(),
            Some(now + Duration::from_nanos(8_333_333))
        );
        pacing.request_frame(now + Duration::from_millis(5));
        assert_eq!(
            pacing.deadline(),
            Some(now + Duration::from_nanos(13_333_333))
        );
        pacing.frame_started();
        assert_eq!(
            pacing.deadline(),
            Some(now + Duration::from_nanos(8_333_333))
        );
        pacing.input(now + Duration::from_millis(22));
        assert_eq!(pacing.deadline(), None);
    }

    #[test]
    fn caps_frame_overlap_during_a_pick_backlog_and_recovers() {
        let now = Instant::now();
        let mut pacing = PreviewPacing::new(Some(60_000));
        let fast = now + Duration::from_millis(10);
        let late = now + Duration::from_millis(20);
        assert!(!pacing.gpu_backlogged(fast, Some(now)));
        pacing.pick_ready(now, fast);
        assert!(!pacing.gpu_backlogged(fast, None));
        assert!(pacing.gpu_backlogged(late, Some(now)));
        pacing.pick_ready(now, late);
        assert!(pacing.gpu_backlogged(late, None));
        assert!(!pacing.gpu_backlogged(now + Duration::from_millis(100), None));
    }
}
