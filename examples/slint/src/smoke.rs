//! Exercise the native Slint compositor and viewport event handlers.

use std::cell::Cell;
use std::rc::Rc;
use std::time::{Duration, Instant};

use slint::platform::{PointerEventButton, WindowEvent};
use slint::{
    ComponentHandle, LogicalPosition, LogicalSize, PhysicalSize, Timer, TimerMode, Window,
};

use super::{CENTER_ID, Demo, Result};

pub(super) fn run(app: &Demo) -> Result<()> {
    app.show()?;
    let weak = app.as_weak();
    let outcome = Rc::new(Cell::new(None));
    let done = outcome.clone();
    let deadline = Instant::now() + Duration::from_secs(15);
    let timer = Timer::default();
    let mut stage = Stage::Frame;
    timer.start(TimerMode::Repeated, Duration::from_millis(32), move || {
        let Some(app) = weak.upgrade() else {
            let _ = slint::quit_event_loop();
            return;
        };
        let result = if Instant::now() >= deadline {
            Err(format!(
                "timed out at {stage:?} with window size {:?}: {}",
                app.window().size(),
                app.get_status()
            )
            .into())
        } else {
            stage.advance(&app)
        };
        match result {
            Ok(false) => {}
            Ok(true) => {
                done.set(Some(Ok(())));
                let _ = slint::quit_event_loop();
            }
            Err(error) => {
                done.set(Some(Err(error)));
                let _ = slint::quit_event_loop();
            }
        }
    });
    slint::run_event_loop()?;
    timer.stop();
    outcome
        .take()
        .ok_or("Slint closed before the test completed")??;
    println!(
        "Slint WGPU30 rendered the viewport, picked {CENTER_ID:?}, resized, panned, and zoomed"
    );
    Ok(())
}

#[derive(Debug)]
enum Stage {
    Frame,
    Pick,
    Resize(PhysicalSize),
    Pan,
    Zoom,
}

impl Stage {
    fn advance(&mut self, app: &Demo) -> Result<bool> {
        let window = app.window();
        let size = window.size();
        let scale = window.scale_factor();
        let logical = size.to_logical(scale);
        let center = LogicalPosition::new(logical.width / 2.0, logical.height / 2.0);
        let panned = LogicalPosition::new(center.x + 80.0, center.y);
        let zoom_probe = LogicalPosition::new(panned.x - 56.0, center.y);

        match self {
            Self::Frame if app.get_frame_width() == 0 => {}
            Self::Frame => {
                check_pixel(window, center, true)?;
                button(window, center, PointerEventButton::Left, true)?;
                button(window, center, PointerEventButton::Left, false)?;
                *self = Self::Pick;
            }
            Self::Pick => {
                let expected = format!(
                    "Picked {}:{}:{}",
                    CENTER_ID.layer, CENTER_ID.object, CENTER_ID.sub
                );
                if app.get_status() == expected {
                    window.set_size(LogicalSize::new(477.0, 319.0).to_physical(scale));
                    *self = Self::Resize(size);
                }
            }
            Self::Resize(previous) if size == *previous => {}
            Self::Resize(_) => {
                check_pixel(window, center, true)?;
                button(window, center, PointerEventButton::Middle, true)?;
                window
                    .dispatch_event_with_result(WindowEvent::PointerMoved { position: panned })?;
                button(window, panned, PointerEventButton::Middle, false)?;
                *self = Self::Pan;
            }
            Self::Pan => {
                check_pixel(window, center, false)?;
                check_pixel(window, panned, true)?;
                check_pixel(window, zoom_probe, false)?;
                window.dispatch_event_with_result(WindowEvent::PointerScrolled {
                    position: panned,
                    delta_x: 0.0,
                    delta_y: 220.0,
                })?;
                *self = Self::Zoom;
            }
            Self::Zoom => {
                check_pixel(window, zoom_probe, true)?;
                return Ok(true);
            }
        }
        Ok(false)
    }
}

fn button(
    window: &Window,
    position: LogicalPosition,
    button: PointerEventButton,
    down: bool,
) -> Result<()> {
    let event = if down {
        window.dispatch_event_with_result(WindowEvent::PointerMoved { position })?;
        WindowEvent::PointerPressed { position, button }
    } else {
        WindowEvent::PointerReleased { position, button }
    };
    window.dispatch_event_with_result(event)?;
    Ok(())
}

fn check_pixel(window: &Window, position: LogicalPosition, green_expected: bool) -> Result<()> {
    let frame = window.take_snapshot()?;
    let size = window.size();
    if frame.width() != size.width || frame.height() != size.height {
        return Err("Slint snapshot size differs from the window size".into());
    }
    let scale = window.scale_factor();
    let x = (position.x * scale) as usize;
    let y = (position.y * scale) as usize;
    if x >= frame.width() as usize || y >= frame.height() as usize {
        return Err("sample is outside the Slint snapshot".into());
    }
    let pixel = y
        .checked_mul(frame.width() as usize)
        .and_then(|row| row.checked_add(x))
        .and_then(|i| frame.as_slice().get(i))
        .ok_or("sample is outside the Slint snapshot")?;
    let green = pixel.g.saturating_sub(pixel.r) > 45 && pixel.g.saturating_sub(pixel.b) > 35;
    if green != green_expected {
        return Err(format!(
            "unexpected Slint pixel at ({x}, {y}): {pixel:?}, expected green={green_expected}"
        )
        .into());
    }
    Ok(())
}
