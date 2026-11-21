//! Thin `egui` / `eframe` integration adapter for [`iletki_wgpu::ViewportHost`].
//!
//! Keeps `iletki-core`, `iletki-wgpu`, `iletki-2d` and `iletki-3d` 100% UI-framework-agnostic
//! while giving `egui`/`eframe` users a ~10-line drop-in integration:
//! - [`translate_egui_input`] converts an `egui::Response` into [`ViewportInput`] events.
//! - [`apply_viewport_commands`] maps [`ViewportCommand`]s to `egui` cursor icons and redraws.
//! - [`ViewportPaintCallback`] wraps any `ViewportHost`-owning resource inside
//!   `egui_wgpu::CallbackResources` via `Path B (BorrowedPass)`.

use iletki_core::{
    CursorKind, KeyCode, Modifiers, PointerButton, ScrollUnit, Vec2, ViewportCommand, ViewportInput,
};
use iletki_wgpu::wgpu;
use iletki_wgpu::{PassTargetDesc, ViewportHost};

/// Re-export `egui` for callers building against the same version.
pub use egui;
/// Re-export `egui_wgpu` for callers building against the same version.
pub use egui_wgpu;

/// Trait implemented by an application resource stored in `egui_wgpu::CallbackResources` that owns
/// one or more [`ViewportHost`] instances and scene layers.
pub trait EguiViewportResource: Send + Sync + 'static {
    /// Returns a reference to the active [`ViewportHost`] so [`ViewportPaintCallback`] can blit it
    /// during `paint`.
    fn host(&self) -> &ViewportHost;

    /// Returns a mutable reference to the active [`ViewportHost`].
    fn host_mut(&mut self) -> &mut ViewportHost;

    /// Returns the [`PassTargetDesc`] matching the `eframe`/`egui_wgpu` render pass.
    fn pass_target(&self) -> PassTargetDesc;

    /// Runs `host.prepare_for_pass(...)` with the application's layers and world state.
    fn prepare_viewport(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
    );
}

/// Recommended [`PassTargetDesc`] for embedding `iletki` inside `eframe` (`multisampling: 0`,
/// `depth_buffer: 0`) where `iletki` performs its own offscreen 4x MSAA + Reversed-Z depth pass
/// and blits the resolved texture into `egui`'s render pass.
#[must_use]
pub const fn egui_pass_target(format: wgpu::TextureFormat, blend_alpha: bool) -> PassTargetDesc {
    PassTargetDesc {
        format,
        sample_count: 1,
        depth_format: None,
        blend_alpha,
    }
}

/// Ready-to-use `egui_wgpu::CallbackTrait` implementation for any [`EguiViewportResource`] stored
/// in `egui_wgpu::CallbackResources`.
///
/// Construct via [`ViewportPaintCallback::new`] and pass to `egui_wgpu::Callback::new_paint_callback`.
pub struct ViewportPaintCallback<R: EguiViewportResource> {
    physical_px: [u32; 2],
    dpi_scale: f32,
    _marker: std::marker::PhantomData<fn() -> R>,
}

impl<R: EguiViewportResource> std::fmt::Debug for ViewportPaintCallback<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ViewportPaintCallback")
            .field("physical_px", &self.physical_px)
            .field("dpi_scale", &self.dpi_scale)
            .finish()
    }
}

impl<R: EguiViewportResource> ViewportPaintCallback<R> {
    /// Creates a paint callback sized to `rect` at `pixels_per_point`.
    #[must_use]
    pub fn new(rect: egui::Rect, pixels_per_point: f32) -> Self {
        let dpi = if pixels_per_point.is_finite() && pixels_per_point > 0.0 {
            pixels_per_point
        } else {
            1.0
        };
        let w = (rect.width().max(1.0) * dpi).round().max(1.0) as u32;
        let h = (rect.height().max(1.0) * dpi).round().max(1.0) as u32;
        Self {
            physical_px: [w, h],
            dpi_scale: dpi,
            _marker: std::marker::PhantomData,
        }
    }

    /// Wraps this callback in an `egui::PaintCallback` ready for `ui.painter().add(...)`.
    #[must_use]
    pub fn into_paint_callback(self, rect: egui::Rect) -> egui::PaintCallback {
        egui_wgpu::Callback::new_paint_callback(rect, self)
    }
}

impl<R: EguiViewportResource> egui_wgpu::CallbackTrait for ViewportPaintCallback<R> {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(res) = callback_resources.get_mut::<R>() {
            res.host_mut()
                .resize(device, self.physical_px, self.dpi_scale);
            res.prepare_viewport(device, queue, egui_encoder);
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        if let Some(res) = callback_resources.get::<R>() {
            res.host().paint_into_pass(render_pass, res.pass_target());
        }
    }
}

/// Converts `egui::Modifiers` to [`Modifiers`].
#[must_use]
pub const fn modifiers_from_egui(m: egui::Modifiers) -> Modifiers {
    Modifiers {
        shift: m.shift,
        ctrl: m.ctrl,
        alt: m.alt,
        meta: m.mac_cmd || m.command,
    }
}

/// Maps an `iletki` [`CursorKind`] to an `egui::CursorIcon`.
#[must_use]
pub const fn cursor_to_egui(kind: CursorKind) -> egui::CursorIcon {
    match kind {
        CursorKind::Default => egui::CursorIcon::Default,
        CursorKind::Crosshair => egui::CursorIcon::Crosshair,
        CursorKind::Grab => egui::CursorIcon::Grab,
        CursorKind::Grabbing => egui::CursorIcon::Grabbing,
        CursorKind::Move => egui::CursorIcon::Move,
        CursorKind::Pointer => egui::CursorIcon::PointingHand,
        CursorKind::Hidden => egui::CursorIcon::None,
    }
}

/// Applies window-wide viewport commands to `egui::Context`.
/// Use [`apply_viewport_commands_for_response`] for widget-local cursor wrapping and raw motion.
pub fn apply_viewport_commands(ctx: &egui::Context, commands: &[ViewportCommand]) {
    for cmd in commands {
        match cmd {
            ViewportCommand::SetCursor(kind) => ctx.set_cursor_icon(cursor_to_egui(*kind)),
            ViewportCommand::RequestRedraw => ctx.request_repaint(),
            ViewportCommand::GrabPointer(grab) => {
                let mode = if *grab {
                    egui::CursorGrab::Locked
                } else {
                    egui::CursorGrab::None
                };
                ctx.send_viewport_cmd(egui::ViewportCommand::CursorGrab(mode));
            }
            ViewportCommand::WrapPointer(_) => {}
        }
    }
}

/// Applies viewport commands with a widget responsible for pointer lock and cursor wrapping.
pub fn apply_viewport_commands_for_response(
    ctx: &egui::Context,
    response: &egui::Response,
    commands: &[ViewportCommand],
) {
    let owner_id = grab_owner_id(ctx);
    for command in commands {
        match command {
            ViewportCommand::GrabPointer(true) => {
                ctx.data_mut(|data| data.insert_temp(owner_id, response.id));
                response.request_focus();
            }
            ViewportCommand::GrabPointer(false) => {
                ctx.data_mut(|data| data.remove::<egui::Id>(owner_id));
            }
            ViewportCommand::WrapPointer(pos) if pos.is_finite() => {
                let offset = egui::vec2(pos.x as f32, pos.y as f32);
                if offset.is_finite() {
                    ctx.send_viewport_cmd(egui::ViewportCommand::CursorPosition(
                        response.rect.min + offset,
                    ));
                }
            }
            _ => {}
        }
        apply_viewport_commands(ctx, std::slice::from_ref(command));
    }
}

fn grab_owner_id(ctx: &egui::Context) -> egui::Id {
    egui::Id::unique(("iletki pointer grab", ctx.viewport_id()))
}

fn navigation_key(key: egui::Key) -> Option<KeyCode> {
    Some(match key {
        egui::Key::Space => KeyCode::Space,
        egui::Key::Escape => KeyCode::Escape,
        egui::Key::W => KeyCode::W,
        egui::Key::A => KeyCode::A,
        egui::Key::S => KeyCode::S,
        egui::Key::D => KeyCode::D,
        egui::Key::Q => KeyCode::Q,
        egui::Key::E => KeyCode::E,
        _ => return None,
    })
}

/// Translates `egui` pointer, drag, wheel, pinch, navigation and focus events on `response` into
/// viewport-local [`ViewportInput`] events (logical points, `(0, 0)` at `rect.min`).
#[must_use]
pub fn translate_egui_input(
    ctx: &egui::Context,
    response: &egui::Response,
    rect: egui::Rect,
) -> Vec<ViewportInput> {
    let mut events = Vec::new();
    let owner_id = grab_owner_id(ctx);
    let grabbed = ctx.data(|data| data.get_temp::<egui::Id>(owner_id)) == Some(response.id);
    let hovered = response.hovered();
    let dragged = response.dragged();
    let focused = response.has_focus();
    let to_local =
        |p: egui::Pos2| Vec2::new(f64::from(p.x - rect.min.x), f64::from(p.y - rect.min.y));

    ctx.input(|input| {
        let modifiers = modifiers_from_egui(input.modifiers);
        let center = to_local(rect.center());

        for ev in &input.events {
            match ev {
                egui::Event::Key {
                    key,
                    physical_key,
                    pressed,
                    modifiers,
                    ..
                } if grabbed || focused || (hovered && *key == egui::Key::Space) => {
                    if let Some(key) = navigation_key(physical_key.unwrap_or(*key)) {
                        let modifiers = modifiers_from_egui(*modifiers);
                        events.push(if *pressed {
                            ViewportInput::KeyDown { key, modifiers }
                        } else {
                            ViewportInput::KeyUp { key, modifiers }
                        });
                    }
                }
                egui::Event::PointerButton {
                    pos,
                    button,
                    pressed,
                    ..
                } if grabbed || hovered || dragged || !*pressed => {
                    let mapped = match button {
                        egui::PointerButton::Primary => Some(PointerButton::Primary),
                        egui::PointerButton::Secondary => Some(PointerButton::Secondary),
                        egui::PointerButton::Middle => Some(PointerButton::Middle),
                        egui::PointerButton::Extra1 => Some(PointerButton::Auxiliary(1)),
                        egui::PointerButton::Extra2 => Some(PointerButton::Auxiliary(2)),
                    };
                    if let Some(btn) = mapped {
                        let local = to_local(*pos);
                        events.push(if *pressed {
                            ViewportInput::PointerDown {
                                pos: local,
                                button: btn,
                                modifiers,
                            }
                        } else {
                            ViewportInput::PointerUp {
                                pos: local,
                                button: btn,
                                modifiers,
                            }
                        });
                    }
                }
                egui::Event::MouseWheel { unit, delta, .. } if hovered || grabbed => {
                    let pos = input.pointer.hover_pos().map_or(
                        Vec2::new(
                            f64::from(rect.width()) * 0.5,
                            f64::from(rect.height()) * 0.5,
                        ),
                        to_local,
                    );
                    let scroll_unit = match unit {
                        egui::MouseWheelUnit::Point => ScrollUnit::Pixel,
                        egui::MouseWheelUnit::Line | egui::MouseWheelUnit::Page => ScrollUnit::Line,
                    };
                    events.push(ViewportInput::Scroll {
                        pos,
                        delta: Vec2::new(f64::from(delta.x), f64::from(delta.y)),
                        unit: scroll_unit,
                        modifiers,
                    });
                }
                egui::Event::Zoom(factor) if hovered && factor.is_finite() => {
                    let pos = input.pointer.hover_pos().map_or(
                        Vec2::new(
                            f64::from(rect.width()) * 0.5,
                            f64::from(rect.height()) * 0.5,
                        ),
                        to_local,
                    );
                    events.push(ViewportInput::Pinch {
                        pos,
                        scale_factor: f64::from(*factor),
                    });
                }
                egui::Event::MouseMoved(delta) if grabbed => {
                    events.push(ViewportInput::PointerMove {
                        pos: center,
                        delta: Vec2::new(f64::from(delta.x), f64::from(delta.y)),
                        modifiers,
                    });
                }
                egui::Event::WindowFocused(false) => events.push(ViewportInput::FocusLost),
                _ => {}
            }
        }

        if !grabbed
            && input.focused
            && let Some(pos) = input.pointer.interact_pos()
        {
            let delta = input.pointer.delta();
            if (hovered || dragged) && (delta.x != 0.0 || delta.y != 0.0) {
                events.push(ViewportInput::PointerMove {
                    pos: to_local(pos),
                    delta: Vec2::new(f64::from(delta.x), f64::from(delta.y)),
                    modifiers,
                });
            }
        }
    });

    if response.lost_focus() && !events.contains(&ViewportInput::FocusLost) {
        events.push(ViewportInput::FocusLost);
    }

    events
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursor_and_modifier_mapping_cover_standard_cases() {
        assert_eq!(
            cursor_to_egui(CursorKind::Grabbing),
            egui::CursorIcon::Grabbing
        );
        assert_eq!(
            cursor_to_egui(CursorKind::Crosshair),
            egui::CursorIcon::Crosshair
        );
        let m = modifiers_from_egui(egui::Modifiers {
            alt: true,
            ctrl: true,
            shift: false,
            mac_cmd: false,
            command: true,
        });
        assert!(m.alt && m.ctrl && m.meta && !m.shift);
    }

    #[test]
    fn pass_target_defaults_to_single_sample_no_depth() {
        let desc = egui_pass_target(wgpu::TextureFormat::Bgra8UnormSrgb, true);
        assert_eq!(desc.sample_count, 1);
        assert_eq!(desc.depth_format, None);
        assert!(desc.blend_alpha);
    }

    #[test]
    fn widget_pointer_grab_routes_motion_and_focus_loss() {
        let ctx = egui::Context::default();
        let input = egui::RawInput {
            events: vec![
                egui::Event::MouseMoved(egui::vec2(3.0, -2.0)),
                egui::Event::WindowFocused(false),
            ],
            ..egui::RawInput::default()
        };
        let mut output = ctx.run_ui(input, |ui| {
            let (_, response) = ui.allocate_exact_size(egui::vec2(100.0, 80.0), egui::Sense::drag());
            apply_viewport_commands_for_response(&ctx, &response, &[ViewportCommand::GrabPointer(true)]);
            let events = translate_egui_input(&ctx, &response, response.rect);
            assert!(events.contains(&ViewportInput::FocusLost));
            assert!(events.iter().any(|event| matches!(event, ViewportInput::PointerMove { delta, .. } if *delta == Vec2::new(3.0, -2.0))));
            apply_viewport_commands_for_response(&ctx, &response, &[ViewportCommand::GrabPointer(false)]);
        });
        output.textures_delta.clear();
        let commands = &output.viewport_output[&egui::ViewportId::ROOT].commands;
        assert!(commands.contains(&egui::ViewportCommand::CursorGrab(egui::CursorGrab::Locked)));
        assert!(commands.contains(&egui::ViewportCommand::CursorGrab(egui::CursorGrab::None)));
    }
}
