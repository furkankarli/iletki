//! Pluggable camera navigation controllers (rigs) for 2D CAD, 3D orbit and 3D fly modes.

use crate::camera::{Camera, Projection};
use crate::input::{
    CursorKind, KeyCode, PointerButton, ScrollUnit, ViewportCommand, ViewportInput,
};
use crate::math::{Quat, Vec2, Vec3};

/// Result returned by a [`CameraRig`] after processing an input event.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RigOutcome {
    /// `true` if the rig consumed the event (so lower-priority scene handlers should ignore it).
    pub consumed: bool,
    /// `true` if the camera changed (so the viewport should mark `DirtyFlags::CAMERA`).
    pub camera_changed: bool,
    /// Host commands requested by the rig (cursor change, pointer grab, redraw).
    pub commands: Vec<ViewportCommand>,
}

/// Trait implemented by pluggable 2D/3D camera controllers.
pub trait CameraRig: Send + Sync {
    /// Processes a single [`ViewportInput`] event and updates `camera` if applicable.
    fn handle_input(
        &mut self,
        input: &ViewportInput,
        camera: &mut Camera,
        viewport_size: Vec2,
    ) -> RigOutcome;

    /// Advances time-based motion (smooth damping or keyboard fly movement).
    ///
    /// Returns `true` if the camera moved and another frame should be scheduled.
    fn tick(&mut self, dt_secs: f64, camera: &mut Camera) -> bool;
}

/// Standard 2D CAD / vector canvas controller.
///
/// - Middle mouse drag (or Space + primary drag) pans.
/// - Mouse wheel or trackpad pinch zooms anchored at the cursor.
/// - Zero damping by default for crisp drafting response.
#[derive(Debug, Clone)]
pub struct Cad2DRig {
    /// Zoom multiplier per discrete mouse-wheel line tick (default `1.15`).
    pub wheel_zoom_step: f64,
    /// Smooth pixel scroll sensitivity for zoom (default `0.0025`).
    pub pixel_zoom_sensitivity: f64,
    /// If `true`, smooth pixel scroll pans instead of zooming (trackpad-friendly).
    pub pixel_scroll_pans: bool,
    panning: bool,
    space_held: bool,
}

impl Default for Cad2DRig {
    fn default() -> Self {
        Self {
            wheel_zoom_step: 1.15,
            pixel_zoom_sensitivity: 0.0025,
            pixel_scroll_pans: false,
            panning: false,
            space_held: false,
        }
    }
}

impl CameraRig for Cad2DRig {
    fn handle_input(
        &mut self,
        input: &ViewportInput,
        camera: &mut Camera,
        viewport_size: Vec2,
    ) -> RigOutcome {
        let mut out = RigOutcome::default();
        match input {
            ViewportInput::KeyDown {
                key: KeyCode::Space,
                ..
            } => {
                self.space_held = true;
            }
            ViewportInput::KeyUp {
                key: KeyCode::Space,
                ..
            } => {
                self.space_held = false;
            }
            ViewportInput::PointerDown { button, .. }
                if *button == PointerButton::Middle
                    || (*button == PointerButton::Primary && self.space_held) =>
            {
                self.panning = true;
                out.consumed = true;
                out.commands
                    .push(ViewportCommand::SetCursor(CursorKind::Grabbing));
            }
            ViewportInput::PointerUp { button, .. }
                if self.panning
                    && (*button == PointerButton::Middle || *button == PointerButton::Primary) =>
            {
                self.panning = false;
                out.consumed = true;
                out.commands
                    .push(ViewportCommand::SetCursor(CursorKind::Default));
            }
            ViewportInput::PointerMove { delta, .. } if self.panning => {
                out.consumed = true;
                if camera.pan_2d(*delta) {
                    out.camera_changed = true;
                    out.commands.push(ViewportCommand::RequestRedraw);
                }
            }
            ViewportInput::Scroll {
                pos, delta, unit, ..
            } => {
                out.consumed = true;
                let changed = match unit {
                    ScrollUnit::Pixel if self.pixel_scroll_pans => camera.pan_2d(*delta),
                    ScrollUnit::Pixel => {
                        let factor = (delta.y * self.pixel_zoom_sensitivity).exp();
                        camera.zoom_at_2d(*pos, viewport_size, factor)
                    }
                    ScrollUnit::Line => {
                        let factor = self.wheel_zoom_step.powf(delta.y);
                        camera.zoom_at_2d(*pos, viewport_size, factor)
                    }
                };
                if changed {
                    out.camera_changed = true;
                    out.commands.push(ViewportCommand::RequestRedraw);
                }
            }
            ViewportInput::Pinch { pos, scale_factor } => {
                out.consumed = true;
                if camera.zoom_at_2d(*pos, viewport_size, *scale_factor) {
                    out.camera_changed = true;
                    out.commands.push(ViewportCommand::RequestRedraw);
                }
            }
            ViewportInput::FocusLost => {
                if self.panning {
                    self.panning = false;
                    out.commands
                        .push(ViewportCommand::SetCursor(CursorKind::Default));
                }
                self.space_held = false;
            }
            _ => {}
        }
        out
    }

    fn tick(&mut self, _dt_secs: f64, _camera: &mut Camera) -> bool {
        false
    }
}

/// Rotation style for [`OrbitRig`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OrbitMode {
    /// Turntable orbit around a fixed world up axis (Blender / Rhino / CAD default).
    #[default]
    Turntable,
    /// Free trackball orbit without a locked up axis.
    Trackball,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OrbitDrag {
    None,
    Orbit,
    Pan,
}

/// 3D orbit/pan/zoom controller around a `f64` pivot point.
///
/// - Middle mouse drag orbits around [`OrbitRig::pivot`].
/// - Shift + middle mouse drag pans pivot and camera together.
/// - Scroll wheel dollies in/out (or scales orthographic `view_height`).
#[derive(Debug, Clone)]
pub struct OrbitRig {
    /// World-space point the camera orbits around.
    pub pivot: Vec3,
    /// World up axis used in [`OrbitMode::Turntable`] mode (default `+Z` for CAD/BIM).
    pub up_axis: Vec3,
    /// Orbit rotation mode.
    pub mode: OrbitMode,
    /// Radians rotated per logical point dragged.
    pub rotate_sensitivity: f64,
    /// Distance multiplier per wheel line tick.
    pub dolly_step: f64,
    drag: OrbitDrag,
}

impl Default for OrbitRig {
    fn default() -> Self {
        Self {
            pivot: Vec3::ZERO,
            up_axis: Vec3::Z,
            mode: OrbitMode::Turntable,
            rotate_sensitivity: 0.006,
            dolly_step: 1.15,
            drag: OrbitDrag::None,
        }
    }
}

impl OrbitRig {
    /// Creates an orbit rig focused on `pivot` with `up_axis`.
    #[must_use]
    pub fn new(pivot: Vec3, up_axis: Vec3) -> Self {
        Self {
            pivot: if pivot.is_finite() { pivot } else { Vec3::ZERO },
            up_axis: up_axis.normalized().unwrap_or(Vec3::Z),
            ..Self::default()
        }
    }

    fn apply_orbit(&self, delta: Vec2, camera: &mut Camera) -> bool {
        if !delta.is_finite() {
            return false;
        }
        let yaw = -delta.x * self.rotate_sensitivity;
        let pitch = -delta.y * self.rotate_sensitivity;
        let (position, rotation) = match &mut camera.projection {
            Projection::Perspective3D {
                position, rotation, ..
            }
            | Projection::Ortho3D {
                position, rotation, ..
            } => (position, rotation),
            Projection::Ortho2D { .. } => return false,
        };
        let offset = *position - self.pivot;
        let right = rotation.rotate(Vec3::X);
        let q_yaw = match self.mode {
            OrbitMode::Turntable => Quat::from_axis_angle(self.up_axis, yaw),
            OrbitMode::Trackball => Quat::from_axis_angle(rotation.rotate(Vec3::Y), yaw),
        };
        let q_pitch = Quat::from_axis_angle(right, pitch);
        let q_delta = q_yaw.mul_quat(q_pitch);
        let new_rot = q_delta.mul_quat(*rotation);
        let new_offset = q_delta.rotate(offset);
        *rotation = new_rot;
        *position = self.pivot + new_offset;
        true
    }

    fn apply_pan(&mut self, delta: Vec2, camera: &mut Camera, viewport_size: Vec2) -> bool {
        if !delta.is_finite() || viewport_size.y <= 0.0 {
            return false;
        }
        let right = camera.right_3d();
        let up = camera.up_3d();
        let world_per_pt = match camera.projection {
            Projection::Perspective3D {
                position, fov_y, ..
            } => {
                let dist = (position - self.pivot).length().max(1e-3);
                (2.0 * dist * (0.5 * fov_y).tan()) / viewport_size.y
            }
            Projection::Ortho3D { view_height, .. } => view_height / viewport_size.y,
            Projection::Ortho2D { .. } => return false,
        };
        let shift = right * (-delta.x * world_per_pt) + up * (delta.y * world_per_pt);
        self.pivot += shift;
        match &mut camera.projection {
            Projection::Perspective3D { position, .. } | Projection::Ortho3D { position, .. } => {
                *position += shift;
                true
            }
            Projection::Ortho2D { .. } => false,
        }
    }

    fn apply_dolly(&self, factor: f64, camera: &mut Camera) -> bool {
        if !factor.is_finite() || factor <= 0.0 {
            return false;
        }
        match &mut camera.projection {
            Projection::Perspective3D { position, .. } => {
                let offset = *position - self.pivot;
                let dist = offset.length();
                let new_dist = (dist / factor).clamp(1e-4, 1e11);
                let dir = offset.normalized().unwrap_or(Vec3::Z);
                *position = self.pivot + dir * new_dist;
                true
            }
            Projection::Ortho3D { view_height, .. } => {
                *view_height = (*view_height / factor).clamp(1e-6, 1e11);
                true
            }
            Projection::Ortho2D { .. } => false,
        }
    }
}

impl CameraRig for OrbitRig {
    fn handle_input(
        &mut self,
        input: &ViewportInput,
        camera: &mut Camera,
        viewport_size: Vec2,
    ) -> RigOutcome {
        let mut out = RigOutcome::default();
        match input {
            ViewportInput::PointerDown {
                button: PointerButton::Middle,
                modifiers,
                ..
            } => {
                self.drag = if modifiers.shift {
                    OrbitDrag::Pan
                } else {
                    OrbitDrag::Orbit
                };
                out.consumed = true;
                out.commands
                    .push(ViewportCommand::SetCursor(CursorKind::Grabbing));
            }
            ViewportInput::PointerUp {
                button: PointerButton::Middle,
                ..
            } if self.drag != OrbitDrag::None => {
                self.drag = OrbitDrag::None;
                out.consumed = true;
                out.commands
                    .push(ViewportCommand::SetCursor(CursorKind::Default));
            }
            ViewportInput::PointerMove { delta, .. } if self.drag != OrbitDrag::None => {
                out.consumed = true;
                let changed = match self.drag {
                    OrbitDrag::Orbit => self.apply_orbit(*delta, camera),
                    OrbitDrag::Pan => self.apply_pan(*delta, camera, viewport_size),
                    OrbitDrag::None => false,
                };
                if changed {
                    out.camera_changed = true;
                    out.commands.push(ViewportCommand::RequestRedraw);
                }
            }
            ViewportInput::Scroll { delta, unit, .. } => {
                out.consumed = true;
                let factor = match unit {
                    ScrollUnit::Line => self.dolly_step.powf(delta.y),
                    ScrollUnit::Pixel => (delta.y * 0.0025).exp(),
                };
                if self.apply_dolly(factor, camera) {
                    out.camera_changed = true;
                    out.commands.push(ViewportCommand::RequestRedraw);
                }
            }
            ViewportInput::Pinch { scale_factor, .. } => {
                out.consumed = true;
                if self.apply_dolly(*scale_factor, camera) {
                    out.camera_changed = true;
                    out.commands.push(ViewportCommand::RequestRedraw);
                }
            }
            ViewportInput::FocusLost if self.drag != OrbitDrag::None => {
                self.drag = OrbitDrag::None;
                out.commands
                    .push(ViewportCommand::SetCursor(CursorKind::Default));
            }
            _ => {}
        }
        out
    }

    fn tick(&mut self, _dt_secs: f64, _camera: &mut Camera) -> bool {
        false
    }
}

/// First-person 3D fly controller (Unreal / architectural walkthrough style).
///
/// Hold secondary (right) mouse button to look around and fly with `W/A/S/D/Q/E`; scroll adjusts
/// movement speed.
#[derive(Debug, Clone)]
pub struct FlyRig {
    /// Base flight speed in world units per second.
    pub speed: f64,
    /// Radians per logical point of mouse look.
    pub look_sensitivity: f64,
    /// World up axis for yaw rotation.
    pub up_axis: Vec3,
    looking: bool,
    move_fwd: bool,
    move_back: bool,
    move_left: bool,
    move_right: bool,
    move_up: bool,
    move_down: bool,
    boost: bool,
}

impl Default for FlyRig {
    fn default() -> Self {
        Self {
            speed: 10.0,
            look_sensitivity: 0.003,
            up_axis: Vec3::Z,
            looking: false,
            move_fwd: false,
            move_back: false,
            move_left: false,
            move_right: false,
            move_up: false,
            move_down: false,
            boost: false,
        }
    }
}

impl FlyRig {
    fn clear_keys(&mut self) {
        self.move_fwd = false;
        self.move_back = false;
        self.move_left = false;
        self.move_right = false;
        self.move_up = false;
        self.move_down = false;
        self.boost = false;
    }

    fn set_key(&mut self, key: KeyCode, pressed: bool) -> bool {
        match key {
            KeyCode::W => self.move_fwd = pressed,
            KeyCode::S => self.move_back = pressed,
            KeyCode::A => self.move_left = pressed,
            KeyCode::D => self.move_right = pressed,
            KeyCode::E => self.move_up = pressed,
            KeyCode::Q => self.move_down = pressed,
            KeyCode::Shift => self.boost = pressed,
            _ => return false,
        }
        true
    }
}

impl CameraRig for FlyRig {
    fn handle_input(
        &mut self,
        input: &ViewportInput,
        camera: &mut Camera,
        _viewport_size: Vec2,
    ) -> RigOutcome {
        let mut out = RigOutcome::default();
        match input {
            ViewportInput::PointerDown {
                button: PointerButton::Secondary,
                ..
            } => {
                self.looking = true;
                out.consumed = true;
                out.commands.push(ViewportCommand::GrabPointer(true));
                out.commands
                    .push(ViewportCommand::SetCursor(CursorKind::Hidden));
            }
            ViewportInput::PointerUp {
                button: PointerButton::Secondary,
                ..
            } if self.looking => {
                self.looking = false;
                self.clear_keys();
                out.consumed = true;
                out.commands.push(ViewportCommand::GrabPointer(false));
                out.commands
                    .push(ViewportCommand::SetCursor(CursorKind::Default));
            }
            ViewportInput::PointerMove { delta, .. } if self.looking => {
                out.consumed = true;
                if delta.is_finite()
                    && let Projection::Perspective3D { rotation, .. } = &mut camera.projection
                {
                    let yaw = -delta.x * self.look_sensitivity;
                    let pitch = -delta.y * self.look_sensitivity;
                    let right = rotation.rotate(Vec3::X);
                    let q_yaw = Quat::from_axis_angle(self.up_axis, yaw);
                    let q_pitch = Quat::from_axis_angle(right, pitch);
                    *rotation = q_yaw.mul_quat(q_pitch).mul_quat(*rotation);
                    out.camera_changed = true;
                    out.commands.push(ViewportCommand::RequestRedraw);
                }
            }
            ViewportInput::Scroll { delta, .. } if self.looking => {
                out.consumed = true;
                if delta.y.is_finite() {
                    self.speed = (self.speed * 1.15_f64.powf(delta.y)).clamp(0.05, 100_000.0);
                }
            }
            ViewportInput::KeyDown { key, .. } if self.looking => {
                if self.set_key(*key, true) {
                    out.consumed = true;
                    out.commands.push(ViewportCommand::RequestRedraw);
                }
            }
            ViewportInput::KeyUp { key, .. } => {
                if self.set_key(*key, false) && self.looking {
                    out.consumed = true;
                }
            }
            ViewportInput::FocusLost if self.looking => {
                self.looking = false;
                self.clear_keys();
                out.commands.push(ViewportCommand::GrabPointer(false));
                out.commands
                    .push(ViewportCommand::SetCursor(CursorKind::Default));
            }
            _ => {}
        }
        out
    }

    fn tick(&mut self, dt_secs: f64, camera: &mut Camera) -> bool {
        if !self.looking || !dt_secs.is_finite() || dt_secs <= 0.0 {
            return false;
        }
        let fwd = i32::from(self.move_fwd) - i32::from(self.move_back);
        let right = i32::from(self.move_right) - i32::from(self.move_left);
        let up = i32::from(self.move_up) - i32::from(self.move_down);
        if fwd == 0 && right == 0 && up == 0 {
            return false;
        }
        let dir = camera.forward_3d() * f64::from(fwd)
            + camera.right_3d() * f64::from(right)
            + self.up_axis * f64::from(up);
        let Some(unit_dir) = dir.normalized() else {
            return false;
        };
        let mult = if self.boost { 3.0 } else { 1.0 };
        let step = unit_dir * (self.speed * mult * dt_secs);
        if let Projection::Perspective3D { position, .. } = &mut camera.projection {
            *position += step;
            return true;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::Modifiers;

    #[test]
    fn cad_2d_rig_pans_and_zooms() {
        let mut rig = Cad2DRig::default();
        let mut cam = Camera::ortho_2d(Vec2::ZERO, 2.0);
        let size = Vec2::new(800.0, 600.0);

        let down = rig.handle_input(
            &ViewportInput::PointerDown {
                pos: Vec2::new(400.0, 300.0),
                button: PointerButton::Middle,
                modifiers: Modifiers::default(),
            },
            &mut cam,
            size,
        );
        assert!(down.consumed);

        let mv = rig.handle_input(
            &ViewportInput::PointerMove {
                pos: Vec2::new(420.0, 300.0),
                delta: Vec2::new(20.0, 0.0),
                modifiers: Modifiers::default(),
            },
            &mut cam,
            size,
        );
        assert!(mv.camera_changed);
        let Projection::Ortho2D { center, .. } = cam.projection else {
            panic!("expected Ortho2D");
        };
        assert!(center.approx_eq(Vec2::new(-10.0, 0.0), 1e-12));

        let scroll = rig.handle_input(
            &ViewportInput::Scroll {
                pos: Vec2::new(400.0, 300.0),
                delta: Vec2::new(0.0, 1.0),
                unit: ScrollUnit::Line,
                modifiers: Modifiers::default(),
            },
            &mut cam,
            size,
        );
        assert!(scroll.camera_changed);
    }

    #[test]
    fn orbit_rig_preserves_distance_to_pivot() {
        let pivot = Vec3::new(100.0, 200.0, 0.0);
        let mut rig = OrbitRig::new(pivot, Vec3::Z);
        let mut cam = Camera::perspective_3d(
            pivot + Vec3::new(0.0, -50.0, 20.0),
            pivot,
            Vec3::Z,
            std::f64::consts::FRAC_PI_3,
            0.05,
        );
        let initial_dist = (cam.position_3d() - pivot).length();
        let size = Vec2::new(1000.0, 800.0);

        rig.handle_input(
            &ViewportInput::PointerDown {
                pos: Vec2::new(500.0, 400.0),
                button: PointerButton::Middle,
                modifiers: Modifiers::default(),
            },
            &mut cam,
            size,
        );
        let mv = rig.handle_input(
            &ViewportInput::PointerMove {
                pos: Vec2::new(620.0, 340.0),
                delta: Vec2::new(120.0, -60.0),
                modifiers: Modifiers::default(),
            },
            &mut cam,
            size,
        );
        assert!(mv.camera_changed);
        let new_dist = (cam.position_3d() - pivot).length();
        assert!((new_dist - initial_dist).abs() < 1e-9);
    }

    #[test]
    fn fly_rig_moves_forward_when_w_held() {
        let mut rig = FlyRig::default();
        let mut cam = Camera::perspective_3d(
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::Z,
            std::f64::consts::FRAC_PI_3,
            0.05,
        );
        let size = Vec2::new(800.0, 600.0);
        rig.handle_input(
            &ViewportInput::PointerDown {
                pos: Vec2::ZERO,
                button: PointerButton::Secondary,
                modifiers: Modifiers::default(),
            },
            &mut cam,
            size,
        );
        rig.handle_input(
            &ViewportInput::KeyDown {
                key: KeyCode::W,
                modifiers: Modifiers::default(),
            },
            &mut cam,
            size,
        );
        assert!(rig.tick(0.5, &mut cam));
        assert!(cam.position_3d().approx_eq(Vec3::new(0.0, 5.0, 0.0), 1e-9));
    }
}
