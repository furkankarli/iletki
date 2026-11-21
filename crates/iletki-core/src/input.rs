//! UI-agnostic pointer, wheel, gesture, pen and keyboard inputs and host commands.

use crate::math::Vec2;

/// Keyboard modifier state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Modifiers {
    /// Shift key pressed.
    pub shift: bool,
    /// Control key pressed.
    pub ctrl: bool,
    /// Alt / Option key pressed.
    pub alt: bool,
    /// Command / Super / Meta key pressed.
    pub meta: bool,
}

/// Mouse or pointer button.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PointerButton {
    /// Primary (usually left) button.
    Primary,
    /// Secondary (usually right) button.
    Secondary,
    /// Middle (wheel click) button.
    Middle,
    /// Extra auxiliary button (e.g. Back/Forward).
    Auxiliary(u16),
}

/// Unit of a scroll event delta.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScrollUnit {
    /// Discrete wheel ticks / lines (typical mouse wheel).
    Line,
    /// Smooth pixel/point deltas (typical trackpad).
    Pixel,
}

/// Navigation-relevant keyboard keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyCode {
    /// W key (fly forward).
    W,
    /// A key (fly left).
    A,
    /// S key (fly backward).
    S,
    /// D key (fly right).
    D,
    /// Q key (fly down).
    Q,
    /// E key (fly up).
    E,
    /// Space bar.
    Space,
    /// Escape key.
    Escape,
    /// Shift modifier key.
    Shift,
    /// Control modifier key.
    Control,
    /// Alt modifier key.
    Alt,
    /// Any other application-specific key.
    Other(u32),
}

/// Cursor appearance requested by the viewport or an active rig/gizmo.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum CursorKind {
    /// Default arrow cursor.
    #[default]
    Default,
    /// Precision drafting crosshair.
    Crosshair,
    /// Open hand (ready to pan).
    Grab,
    /// Closed hand (actively panning/orbiting).
    Grabbing,
    /// Four-way move cursor.
    Move,
    /// Pointer / clickable handle.
    Pointer,
    /// Hidden cursor (during pointer-locked 3D fly/orbit).
    Hidden,
}

/// Input event forwarded from any UI framework or window host into a viewport.
///
/// Positions are in viewport-local logical points with `(0, 0)` at the top-left corner and `+Y`
/// pointing down.
#[derive(Debug, Clone, PartialEq)]
pub enum ViewportInput {
    /// Pointer moved to `pos` with relative displacement `delta` (in logical points).
    PointerMove {
        /// Current pointer position in viewport-local logical points.
        pos: Vec2,
        /// Incremental displacement since the previous event.
        delta: Vec2,
        /// Active keyboard modifiers.
        modifiers: Modifiers,
    },
    /// A pointer button was pressed.
    PointerDown {
        /// Pointer position at press time.
        pos: Vec2,
        /// Button pressed.
        button: PointerButton,
        /// Active keyboard modifiers.
        modifiers: Modifiers,
    },
    /// A pointer button was released.
    PointerUp {
        /// Pointer position at release time.
        pos: Vec2,
        /// Button released.
        button: PointerButton,
        /// Active keyboard modifiers.
        modifiers: Modifiers,
    },
    /// Mouse wheel or trackpad scroll event.
    Scroll {
        /// Pointer position during scroll.
        pos: Vec2,
        /// Scroll delta (`+y` zooms in / scrolls up).
        delta: Vec2,
        /// Whether `delta` is measured in wheel lines or smooth pixels.
        unit: ScrollUnit,
        /// Active keyboard modifiers.
        modifiers: Modifiers,
    },
    /// Trackpad or touchscreen pinch-to-zoom gesture.
    Pinch {
        /// Gesture focal point in viewport-local logical points.
        pos: Vec2,
        /// Multiplicative scale factor for this step (`> 1.0` zooms in).
        scale_factor: f64,
    },
    /// Keyboard key pressed.
    KeyDown {
        /// Key pressed.
        key: KeyCode,
        /// Active keyboard modifiers.
        modifiers: Modifiers,
    },
    /// Keyboard key released.
    KeyUp {
        /// Key released.
        key: KeyCode,
        /// Active keyboard modifiers.
        modifiers: Modifiers,
    },
    /// Stylus / graphics tablet sample.
    Pen {
        /// Pen position in viewport-local logical points.
        pos: Vec2,
        /// Normalized pressure in `[0.0, 1.0]`.
        pressure: f32,
        /// Pen tilt `[x_rad, y_rad]`.
        tilt: [f32; 2],
    },
    /// The viewport lost keyboard/pointer focus (cancel active drags/keys).
    FocusLost,
}

/// Commands emitted by a viewport or camera rig back to the hosting UI/window layer.
#[derive(Debug, Clone, PartialEq)]
pub enum ViewportCommand {
    /// Request the host to schedule a new frame.
    RequestRedraw,
    /// Ask the host to update the cursor shape.
    SetCursor(CursorKind),
    /// Ask the host to lock/capture or release the pointer (for continuous 3D orbit/fly).
    GrabPointer(bool),
    /// Ask the host to wrap the pointer to `pos` (in viewport-local logical points) when dragging
    /// past an edge.
    WrapPointer(Vec2),
}
