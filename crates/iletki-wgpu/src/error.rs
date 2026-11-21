//! Error types returned by `iletki-wgpu`.

use std::fmt;

/// Errors that can occur during GPU viewport operations, picking or readback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ViewportError {
    /// Requested region or viewport has zero width or height.
    ZeroSize,
    /// Mapping a GPU staging buffer for CPU readback failed.
    BufferMapFailed,
    /// The readback texture format is not supported by [`crate::ViewportHost::readback_rgba_blocking`].
    UnsupportedReadbackFormat,
    /// Requested readback or pick region exceeds safe byte limits.
    RegionTooLarge,
}

impl fmt::Display for ViewportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSize => f.write_str("viewport or pick region has zero width or height"),
            Self::BufferMapFailed => f.write_str("failed to map GPU staging buffer for readback"),
            Self::UnsupportedReadbackFormat => {
                f.write_str("readback requires an 8-bit 4-channel RGBA or BGRA texture format")
            }
            Self::RegionTooLarge => f.write_str("requested GPU readback region exceeds size limit"),
        }
    }
}

impl std::error::Error for ViewportError {}
