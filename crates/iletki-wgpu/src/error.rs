//! Error types returned by `iletki-wgpu`.

use std::fmt;

/// Errors that can occur during GPU viewport operations, picking or readback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ViewportError {
    /// The color attachment format cannot be used by the viewport.
    UnsupportedColorFormat(wgpu::TextureFormat),
    /// The depth attachment format is unavailable or has no depth component.
    UnsupportedDepthFormat(wgpu::TextureFormat),
    /// The attachment format does not support the requested multisampling count.
    UnsupportedSampleCount {
        /// Format of the attachment.
        format: wgpu::TextureFormat,
        /// Requested number of samples per pixel.
        sample_count: u32,
    },
    /// Requested region or viewport has zero width or height.
    ZeroSize,
    /// Mapping a GPU staging buffer for CPU readback failed.
    BufferMapFailed,
    /// The readback texture format is not supported by [`crate::ViewportHost::readback_rgba_blocking`].
    UnsupportedReadbackFormat,
    /// Requested readback or pick region exceeds safe byte limits.
    RegionTooLarge,
    /// This pick request's result has already been consumed.
    PickAlreadyRead,
}

impl fmt::Display for ViewportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedColorFormat(format) => {
                write!(f, "unsupported viewport color format {format:?}")
            }
            Self::UnsupportedDepthFormat(format) => {
                write!(f, "unsupported viewport depth format {format:?}")
            }
            Self::UnsupportedSampleCount {
                format,
                sample_count,
            } => write!(
                f,
                "{format:?} does not support {sample_count} samples for this viewport"
            ),
            Self::ZeroSize => f.write_str("viewport or pick region has zero width or height"),
            Self::BufferMapFailed => f.write_str("failed to map GPU staging buffer for readback"),
            Self::UnsupportedReadbackFormat => {
                f.write_str("readback requires an 8-bit 4-channel RGBA or BGRA texture format")
            }
            Self::RegionTooLarge => f.write_str("requested GPU readback region exceeds size limit"),
            Self::PickAlreadyRead => f.write_str("pick result has already been read"),
        }
    }
}

impl std::error::Error for ViewportError {}
