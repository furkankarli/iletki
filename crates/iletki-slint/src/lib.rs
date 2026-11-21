//! Integrates an iletki viewport with the `wgpu 30` renderer in a Slint application.
//!
//! Create a [`SlintViewport`] with the device and queue from `GraphicsAPI::WGPU30`.
//! In `BeforeRendering`, resize it to the Slint viewport's physical extent and render with
//! `slint::Image::try_from` as the importer. Bind [`SlintFrame::source_size`] to the image
//! element's source clip so it samples the active region of the padded texture.
//! The application owns the Slint rendering notifier, UI properties and pointer events.

use iletki_core::Vec2;
use iletki_wgpu::{
    FrameStats, PendingPick, ViewportConfig, ViewportError, ViewportHost, ViewportLayer, wgpu,
};

/// A rendered Slint viewport frame.
#[derive(Debug)]
pub struct SlintFrame<Image> {
    /// The replacement Slint image when the backing texture changes. Retain the existing image
    /// in the UI when this is `None`.
    pub new_image: Option<Image>,
    /// Active width and height in source pixels for `source-clip-width` and `source-clip-height`.
    pub source_size: [i32; 2],
    /// Statistics and follow-up redraw information from iletki.
    pub stats: FrameStats,
}

/// Owns an iletki viewport and tracks the resolved texture currently imported by Slint.
#[derive(Debug)]
pub struct SlintViewport {
    host: ViewportHost,
    device: wgpu::Device,
    queue: wgpu::Queue,
    imported_texture: Option<wgpu::Texture>,
    widget_px: [u32; 2],
    scale_factor: f32,
}

impl SlintViewport {
    /// Creates a viewport on the same device and queue that Slint uses.
    ///
    /// The color format must be `Rgba8Unorm` or `Rgba8UnormSrgb`, the formats accepted by the
    /// Slint `wgpu 30` image importer.
    ///
    /// # Errors
    ///
    /// Returns a [`ViewportError`] if the color format or GPU configuration is unsupported.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: ViewportConfig,
    ) -> Result<Self, ViewportError> {
        if !matches!(
            config.color_format,
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb
        ) {
            return Err(ViewportError::UnsupportedColorFormat(config.color_format));
        }
        Ok(Self {
            host: ViewportHost::new(device, config)?,
            device: device.clone(),
            queue: queue.clone(),
            imported_texture: None,
            widget_px: config.initial_px,
            scale_factor: valid_scale(config.dpi_scale).unwrap_or(1.0),
        })
    }

    /// Provides access to the camera and viewport state.
    #[must_use]
    pub const fn host(&self) -> &ViewportHost {
        &self.host
    }

    /// Provides mutable access for camera updates, input routing and scene invalidation.
    /// Call [`Self::resize`] for size and DPI changes so picking uses the same widget extent.
    pub const fn host_mut(&mut self) -> &mut ViewportHost {
        &mut self.host
    }

    /// Updates the physical size and DPI of the Slint image element.
    /// Use the element's size when the image occupies only part of the window.
    pub fn resize(&mut self, physical_px: [u32; 2], scale_factor: f32) {
        self.widget_px = physical_px;
        if let Some(scale) = valid_scale(scale_factor) {
            self.scale_factor = scale;
        }
        self.host
            .resize(&self.device, physical_px, self.scale_factor);
    }

    /// Renders on Slint's GPU queue and imports the texture only when its handle changes.
    /// Call with `slint::Image::try_from` inside the window's `BeforeRendering` notifier.
    /// Skip rendering when the image element has zero area.
    ///
    /// # Errors
    ///
    /// Returns the application importer's error. A failed import is retried on the next render.
    pub fn render<W, Image, E>(
        &mut self,
        world: &W,
        layers: &mut [&mut dyn ViewportLayer<W>],
        import_image: impl FnOnce(wgpu::Texture) -> Result<Image, E>,
    ) -> Result<SlintFrame<Image>, E> {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let frame =
            self.host
                .render_to_texture(&self.device, &self.queue, &mut encoder, world, layers);
        self.queue.submit([encoder.finish()]);
        let texture = frame.texture_view.texture();
        let new_image = if self.imported_texture.as_ref() == Some(texture) {
            None
        } else {
            let image = import_image(texture.clone())?;
            self.imported_texture = Some(texture.clone());
            Some(image)
        };
        Ok(SlintFrame {
            new_image,
            source_size: frame
                .active_px
                .map(|px| i32::try_from(px).unwrap_or(i32::MAX)),
            stats: frame.stats,
        })
    }

    /// Starts reading the object at a widget-local Slint point in logical pixels.
    ///
    /// Returns `Ok(None)` when the point falls outside the widget. Poll the request with
    /// [`PendingPick::try_read`] or await it according to the iletki picking API.
    ///
    /// # Errors
    ///
    /// Returns a [`ViewportError`] if the GPU pick cannot be submitted.
    pub fn begin_pick<W>(
        &mut self,
        world: &W,
        layers: &mut [&mut dyn ViewportLayer<W>],
        point: Vec2,
    ) -> Result<Option<PendingPick>, ViewportError> {
        let [widget_w, widget_h] = self.widget_px;
        let [active_w, active_h] = self.host.size().active_px;
        let scale = f64::from(self.scale_factor);
        let Some((x, y)) = texel_at(point.x, scale, widget_w, active_w)
            .zip(texel_at(point.y, scale, widget_h, active_h))
        else {
            return Ok(None);
        };
        self.host
            .begin_pick_rect(&self.device, &self.queue, world, layers, [x, y, 1, 1])
            .map(Some)
    }
}

fn valid_scale(scale: f32) -> Option<f32> {
    (scale.is_finite() && scale > 0.0).then_some(scale)
}

fn texel_at(point: f64, scale: f64, widget: u32, active: u32) -> Option<u32> {
    let physical = point * scale;
    let extent = f64::from(widget);
    if !(0.0..extent).contains(&physical) {
        return None;
    }
    let texel = (physical / extent * f64::from(active)) as u32;
    (texel < active).then_some(texel)
}

#[cfg(test)]
mod tests {
    use super::texel_at;

    #[test]
    fn picking_maps_high_dpi_and_device_clamped_image_extents() {
        assert_eq!(texel_at(100.0, 2.0, 400, 400), Some(200));
        assert_eq!(texel_at(100.0, 2.0, 400, 256), Some(128));
        assert_eq!(texel_at(199.9, 2.0, 400, 256), Some(255));
        assert_eq!(texel_at(200.0, 2.0, 400, 256), None);
        assert_eq!(texel_at(-0.1, 2.0, 400, 256), None);
        assert_eq!(texel_at(f64::NAN, 2.0, 400, 256), None);
        assert_eq!(texel_at(f64::INFINITY, 2.0, 400, 256), None);
        assert_eq!(texel_at(0.0, 2.0, 0, 1), None);
    }
}
