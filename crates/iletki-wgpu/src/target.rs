//! Over-allocated offscreen MSAA, depth, static-cache and picking render targets.

use crate::ViewportError;
use crate::blit::BlitResources;
use iletki_core::{BlockSizing, OutlineConfig, TargetSize, UploadBudget, ViewportMode};

/// Standard depth format used across all `iletki-wgpu` viewports (32-bit Reversed-Z float).
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Optional depth-stencil format lazily allocated only when a layer requests stencil support
/// (for example, GPU stencil section caps).
pub const DEPTH_STENCIL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32FloatStencil8;
/// Standard GPU picking ID texture format (64-bit packed [`iletki_core::PickId`]).
pub const PICK_ID_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Uint;
/// Compact 1-based 32-bit slot ID attachment format (`Rg16Uint`, guaranteed 4× MSAA in WebGPU core)
/// for color-frame selection previews.
pub const SELECTION_ID_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Uint;

/// Configuration for creating a [`crate::ViewportHost`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewportConfig {
    /// Filterable, blendable floating-point or normalized color attachment with render, sampling
    /// and copy-source usage. For portable RGBA/BGRA output, use the 8-bit unorm or sRGB formats.
    /// The default is `Rgba8UnormSrgb`.
    pub color_format: wgpu::TextureFormat,
    /// Multisample anti-aliasing count (`1` or `4`; default `4`).
    pub msaa_samples: u32,
    /// Background clear color `[r, g, b, a]`. RGB is interpreted as sRGB when the target is sRGB.
    pub clear_color: [f64; 4],
    /// Initial scheduling mode (`OnDemand` for CAD/editors, `Continuous` for games/animation).
    pub mode: ViewportMode,
    /// Per-frame GPU upload budget passed to layers during `prepare`.
    pub upload_budget: UploadBudget,
    /// Visual configuration for the Jump-Flooding selection outline pass.
    pub outline: OutlineConfig,
    /// Initial physical pixel extent `[width, height]`.
    pub initial_px: [u32; 2],
    /// Initial logical-to-physical DPI scale.
    pub dpi_scale: f32,
}

impl Default for ViewportConfig {
    fn default() -> Self {
        Self {
            color_format: wgpu::TextureFormat::Rgba8UnormSrgb,
            msaa_samples: 4,
            clear_color: [0.09, 0.09, 0.10, 1.0],
            mode: ViewportMode::OnDemand,
            upload_budget: UploadBudget::default(),
            outline: OutlineConfig::default(),
            initial_px: [256, 256],
            dpi_scale: 1.0,
        }
    }
}

impl ViewportConfig {
    pub(crate) fn validate(&self, device: &wgpu::Device) -> Result<(), ViewportError> {
        use wgpu::{TextureFormatFeatureFlags as Flags, TextureUsages as Usage};

        let features = color_attachment_features(device, self.color_format, true)?;
        if !features
            .allowed_usages
            .contains(Usage::TEXTURE_BINDING | Usage::COPY_SRC)
            || !features.flags.contains(Flags::FILTERABLE)
        {
            return Err(ViewportError::UnsupportedColorFormat(self.color_format));
        }
        validate_samples(self.color_format, features.flags, self.msaa_samples)?;
        if self.msaa_samples > 1 && !features.flags.contains(Flags::MULTISAMPLE_RESOLVE) {
            return Err(ViewportError::UnsupportedSampleCount {
                format: self.color_format,
                sample_count: self.msaa_samples,
            });
        }
        validate_depth_attachment(device, DEPTH_FORMAT, self.msaa_samples)
    }

    /// Returns `wgpu::Color` from sRGB `clear_color`, converting to linear when targeting an sRGB
    /// texture format so colors stay perceptually identical across linear and sRGB targets.
    pub(crate) fn wgpu_clear_color(&self) -> wgpu::Color {
        let [r, g, b, a] = self
            .clear_color
            .map(|channel| if channel.is_finite() { channel } else { 0.0 });
        let a = a.clamp(0.0, 1.0);
        if self.color_format.is_srgb() {
            let lin = |c: f64| {
                let c = c.clamp(0.0, 1.0);
                if c <= 0.04045 {
                    c / 12.92
                } else {
                    ((c + 0.055) / 1.055).powf(2.4)
                }
            };
            wgpu::Color {
                r: lin(r),
                g: lin(g),
                b: lin(b),
                a,
            }
        } else {
            wgpu::Color { r, g, b, a }
        }
    }
}

pub(crate) fn color_attachment_features(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    blend: bool,
) -> Result<wgpu::TextureFormatFeatures, ViewportError> {
    let enabled = device.features();
    let features = format.guaranteed_format_features(enabled);
    let fits_color_limit = format
        .target_pixel_byte_cost()
        .is_some_and(|bytes| bytes <= device.limits().max_color_attachment_bytes_per_sample);
    if enabled.contains(format.required_features())
        && fits_color_limit
        && matches!(
            format.sample_type(None, Some(enabled)),
            Some(wgpu::TextureSampleType::Float { .. })
        )
        && features
            .allowed_usages
            .contains(wgpu::TextureUsages::RENDER_ATTACHMENT)
        && (!blend
            || features
                .flags
                .contains(wgpu::TextureFormatFeatureFlags::BLENDABLE))
    {
        Ok(features)
    } else {
        Err(ViewportError::UnsupportedColorFormat(format))
    }
}

pub(crate) fn validate_samples(
    format: wgpu::TextureFormat,
    flags: wgpu::TextureFormatFeatureFlags,
    sample_count: u32,
) -> Result<(), ViewportError> {
    if matches!(sample_count, 1 | 4) && flags.sample_count_supported(sample_count) {
        Ok(())
    } else {
        Err(ViewportError::UnsupportedSampleCount {
            format,
            sample_count,
        })
    }
}

pub(crate) fn validate_depth_attachment(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    sample_count: u32,
) -> Result<(), ViewportError> {
    let enabled = device.features();
    let features = format.guaranteed_format_features(enabled);
    if !format.has_depth_aspect()
        || !enabled.contains(format.required_features())
        || !features
            .allowed_usages
            .contains(wgpu::TextureUsages::RENDER_ATTACHMENT)
    {
        return Err(ViewportError::UnsupportedDepthFormat(format));
    }
    validate_samples(format, features.flags, sample_count)
}

/// Backing textures for a single viewport instance.
pub(crate) struct TargetSet {
    sizing: BlockSizing,
    format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
    msaa_samples: u32,
    depth_sampleable: bool,
    can_copy_output: bool,
    msaa_view: Option<wgpu::TextureView>,
    depth_view: wgpu::TextureView,
    depth_sample_view: Option<wgpu::TextureView>,
    static_texture: wgpu::Texture,
    static_view: wgpu::TextureView,
    static_identity_bg: wgpu::BindGroup,
    static_crop_bg: wgpu::BindGroup,
    output_texture: wgpu::Texture,
    output_view: wgpu::TextureView,
    output_crop_bg: wgpu::BindGroup,
}

impl std::fmt::Debug for TargetSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TargetSet")
            .field("sizing", &self.sizing.current())
            .field("format", &self.format)
            .field("depth_format", &self.depth_format)
            .field("msaa_samples", &self.msaa_samples)
            .finish()
    }
}

fn allocate_depth(
    device: &wgpu::Device,
    size: wgpu::Extent3d,
    depth_format: wgpu::TextureFormat,
    msaa_samples: u32,
) -> (wgpu::TextureView, Option<wgpu::TextureView>, bool) {
    let depth_features = depth_format.guaranteed_format_features(device.features());
    let depth_sampleable = device.adapter_info().backend != wgpu::Backend::Gl
        && depth_features
            .allowed_usages
            .contains(wgpu::TextureUsages::TEXTURE_BINDING)
        && depth_features.flags.sample_count_supported(msaa_samples);
    let depth_usage = if depth_sampleable {
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING
    } else {
        wgpu::TextureUsages::RENDER_ATTACHMENT
    };
    let label = if depth_format.has_stencil_aspect() {
        "iletki depth32 stencil8"
    } else {
        "iletki depth32"
    };
    let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: msaa_samples,
        dimension: wgpu::TextureDimension::D2,
        format: depth_format,
        usage: depth_usage,
        view_formats: &[],
    });
    let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let depth_sample_view = depth_format.has_stencil_aspect().then(|| {
        depth_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("iletki depth32 sample view"),
            aspect: wgpu::TextureAspect::DepthOnly,
            ..wgpu::TextureViewDescriptor::default()
        })
    });
    (depth_view, depth_sample_view, depth_sampleable)
}

impl TargetSet {
    pub(crate) fn new(
        device: &wgpu::Device,
        blit: &BlitResources,
        initial_px: [u32; 2],
        format: wgpu::TextureFormat,
        msaa_samples: u32,
    ) -> Self {
        let sizing =
            BlockSizing::with_max_dimension(initial_px, device.limits().max_texture_dimension_2d);
        Self::allocate(device, blit, sizing, format, DEPTH_FORMAT, msaa_samples)
    }

    /// Updates active size and reallocates backing textures only when crossing a 256 px block or
    /// after shrink hysteresis. Returns `true` if textures were reallocated.
    pub(crate) fn resize(
        &mut self,
        device: &wgpu::Device,
        blit: &BlitResources,
        requested_px: [u32; 2],
    ) -> bool {
        if !self.sizing.request(requested_px) {
            return false;
        }
        *self = Self::allocate(
            device,
            blit,
            self.sizing,
            self.format,
            self.depth_format,
            self.msaa_samples,
        );
        true
    }

    /// Lazily switches the depth attachment format (for example `Depth32Float` ↔
    /// `Depth32FloatStencil8`), reallocating only the depth texture when the format changes.
    pub(crate) fn set_depth_format(
        &mut self,
        device: &wgpu::Device,
        depth_format: wgpu::TextureFormat,
    ) -> bool {
        if self.depth_format == depth_format {
            return false;
        }
        let [width, height] = self.sizing.current().allocated_px;
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let (depth_view, depth_sample_view, depth_sampleable) =
            allocate_depth(device, size, depth_format, self.msaa_samples);
        self.depth_format = depth_format;
        self.depth_view = depth_view;
        self.depth_sample_view = depth_sample_view;
        self.depth_sampleable = depth_sampleable;
        true
    }

    fn allocate(
        device: &wgpu::Device,
        blit: &BlitResources,
        sizing: BlockSizing,
        format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        msaa_samples: u32,
    ) -> Self {
        let [width, height] = sizing.current().allocated_px;
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let msaa_view = (msaa_samples > 1).then(|| {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("iletki msaa color"),
                size,
                mip_level_count: 1,
                sample_count: msaa_samples,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            tex.create_view(&wgpu::TextureViewDescriptor::default())
        });

        let (depth_view, depth_sample_view, depth_sampleable) =
            allocate_depth(device, size, depth_format, msaa_samples);

        let single_usage = wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC;
        let can_copy_output = format
            .guaranteed_format_features(device.features())
            .allowed_usages
            .contains(wgpu::TextureUsages::COPY_DST);
        let output_usage = if can_copy_output {
            single_usage | wgpu::TextureUsages::COPY_DST
        } else {
            single_usage
        };

        let static_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("iletki static cache"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: single_usage,
            view_formats: &[],
        });
        let static_view = static_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let static_identity_bg = blit.create_identity_bind_group(device, &static_view);
        let static_crop_bg = blit.create_crop_bind_group(device, &static_view);

        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("iletki output"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: output_usage,
            view_formats: &[],
        });
        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let output_crop_bg = blit.create_crop_bind_group(device, &output_view);

        Self {
            sizing,
            format,
            depth_format,
            msaa_samples,
            depth_sampleable,
            can_copy_output,
            msaa_view,
            depth_view,
            depth_sample_view,
            static_texture,
            static_view,
            static_identity_bg,
            static_crop_bg,
            output_texture,
            output_view,
            output_crop_bg,
        }
    }

    pub(crate) fn size(&self) -> TargetSize {
        self.sizing.current()
    }

    pub(crate) fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    pub(crate) fn depth_format(&self) -> wgpu::TextureFormat {
        self.depth_format
    }

    pub(crate) fn msaa_samples(&self) -> u32 {
        self.msaa_samples
    }

    pub(crate) fn depth_view(&self) -> &wgpu::TextureView {
        &self.depth_view
    }

    pub(crate) fn depth_sample_view(&self) -> &wgpu::TextureView {
        self.depth_sample_view.as_ref().unwrap_or(&self.depth_view)
    }

    pub(crate) const fn depth_sampleable(&self) -> bool {
        self.depth_sampleable
    }

    pub(crate) const fn can_copy_output(&self) -> bool {
        self.can_copy_output
    }

    /// Returns `(color_attachment_view, resolve_target)` targeting the static cache texture.
    pub(crate) fn static_color_views(&self) -> (&wgpu::TextureView, Option<&wgpu::TextureView>) {
        match &self.msaa_view {
            Some(msaa) => (msaa, Some(&self.static_view)),
            None => (&self.static_view, None),
        }
    }

    /// Returns `(color_attachment_view, resolve_target)` targeting the final output texture.
    pub(crate) fn output_color_views(&self) -> (&wgpu::TextureView, Option<&wgpu::TextureView>) {
        match &self.msaa_view {
            Some(msaa) => (msaa, Some(&self.output_view)),
            None => (&self.output_view, None),
        }
    }

    pub(crate) fn static_texture(&self) -> &wgpu::Texture {
        &self.static_texture
    }

    pub(crate) fn static_view(&self) -> &wgpu::TextureView {
        &self.static_view
    }

    pub(crate) fn static_identity_bg(&self) -> &wgpu::BindGroup {
        &self.static_identity_bg
    }

    pub(crate) fn static_crop_bg(&self) -> &wgpu::BindGroup {
        &self.static_crop_bg
    }

    pub(crate) fn output_texture(&self) -> &wgpu::Texture {
        &self.output_texture
    }

    pub(crate) fn output_view(&self) -> &wgpu::TextureView {
        &self.output_view
    }

    pub(crate) fn output_crop_bg(&self) -> &wgpu::BindGroup {
        &self.output_crop_bg
    }
}

/// Lazily allocated 1x picking targets (`Rg32Uint` ID + `Depth32Float`) for on-demand GPU picking.
pub(crate) struct PickTargets {
    extent: [u32; 2],
    id_texture: wgpu::Texture,
    id_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
}

impl std::fmt::Debug for PickTargets {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PickTargets")
            .field("extent", &self.extent)
            .finish()
    }
}

impl PickTargets {
    pub(crate) fn ensure<'a>(
        slot: &'a mut Option<Self>,
        device: &wgpu::Device,
        allocated_px: [u32; 2],
    ) -> &'a Self {
        let needs_alloc = slot.as_ref().is_none_or(|p| p.extent != allocated_px);
        if needs_alloc {
            *slot = None;
        }
        slot.get_or_insert_with(|| {
            let [width, height] = allocated_px;
            let size = wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            };
            let id_texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("iletki pick id"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: PICK_ID_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let id_view = id_texture.create_view(&wgpu::TextureViewDescriptor::default());
            let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("iletki pick depth"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
            Self {
                extent: allocated_px,
                id_texture,
                id_view,
                depth_view,
            }
        })
    }

    pub(crate) fn id_texture(&self) -> &wgpu::Texture {
        &self.id_texture
    }

    pub(crate) fn id_view(&self) -> &wgpu::TextureView {
        &self.id_view
    }

    pub(crate) fn depth_view(&self) -> &wgpu::TextureView {
        &self.depth_view
    }
}

/// Lazily allocated `Rg16Uint` MRT attachment matching the color pass sample count for active
/// selection previews.
pub(crate) struct SelectionPreviewTargets {
    extent: [u32; 2],
    msaa_samples: u32,
    id_view: wgpu::TextureView,
}

impl std::fmt::Debug for SelectionPreviewTargets {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelectionPreviewTargets")
            .field("extent", &self.extent)
            .field("msaa_samples", &self.msaa_samples)
            .finish()
    }
}

impl SelectionPreviewTargets {
    pub(crate) fn ensure<'a>(
        slot: &'a mut Option<Self>,
        device: &wgpu::Device,
        allocated_px: [u32; 2],
        msaa_samples: u32,
    ) -> &'a Self {
        let needs_alloc = slot
            .as_ref()
            .is_none_or(|t| t.extent != allocated_px || t.msaa_samples != msaa_samples);
        if needs_alloc {
            *slot = None;
        }
        slot.get_or_insert_with(|| {
            let [width, height] = allocated_px;
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("iletki selection preview slot ids"),
                size: wgpu::Extent3d {
                    width: width.max(1),
                    height: height.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: msaa_samples,
                dimension: wgpu::TextureDimension::D2,
                format: SELECTION_ID_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let id_view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            Self {
                extent: allocated_px,
                msaa_samples,
                id_view,
            }
        })
    }

    pub(crate) fn id_view(&self) -> &wgpu::TextureView {
        &self.id_view
    }
}
