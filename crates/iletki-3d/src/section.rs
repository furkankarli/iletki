//! Shader variants for antialiased section cuts.

use iletki_wgpu::PrepareCtx;
use iletki_wgpu::wgpu;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum SectionAa {
    Sampled,
    Raster,
}

impl SectionAa {
    pub(crate) fn for_viewport<W>(ctx: &PrepareCtx<'_, W>) -> Option<Self> {
        ctx.supports_section_antialiasing().then(|| {
            if ctx
                .device
                .features()
                .contains(wgpu::Features::CLIP_DISTANCES)
            {
                Self::Raster
            } else {
                Self::Sampled
            }
        })
    }

    pub(crate) fn shader(
        self,
        device: &wgpu::Device,
        base: &str,
        sampled: &str,
        raster: &str,
    ) -> wgpu::ShaderModule {
        let source = match self {
            Self::Sampled => format!(
                "{base}\n{}\n{sampled}",
                include_str!("shaders/sampled_section.wgsl")
            ),
            Self::Raster => format!(
                "enable clip_distances;\n{base}\n{}\n{raster}",
                include_str!("shaders/raster_section.wgsl")
            ),
        };
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("iletki-3d antialiased section shader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        })
    }
}
