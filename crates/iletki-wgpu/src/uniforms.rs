//! Shared GPU camera uniform buffer and WGSL helper source (`@group(0) @binding(0)`).

use bytemuck::{Pod, Zeroable};
use iletki_core::{CameraUniformData, MAX_CLIP_PLANES};
use wgpu::util::DeviceExt;

/// Reusable WGSL source declaring `CameraUniforms` at `@group(0) @binding(0)` and projection/clip
/// helpers (`iletki_to_clip_2d`, `iletki_project_3d`, `iletki_is_clipped`).
pub const CAMERA_WGSL: &str = include_str!("shaders/camera.wgsl");

/// GPU-uploadable camera uniform struct matching [`CAMERA_WGSL`].
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq)]
pub struct CameraUniformsGpu {
    /// Camera-relative Reversed-Z view-projection matrix.
    pub view_proj_rel: [[f32; 4]; 4],
    /// Inverse of `view_proj_rel`.
    pub inv_view_proj_rel: [[f32; 4]; 4],
    /// High `f32` part of 2D camera center.
    pub camera_hi: [f32; 2],
    /// Low `f32` remainder of 2D camera center.
    pub camera_lo: [f32; 2],
    /// 2D world-to-clip scale.
    pub scale_2d: [f32; 2],
    /// Clip-space units per logical point.
    pub point_size: [f32; 2],
    /// Physical viewport extent in pixels.
    pub viewport_px: [f32; 2],
    /// Logical-to-physical DPI scale.
    pub dpi_scale: f32,
    /// Bit 0: `is_perspective`, bit 1: `is_3d`, bit 2: `reversed_z`.
    pub flags: u32,
    /// Camera-relative section clip planes `[nx, ny, nz, d_rel]`.
    pub clip_planes: [[f32; 4]; MAX_CLIP_PLANES],
    /// Active clip plane count (`0..=6`).
    pub clip_plane_count: u32,
    /// Padding to 16-byte WGSL uniform alignment.
    pub _pad: [u32; 3],
}

impl From<CameraUniformData> for CameraUniformsGpu {
    fn from(d: CameraUniformData) -> Self {
        Self {
            view_proj_rel: d.view_proj_rel,
            inv_view_proj_rel: d.inv_view_proj_rel,
            camera_hi: d.camera_hi,
            camera_lo: d.camera_lo,
            scale_2d: d.scale_2d,
            point_size: d.point_size,
            viewport_px: d.viewport_px,
            dpi_scale: d.dpi_scale,
            flags: d.flags,
            clip_planes: d.clip_planes,
            clip_plane_count: d.clip_plane_count,
            _pad: d._pad,
        }
    }
}

/// GPU buffer and bind group for the viewport's camera uniforms (`@group(0)`).
pub(crate) struct CameraGpu {
    buffer: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

impl CameraGpu {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("iletki camera uniforms"),
            contents: bytemuck::bytes_of(&CameraUniformsGpu::zeroed()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("iletki camera layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("iletki camera bind group"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        Self {
            buffer,
            layout,
            bind_group,
        }
    }

    pub(crate) fn write(&self, queue: &wgpu::Queue, data: CameraUniformData) {
        let gpu = CameraUniformsGpu::from(data);
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&gpu));
    }

    pub(crate) fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    pub(crate) fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_struct_is_16_byte_aligned_for_wgsl() {
        assert_eq!(std::mem::size_of::<CameraUniformsGpu>() % 16, 0);
        // 2 * 64 (mat4) + 5 * 8 (vec2) + 8 (dpi+flags) + 6 * 16 (clip) + 16 (count+pad) = 288
        assert_eq!(std::mem::size_of::<CameraUniformsGpu>(), 288);
    }
}
