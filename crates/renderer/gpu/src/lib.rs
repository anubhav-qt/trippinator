//! # GPU Abstraction Layer
//!
//! Provides the core GPU primitives: uniform buffers, storage buffers,
//! render targets, texture management, and shader module helpers.

use bytemuck::Pod;
use wgpu::{Device, Queue, Buffer, BufferUsages};

/// A GPU uniform buffer that can be updated each frame.
pub struct UniformBuffer<T: Pod> {
    buffer: Buffer,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Pod> UniformBuffer<T> {
    pub fn new(device: &Device, label: &str) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: std::mem::size_of::<T>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn update(&self, queue: &Queue, data: &T) {
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(data));
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }
}

/// A GPU render target (offscreen texture for feedback / multi-pass).
pub struct RenderTarget {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
}

impl RenderTarget {
    pub fn new(
        device: &Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        label: &str,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            texture,
            view,
            width,
            height,
        }
    }
}

/// A fullscreen quad vertex for post-processing passes.
#[derive(Debug, Clone, Copy, Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct FullscreenVertex {
    pub position: [f32; 2],
    pub uv: [f32; 2],
}

impl FullscreenVertex {
    /// The 6 vertices for a fullscreen triangle-strip quad.
    pub const VERTICES: &[Self] = &[
        Self { position: [-1.0, -1.0], uv: [0.0, 1.0] },
        Self { position: [1.0, -1.0], uv: [1.0, 1.0] },
        Self { position: [-1.0, 1.0], uv: [0.0, 0.0] },
        Self { position: [1.0, 1.0], uv: [1.0, 0.0] },
    ];

    pub const INDICES: &[u16] = &[0, 1, 2, 2, 1, 3];

    pub fn buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}
