//! # GPU Abstraction Layer
//!
//! Small, reusable wgpu primitives: typed uniform buffers and offscreen
//! render targets. Nothing here knows what we are drawing.

use bytemuck::{Pod, Zeroable};
use wgpu::{Buffer, BufferUsages, Device, Queue};

/// A GPU uniform buffer that can be updated each frame.
pub struct UniformBuffer<T: Pod + Zeroable> {
    buffer: Buffer,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Pod + Zeroable> UniformBuffer<T> {
    /// Create a new uniform buffer initialized with default data.
    pub fn new(device: &Device, label: &str) -> Self {
        use wgpu::util::DeviceExt;
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(&T::zeroed()),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        Self {
            buffer,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Upload new data to the GPU buffer.
    pub fn update(&self, queue: &Queue, data: &T) {
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(data));
    }

    /// Get a reference to the underlying wgpu buffer.
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }
}

/// Helper for managing offscreen HDR textures and sampler states.
pub struct RenderTarget {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub format: wgpu::TextureFormat,
    pub width: u32,
    pub height: u32,
}

impl RenderTarget {
    pub fn new(device: &Device, width: u32, height: u32, format: wgpu::TextureFormat, label: &str) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            texture,
            view,
            format,
            width,
            height,
        }
    }

    pub fn create_linear_sampler(device: &Device) -> wgpu::Sampler {
        device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        })
    }
}
