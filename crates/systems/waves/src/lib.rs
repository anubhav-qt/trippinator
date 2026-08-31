//! # Wave Systems
//!
//! Oscillatory visual primitives: harmonic interference, standing waves,
//! domain-warped fields, and state-driven dynamic phase synthesis.

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use trippinator_analysis_audio::AudioFeatures;
use trippinator_dynamics_state::GlobalState;
use trippinator_renderer_gpu::UniformBuffer;
use wgpu::{
    BindGroup, ColorTargetState, ColorWrites, Device, FragmentState, MultisampleState,
    PipelineCompilationOptions, PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology, Queue,
    RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor,
    ShaderModuleDescriptor, ShaderSource, TextureFormat, VertexState,
};

/// 128-byte uniform struct uploaded to the wave shader.
/// Exactly 8 x vec4<f32> for clean WGSL alignment.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct AudioVisualUniforms {
    // GlobalState (12 floats = 48 bytes)
    pub energy: f32,
    pub tension: f32,
    pub chaos: f32,
    pub coherence: f32,
    pub turbulence: f32,
    pub density: f32,
    pub flow: f32,
    pub compression: f32,
    pub expansion: f32,
    pub entropy: f32,
    pub chroma: f32,
    pub temporal_memory: f32,

    // Audio frequency bands (8 floats = 32 bytes)
    pub sub_bass: f32,
    pub bass: f32,
    pub low_mid: f32,
    pub mid: f32,
    pub high_mid: f32,
    pub treble: f32,
    pub brilliance: f32,
    pub rms: f32,

    // Audio dynamics & spectral character (4 floats = 16 bytes)
    pub attack: f32,
    pub decay: f32,
    pub energy_velocity: f32,
    pub spectral_centroid: f32,

    // Screen color & palette (4 floats = 16 bytes)
    pub screen_r: f32,
    pub screen_g: f32,
    pub screen_b: f32,
    pub screen_saturation: f32,

    // Dynamic Integrated Phases & Viewport (4 floats = 16 bytes)
    pub flow_phase: f32,
    pub color_phase: f32,
    pub harmonic_phase: f32,
    pub dt: f32,
}

impl Default for AudioVisualUniforms {
    fn default() -> Self {
        Self {
            energy: 0.1,
            tension: 0.0,
            chaos: 0.05,
            coherence: 0.8,
            turbulence: 0.05,
            density: 0.3,
            flow: 0.2,
            compression: 0.0,
            expansion: 0.3,
            entropy: 0.1,
            chroma: 0.5,
            temporal_memory: 0.7,

            sub_bass: 0.0,
            bass: 0.0,
            low_mid: 0.0,
            mid: 0.0,
            high_mid: 0.0,
            treble: 0.0,
            brilliance: 0.0,
            rms: 0.0,

            attack: 0.0,
            decay: 0.0,
            energy_velocity: 0.0,
            spectral_centroid: 0.3,

            screen_r: 0.2,
            screen_g: 0.4,
            screen_b: 0.8,
            screen_saturation: 0.5,

            flow_phase: 0.0,
            color_phase: 0.0,
            harmonic_phase: 0.0,
            dt: 1.0 / 60.0,
        }
    }
}

/// The GPU wave rendering system.
pub struct WaveSystem {
    pipeline: RenderPipeline,
    bind_group: BindGroup,
    uniform_buffer: UniformBuffer<AudioVisualUniforms>,
    uniforms: AudioVisualUniforms,
}

impl WaveSystem {
    /// Create a new wave system targeting the given surface texture format.
    pub fn new(device: &Device, format: TextureFormat) -> Result<Self> {
        let shader_source = include_str!("../../../../shaders/waves/wave.wgsl");

        let shader_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("wave-shader-module"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("wave-bind-group-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("wave-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("wave-render-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader_module,
                entry_point: Some("vs_main"),
                buffers: &[], // Vertex-less fullscreen triangle
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(FragmentState {
                module: &shader_module,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: PipelineCompilationOptions::default(),
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let uniforms = AudioVisualUniforms::default();
        let uniform_buffer = UniformBuffer::new(device, "wave-uniform-buffer");

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wave-bind-group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.buffer().as_entire_binding(),
            }],
        });

        Ok(Self {
            pipeline,
            bind_group,
            uniform_buffer,
            uniforms,
        })
    }

    /// Update uniforms and upload to GPU.
    pub fn update(
        &mut self,
        queue: &Queue,
        state: &GlobalState,
        audio: &AudioFeatures,
        screen_color: [f32; 4],
        flow_phase: f32,
        color_phase: f32,
        harmonic_phase: f32,
        dt: f32,
    ) {
        self.uniforms.energy = state.energy;
        self.uniforms.tension = state.tension;
        self.uniforms.chaos = state.chaos;
        self.uniforms.coherence = state.coherence;
        self.uniforms.turbulence = state.turbulence;
        self.uniforms.density = state.density;
        self.uniforms.flow = state.flow;
        self.uniforms.compression = state.compression;
        self.uniforms.expansion = state.expansion;
        self.uniforms.entropy = state.entropy;
        self.uniforms.chroma = state.chroma;
        self.uniforms.temporal_memory = state.temporal_memory;

        self.uniforms.sub_bass = audio.bands.energy[0];
        self.uniforms.bass = audio.bands.energy[1];
        self.uniforms.low_mid = audio.bands.energy[2];
        self.uniforms.mid = audio.bands.energy[3];
        self.uniforms.high_mid = audio.bands.energy[4];
        self.uniforms.treble = audio.bands.energy[5];
        self.uniforms.brilliance = audio.bands.energy[6];
        self.uniforms.rms = audio.instant.rms;

        self.uniforms.attack = audio.short_term.attack;
        self.uniforms.decay = audio.short_term.decay;
        self.uniforms.energy_velocity = audio.short_term.energy_velocity;
        self.uniforms.spectral_centroid = audio.instant.spectral_centroid;

        self.uniforms.screen_r = screen_color[0];
        self.uniforms.screen_g = screen_color[1];
        self.uniforms.screen_b = screen_color[2];
        self.uniforms.screen_saturation = screen_color[3];

        self.uniforms.flow_phase = flow_phase;
        self.uniforms.color_phase = color_phase;
        self.uniforms.harmonic_phase = harmonic_phase;
        self.uniforms.dt = dt;

        self.uniform_buffer.update(queue, &self.uniforms);
    }

    /// Render wave visual system directly to the given texture view.
    pub fn render(&self, encoder: &mut wgpu::CommandEncoder, target_view: &wgpu::TextureView) {
        let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("wave-render-pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..3, 0..1); // 3 vertices for fullscreen triangle
    }

    pub fn uniforms(&self) -> &AudioVisualUniforms {
        &self.uniforms
    }
}
