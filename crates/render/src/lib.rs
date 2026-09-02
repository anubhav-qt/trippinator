//! # Feedback Render Pipeline
//!
//! Two GPU passes over a ping-ponged pair of HDR textures:
//!
//! 1. **feedback** — samples the previous frame through a warp (5 selectable variants),
//!    decays it, and adds new audio-driven injection (4 selectable variants).
//! 2. **color** — palette maps, tone maps, and dithers the result to the swapchain.
//!
//! Inject and warp are combined into one pass rather than three, because injection needs
//! the same domain-warped coordinate space as the feedback sample to look coherent — doing
//! them as separate render-to-texture passes would just add a texture round-trip with no
//! benefit. "One pipeline with swappable stages," not three data-dependent pipelines.
//!
//! Structural variety comes from two independent axes (see DESIGN.md):
//! - `warp_mode` / `inject_mode` / `symmetry` — discrete, key-switchable "kind" of look.
//! - spectral balance (bass/mid/treble weights) — continuous, always-on "genre" of look.

use bytemuck::{Pod, Zeroable};
use trippinator_analysis_audio::AudioFeatures;
use trippinator_renderer_gpu::{RenderTarget, UniformBuffer};
use wgpu::{Device, Queue, TextureFormat};

const FEEDBACK_SHADER_SRC: &str = concat!(
    include_str!("../../../shaders/common/fullscreen_quad.wgsl"),
    include_str!("../../../shaders/render/feedback.wgsl"),
);
const COLOR_SHADER_SRC: &str = concat!(
    include_str!("../../../shaders/common/fullscreen_quad.wgsl"),
    include_str!("../../../shaders/render/color.wgsl"),
);

const HDR_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// Hand-switchable structural state (Phase 1: set by keyboard; Phase 2: driven by
/// Poincare-section crossings of the chaotic attractor).
///
/// The warp is always the kaleidoscope fold and all four injection layers render
/// together, so symmetry order is the only structural degree of freedom left here.
#[derive(Debug, Clone, Copy)]
pub struct StructuralState {
    /// Kaleidoscope fold order.
    pub symmetry: u32,
}
impl Default for StructuralState {
    fn default() -> Self {
        Self { symmetry: 9 }
    }
}

/// GPU-side uniform block. Every field here is either a per-frame reading or a
/// hand-tuned safe-range constant — see DESIGN.md's safety contract. No chaotic dynamics
/// yet; those land in Phase 2 as additional fields, not a replacement of this struct.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    resolution: [f32; 2],
    time: f32,
    dt: f32,

    /// 7 bands used, 8th is padding for 16-byte alignment.
    band_energy: [f32; 8],

    rms: f32,
    energy_velocity: f32,
    attack: f32,
    decay: f32,

    /// Normalized spectral balance — see DESIGN.md "Spectral balance".
    bass_w: f32,
    mid_w: f32,
    treble_w: f32,
    spectral_centroid: f32,

    symmetry: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,

    // Hand-tuned safe-range constants (Phase 1: static; Phase 2: slow drift).
    zoom: f32,
    /// Per-frame retention, derived from a half-life and `dt` — NOT a fixed constant.
    /// A fixed per-frame value makes brightness depend on frame rate and drives the
    /// loop to `inject / (1 - decay)`, which saturates the tonemapper to white.
    feedback_decay: f32,
    hue_shift: f32,
    /// 0..1 "this is a big moment" envelope, from energy vs. its own rolling baseline.
    grandness: f32,

    /// Injection is energy-per-second; the shader multiplies by `dt`.
    inject_gain: f32,
    exposure: f32,
    _pad3: f32,
    _pad4: f32,
}

/// Same curve as WGSL/GLSL `smoothstep`.
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Owns the ping-pong HDR targets and the two render pipelines.
pub struct Renderer {
    width: u32,
    height: u32,
    time: f32,
    hue_shift: f32,
    /// Smoothed "big moment" envelope — fast to rise, slow to fall, so a peak sustains
    /// instead of flickering frame to frame.
    grandness_env: f32,

    /// Live-tunable overall injection brightness, bound to keys so it can be dialed in
    /// against real music. Worth noting the tuned value came out *higher* than the
    /// single-layer default, not lower — predicting it from the equilibrium arithmetic
    /// alone was wrong, because the four layers barely overlap in practice.
    inject_gain: f32,
    /// Trail length, in seconds to fade to half brightness.
    feedback_half_life: f32,

    ping: RenderTarget,
    pong: RenderTarget,
    /// True when `ping` holds the most recent frame (the one to sample from next).
    ping_is_latest: bool,

    sampler: wgpu::Sampler,
    uniforms: UniformBuffer<Uniforms>,

    feedback_pipeline: wgpu::RenderPipeline,
    feedback_bind_layout: wgpu::BindGroupLayout,
    color_pipeline: wgpu::RenderPipeline,
    color_bind_layout: wgpu::BindGroupLayout,

    quad_vbuf: wgpu::Buffer,
}

/// position (xy) + uv, matching `shaders/common/fullscreen_quad.wgsl`'s `VertexInput`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct QuadVertex {
    position: [f32; 2],
    uv: [f32; 2],
}

const QUAD_VERTICES: [QuadVertex; 4] = [
    QuadVertex { position: [-1.0, -1.0], uv: [0.0, 1.0] },
    QuadVertex { position: [1.0, -1.0], uv: [1.0, 1.0] },
    QuadVertex { position: [-1.0, 1.0], uv: [0.0, 0.0] },
    QuadVertex { position: [1.0, 1.0], uv: [1.0, 0.0] },
];

impl Renderer {
    pub fn new(device: &Device, width: u32, height: u32, output_format: TextureFormat) -> Self {
        use wgpu::util::DeviceExt;

        let ping = RenderTarget::new(device, width, height, HDR_FORMAT, "feedback-ping");
        let pong = RenderTarget::new(device, width, height, HDR_FORMAT, "feedback-pong");
        let sampler = RenderTarget::create_linear_sampler(device);
        let uniforms = UniformBuffer::<Uniforms>::new(device, "visual-uniforms");

        let quad_vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fullscreen-quad"),
            contents: bytemuck::bytes_of(&QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<QuadVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2],
        };

        let (feedback_pipeline, feedback_bind_layout) = build_pipeline(
            device,
            "feedback",
            FEEDBACK_SHADER_SRC,
            HDR_FORMAT,
            &vertex_layout,
        );
        let (color_pipeline, color_bind_layout) = build_pipeline(
            device,
            "color",
            COLOR_SHADER_SRC,
            output_format,
            &vertex_layout,
        );

        Self {
            width,
            height,
            time: 0.0,
            hue_shift: 0.0,
            grandness_env: 0.0,
            // Tuned by ear against real music with all four layers running. Notably
            // ~2x higher than the value predicted from equilibrium alone: the layers
            // overlap far less than worst-case, so the loop sits well below saturation
            // and needs more drive than the arithmetic suggested.
            inject_gain: 3.43,
            feedback_half_life: 0.37,
            ping,
            pong,
            ping_is_latest: true,
            sampler,
            uniforms,
            feedback_pipeline,
            feedback_bind_layout,
            color_pipeline,
            color_bind_layout,
            quad_vbuf,
        }
    }

    /// Current "big moment" envelope (0..1), for logging/tuning.
    pub fn grandness(&self) -> f32 {
        self.grandness_env
    }

    /// Scale overall injected brightness. Clamped to a range that cannot go black or
    /// blow the tonemapper out.
    pub fn adjust_inject_gain(&mut self, factor: f32) {
        // Ceiling raised from 6.0 after tuning ran into it while exploring.
        self.inject_gain = (self.inject_gain * factor).clamp(0.15, 12.0);
        log::info!("inject_gain -> {:.2}", self.inject_gain);
    }

    /// Scale trail length (seconds to half brightness).
    pub fn adjust_trail(&mut self, factor: f32) {
        self.feedback_half_life = (self.feedback_half_life * factor).clamp(0.05, 4.0);
        log::info!("trail half-life -> {:.2}s", self.feedback_half_life);
    }

    pub fn resize(&mut self, device: &Device, width: u32, height: u32) {
        if width == 0 || height == 0 || (width == self.width && height == self.height) {
            return;
        }
        self.width = width;
        self.height = height;
        self.ping = RenderTarget::new(device, width, height, HDR_FORMAT, "feedback-ping");
        self.pong = RenderTarget::new(device, width, height, HDR_FORMAT, "feedback-pong");
        self.ping_is_latest = true;
    }

    /// Render one frame: feedback pass into the back buffer, then color pass to `target`.
    pub fn render(
        &mut self,
        device: &Device,
        queue: &Queue,
        target: &wgpu::TextureView,
        dt: f32,
        audio: &AudioFeatures,
        structural: StructuralState,
    ) {
        self.time += dt;
        // Slow, always-drifting hue independent of any single track — see DESIGN.md
        // "timescale hierarchy." ~90s per full rotation.
        self.hue_shift = (self.hue_shift + dt / 90.0) % 1.0;

        let bands = &audio.bands.energy;
        let bass_raw = bands[0] + bands[1];
        let mid_raw = bands[2] + bands[3] + bands[4];
        let treble_raw = bands[5] + bands[6];
        let total = (bass_raw + mid_raw + treble_raw).max(1e-4);

        let mut band_energy = [0.0f32; 8];
        band_energy[..7].copy_from_slice(bands);

        // Frame-rate-independent trail retention. Over `feedback_half_life` seconds the
        // image fades to half, whether we are running at 60 or 178 fps.
        let feedback_decay = (-dt * std::f32::consts::LN_2 / self.feedback_half_life).exp();

        // "Grand moment" detection: loud *relative to this track's own recent baseline*,
        // and loud in absolute terms. Both gates matter — the ratio alone fires on every
        // note in a quiet passage, the absolute level alone fires on all of a loud track.
        let baseline = audio.long_term.avg_energy.max(1e-3);
        let ratio = audio.instant.rms / baseline;
        let target_grand =
            smoothstep(1.5, 2.8, ratio) * smoothstep(0.04, 0.12, audio.instant.rms);
        let tau = if target_grand > self.grandness_env { 0.4 } else { 2.5 };
        self.grandness_env += (1.0 - (-dt / tau).exp()) * (target_grand - self.grandness_env);

        let uniform_data = Uniforms {
            resolution: [self.width as f32, self.height as f32],
            time: self.time,
            dt,
            band_energy,
            rms: audio.instant.rms,
            energy_velocity: audio.short_term.energy_velocity,
            attack: audio.short_term.attack,
            decay: audio.short_term.decay,
            bass_w: bass_raw / total,
            mid_w: mid_raw / total,
            treble_w: treble_raw / total,
            spectral_centroid: audio.instant.spectral_centroid,
            symmetry: structural.symmetry.max(2),
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            zoom: 1.008,
            feedback_decay,
            hue_shift: self.hue_shift,
            grandness: self.grandness_env,
            inject_gain: self.inject_gain,
            exposure: 1.0,
            _pad3: 0.0,
            _pad4: 0.0,
        };
        self.uniforms.update(queue, &uniform_data);

        let (src, dst) = if self.ping_is_latest {
            (&self.ping, &self.pong)
        } else {
            (&self.pong, &self.ping)
        };

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render-frame"),
        });

        // Pass 1: feedback (warp + decay + inject) — src -> dst.
        {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("feedback-bind-group"),
                layout: &self.feedback_bind_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.uniforms.buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&src.view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("feedback-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dst.view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.feedback_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_vertex_buffer(0, self.quad_vbuf.slice(..));
            pass.draw(0..4, 0..1);
        }

        // Pass 2: color (palette + tonemap + dither) — dst -> swapchain.
        {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("color-bind-group"),
                layout: &self.color_bind_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.uniforms.buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&dst.view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("color-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.color_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_vertex_buffer(0, self.quad_vbuf.slice(..));
            pass.draw(0..4, 0..1);
        }

        queue.submit(Some(encoder.finish()));
        self.ping_is_latest = !self.ping_is_latest;
    }
}

fn build_pipeline(
    device: &Device,
    label: &str,
    shader_src: &str,
    target_format: TextureFormat,
    vertex_layout: &wgpu::VertexBufferLayout,
) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(&format!("{label}-bind-layout")),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{label}-pipeline-layout")),
        bind_group_layouts: &[&bind_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[vertex_layout.clone()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    (pipeline, bind_layout)
}
