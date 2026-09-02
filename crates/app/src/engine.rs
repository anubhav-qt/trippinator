//! GPU engine — owns the wgpu device/surface, the audio loopback, and the feedback
//! render pipeline.
//!
//! v1 is audio-only (see DESIGN.md). Screen capture/analysis are v2 scope and are not
//! wired in here.

use anyhow::Result;
use std::sync::Arc;
use std::time::Instant;
use trippinator_analysis_audio::{AudioFeatures, DspEngine};
use trippinator_capture_audio::{AudioCapture, DeviceSelection, DEFAULT_FFT_SIZE};
use trippinator_render::{Renderer, StructuralState};
use wgpu::{
    Device, Instance, InstanceDescriptor, Queue, Surface, SurfaceConfiguration, TextureUsages,
};
use winit::window::Window;

/// Core engine. Owns the GPU context, audio perception, and the visual pipeline.
pub struct Engine {
    _window: Arc<Window>,
    surface: Surface<'static>,
    device: Device,
    queue: Queue,
    config: SurfaceConfiguration,

    // Perception
    audio_capture: Option<AudioCapture>,
    dsp: DspEngine,
    audio_sample_rate: u32,
    audio: AudioFeatures,

    // Visuals
    renderer: Renderer,
    structural: StructuralState,

    // Timing
    last_frame: Instant,
    last_report: Instant,
    frames_since_report: u32,
}

impl Engine {
    /// Create a new engine bound to the given window.
    pub fn new(window: Arc<Window>) -> Result<Self> {
        let size = window.inner_size();
        log::info!("Initializing engine for {}x{}", size.width, size.height);

        let instance = Instance::new(&InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        // SAFETY: the surface is tied to `_window` (Arc<Window>), which outlives it.
        let surface = unsafe {
            let target = wgpu::SurfaceTargetUnsafe::from_window(&*window)?;
            instance.create_surface_unsafe(target)?
        };

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| anyhow::anyhow!("No suitable GPU adapter found"))?;

        let info = adapter.get_info();
        log::info!("GPU adapter: {} ({:?})", info.name, info.backend);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("trippinator-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            },
            None,
        ))?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
            wgpu::PresentMode::Mailbox
        } else {
            wgpu::PresentMode::AutoVsync
        };
        log::info!("Surface format {format:?}, present mode {present_mode:?}");

        let config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 1,
        };
        surface.configure(&device, &config);

        // Audio loopback.
        let audio_capture = match AudioCapture::start(DeviceSelection::Default) {
            Ok(c) => Some(c),
            Err(e) => {
                log::warn!("Audio capture unavailable ({e}) — running silent");
                None
            }
        };

        let renderer = Renderer::new(&device, config.width, config.height, format);

        let now = Instant::now();
        Ok(Self {
            _window: window,
            surface,
            device,
            queue,
            config,
            audio_capture,
            dsp: DspEngine::new(DEFAULT_FFT_SIZE),
            audio_sample_rate: 48_000,
            audio: AudioFeatures::default(),
            renderer,
            structural: StructuralState::default(),
            last_frame: now,
            last_report: now,
            frames_since_report: 0,
        })
    }

    /// Reconfigure the surface (and the render targets) after a window resize.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.renderer.resize(&self.device, width, height);
        log::info!("Surface resized to {width}x{height}");
    }

    /// Scale overall injected brightness. Bound to `-` / `=`.
    pub fn adjust_inject_gain(&mut self, factor: f32) {
        self.renderer.adjust_inject_gain(factor);
    }

    /// Scale trail length. Bound to `,` / `.`.
    pub fn adjust_trail(&mut self, factor: f32) {
        self.renderer.adjust_trail(factor);
    }

    /// Nudge the kaleidoscope symmetry order.
    pub fn adjust_organic(&mut self, delta: f32) {
        self.renderer.adjust_organic(delta);
    }

    pub fn adjust_core_scale(&mut self, factor: f32) {
        self.renderer.adjust_core_scale(factor);
    }

    pub fn nudge_symmetry(&mut self, delta: i32) {
        let current = self.structural.symmetry as i32;
        self.structural.symmetry = (current + delta).clamp(3, 12) as u32;
        log::info!("Symmetry -> {}", self.structural.symmetry);
    }

    /// Pull the latest audio into `self.audio`.
    fn sense(&mut self, dt: f32) {
        if let Some(capture) = &self.audio_capture {
            while let Some(buf) = capture.try_recv() {
                self.audio_sample_rate = buf.sample_rate;
                self.dsp.push_samples(&buf.samples, buf.channels);
            }
        }
        self.audio = self.dsp.process(self.audio_sample_rate, dt);
    }

    /// Advance one frame and present.
    pub fn render(&mut self) -> Result<()> {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;

        self.sense(dt);

        let frame = self.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.renderer.render(
            &self.device,
            &self.queue,
            &view,
            dt,
            &self.audio,
            self.structural,
        );

        frame.present();

        self.report(now);
        Ok(())
    }

    /// Log FPS and raw sensor levels once a second, so we can see the inputs are alive.
    fn report(&mut self, now: Instant) {
        self.frames_since_report += 1;
        let elapsed = (now - self.last_report).as_secs_f32();
        if elapsed < 1.0 {
            return;
        }

        let fps = self.frames_since_report as f32 / elapsed;
        log::info!(
            "{fps:.0} fps | rms {:.3} | bass {:.3} | mid {:.3} | treble {:.3} | grand {:.2} | organic {:.2} | core {:.2}",
            self.audio.instant.rms,
            self.audio.bands.energy[1],
            self.audio.bands.energy[3],
            self.audio.bands.energy[5],
            self.renderer.grandness(),
            self.renderer.organic(),
            self.renderer.core_scale(),
        );

        self.last_report = now;
        self.frames_since_report = 0;
    }
}
