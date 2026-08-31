//! GPU engine — manages wgpu device, audio capture, screen perception, state dynamics, and the render loop.

use anyhow::Result;
use std::sync::Arc;
use std::time::Instant;
use trippinator_analysis_audio::DspEngine;
use trippinator_analysis_screen::ScreenAnalyzer;
use trippinator_capture_audio::{AudioCapture, DeviceSelection};
use trippinator_capture_screen::{ScreenCapture, DEFAULT_CAPTURE_HEIGHT, DEFAULT_CAPTURE_WIDTH};
use trippinator_dynamics_state::{Observations, StateDynamics};
use trippinator_systems_waves::WaveSystem;
use wgpu::{
    Device, Instance, InstanceDescriptor, Queue, Surface, SurfaceConfiguration, TextureUsages,
};
use winit::window::Window;

/// Core GPU engine. Owns the wgpu device, perception streams, dynamics, and visual systems.
pub struct Engine {
    _window: Arc<Window>,
    surface: Surface<'static>,
    device: Device,
    queue: Queue,
    config: SurfaceConfiguration,

    // Audio Perception
    audio_capture: Option<AudioCapture>,
    dsp_engine: DspEngine,
    audio_sample_rate: u32,

    // Screen Perception (captured on main thread)
    screen_capture: Option<ScreenCapture>,
    screen_analyzer: ScreenAnalyzer,
    smoothed_screen_color: [f32; 4],
    smoothed_screen_motion: f32,
    last_screen_capture_time: Instant,

    // Dynamics
    state_dynamics: StateDynamics,

    // Visual Systems
    wave_system: WaveSystem,

    // Timing & Metrics
    start_time: Instant,
    last_frame_time: Instant,
    frame_count: u64,
}

impl Engine {
    /// Create a new engine bound to the given window.
    pub fn new(window: Arc<Window>) -> Result<Self> {
        let size = window.inner_size();
        log::info!("Initializing Trippinator engine for {}x{} display", size.width, size.height);

        let instance = Instance::new(&InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        // SAFETY: The surface is tied to `_window` (Arc<Window>) which lives as long as the Engine.
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
        log::info!("GPU Adapter: {} ({:?})", info.name, info.backend);

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
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);

        // Prioritize Mailbox (uncapped, low latency for 165Hz)
        let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
            log::info!("Present mode: Mailbox (low-latency uncapped)");
            wgpu::PresentMode::Mailbox
        } else {
            log::info!("Present mode: AutoVsync");
            wgpu::PresentMode::AutoVsync
        };

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

        // 1. Audio Loopback Capture
        let audio_capture = match AudioCapture::start(DeviceSelection::Default) {
            Ok(capture) => {
                log::info!("System audio loopback capture active");
                Some(capture)
            }
            Err(e) => {
                log::warn!("Could not start audio capture ({e}). Running in autonomous mode.");
                None
            }
        };

        // 2. Audio DSP Engine (2048 FFT)
        let dsp_engine = DspEngine::new(2048);

        // 3. Screen Capture & Analyzer
        let screen_capture = match ScreenCapture::new(DEFAULT_CAPTURE_WIDTH, DEFAULT_CAPTURE_HEIGHT) {
            Ok(cap) => {
                log::info!("Desktop screen perception initialized (128x72)");
                Some(cap)
            }
            Err(e) => {
                log::warn!("Could not initialize screen capture ({e}). Using default palette.");
                None
            }
        };

        let screen_analyzer = ScreenAnalyzer::new(DEFAULT_CAPTURE_WIDTH, DEFAULT_CAPTURE_HEIGHT);

        // 4. Dynamic State Engine
        let state_dynamics = StateDynamics::new();

        // 5. Wave Visual System
        let wave_system = WaveSystem::new(&device, format)?;

        let now = Instant::now();

        log::info!(
            "Trippinator Engine Ready: {}x{} @ {:?}",
            config.width, config.height, config.present_mode
        );

        Ok(Self {
            _window: window,
            surface,
            device,
            queue,
            config,
            audio_capture,
            dsp_engine,
            audio_sample_rate: 48000,
            screen_capture,
            screen_analyzer,
            smoothed_screen_color: [0.2, 0.4, 0.8, 0.5],
            smoothed_screen_motion: 0.0,
            last_screen_capture_time: now,
            state_dynamics,
            wave_system,
            start_time: now,
            last_frame_time: now,
            frame_count: 0,
        })
    }

    /// Handle window resize.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            log::info!("Surface resized to {width}x{height}");
        }
    }

    /// Render a single frame.
    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let now = Instant::now();
        let dt = (now - self.last_frame_time).as_secs_f32().clamp(0.0001, 0.1);
        self.last_frame_time = now;
        let time = (now - self.start_time).as_secs_f32();

        // 1. Drain Audio Samples & Update DSP
        if let Some(audio_cap) = &self.audio_capture {
            while let Some(buf) = audio_cap.try_recv() {
                self.audio_sample_rate = buf.sample_rate;
                self.dsp_engine.push_samples(&buf.samples, buf.channels);
            }
        }

        let audio_features = self.dsp_engine.process(self.audio_sample_rate, dt);

        // 2. Process Desktop Screen Frames (~30-60 Hz on main thread)
        let mut screen_brightness = 0.5f32;
        let mut screen_saturation = 0.5f32;
        let mut screen_contrast = 0.5f32;
        let mut screen_dominant_hue = 210.0f32;

        if (now - self.last_screen_capture_time).as_millis() >= 25 {
            self.last_screen_capture_time = now;
            if let Some(screen_cap) = &mut self.screen_capture {
                match screen_cap.capture() {
                    Ok(frame) => {
                        let features = self.screen_analyzer.analyze(&frame);
                        screen_brightness = features.avg_brightness;
                        screen_saturation = features.avg_saturation;
                        screen_contrast = features.contrast;
                        screen_dominant_hue = features.dominant_hue;

                        let color_alpha = 1.0 - (-dt / 0.4).exp();
                        self.smoothed_screen_color[0] += color_alpha * (features.avg_rgb[0] - self.smoothed_screen_color[0]);
                        self.smoothed_screen_color[1] += color_alpha * (features.avg_rgb[1] - self.smoothed_screen_color[1]);
                        self.smoothed_screen_color[2] += color_alpha * (features.avg_rgb[2] - self.smoothed_screen_color[2]);
                        self.smoothed_screen_color[3] += color_alpha * (features.avg_saturation - self.smoothed_screen_color[3]);

                        let motion_alpha = 1.0 - (-dt / 0.08).exp();
                        self.smoothed_screen_motion += motion_alpha * (features.motion_intensity - self.smoothed_screen_motion);
                    }
                    Err(e) => {
                        log::debug!("Main-thread screen capture tick error: {e}");
                    }
                }
            }
        }

        // 3. Evolve Dynamic State
        let obs = Observations {
            audio_energy: audio_features.instant.rms,
            audio_energy_velocity: audio_features.short_term.energy_velocity,
            audio_attack: audio_features.short_term.attack,
            audio_bass: audio_features.bands.energy[0] * 0.5 + audio_features.bands.energy[1] * 0.5,
            audio_mid: audio_features.bands.energy[2] * 0.3 + audio_features.bands.energy[3] * 0.4 + audio_features.bands.energy[4] * 0.3,
            audio_treble: audio_features.bands.energy[5] * 0.5 + audio_features.bands.energy[6] * 0.5,
            audio_spectral_centroid: audio_features.instant.spectral_centroid,
            audio_spectral_flux: audio_features.instant.spectral_flux,
            audio_volatility: audio_features.long_term.volatility,
            audio_zcr: audio_features.instant.zcr,

            screen_brightness,
            screen_saturation,
            screen_contrast,
            screen_motion: self.smoothed_screen_motion,
            screen_dominant_hue,
        };

        self.state_dynamics.update(&obs, dt);
        let global_state = *self.state_dynamics.state();

        let flow_phase = self.state_dynamics.dynamic_flow_phase;
        let color_phase = self.state_dynamics.dynamic_color_phase;
        let harmonic_phase = self.state_dynamics.dynamic_harmonic_phase;

        // 4. Update Visual Systems
        self.wave_system.update(
            &self.queue,
            &global_state,
            &audio_features,
            self.smoothed_screen_color,
            flow_phase,
            color_phase,
            harmonic_phase,
            dt,
        );

        // 5. GPU Render Pass
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("trippinator-command-encoder"),
            });

        // Render Wave System to surface view
        self.wave_system.render(&mut encoder, &view);

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        self.frame_count += 1;

        // Diagnostic log every 5 seconds
        if self.frame_count % 800 == 0 {
            let fps = self.frame_count as f32 / time;
            log::info!(
                "FPS: {:.1} | Energy: {:.2} | Bass: {:.2} | Mid: {:.2} | Centroid: {:.2} | Chaos: {:.2}",
                fps,
                global_state.energy,
                audio_features.bands.energy[1],
                audio_features.bands.energy[3],
                audio_features.instant.spectral_centroid,
                global_state.chaos
            );
        }

        Ok(())
    }
}
