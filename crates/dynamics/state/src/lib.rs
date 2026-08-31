//! # Dynamic State
//!
//! The central nervous system of trippinator.
//!
//! Input does not control the image. Input perturbs a system that controls itself.
//!
//! The state evolves via continuous coupled differential equations:
//! ```text
//! x_{t+1} = x_t + dt * f(x_t, u_t)
//! ```
//! where u_t contains audio + screen observations.

use bytemuck::{Pod, Zeroable};

/// The global dynamic state of the organism.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct GlobalState {
    /// Overall energy level (0..1). Influenced by audio RMS and motion.
    pub energy: f32,
    /// Tension between opposing systems (0..1). High = visual conflict.
    pub tension: f32,
    /// Degree of disorder (0..1). Low = structured, high = chaotic.
    pub chaos: f32,
    /// How synchronized/aligned systems are (0..1).
    pub coherence: f32,
    /// Fluid motion intensity (0..1).
    pub turbulence: f32,
    /// Visual density / complexity (0..1).
    pub density: f32,
    /// Directional flow strength (0..1).
    pub flow: f32,
    /// Inward pressure (0..1).
    pub compression: f32,
    /// Outward pressure (0..1).
    pub expansion: f32,
    /// Information content / unpredictability (0..1).
    pub entropy: f32,
    /// Color intensity / saturation drive (0..1).
    pub chroma: f32,
    /// How much the system remembers vs forgets (0..1).
    pub temporal_memory: f32,
}

impl Default for GlobalState {
    fn default() -> Self {
        Self {
            energy: 0.08,
            tension: 0.0,
            chaos: 0.05,
            coherence: 0.85,
            turbulence: 0.05,
            density: 0.3,
            flow: 0.2,
            compression: 0.0,
            expansion: 0.2,
            entropy: 0.1,
            chroma: 0.5,
            temporal_memory: 0.8,
        }
    }
}

/// Observations from the perception layer that perturb the state.
#[derive(Debug, Clone, Default)]
pub struct Observations {
    // Audio Features
    pub audio_energy: f32,
    pub audio_energy_velocity: f32,
    pub audio_attack: f32,
    pub audio_bass: f32,
    pub audio_mid: f32,
    pub audio_treble: f32,
    pub audio_spectral_centroid: f32,
    pub audio_spectral_flux: f32,
    pub audio_volatility: f32,
    pub audio_zcr: f32,

    // Screen Features
    pub screen_brightness: f32,
    pub screen_saturation: f32,
    pub screen_contrast: f32,
    pub screen_motion: f32,
    pub screen_dominant_hue: f32,
}

/// The state dynamics engine.
///
/// Evolves GlobalState based on observations using coupled ODEs with dynamic phase integration.
pub struct StateDynamics {
    state: GlobalState,
    /// Asymmetric smoothing rates (fast rise on attack, organic relaxation).
    inertia: GlobalState,
    /// Integrated dynamic phase accumulators (evolve with metabolism instead of fixed clocks).
    pub dynamic_flow_phase: f32,
    pub dynamic_color_phase: f32,
    pub dynamic_harmonic_phase: f32,
}

impl StateDynamics {
    pub fn new() -> Self {
        Self {
            state: GlobalState::default(),
            inertia: GlobalState {
                energy: 0.82,
                tension: 0.88,
                chaos: 0.86,
                coherence: 0.90,
                turbulence: 0.84,
                density: 0.92,
                flow: 0.85,
                compression: 0.86,
                expansion: 0.86,
                entropy: 0.88,
                chroma: 0.90,
                temporal_memory: 0.94,
            },
            dynamic_flow_phase: 0.0,
            dynamic_color_phase: 0.0,
            dynamic_harmonic_phase: 0.0,
        }
    }

    /// Evolve the state by one timestep given current observations.
    pub fn update(&mut self, obs: &Observations, dt: f32) {
        let s = &self.state;
        let k = &self.inertia;

        let dt_clamped = dt.clamp(0.0001, 0.1);

        // 1. Energy: driven by audio loudness & screen motion (fast attack, smooth release)
        let energy_target = (obs.audio_energy * 0.75 + obs.screen_motion * 0.25).clamp(0.0, 1.0);
        let energy_rate = if energy_target > s.energy {
            1.0 - (1.0 - k.energy).powf(dt_clamped * 120.0) // Fast excitation (~20ms)
        } else {
            1.0 - k.energy.powf(dt_clamped * 35.0) // Organic cooldown (~300ms)
        };
        let energy = lerp(s.energy, energy_target, energy_rate);

        // 2. Tension: rises when audio is volatile or high energy flux is detected
        let tension_drive = (obs.audio_volatility * 0.5 + obs.audio_attack * 0.35 + obs.audio_spectral_flux * 0.35)
            .clamp(0.0, 1.0);
        let tension = lerp(s.tension, tension_drive, 1.0 - k.tension.powf(dt_clamped * 45.0));

        // 3. Chaos: sudden changes, dissonance (ZCR / high spectral flux), and high tension
        let chaos_drive = (tension * 0.4 + obs.audio_energy_velocity.abs() * 0.3 + obs.audio_zcr * 0.3)
            .clamp(0.0, 1.0);
        let chaos = lerp(s.chaos, chaos_drive, 1.0 - k.chaos.powf(dt_clamped * 50.0));

        // 4. Coherence: high when music is stable & harmonic; drops when chaotic
        let coherence_drive = ((1.0 - chaos * 0.75) * (1.0 - tension * 0.35)).clamp(0.05, 1.0);
        let coherence = lerp(s.coherence, coherence_drive, 1.0 - k.coherence.powf(dt_clamped * 40.0));

        // 5. Turbulence: fluid motion driven by mid-frequency energy + chaos + motion
        let turbulence_drive = (obs.audio_mid * 0.45 + obs.screen_motion * 0.25 + chaos * 0.3).clamp(0.0, 1.0);
        let turbulence = lerp(s.turbulence, turbulence_drive, 1.0 - k.turbulence.powf(dt_clamped * 50.0));

        // 6. Density: visual richness / layer complexity
        let density_drive = (energy * 0.5 + coherence * 0.3 + obs.screen_contrast * 0.2).clamp(0.1, 1.0);
        let density = lerp(s.density, density_drive, 1.0 - k.density.powf(dt_clamped * 30.0));

        // 7. Flow: driven by bass weight and turbulence
        let flow_drive = (obs.audio_bass * 0.55 + turbulence * 0.3 + energy * 0.15).clamp(0.05, 1.0);
        let flow = lerp(s.flow, flow_drive, 1.0 - k.flow.powf(dt_clamped * 45.0));

        // 8. Compression vs Expansion
        let compression_drive = (energy * (1.0 - chaos * 0.8)).clamp(0.0, 1.0);
        let compression = lerp(s.compression, compression_drive, 1.0 - k.compression.powf(dt_clamped * 40.0));

        let expansion_drive = (energy * (0.3 + chaos * 0.7)).clamp(0.0, 1.0);
        let expansion = lerp(s.expansion, expansion_drive, 1.0 - k.expansion.powf(dt_clamped * 40.0));

        // 9. Entropy: information content and tonal unpredictability
        let entropy_drive = (obs.audio_spectral_flux * 0.5 + chaos * 0.3 + (1.0 - coherence) * 0.2).clamp(0.0, 1.0);
        let entropy = lerp(s.entropy, entropy_drive, 1.0 - k.entropy.powf(dt_clamped * 35.0));

        // 10. Chroma: color saturation and vibrancy drive
        let chroma_drive = (0.4 + energy * 0.45 + obs.screen_saturation * 0.25 - chaos * 0.1).clamp(0.2, 1.0);
        let chroma = lerp(s.chroma, chroma_drive, 1.0 - k.chroma.powf(dt_clamped * 35.0));

        // 11. Temporal Memory: high when calm; low when chaotic (rapid overwrite)
        let memory_drive = (0.9 - energy * 0.4 - chaos * 0.35).clamp(0.1, 0.95);
        let temporal_memory = lerp(s.temporal_memory, memory_drive, 1.0 - k.temporal_memory.powf(dt_clamped * 20.0));

        self.state = GlobalState {
            energy,
            tension,
            chaos,
            coherence,
            turbulence,
            density,
            flow,
            compression,
            expansion,
            entropy,
            chroma,
            temporal_memory,
        };

        // 12. Dynamic Phase Integration (State-Driven Clocks)
        // Instead of fixed time * speed, phase velocity is dynamically driven by the organism's energy & mood!
        let flow_velocity = 0.2 + flow * 0.8 + energy * 1.2 + obs.audio_bass * 0.6;
        self.dynamic_flow_phase += dt_clamped * flow_velocity;

        // Color drift velocity: peaceful and gliding in piano, accelerating into burning ionization in metal
        let color_velocity = 0.03 + energy * 0.25 + tension * 0.35 + obs.audio_spectral_flux * 0.4;
        self.dynamic_color_phase += dt_clamped * color_velocity;

        let harmonic_velocity = 0.4 + turbulence * 0.6 + obs.audio_treble * 0.8;
        self.dynamic_harmonic_phase += dt_clamped * harmonic_velocity;
    }

    pub fn state(&self) -> &GlobalState {
        &self.state
    }
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}
