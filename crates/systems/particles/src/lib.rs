//! # Particle Systems
//!
//! GPU-driven particle simulation using compute shaders.
//! Particles are born, live, and die on the GPU — the CPU only sets parameters.

use bytemuck::{Pod, Zeroable};

/// Per-particle data stored in GPU storage buffer.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct Particle {
    pub position: [f32; 2],
    pub velocity: [f32; 2],
    pub life: f32,
    pub max_life: f32,
    pub size: f32,
    pub _padding: f32,
}

/// Parameters for the particle compute shader.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct ParticleParams {
    pub time: f32,
    pub dt: f32,
    pub particle_count: u32,
    pub spawn_rate: f32,
    pub gravity: [f32; 2],
    pub turbulence: f32,
    pub energy: f32,
    pub chaos: f32,
    pub expansion: f32,
    pub flow_x: f32,
    pub flow_y: f32,
}

pub struct ParticleSystem {
    params: ParticleParams,
    max_particles: u32,
}

impl ParticleSystem {
    pub fn new(max_particles: u32) -> Self {
        Self {
            params: ParticleParams {
                time: 0.0,
                dt: 1.0 / 60.0,
                particle_count: 0,
                spawn_rate: 100.0,
                gravity: [0.0, -0.1],
                turbulence: 0.0,
                energy: 0.0,
                chaos: 0.0,
                expansion: 0.0,
                flow_x: 0.0,
                flow_y: 0.0,
            },
            max_particles,
        }
    }

    pub fn update(&mut self, state: &trippinator_dynamics_state::GlobalState, dt: f32) {
        self.params.time += dt;
        self.params.dt = dt;
        self.params.energy = state.energy;
        self.params.chaos = state.chaos;
        self.params.expansion = state.expansion;
        self.params.turbulence = state.turbulence;
        self.params.spawn_rate = 50.0 + state.energy * 500.0;
        self.params.gravity[1] = -0.05 - state.flow * 0.2;
    }
}
