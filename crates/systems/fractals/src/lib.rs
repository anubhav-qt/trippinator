//! # Fractal Systems
//!
//! Fractals, recursive transformations, iterated domain warps.

use bytemuck::{Pod, Zeroable};

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct FractalParams {
    pub time: f32,
    pub iterations: f32,
    pub zoom: f32,
    pub rotation: f32,
    pub warp_intensity: f32,
    pub warp_frequency: f32,
    pub color_shift: f32,
    pub energy: f32,
}

pub struct FractalSystem {
    params: FractalParams,
}

impl FractalSystem {
    pub fn new() -> Self {
        Self {
            params: FractalParams {
                time: 0.0,
                iterations: 8.0,
                zoom: 1.0,
                rotation: 0.0,
                warp_intensity: 0.3,
                warp_frequency: 2.0,
                color_shift: 0.0,
                energy: 0.0,
            },
        }
    }

    pub fn update(&mut self, state: &trippinator_dynamics_state::GlobalState, dt: f32) {
        self.params.time += dt;
        self.params.energy = state.energy;
        self.params.iterations = 4.0 + state.density * 12.0;
        self.params.warp_intensity = state.chaos * 0.8;
        self.params.zoom = 0.5 + state.compression * 2.0;
        self.params.rotation += state.flow * dt * 0.5;
    }
}
