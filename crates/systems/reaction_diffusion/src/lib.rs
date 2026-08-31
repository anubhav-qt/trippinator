//! # Reaction-Diffusion Systems
//!
//! Gray-Scott model and variations, computed on GPU via compute shaders.
//! Produces organic, self-organizing patterns.

use bytemuck::{Pod, Zeroable};

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct ReactionDiffusionParams {
    pub time: f32,
    pub feed_rate: f32,
    pub kill_rate: f32,
    pub diffusion_a: f32,
    pub diffusion_b: f32,
    pub steps_per_frame: f32,
    pub energy: f32,
    pub tension: f32,
}

pub struct ReactionDiffusionSystem {
    params: ReactionDiffusionParams,
}

impl ReactionDiffusionSystem {
    pub fn new() -> Self {
        Self {
            params: ReactionDiffusionParams {
                time: 0.0,
                feed_rate: 0.055,
                kill_rate: 0.062,
                diffusion_a: 1.0,
                diffusion_b: 0.5,
                steps_per_frame: 4.0,
                energy: 0.0,
                tension: 0.0,
            },
        }
    }

    pub fn update(&mut self, state: &trippinator_dynamics_state::GlobalState, dt: f32) {
        self.params.time += dt;
        self.params.energy = state.energy;
        self.params.tension = state.tension;
        // Slowly drift through parameter space
        self.params.feed_rate = 0.03 + state.density * 0.04;
        self.params.kill_rate = 0.055 + state.chaos * 0.015;
        self.params.steps_per_frame = 2.0 + state.energy * 6.0;
    }
}
