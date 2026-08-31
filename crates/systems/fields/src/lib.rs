//! # Field Systems
//!
//! Vector fields, curl fields, vortex fields, gravitational fields, turbulence.
//! Computed entirely on GPU via compute shaders.

use bytemuck::{Pod, Zeroable};

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct FieldParams {
    pub time: f32,
    pub vortex_count: f32,
    pub curl_strength: f32,
    pub turbulence_scale: f32,
    pub turbulence_speed: f32,
    pub flow_strength: f32,
    pub energy: f32,
    pub chaos: f32,
}

pub struct FieldSystem {
    params: FieldParams,
}

impl FieldSystem {
    pub fn new() -> Self {
        Self {
            params: FieldParams {
                time: 0.0,
                vortex_count: 3.0,
                curl_strength: 0.5,
                turbulence_scale: 4.0,
                turbulence_speed: 0.3,
                flow_strength: 0.2,
                energy: 0.0,
                chaos: 0.0,
            },
        }
    }

    pub fn update(&mut self, state: &trippinator_dynamics_state::GlobalState, dt: f32) {
        self.params.time += dt;
        self.params.energy = state.energy;
        self.params.chaos = state.chaos;
        self.params.curl_strength = state.flow * 0.5 + state.turbulence * 0.5;
        self.params.turbulence_scale = 2.0 + state.chaos * 8.0;
        self.params.flow_strength = state.flow;
    }
}
