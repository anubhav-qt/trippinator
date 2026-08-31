//! # Temporal Feedback
//!
//! The image has memory.
//!
//! ```text
//! I_t = (1 - λ) * P_t + λ * W(I_{t-1})
//! ```
//!
//! where W is a distortion/warp function and λ is the feedback strength
//! controlled by global state.

use bytemuck::{Pod, Zeroable};

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct FeedbackParams {
    pub time: f32,
    /// Blend factor: how much of the previous frame to keep.
    pub feedback_strength: f32,
    /// Warp intensity applied to previous frame.
    pub warp_amount: f32,
    /// Warp rotation speed.
    pub warp_rotation: f32,
    /// Zoom factor for feedback warp.
    pub warp_zoom: f32,
    /// Color decay rate.
    pub color_decay: f32,
    /// Energy from global state.
    pub energy: f32,
    /// Memory from global state.
    pub temporal_memory: f32,
}

pub struct FeedbackSystem {
    params: FeedbackParams,
}

impl FeedbackSystem {
    pub fn new() -> Self {
        Self {
            params: FeedbackParams {
                time: 0.0,
                feedback_strength: 0.85,
                warp_amount: 0.01,
                warp_rotation: 0.001,
                warp_zoom: 1.001,
                color_decay: 0.98,
                energy: 0.0,
                temporal_memory: 0.7,
            },
        }
    }

    pub fn update(&mut self, state: &trippinator_dynamics_state::GlobalState, dt: f32) {
        self.params.time += dt;
        self.params.energy = state.energy;
        self.params.temporal_memory = state.temporal_memory;

        // Calm: long memory. Chaotic: rapid overwriting.
        self.params.feedback_strength = 0.5 + state.temporal_memory * 0.45;
        self.params.warp_amount = state.turbulence * 0.03;
        self.params.warp_rotation = state.flow * 0.005;
        self.params.warp_zoom = 1.0 + state.expansion * 0.005 - state.compression * 0.003;
        self.params.color_decay = 0.95 + state.coherence * 0.04;
    }
}
