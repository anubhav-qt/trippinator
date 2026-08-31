//! # System Graph
//!
//! A dynamic graph where each node represents a visual system and edges
//! represent coupling weights. The graph itself evolves based on global state.

use trippinator_dynamics_state::GlobalState;

/// Identifies a visual system node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SystemId {
    Waves,
    Particles,
    Fields,
    Fractals,
    ReactionDiffusion,
    Feedback,
}

impl SystemId {
    pub const ALL: &[SystemId] = &[
        SystemId::Waves,
        SystemId::Particles,
        SystemId::Fields,
        SystemId::Fractals,
        SystemId::ReactionDiffusion,
        SystemId::Feedback,
    ];
}

/// A connection between two systems.
#[derive(Debug, Clone)]
pub struct Connection {
    pub from: SystemId,
    pub to: SystemId,
    /// Current coupling weight (0..1).
    pub weight: f32,
    /// Base weight (resting state).
    pub base_weight: f32,
}

/// The system graph: nodes + weighted directed edges that evolve.
pub struct SystemGraph {
    connections: Vec<Connection>,
    /// Per-system activation level (0..1).
    activations: Vec<(SystemId, f32)>,
}

impl SystemGraph {
    /// Create the default system graph with initial connections.
    pub fn new() -> Self {
        use SystemId::*;

        let connections = vec![
            Connection { from: Waves, to: Particles, weight: 0.3, base_weight: 0.3 },
            Connection { from: Waves, to: Fields, weight: 0.4, base_weight: 0.4 },
            Connection { from: Particles, to: Fields, weight: 0.2, base_weight: 0.2 },
            Connection { from: Fields, to: Fractals, weight: 0.1, base_weight: 0.1 },
            Connection { from: Fractals, to: Feedback, weight: 0.3, base_weight: 0.3 },
            Connection { from: Feedback, to: Waves, weight: 0.5, base_weight: 0.5 },
            Connection { from: Fields, to: ReactionDiffusion, weight: 0.15, base_weight: 0.15 },
            Connection { from: ReactionDiffusion, to: Feedback, weight: 0.2, base_weight: 0.2 },
        ];

        let activations = SystemId::ALL
            .iter()
            .map(|&id| (id, 0.5))
            .collect();

        Self {
            connections,
            activations,
        }
    }

    /// Evolve the graph based on global state.
    ///
    /// Connections strengthen/weaken and systems activate/deactivate
    /// based on energy, chaos, coherence, etc.
    pub fn evolve(&mut self, state: &GlobalState, dt: f32) {
        let eta = 0.1 * dt; // learning rate

        for conn in &mut self.connections {
            // High energy strengthens connections, low energy returns to base
            let energy_mod = state.energy * 0.5;
            // Chaos destabilizes connections (push away from base)
            let chaos_mod = state.chaos * 0.3 * ((conn.weight - 0.5).signum());
            // Coherence pulls toward base
            let coherence_mod = state.coherence * 0.2 * (conn.base_weight - conn.weight);

            conn.weight += eta * (energy_mod + chaos_mod + coherence_mod);
            conn.weight = conn.weight.clamp(0.0, 1.0);
        }

        // Update activations
        for (id, activation) in &mut self.activations {
            let target = match id {
                SystemId::Waves => 0.3 + state.energy * 0.4 + state.flow * 0.3,
                SystemId::Particles => state.energy * 0.5 + state.chaos * 0.3 + state.expansion * 0.2,
                SystemId::Fields => state.flow * 0.4 + state.turbulence * 0.4 + state.coherence * 0.2,
                SystemId::Fractals => state.density * 0.3 + state.chaos * 0.3 + state.entropy * 0.4,
                SystemId::ReactionDiffusion => state.tension * 0.4 + state.density * 0.3 + (1.0 - state.chaos) * 0.3,
                SystemId::Feedback => state.temporal_memory * 0.5 + state.coherence * 0.3 + state.energy * 0.2,
            };

            *activation += 0.05 * dt * 60.0 * (target.clamp(0.0, 1.0) - *activation);
        }
    }

    /// Get the coupling weight between two systems.
    pub fn weight(&self, from: SystemId, to: SystemId) -> f32 {
        self.connections
            .iter()
            .find(|c| c.from == from && c.to == to)
            .map_or(0.0, |c| c.weight)
    }

    /// Get the activation level of a system.
    pub fn activation(&self, id: SystemId) -> f32 {
        self.activations
            .iter()
            .find(|(sys_id, _)| *sys_id == id)
            .map_or(0.0, |(_, a)| *a)
    }
}
