//! # Coupling Functions
//!
//! Defines how systems respond differently to the same stimulus.
//! Five systems receive the same signal but respond completely differently:
//!
//! - Direct: A = E
//! - Inverse: A = 1 - E
//! - Resonant: A = sin(π * E)
//! - Quadratic: A = E²
//! - Exponential decay: A = e^(-kE)

use std::f32::consts::PI;

/// A coupling response function.
#[derive(Debug, Clone, Copy)]
pub enum CouplingFunction {
    /// Direct proportional response.
    Direct,
    /// Inverse response.
    Inverse,
    /// Resonant (peaks at 0.5).
    Resonant,
    /// Quadratic (amplifies high values).
    Quadratic,
    /// Exponential decay (strongest at low input).
    ExponentialDecay { k: f32 },
    /// Threshold (activates above a point).
    Threshold { point: f32, sharpness: f32 },
    /// Custom polynomial.
    Custom { coeffs: [f32; 4] },
}

impl CouplingFunction {
    /// Evaluate the coupling function for input value e (0..1).
    pub fn evaluate(&self, e: f32) -> f32 {
        match self {
            Self::Direct => e,
            Self::Inverse => 1.0 - e,
            Self::Resonant => (PI * e).sin(),
            Self::Quadratic => e * e,
            Self::ExponentialDecay { k } => (-k * e).exp(),
            Self::Threshold { point, sharpness } => {
                1.0 / (1.0 + (-sharpness * (e - point)).exp())
            }
            Self::Custom { coeffs } => {
                coeffs[0] + coeffs[1] * e + coeffs[2] * e * e + coeffs[3] * e * e * e
            }
        }
        .clamp(0.0, 1.0)
    }
}

/// A coupling specification: which state dimension, which response function, and weight.
#[derive(Debug, Clone)]
pub struct CouplingSpec {
    /// Human-readable name.
    pub name: String,
    /// The response function.
    pub function: CouplingFunction,
    /// Output weight (0..1).
    pub weight: f32,
}

/// Evaluate multiple opposing couplings to produce tension.
pub fn evaluate_opposing_couplings(specs: &[CouplingSpec], input: f32) -> Vec<f32> {
    specs
        .iter()
        .map(|spec| spec.function.evaluate(input) * spec.weight)
        .collect()
}

/// Compute tension as the variance of coupling outputs.
pub fn coupling_tension(outputs: &[f32]) -> f32 {
    if outputs.is_empty() {
        return 0.0;
    }
    let mean = outputs.iter().sum::<f32>() / outputs.len() as f32;
    let variance = outputs.iter().map(|o| (o - mean) * (o - mean)).sum::<f32>()
        / outputs.len() as f32;
    variance.sqrt()
}
