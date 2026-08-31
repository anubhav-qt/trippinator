//! Audio feature structs across three timescales.

/// Complete audio features for the dynamics layer.
#[derive(Debug, Clone, Default)]
pub struct AudioFeatures {
    pub instant: InstantFeatures,
    pub short_term: ShortTermFeatures,
    pub long_term: LongTermFeatures,
    pub bands: BandFeatures,
}

/// Per-band features:
/// [0] Sub-bass (20-60 Hz)
/// [1] Bass (60-250 Hz)
/// [2] Low-mid (250-500 Hz)
/// [3] Mid (500-2000 Hz)
/// [4] High-mid (2000-4000 Hz)
/// [5] Treble (4000-8000 Hz)
/// [6] Brilliance (8000-20000 Hz)
#[derive(Debug, Clone, Default)]
pub struct BandFeatures {
    /// Smoothed energy per band (0..1).
    pub energy: [f32; 7],
    /// Energy velocity per band (rate of change).
    pub velocity: [f32; 7],
    /// Energy acceleration per band.
    pub acceleration: [f32; 7],
    /// Recent peak per band.
    pub recent_peak: [f32; 7],
    /// Baseline (rolling average) per band.
    pub baseline: [f32; 7],
    /// Stability (inverse variance) per band.
    pub stability: [f32; 7],
}

/// Instantaneous features — what is happening right now.
#[derive(Debug, Clone, Default)]
pub struct InstantFeatures {
    /// Root mean square energy (0..1).
    pub rms: f32,
    /// Spectral centroid (normalized 0..1 across Nyquist).
    pub spectral_centroid: f32,
    /// Spectral flux (rate of spectral change).
    pub spectral_flux: f32,
    /// Zero-crossing rate (0..1).
    pub zcr: f32,
    /// Raw FFT magnitudes (half-spectrum).
    pub fft_magnitudes: Vec<f32>,
}

/// Short-term features — ~100ms to 2s.
#[derive(Debug, Clone, Default)]
pub struct ShortTermFeatures {
    /// Attack strength (onset detection).
    pub attack: f32,
    /// Decay rate.
    pub decay: f32,
    /// Energy velocity (first derivative).
    pub energy_velocity: f32,
    /// Spectral movement (spectral centroid velocity).
    pub spectral_movement: f32,
    /// Transient strength.
    pub transient_strength: f32,
}

/// Long-term features — sustained state.
#[derive(Debug, Clone, Default)]
pub struct LongTermFeatures {
    /// Rolling average energy.
    pub avg_energy: f32,
    /// Dominant frequency profile center.
    pub dominant_frequency: f32,
    /// Energy volatility (variance of energy over time).
    pub volatility: f32,
    /// How long current energy level has been stable in seconds.
    pub stability_duration: f32,
}
