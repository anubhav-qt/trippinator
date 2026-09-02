//! Audio feature structs across three timescales.

/// Complete audio features for the dynamics layer.
#[derive(Debug, Clone, Default)]
pub struct AudioFeatures {
    pub instant: InstantFeatures,
    pub short_term: ShortTermFeatures,
    pub long_term: LongTermFeatures,
    pub bands: BandFeatures,
    pub character: SongCharacter,
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
    /// True on the frame an onset is detected (adaptive-threshold spectral flux).
    pub onset: bool,
}

/// Long-term features — sustained state.
///
/// Everything here is a time-based exponential envelope, not an N-frame ring buffer.
/// The distinction is not cosmetic: the ring buffer this replaced was 180 frames long,
/// which is 1.0 s at the 178 fps this actually runs at — so the "long-term" baseline was
/// a one-second average, and any passage that stayed loud for more than a second was
/// absorbed into its own baseline and stopped registering as anything at all.
#[derive(Debug, Clone, Default)]
pub struct LongTermFeatures {
    /// Rolling average energy, ~1.5 s. The local context.
    pub avg_energy: f32,
    /// Slow average energy, ~25 s. The level this section of the track sits at.
    pub slow_energy: f32,
    /// Peak-following loudness reference, ~0.4 s up and ~60 s down: how loud this track
    /// gets at its loudest. Dividing by it makes every level judgement independent of how
    /// the record was mastered, which is what lets a quiet 1979 mix and a brickwalled
    /// modern one both reach the top of the visual range.
    pub loudness_ref: f32,
    /// Current level as a fraction of `loudness_ref` (0..1). The master-independent
    /// "how loud is it right now, for this track" reading.
    pub level_norm: f32,
    /// How much headroom this track uses: high for a dynamic master, low for a
    /// compressed one.
    pub dynamic_range: f32,
    /// Dominant frequency profile center.
    pub dominant_frequency: f32,
    /// Energy volatility (std-dev of energy over ~4 s, scaled).
    pub volatility: f32,
    /// How long current energy level has been stable in seconds.
    pub stability_duration: f32,
}

/// What *kind* of music this is, on timescales of tens of seconds.
///
/// These describe the material rather than the moment, and they are what the visual
/// side switches configuration on. All are 0..1 and heavily smoothed — they should
/// drift over a track, never move on a beat.
#[derive(Debug, Clone, Default)]
pub struct SongCharacter {
    /// Detected onsets per second, smoothed. Raw units, not normalized — a busy rock
    /// track sits around 4-8, an ambient pad near 0.
    pub onset_rate: f32,
    /// Transient-led: frequent onsets, sharp attacks. Drums and plucked strings.
    pub percussive: f32,
    /// Sustain-led: energy stays up between onsets. Bowed strings, organ, held guitar,
    /// pads. This is the axis that "Comfortably Numb" scores high on and that nothing
    /// in the visual pipeline was previously reading.
    pub sustained: f32,
    /// Slow spectral balance, each ~20 s. Where this track's weight sits.
    pub bass_w: f32,
    pub mid_w: f32,
    pub treble_w: f32,
    /// Slow spectral centroid — overall brightness of the material.
    pub brightness: f32,
    /// How full the spectrum is: a wall of sound vs. a few sparse voices.
    pub density: f32,
}
