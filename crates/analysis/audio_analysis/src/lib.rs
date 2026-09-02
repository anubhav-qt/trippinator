//! # Audio Analysis
//!
//! Multi-timescale audio feature extraction:
//! - **Instantaneous**: FFT, RMS, band energies, spectral centroid/flux
//! - **Short-term** (~100ms–2s): attack, decay, energy velocity, transients
//! - **Long-term** (>2s): average energy, dominant profile, peak history, stability

mod dsp;
mod features;

pub use features::{
    AudioFeatures, BandFeatures, InstantFeatures, LongTermFeatures, ShortTermFeatures, SongCharacter,
};
pub use dsp::DspEngine;
