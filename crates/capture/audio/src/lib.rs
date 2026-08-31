//! # Audio Capture
//!
//! Captures a specific audio output device via WASAPI loopback.
//! On startup, logs all available devices. Device selection is via
//! `DeviceSelection` — partial name match, default, or first non-default.

mod loopback;

pub use loopback::{AudioCapture, AudioBuffer, DeviceSelection, list_output_devices};

/// Default audio FFT buffer size in samples.
pub const DEFAULT_FFT_SIZE: usize = 2048;
