//! DSP engine — FFT, multi-timescale feature extraction, and spectral dynamics.

use crate::features::*;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f32::consts::PI;

/// The audio DSP processing engine.
///
/// Maintains internal sample buffering, Hann windowing, and multi-timescale history.
pub struct DspEngine {
    fft_size: usize,
    planner: FftPlanner<f32>,
    hann_window: Vec<f32>,
    sample_ring_buffer: Vec<f32>,
    ring_write_idx: usize,
    sample_count_accum: usize,

    // Previous frame magnitudes for spectral flux calculation
    prev_magnitudes: Vec<f32>,

    // Smoothed band energies
    smoothed_bands: [f32; 7],
    band_peaks: [f32; 7],
    prev_band_energies: [f32; 7],
    prev_band_velocities: [f32; 7],

    // Rolling history for temporal dynamics (~2-3s history)
    energy_history: Vec<f32>,
    spectral_history: Vec<f32>,
    history_cursor: usize,
    history_len: usize,

    // Derivatives and rolling metrics
    prev_rms: f32,
    prev_spectral_centroid: f32,
    long_term_energy_sum: f32,
    long_term_count: u64,
    stable_time_accum: f32,
}

impl DspEngine {
    /// Create a new DSP engine with specified FFT window size (typically 2048).
    pub fn new(fft_size: usize) -> Self {
        let fft_size = fft_size.max(256).next_power_of_two();
        let history_len = 180; // ~1-2 seconds at 120-165 fps

        // Precompute Hann window
        let hann_window: Vec<f32> = (0..fft_size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (fft_size - 1) as f32).cos()))
            .collect();

        Self {
            fft_size,
            planner: FftPlanner::new(),
            hann_window,
            sample_ring_buffer: vec![0.0; fft_size],
            ring_write_idx: 0,
            sample_count_accum: 0,
            prev_magnitudes: vec![0.0; fft_size / 2],
            smoothed_bands: [0.0; 7],
            band_peaks: [0.0; 7],
            prev_band_energies: [0.0; 7],
            prev_band_velocities: [0.0; 7],
            energy_history: vec![0.0; history_len],
            spectral_history: vec![0.0; history_len],
            history_cursor: 0,
            history_len,
            prev_rms: 0.0,
            prev_spectral_centroid: 0.0,
            long_term_energy_sum: 0.0,
            long_term_count: 0,
            stable_time_accum: 0.0,
        }
    }

    /// Push new interleaved or mono samples into the ring buffer.
    pub fn push_samples(&mut self, samples: &[f32], channels: u16) {
        if samples.is_empty() {
            return;
        }

        let ch = channels.max(1) as usize;
        let num_frames = samples.len() / ch;

        for frame_idx in 0..num_frames {
            let mut mono_sum = 0.0f32;
            for c in 0..ch {
                mono_sum += samples[frame_idx * ch + c];
            }
            let mono_sample = mono_sum / ch as f32;

            self.sample_ring_buffer[self.ring_write_idx] = mono_sample;
            self.ring_write_idx = (self.ring_write_idx + 1) % self.fft_size;
            self.sample_count_accum += 1;
        }
    }

    /// Compute audio features from current accumulated sample buffer.
    /// `dt` is elapsed frame time in seconds (for derivative & decay scaling).
    pub fn process(&mut self, sample_rate: u32, dt: f32) -> AudioFeatures {
        let sample_rate = sample_rate.max(8000);
        let dt = dt.clamp(0.0001, 0.1);

        // Read windowed samples from ring buffer in chronological order
        let mut fft_buffer: Vec<Complex<f32>> = Vec::with_capacity(self.fft_size);
        let mut rms_sum = 0.0f32;
        let mut zero_crossings = 0usize;
        let mut prev_s = 0.0f32;

        let start_idx = self.ring_write_idx; // Oldest sample
        for i in 0..self.fft_size {
            let idx = (start_idx + i) % self.fft_size;
            let raw_sample = self.sample_ring_buffer[idx];

            rms_sum += raw_sample * raw_sample;
            if i > 0 && (raw_sample >= 0.0) != (prev_s >= 0.0) {
                zero_crossings += 1;
            }
            prev_s = raw_sample;

            let windowed = raw_sample * self.hann_window[i];
            fft_buffer.push(Complex::new(windowed, 0.0));
        }

        let raw_rms = (rms_sum / self.fft_size as f32).sqrt();
        let zcr = (zero_crossings as f32 / (self.fft_size - 1) as f32).clamp(0.0, 1.0);

        // Compute FFT
        let fft = self.planner.plan_fft_forward(self.fft_size);
        fft.process(&mut fft_buffer);

        let half = self.fft_size / 2;
        let norm_factor = 2.0 / self.fft_size as f32;
        let mut magnitudes = Vec::with_capacity(half);

        for i in 0..half {
            let mag = fft_buffer[i].norm() * norm_factor;
            magnitudes.push(mag);
        }

        // Spectral Centroid & Spectral Flux
        let freq_resolution = sample_rate as f32 / self.fft_size as f32;
        let total_mag: f32 = magnitudes.iter().sum();

        let spectral_centroid_hz = if total_mag > 1e-7 {
            magnitudes
                .iter()
                .enumerate()
                .map(|(i, &m)| (i as f32 * freq_resolution) * m)
                .sum::<f32>()
                / total_mag
        } else {
            0.0
        };

        // Normalized centroid (0..1 relative to Nyquist)
        let spectral_centroid_norm = (spectral_centroid_hz / (sample_rate as f32 * 0.5)).clamp(0.0, 1.0);

        // Spectral flux: sum of positive spectral differences
        let mut spectral_flux = 0.0f32;
        for i in 0..half {
            let diff = magnitudes[i] - self.prev_magnitudes[i];
            if diff > 0.0 {
                spectral_flux += diff * diff;
            }
        }
        spectral_flux = spectral_flux.sqrt() * 10.0;
        self.prev_magnitudes = magnitudes.clone();

        // 7-Band Decomposition with dynamic attack & decay
        let raw_bands = self.compute_raw_bands(&magnitudes, sample_rate);

        let mut band_velocities = [0.0f32; 7];
        let mut band_accelerations = [0.0f32; 7];

        for i in 0..7 {
            let raw = raw_bands[i];
            let current = self.smoothed_bands[i];

            // Asymmetric attack/decay smoothing:
            // Fast rise on transients (attack), smoother release (decay)
            let alpha = if raw > current {
                1.0 - (-dt / 0.015).exp() // ~15ms attack
            } else {
                1.0 - (-dt / 0.120).exp() // ~120ms release
            };
            let smoothed = current + alpha * (raw - current);
            self.smoothed_bands[i] = smoothed.clamp(0.0, 1.0);

            // Peak tracking with decay (~1.5s half-life)
            if smoothed > self.band_peaks[i] {
                self.band_peaks[i] = smoothed;
            } else {
                let peak_decay = (-dt / 0.8).exp();
                self.band_peaks[i] = (self.band_peaks[i] * peak_decay).max(smoothed);
            }

            let vel = (smoothed - self.prev_band_energies[i]) / dt;
            let accel = (vel - self.prev_band_velocities[i]) / dt;

            band_velocities[i] = vel;
            band_accelerations[i] = accel;

            self.prev_band_energies[i] = smoothed;
            self.prev_band_velocities[i] = vel;
        }

        // Temporal derivatives
        let energy_velocity = (raw_rms - self.prev_rms) / dt;
        let spectral_movement = (spectral_centroid_norm - self.prev_spectral_centroid) / dt;

        let attack_strength = energy_velocity.max(0.0);
        let decay_rate = (-energy_velocity).max(0.0);
        let transient_strength = (attack_strength * 2.0 + spectral_flux * 0.5).clamp(0.0, 1.0);

        // Update long-term history
        self.energy_history[self.history_cursor] = raw_rms;
        self.spectral_history[self.history_cursor] = spectral_centroid_norm;
        self.history_cursor = (self.history_cursor + 1) % self.history_len;
        self.long_term_energy_sum += raw_rms;
        self.long_term_count += 1;

        let rolling_avg_energy = self.energy_history.iter().sum::<f32>() / self.history_len as f32;
        let volatility = self.compute_energy_volatility(rolling_avg_energy);

        if volatility < 0.05 {
            self.stable_time_accum += dt;
        } else {
            self.stable_time_accum = (self.stable_time_accum - dt * 2.0).max(0.0);
        }

        self.prev_rms = raw_rms;
        self.prev_spectral_centroid = spectral_centroid_norm;

        AudioFeatures {
            instant: InstantFeatures {
                rms: raw_rms.clamp(0.0, 1.0),
                spectral_centroid: spectral_centroid_norm,
                spectral_flux: spectral_flux.clamp(0.0, 1.0),
                zcr,
                fft_magnitudes: magnitudes,
            },
            short_term: ShortTermFeatures {
                attack: attack_strength.clamp(0.0, 5.0),
                decay: decay_rate.clamp(0.0, 5.0),
                energy_velocity,
                spectral_movement,
                transient_strength,
            },
            long_term: LongTermFeatures {
                avg_energy: rolling_avg_energy.clamp(0.0, 1.0),
                dominant_frequency: spectral_centroid_norm,
                volatility: volatility.clamp(0.0, 1.0),
                stability_duration: self.stable_time_accum,
            },
            bands: BandFeatures {
                energy: self.smoothed_bands,
                velocity: band_velocities,
                acceleration: band_accelerations,
                recent_peak: self.band_peaks,
                baseline: [rolling_avg_energy; 7],
                stability: [1.0 - volatility; 7],
            },
        }
    }

    /// Compute raw frequency bands with perceptual log-gain curves.
    fn compute_raw_bands(&self, magnitudes: &[f32], sample_rate: u32) -> [f32; 7] {
        let boundaries = [
            (20.0, 60.0),     // Sub-bass
            (60.0, 250.0),    // Bass
            (250.0, 500.0),   // Low-mid
            (500.0, 2000.0),  // Mid
            (2000.0, 4000.0), // High-mid
            (4000.0, 8000.0), // Treble
            (8000.0, 20000.0),// Brilliance
        ];

        let freq_resolution = sample_rate as f32 / self.fft_size as f32;
        let mut bands = [0.0f32; 7];

        // Equal-loudness approximate compensation curve
        let band_gains = [3.5, 2.5, 2.0, 1.6, 2.0, 3.0, 4.0];

        for (i, &(lo_hz, hi_hz)) in boundaries.iter().enumerate() {
            let lo_bin = ((lo_hz / freq_resolution).floor() as usize).min(magnitudes.len());
            let hi_bin = ((hi_hz / freq_resolution).ceil() as usize).min(magnitudes.len());

            if lo_bin < hi_bin {
                let bin_count = (hi_bin - lo_bin) as f32;
                let sum_sq: f32 = magnitudes[lo_bin..hi_bin].iter().map(|&m| m * m).sum();
                let rms_band = (sum_sq / bin_count).sqrt();

                // Apply log/perceptual curve: sqrt compression + frequency weighting
                let perceived = (rms_band * band_gains[i] * 12.0).sqrt();
                bands[i] = perceived.clamp(0.0, 1.0);
            }
        }

        bands
    }

    fn compute_energy_volatility(&self, mean: f32) -> f32 {
        if self.energy_history.is_empty() {
            return 0.0;
        }
        let var_sum: f32 = self
            .energy_history
            .iter()
            .map(|&e| {
                let diff = e - mean;
                diff * diff
            })
            .sum();
        (var_sum / self.energy_history.len() as f32).sqrt() * 5.0
    }
}
