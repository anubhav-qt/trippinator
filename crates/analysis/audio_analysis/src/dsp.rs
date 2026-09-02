//! DSP engine — FFT, multi-timescale feature extraction, and spectral dynamics.

use crate::features::*;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f32::consts::PI;

/// Frame-rate-independent exponential move of `cur` toward `target`, with time constant
/// `tau` seconds. Every long-timescale quantity in this file goes through this — see the
/// note on [`LongTermFeatures`] for why counting frames instead was wrong.
fn ema(cur: f32, target: f32, tau: f32, dt: f32) -> f32 {
    cur + (1.0 - (-dt / tau.max(1e-4)).exp()) * (target - cur)
}

/// Same curve as WGSL/GLSL `smoothstep`.
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

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

    // Energy envelopes at four timescales. Together these are the whole basis for
    // "loud compared to what?", and each answers a different version of the question.
    env_fast: f32,     // ~0.15 s — the note
    env_mid: f32,      // ~1.5 s  — the bar
    env_slow: f32,     // ~25 s   — the section
    env_peak: f32,     // fast up, ~1 s down — the local crest
    loudness_ref: f32,   // fast up, ~60 s down — the whole track's ceiling
    loudness_floor: f32, // ~2 s down, ~90 s up — the whole track's floor

    // Volatility as a running mean/variance pair rather than a windowed std-dev.
    vol_mean: f32,
    vol_var: f32,

    // Onset detection: adaptive threshold on spectral flux with a refractory period.
    flux_ema: f32,
    flux_dev: f32,
    onset_refractory: f32,
    onset_rate: f32,

    // Slow spectral character.
    bass_w_slow: f32,
    mid_w_slow: f32,
    treble_w_slow: f32,
    brightness_slow: f32,
    density_slow: f32,
    percussive_slow: f32,
    sustained_slow: f32,

    // Derivatives and rolling metrics
    prev_rms: f32,
    prev_spectral_centroid: f32,
    stable_time_accum: f32,
}

impl DspEngine {
    /// Create a new DSP engine with specified FFT window size (typically 2048).
    pub fn new(fft_size: usize) -> Self {
        let fft_size = fft_size.max(256).next_power_of_two();

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
            env_fast: 0.0,
            env_mid: 0.0,
            env_slow: 0.0,
            env_peak: 0.0,
            // Starts at a plausible mid-level rather than 0, so the first second of a
            // track does not divide by an almost-zero reference and read as maximally
            // loud before the follower has seen anything.
            loudness_ref: 0.08,
            // Starts equal to the ceiling, i.e. "no dynamic range known yet". The
            // asymmetry matters: the floor drops in seconds, so one quiet bar is enough
            // to establish that a track has range, while it takes a minute and a half of
            // unbroken level to conclude that it does not. Starting it low instead would
            // credit every source with range it has not demonstrated.
            loudness_floor: 0.08,
            vol_mean: 0.0,
            vol_var: 0.0,
            flux_ema: 0.0,
            flux_dev: 0.0,
            onset_refractory: 0.0,
            onset_rate: 0.0,
            bass_w_slow: 0.34,
            mid_w_slow: 0.33,
            treble_w_slow: 0.33,
            brightness_slow: 0.2,
            density_slow: 0.3,
            percussive_slow: 0.5,
            sustained_slow: 0.5,
            prev_rms: 0.0,
            prev_spectral_centroid: 0.0,
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
        let spectral_centroid_norm =
            (spectral_centroid_hz / (sample_rate as f32 * 0.5)).clamp(0.0, 1.0);

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

        // -------------------------------------------------------------------
        // Energy envelopes. Four timescales, all time-based.
        // -------------------------------------------------------------------
        self.env_fast = ema(self.env_fast, raw_rms, 0.15, dt);
        self.env_mid = ema(self.env_mid, raw_rms, 1.5, dt);
        self.env_slow = ema(self.env_slow, raw_rms, 25.0, dt);

        // Local crest follower: rises immediately, releases over ~1 s.
        self.env_peak = if self.env_fast > self.env_peak {
            self.env_fast
        } else {
            ema(self.env_peak, self.env_fast, 1.0, dt).max(self.env_fast)
        };

        // Track loudness ceiling. Rises in ~0.4 s so a chorus sets it almost at once, and
        // falls over ~60 s so a quiet verse does not drag the reference down with it and
        // make the next quiet passage read as loud.
        self.loudness_ref = if self.env_fast > self.loudness_ref {
            ema(self.loudness_ref, self.env_fast, 0.4, dt)
        } else {
            ema(self.loudness_ref, self.env_fast, 60.0, dt)
        }
        .max(0.004);

        let level_norm = (self.env_mid / self.loudness_ref).clamp(0.0, 1.0);

        // Floor follower, the mirror of the ceiling. Gated on there being something
        // playing at all, so silence between tracks does not set the floor to zero and
        // make everything after it look enormously dynamic.
        if self.env_mid > 0.006 {
            self.loudness_floor = if self.env_mid < self.loudness_floor {
                ema(self.loudness_floor, self.env_mid, 2.0, dt)
            } else {
                ema(self.loudness_floor, self.env_mid, 90.0, dt)
            };
        }
        self.loudness_floor = self.loudness_floor.clamp(1e-4, self.loudness_ref);

        // How much range this material actually has, floor against ceiling, both on long
        // timescales. This is the quantity that separates music from a source that is
        // simply *on* — narration, a podcast, a stream — and it has to be measured
        // between two long-memory followers rather than against a rolling average,
        // because a rolling average converges to whatever is playing and so reports no
        // range for a sustained passage and no range for continuous speech alike.
        let dynamic_range = (1.0 - self.loudness_floor / self.loudness_ref).clamp(0.0, 1.0);

        // Volatility over ~4 s as a running mean/variance pair. The x5 scale is kept from
        // the windowed version it replaces so the visual side's tuning still holds.
        self.vol_mean = ema(self.vol_mean, raw_rms, 4.0, dt);
        let dev = raw_rms - self.vol_mean;
        self.vol_var = ema(self.vol_var, dev * dev, 4.0, dt);
        let volatility = self.vol_var.max(0.0).sqrt() * 5.0;

        if volatility < 0.05 {
            self.stable_time_accum += dt;
        } else {
            self.stable_time_accum = (self.stable_time_accum - dt * 2.0).max(0.0);
        }

        // -------------------------------------------------------------------
        // Onset detection — adaptive threshold on spectral flux.
        // -------------------------------------------------------------------
        self.onset_refractory = (self.onset_refractory - dt).max(0.0);
        let threshold = self.flux_ema + 1.6 * self.flux_dev + 0.004;
        let onset = spectral_flux > threshold && self.onset_refractory <= 0.0;
        if onset {
            self.onset_refractory = 0.075;
        }
        self.flux_ema = ema(self.flux_ema, spectral_flux, 0.35, dt);
        self.flux_dev = ema(self.flux_dev, (spectral_flux - self.flux_ema).abs(), 0.5, dt);

        // An onset frame contributes 1/dt to the rate, so the ~3 s average reads directly
        // in onsets per second regardless of frame rate.
        let rate_sample = if onset { 1.0 / dt } else { 0.0 };
        self.onset_rate = ema(self.onset_rate, rate_sample, 3.0, dt).clamp(0.0, 30.0);

        // -------------------------------------------------------------------
        // Song character. Slow enough to describe the material, not the moment.
        // -------------------------------------------------------------------
        let bass_raw = self.smoothed_bands[0] + self.smoothed_bands[1];
        let mid_raw = self.smoothed_bands[2] + self.smoothed_bands[3] + self.smoothed_bands[4];
        let treble_raw = self.smoothed_bands[5] + self.smoothed_bands[6];
        let band_total = (bass_raw + mid_raw + treble_raw).max(1e-4);

        self.bass_w_slow = ema(self.bass_w_slow, bass_raw / band_total, 20.0, dt);
        self.mid_w_slow = ema(self.mid_w_slow, mid_raw / band_total, 20.0, dt);
        self.treble_w_slow = ema(self.treble_w_slow, treble_raw / band_total, 20.0, dt);
        self.brightness_slow = ema(self.brightness_slow, spectral_centroid_norm, 20.0, dt);

        // How much of the spectrum is actually occupied — a wall of sound scores near 1,
        // a solo instrument near 0.
        let occupancy = self
            .smoothed_bands
            .iter()
            .map(|&b| smoothstep(0.12, 0.45, b))
            .sum::<f32>()
            / 7.0;
        self.density_slow = ema(self.density_slow, occupancy, 15.0, dt);

        // Crest: how far the local peaks stand above the local mean. Drums make this
        // large; a held chord makes it ~1.
        let crest = (self.env_peak / self.env_mid.max(1e-4)).clamp(1.0, 6.0);

        // The two axes that decide which visual configuration a track gets. Percussive
        // and sustained are computed independently rather than as `1 - other`, because a
        // dense mix can genuinely be both (a rock band under a held organ chord) and a
        // near-silent passage is neither.
        let percussive_now = (smoothstep(0.8, 5.0, self.onset_rate) * 0.6
            + smoothstep(1.15, 2.4, crest) * 0.4)
            .clamp(0.0, 1.0);
        let sustained_now = ((1.0 - smoothstep(0.6, 4.0, self.onset_rate))
            * (1.0 - smoothstep(1.2, 2.4, crest))
            * smoothstep(0.10, 0.35, level_norm))
        .clamp(0.0, 1.0);

        self.percussive_slow = ema(self.percussive_slow, percussive_now, 12.0, dt);
        self.sustained_slow = ema(self.sustained_slow, sustained_now, 12.0, dt);

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
                onset,
            },
            long_term: LongTermFeatures {
                avg_energy: self.env_mid.clamp(0.0, 1.0),
                slow_energy: self.env_slow.clamp(0.0, 1.0),
                loudness_ref: self.loudness_ref,
                level_norm,
                dynamic_range,
                dominant_frequency: spectral_centroid_norm,
                volatility: volatility.clamp(0.0, 1.0),
                stability_duration: self.stable_time_accum,
            },
            bands: BandFeatures {
                energy: self.smoothed_bands,
                velocity: band_velocities,
                acceleration: band_accelerations,
                recent_peak: self.band_peaks,
                baseline: [self.env_mid; 7],
                stability: [1.0 - volatility; 7],
            },
            character: SongCharacter {
                onset_rate: self.onset_rate,
                percussive: self.percussive_slow,
                sustained: self.sustained_slow,
                bass_w: self.bass_w_slow,
                mid_w: self.mid_w_slow,
                treble_w: self.treble_w_slow,
                brightness: self.brightness_slow,
                density: self.density_slow,
            },
        }
    }

    /// Compute raw frequency bands with perceptual log-gain curves.
    fn compute_raw_bands(&self, magnitudes: &[f32], sample_rate: u32) -> [f32; 7] {
        let boundaries = [
            (20.0, 60.0),      // Sub-bass
            (60.0, 250.0),     // Bass
            (250.0, 500.0),    // Low-mid
            (500.0, 2000.0),   // Mid
            (2000.0, 4000.0),  // High-mid
            (4000.0, 8000.0),  // Treble
            (8000.0, 20000.0), // Brilliance
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
}
