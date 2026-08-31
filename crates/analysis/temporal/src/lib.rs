//! # Temporal Analysis
//!
//! Provides temporal smoothing, exponential moving averages, trend detection,
//! and multi-timescale signal integration utilities used across the system.

/// Exponential moving average filter.
pub struct Ema {
    value: f32,
    alpha: f32,
    initialized: bool,
}

impl Ema {
    /// Create a new EMA with the given smoothing factor (0..1).
    /// Lower alpha = smoother (more lag). Higher alpha = more responsive.
    pub fn new(alpha: f32) -> Self {
        Self {
            value: 0.0,
            alpha: alpha.clamp(0.001, 1.0),
            initialized: false,
        }
    }

    /// Create an EMA from a desired time constant in seconds and frame rate.
    pub fn from_time_constant(time_constant_secs: f32, fps: f32) -> Self {
        let alpha = 1.0 - (-1.0 / (time_constant_secs * fps)).exp();
        Self::new(alpha)
    }

    /// Update with a new sample and return the smoothed value.
    pub fn update(&mut self, sample: f32) -> f32 {
        if !self.initialized {
            self.value = sample;
            self.initialized = true;
        } else {
            self.value += self.alpha * (sample - self.value);
        }
        self.value
    }

    pub fn value(&self) -> f32 {
        self.value
    }

    pub fn reset(&mut self) {
        self.initialized = false;
        self.value = 0.0;
    }
}

/// Tracks a signal's derivative (velocity) with smoothing.
pub struct VelocityTracker {
    prev_value: f32,
    velocity: Ema,
    initialized: bool,
}

impl VelocityTracker {
    pub fn new(smoothing: f32) -> Self {
        Self {
            prev_value: 0.0,
            velocity: Ema::new(smoothing),
            initialized: false,
        }
    }

    pub fn update(&mut self, value: f32, dt: f32) -> f32 {
        if !self.initialized {
            self.prev_value = value;
            self.initialized = true;
            return 0.0;
        }
        let raw_velocity = (value - self.prev_value) / dt.max(1e-6);
        self.prev_value = value;
        self.velocity.update(raw_velocity)
    }

    pub fn velocity(&self) -> f32 {
        self.velocity.value()
    }
}

/// Peak tracker with configurable decay.
pub struct PeakTracker {
    peak: f32,
    decay_rate: f32,
}

impl PeakTracker {
    pub fn new(decay_rate: f32) -> Self {
        Self {
            peak: 0.0,
            decay_rate,
        }
    }

    pub fn update(&mut self, value: f32, dt: f32) -> f32 {
        if value > self.peak {
            self.peak = value;
        } else {
            self.peak *= (-self.decay_rate * dt).exp();
        }
        self.peak
    }

    pub fn peak(&self) -> f32 {
        self.peak
    }
}
