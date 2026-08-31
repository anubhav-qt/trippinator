//! Motion field — detects temporal differences, localized activity, and directional motion.

/// A field of motion magnitudes and spatial concentration.
#[derive(Debug, Clone)]
pub struct MotionField {
    /// Per-cell motion magnitude (0..1).
    pub magnitudes: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

impl MotionField {
    /// Compute motion between two consecutive frames.
    pub fn compute(prev: &[u8], curr: &[u8], width: u32, height: u32) -> Self {
        let cell_count = (width * height) as usize;
        let mut magnitudes = Vec::with_capacity(cell_count);

        for i in 0..cell_count {
            let offset = i * 4;
            if offset + 2 < prev.len() && offset + 2 < curr.len() {
                let dr = (curr[offset] as f32 - prev[offset] as f32).abs();
                let dg = (curr[offset + 1] as f32 - prev[offset + 1] as f32).abs();
                let db = (curr[offset + 2] as f32 - prev[offset + 2] as f32).abs();
                let diff = (dr * 0.299 + dg * 0.587 + db * 0.114) / 255.0;
                // Non-linear response to filter out subtle video compression noise
                let thresholded = if diff > 0.02 { diff * 1.5 } else { 0.0 };
                magnitudes.push(thresholded.clamp(0.0, 1.0));
            } else {
                magnitudes.push(0.0);
            }
        }

        Self {
            magnitudes,
            width,
            height,
        }
    }

    /// Create a zero motion field.
    pub fn zero(width: u32, height: u32) -> Self {
        Self {
            magnitudes: vec![0.0; (width * height) as usize],
            width,
            height,
        }
    }

    /// Global motion intensity (average of all cells, scaled 0..1).
    pub fn global_intensity(&self) -> f32 {
        if self.magnitudes.is_empty() {
            return 0.0;
        }
        let avg = self.magnitudes.iter().sum::<f32>() / self.magnitudes.len() as f32;
        (avg * 4.0).clamp(0.0, 1.0)
    }

    /// Maximum localized motion spike.
    pub fn max_local_motion(&self) -> f32 {
        self.magnitudes.iter().copied().fold(0.0f32, f32::max)
    }

    /// Spatial centroid of motion [x (0..1), y (0..1)].
    pub fn motion_centroid(&self) -> [f32; 2] {
        if self.magnitudes.is_empty() || self.width == 0 || self.height == 0 {
            return [0.5, 0.5];
        }

        let mut x_acc = 0.0f32;
        let mut y_acc = 0.0f32;
        let mut total_m = 0.0f32;

        for y in 0..self.height {
            let ny = y as f32 / self.height as f32;
            for x in 0..self.width {
                let nx = x as f32 / self.width as f32;
                let m = self.magnitudes[(y * self.width + x) as usize];
                x_acc += nx * m;
                y_acc += ny * m;
                total_m += m;
            }
        }

        if total_m > 1e-5 {
            [x_acc / total_m, y_acc / total_m]
        } else {
            [0.5, 0.5]
        }
    }
}
