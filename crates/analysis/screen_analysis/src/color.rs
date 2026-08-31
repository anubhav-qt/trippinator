//! Spatial color map — low-resolution color representation of the desktop with circular hue tracking and focus weighting.

use std::f32::consts::PI;

/// A low-resolution spatial color representation of the desktop.
#[derive(Debug, Clone)]
pub struct SpatialColorMap {
    /// HSL values per cell: [h (0..1), s (0..1), l (0..1)].
    pub cells: Vec<[f32; 3]>,
    /// Average RGB color of the desktop [r, g, b].
    pub avg_rgb: [f32; 3],
    pub width: u32,
    pub height: u32,
}

impl SpatialColorMap {
    /// Build a spatial color map from RGBA pixel data with optional focus weighting.
    pub fn from_pixels(
        pixels: &[u8],
        width: u32,
        height: u32,
        focused_rect: Option<[f32; 4]>,
    ) -> Self {
        let cell_count = (width * height) as usize;
        let mut cells = Vec::with_capacity(cell_count);

        let mut r_sum = 0.0f32;
        let mut g_sum = 0.0f32;
        let mut b_sum = 0.0f32;
        let mut total_weight = 0.0f32;

        let alpha_focus = 1.5f32; // Focus weight boost

        for y in 0..height {
            let norm_y = y as f32 / height as f32;
            for x in 0..width {
                let norm_x = x as f32 / width as f32;
                let i = (y * width + x) as usize;
                let offset = i * 4;

                let (r, g, b) = if offset + 2 < pixels.len() {
                    (
                        pixels[offset] as f32 / 255.0,
                        pixels[offset + 1] as f32 / 255.0,
                        pixels[offset + 2] as f32 / 255.0,
                    )
                } else {
                    (0.0, 0.0, 0.0)
                };

                let mut weight = 1.0f32;
                if let Some([fx, fy, fw, fh]) = focused_rect {
                    if norm_x >= fx && norm_x <= fx + fw && norm_y >= fy && norm_y <= fy + fh {
                        weight += alpha_focus;
                    }
                }

                r_sum += r * weight;
                g_sum += g * weight;
                b_sum += b * weight;
                total_weight += weight;

                cells.push(rgb_to_hsl(r, g, b));
            }
        }

        let denom = total_weight.max(1.0);
        let avg_rgb = [r_sum / denom, g_sum / denom, b_sum / denom];

        Self {
            cells,
            avg_rgb,
            width,
            height,
        }
    }

    /// Average brightness across all cells (0..1).
    pub fn avg_brightness(&self) -> f32 {
        if self.cells.is_empty() {
            return 0.0;
        }
        self.cells.iter().map(|c| c[2]).sum::<f32>() / self.cells.len() as f32
    }

    /// Average saturation across all cells (0..1).
    pub fn avg_saturation(&self) -> f32 {
        if self.cells.is_empty() {
            return 0.0;
        }
        self.cells.iter().map(|c| c[1]).sum::<f32>() / self.cells.len() as f32
    }

    /// Dominant hue computed via saturation-weighted circular mean (in degrees: 0..360).
    pub fn dominant_hue(&self) -> f32 {
        if self.cells.is_empty() {
            return 0.0;
        }

        let mut sin_sum = 0.0f32;
        let mut cos_sum = 0.0f32;

        for c in &self.cells {
            let h_rad = c[0] * 2.0 * PI;
            let weight = c[1] * (0.2 + (0.5 - (c[2] - 0.5).abs()).max(0.0)); // Weight saturated, non-extreme tones

            sin_sum += h_rad.sin() * weight;
            cos_sum += h_rad.cos() * weight;
        }

        let angle = sin_sum.atan2(cos_sum);
        let normalized = if angle < 0.0 {
            angle + 2.0 * PI
        } else {
            angle
        };

        normalized * (180.0 / PI)
    }

    /// Local contrast (difference between max and min luminance).
    pub fn contrast(&self) -> f32 {
        if self.cells.is_empty() {
            return 0.0;
        }
        let min_l = self.cells.iter().map(|c| c[2]).fold(f32::MAX, f32::min);
        let max_l = self.cells.iter().map(|c| c[2]).fold(f32::MIN, f32::max);
        (max_l - min_l).clamp(0.0, 1.0)
    }
}

/// Convert RGB (0..1) to HSL (0..1, 0..1, 0..1).
fn rgb_to_hsl(r: f32, g: f32, b: f32) -> [f32; 3] {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) * 0.5;

    if (max - min).abs() < 1e-6 {
        return [0.0, 0.0, l];
    }

    let d = max - min;
    let s = if l > 0.5 {
        d / (2.0 - max - min)
    } else {
        d / (max + min)
    };

    let h = if (max - r).abs() < 1e-6 {
        ((g - b) / d + if g < b { 6.0 } else { 0.0 }) / 6.0
    } else if (max - g).abs() < 1e-6 {
        ((b - r) / d + 2.0) / 6.0
    } else {
        ((r - g) / d + 4.0) / 6.0
    };

    [h.fract(), s.clamp(0.0, 1.0), l.clamp(0.0, 1.0)]
}
