//! Core screen analyzer that produces spatial features from captured frames.

use crate::color::SpatialColorMap;
use crate::motion::MotionField;
use trippinator_capture_screen::ScreenFrame;

/// Aggregated screen features for the dynamics layer.
#[derive(Debug, Clone)]
pub struct ScreenFeatures {
    /// Average brightness (0..1).
    pub avg_brightness: f32,
    /// Average saturation (0..1).
    pub avg_saturation: f32,
    /// Dominant hue (0..360 degrees).
    pub dominant_hue: f32,
    /// Global contrast (0..1).
    pub contrast: f32,
    /// Global motion intensity (0..1).
    pub motion_intensity: f32,
    /// Maximum local motion spike (0..1).
    pub max_local_motion: f32,
    /// Motion spatial centroid [x (0..1), y (0..1)].
    pub motion_centroid: [f32; 2],
    /// Average RGB color [r, g, b].
    pub avg_rgb: [f32; 3],
    /// Spatial color map.
    pub color_map: SpatialColorMap,
    /// Motion field.
    pub motion_field: MotionField,
}

/// Analyzes screen frames to extract features.
pub struct ScreenAnalyzer {
    prev_frame: Option<Vec<u8>>,
    pub width: u32,
    pub height: u32,
}

impl ScreenAnalyzer {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            prev_frame: None,
            width,
            height,
        }
    }

    /// Analyze a captured frame and produce features.
    pub fn analyze(&mut self, frame: &ScreenFrame) -> ScreenFeatures {
        let color_map = SpatialColorMap::from_pixels(
            &frame.pixels,
            frame.width,
            frame.height,
            frame.focused_rect,
        );

        let motion_field = if let Some(prev) = &self.prev_frame {
            MotionField::compute(prev, &frame.pixels, frame.width, frame.height)
        } else {
            MotionField::zero(frame.width, frame.height)
        };

        let avg_brightness = color_map.avg_brightness();
        let avg_saturation = color_map.avg_saturation();
        let dominant_hue = color_map.dominant_hue();
        let contrast = color_map.contrast();
        let motion_intensity = motion_field.global_intensity();
        let max_local_motion = motion_field.max_local_motion();
        let motion_centroid = motion_field.motion_centroid();
        let avg_rgb = color_map.avg_rgb;

        self.prev_frame = Some(frame.pixels.clone());

        ScreenFeatures {
            avg_brightness,
            avg_saturation,
            dominant_hue,
            contrast,
            motion_intensity,
            max_local_motion,
            motion_centroid,
            avg_rgb,
            color_map,
            motion_field,
        }
    }
}
