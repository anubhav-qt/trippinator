//! # Screen Analysis
//!
//! Extracts spatial color, brightness, saturation, local contrast,
//! edges, gradients, temporal difference, and motion fields from
//! captured desktop frames.

mod analyzer;
mod color;
mod motion;

pub use analyzer::{ScreenAnalyzer, ScreenFeatures};
pub use color::SpatialColorMap;
pub use motion::MotionField;
