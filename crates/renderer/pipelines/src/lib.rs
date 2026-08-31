//! # Render Pipelines
//!
//! Manages wgpu render and compute pipelines, shader module loading,
//! and bind group layouts for the visual systems.

pub struct PipelineCache {
    // TODO: HashMap of pipeline name → compiled pipeline
}

impl PipelineCache {
    pub fn new() -> Self {
        Self {}
    }
}
