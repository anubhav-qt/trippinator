//! # Compositor
//!
//! Blends outputs from all visual systems, applies tonemapping,
//! color grading, and temporal feedback to produce the final image.
//!
//! ```text
//! System outputs → blend → tonemap → color grade → present
//!       ↑                                    │
//!       └──── temporal feedback ─────────────┘
//! ```

pub struct Compositor {
    // TODO: blend weights, tonemap params, feedback state
}

impl Compositor {
    pub fn new() -> Self {
        Self {}
    }
}
