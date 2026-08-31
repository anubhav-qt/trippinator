//! # Screen Capture
//!
//! Captures the entire visible desktop and hardware-downscales to a low-resolution
//! texture (128x72) for the analysis layer.
//!
//! Provides `AsyncScreenCapture` which captures on a dedicated background thread
//! at ~30-60 Hz so the main 165Hz render loop is never blocked.

mod capture;

pub use capture::{ScreenCapture, ScreenFrame};
use anyhow::Result;
use crossbeam_channel::{Receiver, Sender};
use std::time::Duration;

/// The default resolution for the downscaled screen texture.
pub const DEFAULT_CAPTURE_WIDTH: u32 = 128;
pub const DEFAULT_CAPTURE_HEIGHT: u32 = 72;

/// Dedicated background screen capture worker.
pub struct AsyncScreenCapture {
    receiver: Receiver<ScreenFrame>,
    _handle: std::thread::JoinHandle<()>,
}

impl AsyncScreenCapture {
    /// Start async desktop screen capture at the given target resolution and interval.
    pub fn start(width: u32, height: u32, interval_ms: u64) -> Result<Self> {
        let (tx, rx): (Sender<ScreenFrame>, Receiver<ScreenFrame>) = crossbeam_channel::bounded(2);

        let handle = std::thread::Builder::new()
            .name("trippinator-screen-capture".into())
            .spawn(move || {
                log::info!("Screen capture worker thread started ({width}x{height} @ {interval_ms}ms)");
                let mut capture = match ScreenCapture::new(width, height) {
                    Ok(c) => c,
                    Err(e) => {
                        log::error!("Failed to initialize screen capture: {e}");
                        return;
                    }
                };

                let interval = Duration::from_millis(interval_ms.max(8));

                loop {
                    let start = std::time::Instant::now();

                    match capture.capture() {
                        Ok(frame) => {
                            // Drop old frame if receiver buffer is full
                            let _ = tx.try_send(frame);
                        }
                        Err(e) => {
                            log::warn!("Screen capture tick error: {e}");
                        }
                    }

                    let elapsed = start.elapsed();
                    if elapsed < interval {
                        std::thread::sleep(interval - elapsed);
                    }
                }
            })?;

        Ok(Self {
            receiver: rx,
            _handle: handle,
        })
    }

    /// Try to receive the latest captured screen frame.
    pub fn try_recv(&self) -> Option<ScreenFrame> {
        let mut latest = None;
        while let Ok(frame) = self.receiver.try_recv() {
            latest = Some(frame);
        }
        latest
    }
}
