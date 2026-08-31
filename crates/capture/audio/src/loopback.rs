//! WASAPI loopback audio capture with device selection.
//!
//! Captures the output of a specific audio device (or default) via WASAPI
//! loopback mode, which lets us hear exactly what the device is playing.

use anyhow::{anyhow, Result};
use crossbeam_channel::{Receiver, Sender};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Host, SampleFormat, StreamConfig};
use parking_lot::Mutex;
use std::sync::Arc;

/// A buffer of captured audio samples.
pub struct AudioBuffer {
    /// Interleaved f32 samples (converted from any native format).
    pub samples: Vec<f32>,
    /// Number of channels.
    pub channels: u16,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Monotonic frame counter.
    pub frame: u64,
}

/// Strategy for selecting an audio output device to capture.
#[derive(Debug, Clone)]
pub enum DeviceSelection {
    /// Use the system default output device.
    Default,
    /// Match by partial name (case-insensitive).
    ByName(String),
    /// Use the first non-default output device found.
    FirstNonDefault,
}

/// Enumerate available output devices and return their names.
pub fn list_output_devices() -> Result<Vec<String>> {
    let host = cpal::default_host();
    let devices = host.output_devices()?;
    Ok(devices
        .filter_map(|d| d.name().ok())
        .collect())
}

/// Find an output device matching the given selection strategy.
fn find_device(host: &Host, selection: &DeviceSelection) -> Result<Device> {
    match selection {
        DeviceSelection::Default => host
            .default_output_device()
            .ok_or_else(|| anyhow!("No default output device found")),

        DeviceSelection::ByName(name) => {
            let name_lower = name.to_lowercase();
            let devices = host.output_devices()?;
            for device in devices {
                if let Ok(device_name) = device.name() {
                    if device_name.to_lowercase().contains(&name_lower) {
                        log::info!("Matched audio device: {}", device_name);
                        return Ok(device);
                    }
                }
            }
            Err(anyhow!("No output device matching '{name}' found"))
        }

        DeviceSelection::FirstNonDefault => {
            let default_name = host
                .default_output_device()
                .and_then(|d| d.name().ok())
                .unwrap_or_default();

            let devices = host.output_devices()?;
            for device in devices {
                if let Ok(device_name) = device.name() {
                    if device_name != default_name {
                        log::info!("Selected non-default audio device: {}", device_name);
                        return Ok(device);
                    }
                }
            }
            // Fallback to default
            host.default_output_device()
                .ok_or_else(|| anyhow!("No output devices found"))
        }
    }
}

/// Captures system audio from a specific output device via WASAPI loopback.
///
/// Runs on a dedicated thread. Samples arrive via `drain_latest()`.
pub struct AudioCapture {
    receiver: Receiver<AudioBuffer>,
    frame_counter: Arc<Mutex<u64>>,
    _stream: cpal::Stream,
}

impl AudioCapture {
    /// Start capturing from the specified device.
    pub fn start(selection: DeviceSelection) -> Result<Self> {
        let host = cpal::default_host();

        // Log all available devices
        log::info!("Available audio output devices:");
        if let Ok(devices) = host.output_devices() {
            for device in devices {
                if let Ok(name) = device.name() {
                    log::info!("  - {}", name);
                }
            }
        }

        let device = find_device(&host, &selection)?;
        let device_name = device.name().unwrap_or_else(|_| "unknown".into());
        log::info!("Capturing audio from: {}", device_name);

        let config = device.default_output_config()?;
        log::info!(
            "Audio config: {} Hz, {} ch, {:?}",
            config.sample_rate().0,
            config.channels(),
            config.sample_format()
        );

        let sample_rate = config.sample_rate().0;
        let channels = config.channels();

        let (tx, rx): (Sender<AudioBuffer>, Receiver<AudioBuffer>) =
            crossbeam_channel::bounded(64);

        let frame_counter = Arc::new(Mutex::new(0u64));
        let frame_counter_clone = frame_counter.clone();

        // Build the loopback stream
        let stream = match config.sample_format() {
            SampleFormat::F32 => {
                let tx = tx.clone();
                device.build_input_stream(
                    &StreamConfig {
                        channels,
                        sample_rate: config.sample_rate(),
                        buffer_size: cpal::BufferSize::Default,
                    },
                    move |data: &[f32], _| {
                        let mut frame = frame_counter_clone.lock();
                        *frame += 1;
                        let buf = AudioBuffer {
                            samples: data.to_vec(),
                            channels,
                            sample_rate,
                            frame: *frame,
                        };
                        // Drop old buffers if the receiver is full
                        let _ = tx.try_send(buf);
                    },
                    |err| log::error!("Audio stream error: {err}"),
                    None,
                )?
            }
            SampleFormat::I16 => {
                let tx = tx.clone();
                device.build_input_stream(
                    &StreamConfig {
                        channels,
                        sample_rate: config.sample_rate(),
                        buffer_size: cpal::BufferSize::Default,
                    },
                    move |data: &[i16], _| {
                        let mut frame = frame_counter_clone.lock();
                        *frame += 1;
                        let samples: Vec<f32> =
                            data.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
                        let buf = AudioBuffer {
                            samples,
                            channels,
                            sample_rate,
                            frame: *frame,
                        };
                        let _ = tx.try_send(buf);
                    },
                    |err| log::error!("Audio stream error: {err}"),
                    None,
                )?
            }
            SampleFormat::U16 => {
                let tx = tx.clone();
                device.build_input_stream(
                    &StreamConfig {
                        channels,
                        sample_rate: config.sample_rate(),
                        buffer_size: cpal::BufferSize::Default,
                    },
                    move |data: &[u16], _| {
                        let mut frame = frame_counter_clone.lock();
                        *frame += 1;
                        let samples: Vec<f32> = data
                            .iter()
                            .map(|&s| (s as f32 / u16::MAX as f32) * 2.0 - 1.0)
                            .collect();
                        let buf = AudioBuffer {
                            samples,
                            channels,
                            sample_rate,
                            frame: *frame,
                        };
                        let _ = tx.try_send(buf);
                    },
                    |err| log::error!("Audio stream error: {err}"),
                    None,
                )?
            }
            fmt => return Err(anyhow!("Unsupported audio format: {fmt:?}")),
        };

        stream.play()?;
        log::info!("Audio capture stream started");

        Ok(Self {
            receiver: rx,
            frame_counter,
            _stream: stream,
        })
    }

    /// Try to get the latest audio buffer without blocking.
    pub fn try_recv(&self) -> Option<AudioBuffer> {
        self.receiver.try_recv().ok()
    }

    /// Drain all pending buffers, returning only the most recent one.
    /// Use this when you want to stay current and skip stale data.
    pub fn drain_latest(&self) -> Option<AudioBuffer> {
        let mut latest = None;
        while let Ok(buf) = self.receiver.try_recv() {
            latest = Some(buf);
        }
        latest
    }

    /// Current frame count.
    pub fn frame(&self) -> u64 {
        *self.frame_counter.lock()
    }
}
