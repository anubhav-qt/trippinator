//! # Trippinator
//!
//! A real-time generative audiovisual organism.
//!
//! Input does not control the image. Input perturbs a system that controls itself.

mod engine;
mod monitor;

use anyhow::Result;
use log::info;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Fullscreen, Window, WindowAttributes, WindowId};

use crate::engine::Engine;
use crate::monitor::find_secondary_portrait_monitor;

/// The top-level application state.
struct App {
    window: Option<Arc<Window>>,
    engine: Option<Engine>,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            engine: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        info!("Application resumed — setting up window");

        let monitor = find_secondary_portrait_monitor(event_loop);

        let (target_pos, target_size) = if let Some(m) = &monitor {
            (m.position(), m.size())
        } else {
            (
                winit::dpi::PhysicalPosition::new(-1080, -452),
                winit::dpi::PhysicalSize::new(1080, 1920),
            )
        };

        info!(
            "Creating window for portrait monitor at position {:?}, size {:?}",
            target_pos, target_size
        );

        let attrs = WindowAttributes::default()
            .with_title("trippinator")
            .with_position(target_pos)
            .with_inner_size(target_size)
            .with_decorations(false)
            .with_resizable(false)
            .with_active(true)
            .with_visible(true);

        match event_loop.create_window(attrs) {
            Ok(window) => {
                let window = Arc::new(window);
                window.set_outer_position(target_pos);
                window.set_visible(true);
                window.focus_window();

                info!(
                    "Window created: {:?} (size: {:?}, pos: {:?})",
                    window.id(),
                    window.inner_size(),
                    window.outer_position()
                );

                // Initialize GPU engine
                match Engine::new(Arc::clone(&window)) {
                    Ok(engine) => {
                        self.engine = Some(engine);
                        info!("GPU engine initialized");
                    }
                    Err(e) => {
                        log::error!("Failed to initialize GPU engine: {e}");
                        event_loop.exit();
                        return;
                    }
                }

                // Initial redraw request
                window.request_redraw();
                self.window = Some(window);
            }
            Err(e) => {
                log::error!("Failed to create window: {e}");
                event_loop.exit();
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                info!("Close requested — exiting");
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                use winit::keyboard::{Key, NamedKey};
                if event.logical_key == Key::Named(NamedKey::Escape) {
                    info!("Escape pressed — exiting");
                    event_loop.exit();
                } else if event.logical_key == Key::Named(NamedKey::F11) {
                    if let Some(window) = &self.window {
                        if window.fullscreen().is_some() {
                            window.set_fullscreen(None);
                        } else {
                            window.set_fullscreen(Some(Fullscreen::Borderless(None)));
                        }
                    }
                }
            }
            WindowEvent::Resized(new_size) => {
                if let Some(engine) = &mut self.engine {
                    engine.resize(new_size.width, new_size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(engine) = &mut self.engine {
                    match engine.render() {
                        Ok(()) => {}
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            if let Some(window) = &self.window {
                                let size = window.inner_size();
                                engine.resize(size.width, size.height);
                            }
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            log::error!("Out of GPU memory — exiting");
                            event_loop.exit();
                        }
                        Err(e) => {
                            log::warn!("Render error (continuing): {e}");
                        }
                    }
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    info!("╔══════════════════════════════════════════════╗");
    info!("║         TRIPPINATOR — Starting Up            ║");
    info!("║  Input perturbs a system that controls itself ║");
    info!("╚══════════════════════════════════════════════╝");

    let event_loop = EventLoop::new()?;
    let mut app = App::new();
    event_loop.run_app(&mut app)?;

    Ok(())
}
