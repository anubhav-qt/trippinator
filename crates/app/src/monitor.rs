//! Monitor detection — finds secondary portrait displays.

use log::info;
use winit::event_loop::ActiveEventLoop;
use winit::monitor::MonitorHandle;

/// Attempts to find a secondary monitor in portrait orientation.
///
/// Falls back to the primary monitor if no portrait secondary is found.
pub fn find_secondary_portrait_monitor(event_loop: &ActiveEventLoop) -> Option<MonitorHandle> {
    let monitors: Vec<MonitorHandle> = event_loop.available_monitors().collect();

    info!("Detected {} monitor(s):", monitors.len());
    for (i, monitor) in monitors.iter().enumerate() {
        let size = monitor.size();
        let orientation = if size.height > size.width {
            "portrait"
        } else {
            "landscape"
        };
        info!(
            "  Monitor {}: {}x{} ({}) — {:?}",
            i,
            size.width,
            size.height,
            orientation,
            monitor.name().unwrap_or_else(|| "unnamed".into())
        );
    }

    // Strategy:
    // 1. Look for a non-primary portrait monitor
    // 2. Fall back to any secondary monitor
    // 3. Fall back to primary

    let primary = event_loop.primary_monitor();

    // Try to find a portrait secondary
    let portrait_secondary = monitors.iter().find(|m| {
        let size = m.size();
        let is_portrait = size.height > size.width;
        let is_secondary = primary.as_ref().map_or(true, |p| *m != p);
        is_portrait && is_secondary
    });

    if let Some(monitor) = portrait_secondary {
        let size = monitor.size();
        info!(
            "Selected portrait secondary monitor: {}x{} — {:?}",
            size.width,
            size.height,
            monitor.name().unwrap_or_else(|| "unnamed".into())
        );
        return Some(monitor.clone());
    }

    // Fall back to any secondary
    let any_secondary = monitors.iter().find(|m| {
        primary.as_ref().map_or(false, |p| *m != p)
    });

    if let Some(monitor) = any_secondary {
        let size = monitor.size();
        info!(
            "No portrait secondary found. Using secondary: {}x{} — {:?}",
            size.width,
            size.height,
            monitor.name().unwrap_or_else(|| "unnamed".into())
        );
        return Some(monitor.clone());
    }

    info!("No secondary monitor found. Will use primary/default.");
    primary
}
