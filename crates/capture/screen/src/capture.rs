//! Windows Desktop Screen Capture.
//!
//! Clean, robust desktop capture via GDI and GetDIBits with downsampling.

use anyhow::{anyhow, Result};
use std::time::Instant;
use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Gdi::{
    BitBlt, CreateCompatibleBitmap, CreateCompatibleDC, DeleteDC, DeleteObject, GetDC, GetDIBits,
    ReleaseDC, SelectObject, BITMAPINFO, BITMAPINFOHEADER, BI_RGB, DIB_RGB_COLORS, HGDIOBJ,
    SRCCOPY,
};
use windows::Win32::UI::WindowsAndMessaging::{
    GetForegroundWindow, GetSystemMetrics, GetWindowRect, SM_CXSCREEN, SM_CYSCREEN,
};

/// A captured screen frame downscaled for analysis.
#[derive(Debug, Clone)]
pub struct ScreenFrame {
    /// RGBA pixel data, row-major (top-to-bottom, left-to-right).
    pub pixels: Vec<u8>,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Timestamp of capture (monotonic milliseconds).
    pub timestamp_ms: u64,
    /// Normalized coordinates of the focused window [x, y, w, h] in 0..1 range.
    pub focused_rect: Option<[f32; 4]>,
}

/// Captures the desktop screen.
pub struct ScreenCapture {
    target_width: u32,
    target_height: u32,
    start_time: Instant,
}

impl ScreenCapture {
    /// Initialize screen capture targeting the given analysis resolution.
    pub fn new(target_width: u32, target_height: u32) -> Result<Self> {
        let target_width = target_width.max(16);
        let target_height = target_height.max(16);
        log::info!("Screen capture initialized targeting {target_width}x{target_height}");

        Ok(Self {
            target_width,
            target_height,
            start_time: Instant::now(),
        })
    }

    /// Capture the current desktop frame.
    pub fn capture(&mut self) -> Result<ScreenFrame> {
        unsafe {
            let screen_w = GetSystemMetrics(SM_CXSCREEN);
            let screen_h = GetSystemMetrics(SM_CYSCREEN);

            if screen_w <= 0 || screen_h <= 0 {
                return Err(anyhow!("Invalid screen dimensions: {screen_w}x{screen_h}"));
            }

            let null_hwnd = HWND::default();
            let screen_dc = GetDC(null_hwnd);
            if screen_dc.is_invalid() {
                return Err(anyhow!("GetDC(NULL) failed"));
            }

            let mem_dc = CreateCompatibleDC(screen_dc);
            if mem_dc.is_invalid() {
                ReleaseDC(null_hwnd, screen_dc);
                return Err(anyhow!("CreateCompatibleDC failed"));
            }

            let h_bitmap = CreateCompatibleBitmap(screen_dc, screen_w, screen_h);
            if h_bitmap.is_invalid() {
                let _ = DeleteDC(mem_dc);
                ReleaseDC(null_hwnd, screen_dc);
                return Err(anyhow!("CreateCompatibleBitmap failed"));
            }

            let old_obj: HGDIOBJ = SelectObject(mem_dc, h_bitmap);
            if old_obj.is_invalid() {
                let _ = DeleteObject(h_bitmap);
                let _ = DeleteDC(mem_dc);
                ReleaseDC(null_hwnd, screen_dc);
                return Err(anyhow!("SelectObject failed"));
            }

            let blt_res = BitBlt(
                mem_dc,
                0,
                0,
                screen_w,
                screen_h,
                screen_dc,
                0,
                0,
                SRCCOPY,
            );

            if let Err(err) = blt_res {
                SelectObject(mem_dc, old_obj);
                let _ = DeleteObject(h_bitmap);
                let _ = DeleteDC(mem_dc);
                ReleaseDC(null_hwnd, screen_dc);
                return Err(anyhow!("BitBlt failed: {err:?}"));
            }

            let mut bmi = BITMAPINFO {
                bmiHeader: BITMAPINFOHEADER {
                    biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                    biWidth: screen_w,
                    biHeight: -screen_h, // Negative for top-down orientation
                    biPlanes: 1,
                    biBitCount: 32,
                    biCompression: BI_RGB.0,
                    biSizeImage: 0,
                    biXPelsPerMeter: 0,
                    biYPelsPerMeter: 0,
                    biClrUsed: 0,
                    biClrImportant: 0,
                },
                bmiColors: [windows::Win32::Graphics::Gdi::RGBQUAD::default(); 1],
            };

            let pixel_count = (screen_w * screen_h) as usize;
            let mut bgra_pixels = vec![0u8; pixel_count * 4];

            GetDIBits(
                mem_dc,
                h_bitmap,
                0,
                screen_h as u32,
                Some(bgra_pixels.as_mut_ptr() as *mut _),
                &mut bmi,
                DIB_RGB_COLORS,
            );

            // Cleanup GDI objects
            SelectObject(mem_dc, old_obj);
            let _ = DeleteObject(h_bitmap);
            let _ = DeleteDC(mem_dc);
            ReleaseDC(null_hwnd, screen_dc);

            // Fast stride downsampling from full screen BGRA to target_width x target_height RGBA
            let tw = self.target_width;
            let th = self.target_height;
            let target_pixels_count = (tw * th) as usize;
            let mut rgba_pixels = vec![0u8; target_pixels_count * 4];

            let step_x = screen_w as f32 / tw as f32;
            let step_y = screen_h as f32 / th as f32;

            for ty in 0..th {
                let sy = ((ty as f32 + 0.5) * step_y) as i32;
                let sy_clamped = sy.clamp(0, screen_h - 1) as usize;

                for tx in 0..tw {
                    let sx = ((tx as f32 + 0.5) * step_x) as i32;
                    let sx_clamped = sx.clamp(0, screen_w - 1) as usize;

                    let src_idx = (sy_clamped * screen_w as usize + sx_clamped) * 4;
                    let dst_idx = (ty as usize * tw as usize + tx as usize) * 4;

                    if src_idx + 2 < bgra_pixels.len() {
                        let b = bgra_pixels[src_idx];
                        let g = bgra_pixels[src_idx + 1];
                        let r = bgra_pixels[src_idx + 2];
                        rgba_pixels[dst_idx] = r;
                        rgba_pixels[dst_idx + 1] = g;
                        rgba_pixels[dst_idx + 2] = b;
                        rgba_pixels[dst_idx + 3] = 255;
                    }
                }
            }

            let focused_rect = get_focused_window_normalized(screen_w as f32, screen_h as f32);
            let timestamp_ms = self.start_time.elapsed().as_millis() as u64;

            Ok(ScreenFrame {
                pixels: rgba_pixels,
                width: self.target_width,
                height: self.target_height,
                timestamp_ms,
                focused_rect,
            })
        }
    }
}

/// Helper to get the normalized coordinates [x, y, w, h] of the focused window on the primary screen.
fn get_focused_window_normalized(screen_w: f32, screen_h: f32) -> Option<[f32; 4]> {
    unsafe {
        let fg_hwnd = GetForegroundWindow();
        if fg_hwnd == HWND::default() {
            return None;
        }

        let mut rect = windows::Win32::Foundation::RECT::default();
        if GetWindowRect(fg_hwnd, &mut rect).is_ok() {
            let x = (rect.left as f32 / screen_w).clamp(0.0, 1.0);
            let y = (rect.top as f32 / screen_h).clamp(0.0, 1.0);
            let w = ((rect.right - rect.left) as f32 / screen_w).clamp(0.0, 1.0);
            let h = ((rect.bottom - rect.top) as f32 / screen_h).clamp(0.0, 1.0);
            Some([x, y, w, h])
        } else {
            None
        }
    }
}
