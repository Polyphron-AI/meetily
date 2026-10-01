//! Win32 helpers: DPI awareness, foreground monitor lookup, clipboard paste.
//! Non-Windows builds get stubs that report `unsupported-platform`.

#[cfg(not(windows))]
use serde_json::Value;

#[cfg(not(windows))]
use crate::protocol::{ErrorCode, ProtoError, ProtoResult};

#[cfg(not(windows))]
fn unsupported() -> ProtoError {
    ProtoError::new(ErrorCode::UnsupportedPlatform, "only supported on Windows")
}

/// Which modifier releases to send before a synthetic Ctrl+V, given the vks
/// the user still holds. A dummy key goes first when Alt or Win is held so
/// their release does not open the menu bar or Start. Pure for testing.
pub fn modifier_release_plan(held: &[u16]) -> Vec<u16> {
    const MASK: u16 = 0xE8;
    let mut plan = Vec::new();
    if held.iter().any(|&k| matches!(k, 0xA4 | 0xA5 | 0x5B | 0x5C)) {
        plan.push(MASK);
    }
    plan.extend_from_slice(held);
    plan
}

/// Clipboard formats whose handle is not an `HGLOBAL` (GDI objects,
/// owner-display), which a byte snapshot cannot capture.
pub fn is_hglobal_format(format: u32) -> bool {
    !matches!(format, 2 | 3 | 9 | 14 | 0x80 | 0x82 | 0x83 | 0x8E)
}

#[cfg(not(windows))]
pub fn init_dpi_awareness() {}

#[cfg(not(windows))]
pub fn foreground_monitor() -> ProtoResult<Value> {
    Err(unsupported())
}

#[cfg(not(windows))]
pub fn insert_text(_text: &str, _restore: bool) -> ProtoResult<()> {
    Err(unsupported())
}

#[cfg(windows)]
pub use imp::*;

#[cfg(windows)]
mod imp {
    use std::time::Duration;

    use serde_json::{json, Value};
    use windows::core::w;
    use windows::Win32::Foundation::{GlobalFree, HANDLE, HGLOBAL, HWND, POINT, RECT};
    use windows::Win32::Graphics::Gdi::{
        GetMonitorInfoW, MonitorFromPoint, MonitorFromWindow, HMONITOR, MONITORINFO,
        MONITOR_DEFAULTTONEAREST,
    };
    use windows::Win32::System::DataExchange::{
        CloseClipboard, EmptyClipboard, EnumClipboardFormats, GetClipboardData, OpenClipboard,
        SetClipboardData,
    };
    use windows::Win32::System::Memory::{
        GlobalAlloc, GlobalLock, GlobalSize, GlobalUnlock, GMEM_MOVEABLE,
    };
    use windows::Win32::System::Ole::CF_UNICODETEXT;
    use windows::Win32::UI::HiDpi::{
        GetDpiForMonitor, SetProcessDpiAwarenessContext,
        DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, MDT_EFFECTIVE_DPI,
    };
    use windows::Win32::UI::Input::KeyboardAndMouse::{
        GetAsyncKeyState, SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBDINPUT, KEYBD_EVENT_FLAGS,
        KEYEVENTF_KEYUP, VIRTUAL_KEY, VK_CONTROL, VK_V,
    };
    use windows::Win32::UI::WindowsAndMessaging::{
        CreateWindowExW, DestroyWindow, GetCursorPos, GetForegroundWindow, HWND_MESSAGE,
        WINDOW_EX_STYLE, WINDOW_STYLE,
    };

    use super::{is_hglobal_format, modifier_release_plan};
    use crate::protocol::{ErrorCode, ProtoError, ProtoResult};

    pub fn init_dpi_awareness() {
        unsafe {
            if let Err(e) =
                SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
            {
                log::warn!("SetProcessDpiAwarenessContext failed: {e}");
            }
        }
    }

    fn rect(r: RECT) -> Value {
        json!({ "x": r.left, "y": r.top, "width": r.right - r.left, "height": r.bottom - r.top })
    }

    pub fn foreground_monitor() -> ProtoResult<Value> {
        unsafe {
            let fg = GetForegroundWindow();
            let (monitor, source): (HMONITOR, &str) = if !fg.is_invalid() {
                (
                    MonitorFromWindow(fg, MONITOR_DEFAULTTONEAREST),
                    "foreground",
                )
            } else {
                let mut pt = POINT::default();
                GetCursorPos(&mut pt).map_err(|e| internal(format!("GetCursorPos: {e}")))?;
                (MonitorFromPoint(pt, MONITOR_DEFAULTTONEAREST), "cursor")
            };
            let mut info = MONITORINFO {
                cbSize: std::mem::size_of::<MONITORINFO>() as u32,
                ..Default::default()
            };
            if !GetMonitorInfoW(monitor, &mut info).as_bool() {
                return Err(internal("GetMonitorInfoW failed".into()));
            }
            let (mut dpi_x, mut dpi_y) = (96u32, 96u32);
            if let Err(e) = GetDpiForMonitor(monitor, MDT_EFFECTIVE_DPI, &mut dpi_x, &mut dpi_y) {
                log::warn!("GetDpiForMonitor failed: {e}");
            }
            Ok(json!({
                "source": source,
                "bounds": rect(info.rcMonitor),
                "workArea": rect(info.rcWork),
                "dpi": dpi_x,
            }))
        }
    }

    fn internal(message: String) -> ProtoError {
        ProtoError::new(ErrorCode::Internal, message)
    }

    fn insert_failed(message: impl Into<String>) -> ProtoError {
        ProtoError::new(ErrorCode::InsertFailed, message)
    }

    /// Message-only window that owns the clipboard while we write to it.
    struct OwnerWindow(HWND);

    impl OwnerWindow {
        fn new() -> ProtoResult<Self> {
            unsafe {
                CreateWindowExW(
                    WINDOW_EX_STYLE(0),
                    w!("STATIC"),
                    w!("pulse-voice-clipboard"),
                    WINDOW_STYLE(0),
                    0,
                    0,
                    0,
                    0,
                    Some(HWND_MESSAGE),
                    None,
                    None,
                    None,
                )
                .map(OwnerWindow)
                .map_err(|e| insert_failed(format!("cannot create clipboard owner window: {e}")))
            }
        }
    }

    impl Drop for OwnerWindow {
        fn drop(&mut self) {
            unsafe {
                let _ = DestroyWindow(self.0);
            }
        }
    }

    /// Open clipboard; closes on drop.
    struct Clipboard;

    impl Clipboard {
        fn open(owner: HWND) -> ProtoResult<Clipboard> {
            let mut last = String::new();
            for attempt in 0..10 {
                if attempt > 0 {
                    std::thread::sleep(Duration::from_millis(20));
                }
                match unsafe { OpenClipboard(Some(owner)) } {
                    Ok(()) => return Ok(Clipboard),
                    Err(e) => last = e.to_string(),
                }
            }
            Err(insert_failed(format!(
                "cannot open clipboard after 10 retries: {last}"
            )))
        }
    }

    impl Drop for Clipboard {
        fn drop(&mut self) {
            unsafe {
                let _ = CloseClipboard();
            }
        }
    }

    fn snapshot() -> Vec<(u32, Vec<u8>)> {
        let mut saved = Vec::new();
        unsafe {
            let mut format = EnumClipboardFormats(0);
            while format != 0 {
                if is_hglobal_format(format) {
                    if let Ok(handle) = GetClipboardData(format) {
                        let global = HGLOBAL(handle.0);
                        let size = GlobalSize(global);
                        let ptr = GlobalLock(global) as *const u8;
                        if !ptr.is_null() {
                            saved.push((format, std::slice::from_raw_parts(ptr, size).to_vec()));
                            let _ = GlobalUnlock(global);
                        }
                    }
                }
                format = EnumClipboardFormats(format);
            }
        }
        saved
    }

    fn put(format: u32, bytes: &[u8]) -> Result<(), String> {
        unsafe {
            let global =
                GlobalAlloc(GMEM_MOVEABLE, bytes.len().max(1)).map_err(|e| e.to_string())?;
            let ptr = GlobalLock(global) as *mut u8;
            if ptr.is_null() {
                let _ = GlobalFree(Some(global));
                return Err("GlobalLock failed".into());
            }
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
            let _ = GlobalUnlock(global);
            if let Err(e) = SetClipboardData(format, Some(HANDLE(global.0))) {
                let _ = GlobalFree(Some(global));
                return Err(e.to_string());
            }
            Ok(())
        }
    }

    fn key_input(vk: u16, up: bool) -> INPUT {
        INPUT {
            r#type: INPUT_KEYBOARD,
            Anonymous: INPUT_0 {
                ki: KEYBDINPUT {
                    wVk: VIRTUAL_KEY(vk),
                    wScan: 0,
                    dwFlags: if up {
                        KEYEVENTF_KEYUP
                    } else {
                        KEYBD_EVENT_FLAGS(0)
                    },
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        }
    }

    fn send_paste() -> ProtoResult<()> {
        const MODIFIERS: [u16; 8] = [0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0x5B, 0x5C];
        let held: Vec<u16> = MODIFIERS
            .iter()
            .copied()
            .filter(|&vk| unsafe { GetAsyncKeyState(vk as i32) } as u16 & 0x8000 != 0)
            .collect();
        let mut inputs = Vec::new();
        for vk in modifier_release_plan(&held) {
            if vk == 0xE8 {
                inputs.push(key_input(vk, false));
            }
            inputs.push(key_input(vk, true));
        }
        inputs.push(key_input(VK_CONTROL.0, false));
        inputs.push(key_input(VK_V.0, false));
        inputs.push(key_input(VK_V.0, true));
        inputs.push(key_input(VK_CONTROL.0, true));
        let sent = unsafe { SendInput(&inputs, std::mem::size_of::<INPUT>() as i32) };
        if sent as usize != inputs.len() {
            return Err(insert_failed(format!(
                "SendInput sent {sent} of {} events",
                inputs.len()
            )));
        }
        Ok(())
    }

    pub fn insert_text(text: &str, restore: bool) -> ProtoResult<()> {
        let owner = OwnerWindow::new()?;
        let saved = {
            let _clip = Clipboard::open(owner.0)?;
            let saved = if restore { snapshot() } else { Vec::new() };
            unsafe { EmptyClipboard() }
                .map_err(|e| insert_failed(format!("EmptyClipboard: {e}")))?;
            let mut utf16: Vec<u16> = text.encode_utf16().collect();
            utf16.push(0);
            let bytes: Vec<u8> = utf16.iter().flat_map(|c| c.to_le_bytes()).collect();
            put(CF_UNICODETEXT.0 as u32, &bytes)
                .map_err(|e| insert_failed(format!("SetClipboardData: {e}")))?;
            saved
        };
        send_paste()?;
        std::thread::sleep(Duration::from_millis(150));
        if restore {
            let _clip = Clipboard::open(owner.0)?;
            unsafe { EmptyClipboard() }
                .map_err(|e| insert_failed(format!("EmptyClipboard: {e}")))?;
            for (format, bytes) in &saved {
                if let Err(e) = put(*format, bytes) {
                    log::warn!("could not restore clipboard format {format}: {e}");
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_plan_masks_alt_and_win_only() {
        assert_eq!(modifier_release_plan(&[]), Vec::<u16>::new());
        assert_eq!(modifier_release_plan(&[0xA2]), vec![0xA2]);
        assert_eq!(modifier_release_plan(&[0xA0, 0x5B]), vec![0xE8, 0xA0, 0x5B]);
        assert_eq!(modifier_release_plan(&[0xA5]), vec![0xE8, 0xA5]);
    }

    #[test]
    fn gdi_formats_are_skipped() {
        assert!(is_hglobal_format(13)); // CF_UNICODETEXT
        assert!(is_hglobal_format(1)); // CF_TEXT
        assert!(is_hglobal_format(0xC123)); // registered format
        assert!(!is_hglobal_format(2)); // CF_BITMAP
        assert!(!is_hglobal_format(14)); // CF_ENHMETAFILE
    }
}
