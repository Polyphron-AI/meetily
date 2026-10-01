//! Global hotkey: chord edge detection (pure) and the `WH_KEYBOARD_LL` hook thread.

use crate::protocol::Key;

/// Virtual-key codes the tracker works with. Left/right variants of a modifier
/// map to the same logical key.
pub mod vk {
    pub const SHIFT: u32 = 0x10;
    pub const CONTROL: u32 = 0x11;
    pub const MENU: u32 = 0x12;
    pub const SPACE: u32 = 0x20;
    pub const LWIN: u32 = 0x5B;
    pub const RWIN: u32 = 0x5C;
    pub const F1: u32 = 0x70;
    pub const LSHIFT: u32 = 0xA0;
    pub const RSHIFT: u32 = 0xA1;
    pub const LCONTROL: u32 = 0xA2;
    pub const RCONTROL: u32 = 0xA3;
    pub const LMENU: u32 = 0xA4;
    pub const RMENU: u32 = 0xA5;
    /// Unassigned; injected to stop a lone Win release from opening Start.
    pub const MASK: u32 = 0xE8;
}

pub fn key_vk(key: Key) -> u32 {
    match key {
        Key::Ctrl => vk::CONTROL,
        Key::Alt => vk::MENU,
        Key::Shift => vk::SHIFT,
        Key::Win => vk::LWIN,
        Key::Space => vk::SPACE,
        Key::Letter(c) => c as u32,
        Key::F(n) => vk::F1 + n as u32 - 1,
    }
}

/// Maps a raw hook vk to the logical vk used for chord matching.
fn logical(raw: u32) -> u32 {
    match raw {
        vk::LSHIFT | vk::RSHIFT => vk::SHIFT,
        vk::LCONTROL | vk::RCONTROL => vk::CONTROL,
        vk::LMENU | vk::RMENU => vk::MENU,
        vk::RWIN => vk::LWIN,
        other => other,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Edge {
    Down,
    Up,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Reaction {
    pub edge: Option<Edge>,
    /// Inject `vk::MASK` before letting this Win release through.
    pub inject_mask: bool,
}

/// Tracks held keys and reports chord edges. Pure, so it is unit tested
/// without a hook.
pub struct ChordTracker {
    chord: Vec<u32>,
    /// Raw vks currently held (physical keys, so LCtrl and RCtrl are distinct).
    held: Vec<u32>,
    active: bool,
    /// The chord fired while Win was held; the next Win release needs masking.
    win_fired: bool,
}

impl ChordTracker {
    pub fn new(keys: &[Key]) -> Self {
        Self {
            chord: keys.iter().map(|&k| key_vk(k)).collect(),
            held: Vec::new(),
            active: false,
            win_fired: false,
        }
    }

    fn logical_down(&self, lvk: u32) -> bool {
        self.held.iter().any(|&r| logical(r) == lvk)
    }

    fn all_down(&self) -> bool {
        self.chord.iter().all(|&k| self.logical_down(k))
    }

    pub fn on_key(&mut self, raw: u32, down: bool) -> Reaction {
        let mut r = Reaction::default();
        if down {
            if self.held.contains(&raw) {
                // Auto-repeat.
                return r;
            }
            self.held.push(raw);
            if !self.active && self.all_down() {
                self.active = true;
                r.edge = Some(Edge::Down);
                if self.chord.contains(&vk::LWIN) {
                    self.win_fired = true;
                }
            }
        } else {
            let Some(pos) = self.held.iter().position(|&h| h == raw) else {
                return r;
            };
            self.held.remove(pos);
            if self.active && !self.all_down() {
                self.active = false;
                r.edge = Some(Edge::Up);
            }
            if logical(raw) == vk::LWIN && self.win_fired {
                r.inject_mask = true;
                if !self.logical_down(vk::LWIN) {
                    self.win_fired = false;
                }
            }
        }
        r
    }
}

#[cfg(windows)]
pub use win_hook::Hook;

#[cfg(windows)]
mod win_hook {
    use std::cell::RefCell;
    use std::sync::mpsc;
    use std::thread::JoinHandle;

    use serde_json::json;
    use windows::Win32::Foundation::{LPARAM, LRESULT, WPARAM};
    use windows::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows::Win32::System::Threading::GetCurrentThreadId;
    use windows::Win32::UI::Input::KeyboardAndMouse::{
        SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBDINPUT, KEYBD_EVENT_FLAGS, KEYEVENTF_KEYUP,
        VIRTUAL_KEY,
    };
    use windows::Win32::UI::WindowsAndMessaging::{
        CallNextHookEx, GetMessageW, PeekMessageW, PostThreadMessageW, SetWindowsHookExW,
        UnhookWindowsHookEx, KBDLLHOOKSTRUCT, LLKHF_INJECTED, MSG, PM_NOREMOVE, WH_KEYBOARD_LL,
        WM_KEYDOWN, WM_KEYUP, WM_QUIT, WM_SYSKEYDOWN, WM_SYSKEYUP,
    };

    use super::{vk, ChordTracker, Edge};
    use crate::protocol::{event, Key, Out};

    struct HookState {
        tracker: ChordTracker,
        out: Out,
    }

    thread_local! {
        static STATE: RefCell<Option<HookState>> = const { RefCell::new(None) };
    }

    /// A running hook thread. Dropping it removes the hook.
    pub struct Hook {
        thread_id: u32,
        handle: Option<JoinHandle<()>>,
    }

    impl Hook {
        pub fn install(keys: Vec<Key>, out: Out) -> Result<Hook, String> {
            let (ready_tx, ready_rx) = mpsc::channel::<Result<u32, String>>();
            let handle = std::thread::Builder::new()
                .name("hotkey".into())
                .spawn(move || hook_thread(keys, out, ready_tx))
                .map_err(|e| e.to_string())?;
            match ready_rx.recv() {
                Ok(Ok(thread_id)) => Ok(Hook {
                    thread_id,
                    handle: Some(handle),
                }),
                Ok(Err(e)) => {
                    let _ = handle.join();
                    Err(e)
                }
                Err(_) => Err("hook thread exited".into()),
            }
        }
    }

    impl Drop for Hook {
        fn drop(&mut self) {
            unsafe {
                let _ = PostThreadMessageW(self.thread_id, WM_QUIT, WPARAM(0), LPARAM(0));
            }
            if let Some(h) = self.handle.take() {
                let _ = h.join();
            }
        }
    }

    fn hook_thread(keys: Vec<Key>, out: Out, ready: mpsc::Sender<Result<u32, String>>) {
        unsafe {
            // Create this thread's message queue before reporting ready so a
            // PostThreadMessageW(WM_QUIT) cannot be lost.
            let mut msg = MSG::default();
            let _ = PeekMessageW(&mut msg, None, 0, 0, PM_NOREMOVE);
            STATE.with(|s| {
                *s.borrow_mut() = Some(HookState {
                    tracker: ChordTracker::new(&keys),
                    out,
                })
            });
            let module = GetModuleHandleW(None).ok();
            let hook = match SetWindowsHookExW(
                WH_KEYBOARD_LL,
                Some(hook_proc),
                module.map(|m| m.into()),
                0,
            ) {
                Ok(h) => h,
                Err(e) => {
                    let _ = ready.send(Err(format!("SetWindowsHookExW failed: {e}")));
                    return;
                }
            };
            let _ = ready.send(Ok(GetCurrentThreadId()));
            // Blocks until WM_QUIT; the hook callback runs inside this call.
            while GetMessageW(&mut msg, None, 0, 0).0 > 0 {}
            let _ = UnhookWindowsHookEx(hook);
            STATE.with(|s| *s.borrow_mut() = None);
        }
    }

    fn send_mask() {
        let key = |flags| INPUT {
            r#type: INPUT_KEYBOARD,
            Anonymous: INPUT_0 {
                ki: KEYBDINPUT {
                    wVk: VIRTUAL_KEY(vk::MASK as u16),
                    wScan: 0,
                    dwFlags: flags,
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        };
        let inputs = [key(KEYBD_EVENT_FLAGS(0)), key(KEYEVENTF_KEYUP)];
        unsafe {
            SendInput(&inputs, std::mem::size_of::<INPUT>() as i32);
        }
    }

    /// Only updates key state and posts to the writer channel; never blocks.
    unsafe extern "system" fn hook_proc(code: i32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
        if code >= 0 {
            let info = &*(lparam.0 as *const KBDLLHOOKSTRUCT);
            if info.flags.0 & LLKHF_INJECTED.0 == 0 {
                let msg = wparam.0 as u32;
                let down = match msg {
                    WM_KEYDOWN | WM_SYSKEYDOWN => Some(true),
                    WM_KEYUP | WM_SYSKEYUP => Some(false),
                    _ => None,
                };
                if let Some(down) = down {
                    STATE.with(|s| {
                        if let Some(state) = s.borrow_mut().as_mut() {
                            let r = state.tracker.on_key(info.vkCode, down);
                            if r.inject_mask {
                                send_mask();
                            }
                            if let Some(edge) = r.edge {
                                let phase = if edge == Edge::Down { "down" } else { "up" };
                                let _ = state.out.send(event("hotkey", json!({ "phase": phase })));
                            }
                        }
                    });
                }
            }
        }
        CallNextHookEx(None, code, wparam, lparam)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::Key;

    fn feed(t: &mut ChordTracker, events: &[(u32, bool)]) -> Vec<Reaction> {
        events.iter().map(|&(k, d)| t.on_key(k, d)).collect()
    }

    fn edges(rs: &[Reaction]) -> Vec<Edge> {
        rs.iter().filter_map(|r| r.edge).collect()
    }

    #[test]
    fn down_once_up_once_with_repeat_ignored() {
        let mut t = ChordTracker::new(&[Key::Ctrl, Key::Space]);
        let rs = feed(
            &mut t,
            &[
                (vk::LCONTROL, true),
                (vk::LCONTROL, true), // repeat
                (vk::SPACE, true),
                (vk::SPACE, true), // repeat
                (vk::LCONTROL, true),
                (vk::SPACE, true),
                (vk::SPACE, false),
                (vk::LCONTROL, false),
            ],
        );
        assert_eq!(edges(&rs), vec![Edge::Down, Edge::Up]);
        assert_eq!(rs[2].edge, Some(Edge::Down));
        assert_eq!(rs[6].edge, Some(Edge::Up));
    }

    #[test]
    fn order_of_presses_does_not_matter_and_extra_keys_are_allowed() {
        let mut t = ChordTracker::new(&[Key::Ctrl, Key::Shift, Key::Letter(b'K')]);
        let rs = feed(
            &mut t,
            &[
                (b'K' as u32, true),
                (vk::RSHIFT, true),
                (b'J' as u32, true),
                (vk::RCONTROL, true),
            ],
        );
        assert_eq!(edges(&rs), vec![Edge::Down]);
        assert_eq!(rs[3].edge, Some(Edge::Down));
    }

    #[test]
    fn releasing_any_chord_key_ends_and_re_pressing_refires() {
        let mut t = ChordTracker::new(&[Key::Alt, Key::F(13)]);
        let f13 = vk::F1 + 12;
        let rs = feed(
            &mut t,
            &[
                (vk::LMENU, true),
                (f13, true),
                (vk::LMENU, false),
                (vk::LMENU, true),
                (f13, false),
            ],
        );
        assert_eq!(edges(&rs), vec![Edge::Down, Edge::Up, Edge::Down, Edge::Up]);
    }

    #[test]
    fn left_and_right_modifiers_are_one_logical_key() {
        let mut t = ChordTracker::new(&[Key::Ctrl, Key::Space]);
        let rs = feed(
            &mut t,
            &[
                (vk::LCONTROL, true),
                (vk::RCONTROL, true),
                (vk::SPACE, true),
                (vk::LCONTROL, false),
                (vk::RCONTROL, false),
            ],
        );
        // Still active after LCtrl release because RCtrl holds Ctrl down.
        assert_eq!(rs[3].edge, None);
        assert_eq!(rs[4].edge, Some(Edge::Up));
    }

    #[test]
    fn unrelated_releases_are_ignored() {
        let mut t = ChordTracker::new(&[Key::Ctrl, Key::Space]);
        let rs = feed(
            &mut t,
            &[
                (vk::SPACE, false),
                (b'A' as u32, true),
                (b'A' as u32, false),
            ],
        );
        assert!(edges(&rs).is_empty());
    }

    #[test]
    fn win_release_after_fire_is_masked_once() {
        let mut t = ChordTracker::new(&[Key::Win, Key::Space]);
        let rs = feed(
            &mut t,
            &[
                (vk::LWIN, true),
                (vk::SPACE, true),
                (vk::SPACE, false),
                (vk::LWIN, false),
            ],
        );
        assert_eq!(edges(&rs), vec![Edge::Down, Edge::Up]);
        assert!(rs[3].inject_mask);
        assert!(!rs[2].inject_mask);
        // A later lone Win tap is not masked.
        let rs = feed(&mut t, &[(vk::LWIN, true), (vk::LWIN, false)]);
        assert!(rs.iter().all(|r| !r.inject_mask && r.edge.is_none()));
    }

    #[test]
    fn win_release_without_fire_is_not_masked() {
        let mut t = ChordTracker::new(&[Key::Win, Key::Shift, Key::Space]);
        let rs = feed(
            &mut t,
            &[
                (vk::RWIN, true),
                (vk::SPACE, true),
                (vk::SPACE, false),
                (vk::RWIN, false),
            ],
        );
        assert!(rs.iter().all(|r| !r.inject_mask && r.edge.is_none()));
    }

    #[test]
    fn win_chord_up_edge_and_mask_on_same_event() {
        let mut t = ChordTracker::new(&[Key::Ctrl, Key::Win]);
        let rs = feed(
            &mut t,
            &[
                (vk::LCONTROL, true),
                (vk::LWIN, true),
                (vk::LWIN, false),
                (vk::LCONTROL, false),
            ],
        );
        assert_eq!(
            rs[2],
            Reaction {
                edge: Some(Edge::Up),
                inject_mask: true
            }
        );
        assert_eq!(rs[3], Reaction::default());
    }

    #[test]
    fn key_vk_mapping() {
        assert_eq!(key_vk(Key::F(24)), 0x87);
        assert_eq!(key_vk(Key::Letter(b'A')), 0x41);
        assert_eq!(key_vk(Key::Win), vk::LWIN);
    }
}
