//! End-to-end smoke test against a built pulse-voice executable.
//!
//! cargo run --release --example smoke -- <pulse-voice.exe> <modelDir> <speech.wav>
//!
//! Sends hello, status, devices.list, foreground.monitor, model.load and
//! transcribe (16-bit PCM mono WAV at any rate), measures idle CPU for 10 s,
//! then shuts down. Every stdout line must be a protocol object. It never
//! installs the hotkey hook or inserts text.
//!
//! With a trailing `--capture`, it also opens the default microphone for a
//! 2 s dictation, then runs a meeting with system audio while playing the WAV
//! through the default output device (audible), to exercise loopback + VAD.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use base64::Engine as _;
use serde_json::{json, Value};

fn read_wav_pcm16_mono(path: &str) -> (Vec<u8>, u32) {
    let bytes = std::fs::read(path).expect("read wav");
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
    let mut pos = 12;
    let mut rate = 0;
    let mut channels = 0;
    let mut bits = 0;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body = &bytes[pos + 8..(pos + 8 + len).min(bytes.len())];
        if id == b"fmt " {
            channels = u16::from_le_bytes([body[2], body[3]]);
            rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
            bits = u16::from_le_bytes([body[14], body[15]]);
        } else if id == b"data" {
            assert_eq!((channels, bits), (1, 16), "expected 16-bit mono PCM");
            return (body.to_vec(), rate);
        }
        pos += 8 + len + (len & 1);
    }
    panic!("no data chunk");
}

#[cfg(windows)]
fn cpu_time(child: &std::process::Child) -> Duration {
    use std::os::windows::io::AsRawHandle;
    use windows::Win32::Foundation::{FILETIME, HANDLE};
    use windows::Win32::System::Threading::GetProcessTimes;
    let (mut c, mut e, mut k, mut u) = (
        FILETIME::default(),
        FILETIME::default(),
        FILETIME::default(),
        FILETIME::default(),
    );
    unsafe {
        GetProcessTimes(
            HANDLE(child.as_raw_handle()),
            &mut c,
            &mut e,
            &mut k,
            &mut u,
        )
        .expect("GetProcessTimes")
    };
    let ticks = |f: FILETIME| ((f.dwHighDateTime as u64) << 32) | f.dwLowDateTime as u64;
    Duration::from_nanos((ticks(k) + ticks(u)) * 100)
}

#[cfg(not(windows))]
fn cpu_time(_child: &std::process::Child) -> Duration {
    Duration::ZERO
}

struct Client {
    stdin: std::process::ChildStdin,
    lines: mpsc::Receiver<Value>,
    next_id: u64,
}

impl Client {
    fn call(&mut self, method: &str, params: Value) -> Value {
        self.next_id += 1;
        let id = self.next_id;
        let req = json!({ "id": id, "method": method, "params": params });
        writeln!(self.stdin, "{req}").unwrap();
        self.stdin.flush().unwrap();
        loop {
            let v = self
                .lines
                .recv_timeout(Duration::from_secs(120))
                .expect("response timeout");
            if v.get("id").and_then(Value::as_u64) == Some(id) {
                return v;
            }
            println!("  event: {v}");
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let capture = args.last().is_some_and(|a| a == "--capture");
    let positional: Vec<&String> = args.iter().skip(1).filter(|a| *a != "--capture").collect();
    let [exe, model_dir, wav] = positional[..] else {
        eprintln!("usage: smoke <pulse-voice.exe> <modelDir> <speech.wav> [--capture]");
        std::process::exit(2);
    };

    let mut child = Command::new(exe)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn pulse-voice");
    let stdout = child.stdout.take().unwrap();
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let line = line.unwrap();
            let v: Value = serde_json::from_str(&line)
                .unwrap_or_else(|_| panic!("non-JSON stdout line: {line}"));
            assert!(
                v.get("id").is_some() || v.get("event").is_some(),
                "non-protocol stdout line: {line}"
            );
            if tx.send(v).is_err() {
                break;
            }
        }
    });
    let ready = rx.recv_timeout(Duration::from_secs(10)).expect("ready");
    assert_eq!(ready["event"], "ready", "first line must be ready: {ready}");
    println!("ready: {ready}");
    let mut client = Client {
        stdin: child.stdin.take().unwrap(),
        lines: rx,
        next_id: 0,
    };

    println!("hello: {}", client.call("hello", json!({})));
    println!("status: {}", client.call("status", json!({})));
    println!("devices.list: {}", client.call("devices.list", json!({})));
    println!(
        "foreground.monitor: {}",
        client.call("foreground.monitor", json!({}))
    );
    println!(
        "transcribe before load: {}",
        client.call("transcribe", json!({ "pcm16": "", "sampleRate": 16000 }))
    );
    println!("bogus: {}", client.call("bogus.method", json!({})));

    let started = Instant::now();
    let load = client.call("model.load", json!({ "modelDir": model_dir }));
    println!(
        "model.load: {load} (wall {} ms)",
        started.elapsed().as_millis()
    );
    assert!(load.get("result").is_some(), "model.load failed");
    println!(
        "model.load again (idempotent): {}",
        client.call("model.load", json!({ "modelDir": model_dir }))
    );
    println!("status: {}", client.call("status", json!({})));

    // Two consecutive windows: the first can include the tail of load cleanup.
    for window in 1..=2 {
        let cpu0 = cpu_time(&child);
        std::thread::sleep(Duration::from_secs(10));
        let cpu1 = cpu_time(&child);
        println!(
            "idle CPU over 10 s after model.load (window {window}): {} ms",
            (cpu1 - cpu0).as_millis()
        );
    }

    let (pcm, rate) = read_wav_pcm16_mono(wav);
    let b64 = base64::engine::general_purpose::STANDARD.encode(&pcm);
    for run in 1..=3 {
        let r = client.call("transcribe", json!({ "pcm16": b64, "sampleRate": rate }));
        println!("transcribe run {run} ({} Hz): {r}", rate);
    }

    let cpu2 = cpu_time(&child);
    std::thread::sleep(Duration::from_secs(10));
    let cpu3 = cpu_time(&child);
    println!(
        "idle CPU over 10 s after transcription: {} ms",
        (cpu3 - cpu2).as_millis()
    );

    if capture {
        println!(
            "dictation.stop idle: {}",
            client.call("dictation.stop", json!({}))
        );
        println!(
            "dictation.start: {}",
            client.call("dictation.start", json!({}))
        );
        println!(
            "dictation.start again: {}",
            client.call("dictation.start", json!({}))
        );
        println!("status: {}", client.call("status", json!({})));
        std::thread::sleep(Duration::from_secs(2));
        println!(
            "dictation.stop: {}",
            client.call("dictation.stop", json!({}))
        );
        println!(
            "dictation.cancel idle: {}",
            client.call("dictation.cancel", json!({}))
        );

        println!(
            "meeting.start: {}",
            client.call("meeting.start", json!({ "systemAudio": true }))
        );
        let mut player = Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                &format!("(New-Object Media.SoundPlayer '{wav}').PlaySync()"),
            ])
            .spawn()
            .expect("play wav");
        player.wait().unwrap();
        std::thread::sleep(Duration::from_millis(1500));
        println!("meeting.stop: {}", client.call("meeting.stop", json!({})));
        let cpu4 = cpu_time(&child);
        std::thread::sleep(Duration::from_secs(5));
        println!(
            "idle CPU over 5 s after captures: {} ms",
            (cpu_time(&child) - cpu4).as_millis()
        );
    }

    println!("shutdown: {}", client.call("shutdown", json!({})));
    let status = child.wait().unwrap();
    println!("exit: {status}");
    assert!(status.success());
}
