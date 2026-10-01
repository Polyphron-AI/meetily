//! pulse-voice: headless voice sidecar for Pulse Code. Speaks the JSON Lines
//! protocol in PROTOCOL.md over stdio.
//!
//! Threads: stdin reader (dispatch, blocks on read), stdout writer (the only
//! code that touches stdout), engine (owns the model), optional hotkey hook,
//! and per-capture stream/processing threads that exist only while recording.

mod audio;
mod capture;
mod engine;
mod hotkey;
mod protocol;
mod vad;
mod win;

use std::io::{BufRead, Write};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use serde_json::{json, Value};

use capture::{Capture, CaptureKind, CaptureRequest, MeetingContext};
use engine::Engine;
use protocol::{
    event, parse_request, platform, response_err, response_ok, ErrorCode, Method, Out, ProtoError,
    ProtoResult, Request, PROTOCOL_VERSION,
};

/// Single owner of stdout. `Value::Null` is never a protocol line; it tells
/// the writer to flush and exit.
fn spawn_writer() -> (Out, JoinHandle<()>) {
    let (tx, rx) = mpsc::channel::<Value>();
    let handle = std::thread::Builder::new()
        .name("stdout".into())
        .spawn(move || {
            let stdout = std::io::stdout();
            while let Ok(value) = rx.recv() {
                if value.is_null() {
                    break;
                }
                let mut lock = stdout.lock();
                let mut line = value.to_string();
                line.push('\n');
                if lock
                    .write_all(line.as_bytes())
                    .and_then(|_| lock.flush())
                    .is_err()
                {
                    break;
                }
            }
        })
        .expect("spawn stdout writer");
    (tx, handle)
}

fn not_recording() -> ProtoError {
    ProtoError::new(ErrorCode::NotRecording, "no matching capture is running")
}

struct App {
    out: Out,
    engine: Engine,
    capture: Arc<Mutex<Option<Capture>>>,
    #[cfg(windows)]
    hook: Option<hotkey::Hook>,
    sessions: u64,
}

impl App {
    fn reply(&self, id: u64, result: ProtoResult<Value>) {
        let _ = self.out.send(match result {
            Ok(v) => response_ok(id, v),
            Err(e) => response_err(Some(id), &e),
        });
    }

    /// Runs `f` off the stdin thread and sends its result as the response.
    fn reply_async(&self, id: u64, f: impl FnOnce() -> ProtoResult<Value> + Send + 'static) {
        let out = self.out.clone();
        std::thread::spawn(move || {
            let _ = out.send(match f() {
                Ok(v) => response_ok(id, v),
                Err(e) => response_err(Some(id), &e),
            });
        });
    }

    fn running_kind(&self) -> Option<CaptureKind> {
        self.capture
            .lock()
            .unwrap()
            .as_ref()
            .filter(|c| !c.has_ended())
            .map(|c| c.kind)
    }

    /// Removes the running capture if it is of `kind`.
    fn take_capture(&self, kind: CaptureKind) -> Option<Capture> {
        let mut slot = self.capture.lock().unwrap();
        if slot.as_ref().is_some_and(|c| c.kind == kind) {
            slot.take()
        } else {
            None
        }
    }

    fn start_capture(&mut self, req: CaptureRequest) -> ProtoResult<Option<String>> {
        if !self.engine.is_loaded() {
            return Err(ProtoError::new(
                ErrorCode::ModelNotLoaded,
                "call model.load first",
            ));
        }
        let mut slot = self.capture.lock().unwrap();
        if let Some(c) = slot.as_ref() {
            if !c.has_ended() {
                return Err(ProtoError::new(
                    ErrorCode::Busy,
                    format!("{} capture is running", c.kind.as_str()),
                ));
            }
            // A capture that died from a device error; reap it.
            slot.take();
        }
        let cap = Capture::start(req, self.out.clone())?;
        let session = cap.session_id.clone();
        *slot = Some(cap);
        Ok(session)
    }

    /// Handles one request. Returns false when the process should exit.
    fn handle(&mut self, Request { id, method }: Request) -> bool {
        match method {
            Method::Hello => self.reply(
                id,
                Ok(json!({ "protocol": PROTOCOL_VERSION, "version": env!("CARGO_PKG_VERSION"), "platform": platform() })),
            ),
            Method::Status => {
                let capture = self.running_kind().map(|k| k.as_str());
                self.reply(
                    id,
                    Ok(json!({ "modelLoaded": self.engine.is_loaded(), "modelDir": self.engine.model_dir(), "capture": capture })),
                );
            }
            Method::ModelLoad { model_dir } => {
                let out = self.out.clone();
                self.engine.load(model_dir, move |r| {
                    let _ = out.send(match r {
                        Ok(load_ms) => response_ok(id, json!({ "loadMs": load_ms })),
                        Err(e) => response_err(Some(id), &e),
                    });
                });
            }
            Method::DevicesList => self.reply_async(id, capture::list_devices),
            Method::Transcribe { pcm16, sample_rate } => {
                let out = self.out.clone();
                let engine = self.engine.clone();
                std::thread::spawn(move || {
                    let samples = audio::decode_pcm16_base64(&pcm16)
                        .map_err(|e| ProtoError::new(ErrorCode::InvalidRequest, e))
                        .and_then(|s| {
                            audio::resample_to_16k(&s, sample_rate).map_err(|e| ProtoError::new(ErrorCode::InvalidRequest, e))
                        });
                    drop(pcm16);
                    match samples {
                        Err(e) => {
                            let _ = out.send(response_err(Some(id), &e));
                        }
                        Ok(samples) => engine.transcribe(samples, move |r| {
                            let _ = out.send(match r {
                                Ok(t) => response_ok(id, t.to_json()),
                                Err(e) => response_err(Some(id), &e),
                            });
                        }),
                    }
                });
            }
            Method::DictationStart { input_device_id } => {
                let r = self.start_capture(CaptureRequest {
                    kind: CaptureKind::Dictation,
                    input_device_id,
                    loopback: None,
                    meeting: None,
                });
                self.reply(id, r.map(|_| json!({})));
            }
            Method::DictationStop => match self.take_capture(CaptureKind::Dictation) {
                None => self.reply(id, Err(not_recording())),
                Some(cap) => {
                    let out = self.out.clone();
                    let engine = self.engine.clone();
                    std::thread::spawn(move || {
                        // The microphone is closed inside stop(), before transcription.
                        let outcome = cap.stop();
                        if outcome.failed {
                            let _ = out.send(response_err(Some(id), &not_recording()));
                            return;
                        }
                        engine.transcribe(outcome.samples, move |r| {
                            let _ = out.send(match r {
                                Ok(t) => response_ok(id, t.to_json()),
                                Err(e) => response_err(Some(id), &e),
                            });
                        });
                    });
                }
            },
            Method::DictationCancel => match self.take_capture(CaptureKind::Dictation) {
                None => self.reply(id, Ok(json!({}))),
                Some(cap) => self.reply_async(id, move || {
                    drop(cap.stop());
                    Ok(json!({}))
                }),
            },
            Method::MeetingStart { input_device_id, system_audio, output_device_id } => {
                self.sessions += 1;
                let epoch_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis())
                    .unwrap_or(0);
                let session_id = format!("{epoch_ms:x}-{}", self.sessions);
                let r = self.start_capture(CaptureRequest {
                    kind: CaptureKind::Meeting,
                    input_device_id,
                    loopback: system_audio.then_some(output_device_id),
                    meeting: Some(MeetingContext { session_id, engine: self.engine.clone() }),
                });
                self.reply(id, r.map(|s| json!({ "sessionId": s })));
            }
            Method::MeetingStop => match self.take_capture(CaptureKind::Meeting) {
                None => self.reply(id, Err(not_recording())),
                Some(cap) => {
                    let out = self.out.clone();
                    let engine = self.engine.clone();
                    std::thread::spawn(move || {
                        let session_id = cap.session_id.clone();
                        // stop() flushes VAD and queues the final segment on the engine.
                        let outcome = cap.stop();
                        let duration_ms = audio::samples_to_ms(outcome.total_samples);
                        // Respond only after every queued segment has been emitted.
                        engine.after_pending(move || {
                            let _ = out.send(if outcome.failed {
                                response_err(Some(id), &not_recording())
                            } else {
                                response_ok(id, json!({ "sessionId": session_id, "durationMs": duration_ms }))
                            });
                        });
                    });
                }
            },
            Method::ForegroundMonitor => self.reply(id, win::foreground_monitor()),
            Method::TextInsert { text, restore_clipboard } => {
                self.reply_async(id, move || win::insert_text(&text, restore_clipboard).map(|_| json!({})))
            }
            Method::HotkeySet(spec) => {
                let r = self.set_hotkey(spec);
                self.reply(id, r.map(|_| json!({})));
            }
            Method::Shutdown => {
                self.reply(id, Ok(json!({})));
                return false;
            }
        }
        true
    }

    #[cfg(windows)]
    fn set_hotkey(&mut self, spec: Option<protocol::HotkeySpec>) -> ProtoResult<()> {
        self.hook = None;
        if let Some(spec) = spec {
            let hook = hotkey::Hook::install(spec.keys, self.out.clone())
                .map_err(|e| ProtoError::new(ErrorCode::Internal, e))?;
            self.hook = Some(hook);
        }
        Ok(())
    }

    #[cfg(not(windows))]
    fn set_hotkey(&mut self, _spec: Option<protocol::HotkeySpec>) -> ProtoResult<()> {
        Err(ProtoError::new(
            ErrorCode::UnsupportedPlatform,
            "only supported on Windows",
        ))
    }

    fn close(&mut self) {
        // Dropping a capture closes its devices.
        drop(self.capture.lock().unwrap().take());
        #[cfg(windows)]
        {
            self.hook = None;
        }
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .target(env_logger::Target::Stderr)
        .init();
    win::init_dpi_awareness();

    let (out, writer) = spawn_writer();
    let _ = out.send(event("ready", json!({})));
    let engine = Engine::spawn(out.clone());
    let mut app = App {
        out: out.clone(),
        engine,
        capture: Arc::new(Mutex::new(None)),
        #[cfg(windows)]
        hook: None,
        sessions: 0,
    };
    log::info!(
        "pulse-voice {} ready (protocol {PROTOCOL_VERSION})",
        env!("CARGO_PKG_VERSION")
    );

    let stdin = std::io::stdin();
    let mut reader = stdin.lock();
    let mut buf = Vec::new();
    loop {
        buf.clear();
        match reader.read_until(b'\n', &mut buf) {
            Ok(0) => {
                log::info!("stdin closed; exiting");
                break;
            }
            Ok(_) => {}
            Err(e) => {
                log::error!("stdin read failed: {e}");
                break;
            }
        }
        let line = String::from_utf8_lossy(&buf);
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match parse_request(line) {
            Ok(req) => {
                if !app.handle(req) {
                    break;
                }
            }
            Err(f) => {
                let _ = out.send(response_err(f.id, &f.error));
            }
        }
    }

    app.close();
    let _ = out.send(Value::Null);
    let _ = writer.join();
    std::process::exit(0);
}
