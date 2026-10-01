// Derived from Meetily (MIT), Zackriya Solutions
// Source: frontend/src-tauri/src/audio/stream.rs (cpal stream construction per
// sample format) and audio/devices/platform/windows.rs (WASAPI loopback on an
// output device). Pipeline, mixing and threading are rewritten for the sidecar.

//! Microphone and system-audio capture.
//!
//! Threads per capture:
//! - stream thread: owns the cpal streams (they are not `Send`), blocks on a
//!   stop channel, and drops the streams (closing the device) when told to.
//! - processing thread: resamples to 16 kHz, mixes, emits `level`, and either
//!   buffers (dictation) or runs VAD and queues segments on the engine
//!   (meeting). It never waits on transcription.
//!
//! Audio callbacks only convert samples and send them over a channel.

use std::collections::VecDeque;
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, SizedSample, Stream};
use serde_json::{json, Value};

use crate::audio::{rms, StreamResampler, TARGET_RATE};
use crate::engine::Engine;
use crate::protocol::{event, ErrorCode, Out, ProtoError, ProtoResult};
use crate::vad::{SegmenterConfig, Silero, Vad};

const LEVEL_INTERVAL: Duration = Duration::from_millis(50);
/// Loopback audio waiting for mic audio to mix with; older audio is dropped.
const MAX_LOOP_BACKLOG: usize = TARGET_RATE as usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureKind {
    Dictation,
    Meeting,
}

impl CaptureKind {
    pub fn as_str(self) -> &'static str {
        match self {
            CaptureKind::Dictation => "dictation",
            CaptureKind::Meeting => "meeting",
        }
    }
}

fn device_unavailable(message: impl Into<String>) -> ProtoError {
    ProtoError::new(ErrorCode::DeviceUnavailable, message)
}

fn device_entry(device: &Device, default_name: &Option<String>) -> Option<Value> {
    let name = device.name().ok()?;
    let is_default = default_name.as_deref() == Some(name.as_str());
    Some(json!({ "id": name, "name": name, "isDefault": is_default }))
}

/// `devices.list`. The id is the device name (cpal exposes no stabler id).
pub fn list_devices() -> ProtoResult<Value> {
    let host = cpal::default_host();
    let default_in = host.default_input_device().and_then(|d| d.name().ok());
    let default_out = host.default_output_device().and_then(|d| d.name().ok());
    let inputs: Vec<Value> = host
        .input_devices()
        .map_err(|e| device_unavailable(e.to_string()))?
        .filter_map(|d| device_entry(&d, &default_in))
        .collect();
    let outputs: Vec<Value> = host
        .output_devices()
        .map_err(|e| device_unavailable(e.to_string()))?
        .filter_map(|d| device_entry(&d, &default_out))
        .collect();
    Ok(json!({ "inputs": inputs, "outputs": outputs }))
}

fn find_device(input: bool, id: Option<&str>) -> ProtoResult<Device> {
    let host = cpal::default_host();
    let kind = if input { "input" } else { "output" };
    match id {
        None => if input {
            host.default_input_device()
        } else {
            host.default_output_device()
        }
        .ok_or_else(|| device_unavailable(format!("no default {kind} device"))),
        Some(id) => {
            let mut devices = if input {
                host.input_devices()
            } else {
                host.output_devices()
            }
            .map_err(|e| device_unavailable(e.to_string()))?;
            devices
                .find(|d| d.name().map(|n| n == id).unwrap_or(false))
                .ok_or_else(|| device_unavailable(format!("{kind} device not found: {id}")))
        }
    }
}

enum Msg {
    Mic(Vec<f32>),
    Loop(Vec<f32>),
    Error(String),
    End,
}

#[derive(Clone, Copy)]
enum Source {
    Mic,
    Loop,
}

fn build_typed<T: SizedSample>(
    device: &Device,
    config: &cpal::StreamConfig,
    source: Source,
    tx: mpsc::Sender<Msg>,
) -> Result<Stream, cpal::BuildStreamError>
where
    f32: cpal::FromSample<T>,
{
    let channels = config.channels.max(1) as usize;
    let err_tx = tx.clone();
    device.build_input_stream(
        config,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            let mono: Vec<f32> = data
                .chunks(channels)
                .map(|frame| {
                    frame.iter().map(|&s| s.to_sample::<f32>()).sum::<f32>() / channels as f32
                })
                .collect();
            let _ = tx.send(match source {
                Source::Mic => Msg::Mic(mono),
                Source::Loop => Msg::Loop(mono),
            });
        },
        move |err| {
            let _ = err_tx.send(Msg::Error(err.to_string()));
        },
        None,
    )
}

/// Builds an input stream. On WASAPI, an input stream on an output device is loopback.
fn build_stream(
    device: &Device,
    source: Source,
    tx: mpsc::Sender<Msg>,
) -> Result<(Stream, u32), String> {
    let supported = match source {
        Source::Mic => device.default_input_config(),
        Source::Loop => device.default_output_config(),
    }
    .map_err(|e| e.to_string())?;
    let config: cpal::StreamConfig = supported.config();
    let rate = config.sample_rate.0;
    let stream = match supported.sample_format() {
        SampleFormat::F32 => build_typed::<f32>(device, &config, source, tx),
        SampleFormat::I16 => build_typed::<i16>(device, &config, source, tx),
        SampleFormat::I32 => build_typed::<i32>(device, &config, source, tx),
        SampleFormat::U16 => build_typed::<u16>(device, &config, source, tx),
        SampleFormat::I8 => build_typed::<i8>(device, &config, source, tx),
        other => return Err(format!("unsupported sample format {other:?}")),
    }
    .map_err(|e| e.to_string())?;
    stream.play().map_err(|e| e.to_string())?;
    Ok((stream, rate))
}

pub struct MeetingContext {
    pub session_id: String,
    pub engine: Engine,
}

pub struct CaptureRequest {
    pub kind: CaptureKind,
    pub input_device_id: Option<String>,
    /// `Some(output id or default)` when system audio should be mixed in.
    pub loopback: Option<Option<String>>,
    pub meeting: Option<MeetingContext>,
}

/// What the processing thread hands back when the capture ends.
pub struct Outcome {
    /// Buffered 16 kHz audio (dictation only).
    pub samples: Vec<f32>,
    /// Captured 16 kHz mic samples, for durations.
    pub total_samples: usize,
    /// The capture ended because of a device error (already reported as `capture.error`).
    pub failed: bool,
}

pub struct Capture {
    pub kind: CaptureKind,
    pub session_id: Option<String>,
    stop_tx: mpsc::Sender<()>,
    stream_thread: Option<JoinHandle<()>>,
    proc_thread: Option<JoinHandle<Outcome>>,
}

impl Capture {
    pub fn start(req: CaptureRequest, out: Out) -> ProtoResult<Capture> {
        let mic = find_device(true, req.input_device_id.as_deref())?;
        let loop_dev = match &req.loopback {
            Some(id) => Some(find_device(false, id.as_deref())?),
            None => None,
        };
        let (tx, rx) = mpsc::channel::<Msg>();
        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<(u32, Option<u32>), String>>();

        let stream_tx = tx.clone();
        let stream_thread = std::thread::Builder::new()
            .name("capture-streams".into())
            .spawn(move || {
                let mic_stream = match build_stream(&mic, Source::Mic, stream_tx.clone()) {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = ready_tx.send(Err(format!("microphone: {e}")));
                        return;
                    }
                };
                let loop_stream = match loop_dev
                    .as_ref()
                    .map(|d| build_stream(d, Source::Loop, stream_tx.clone()))
                {
                    None => None,
                    Some(Ok(s)) => Some(s),
                    Some(Err(e)) => {
                        let _ = ready_tx.send(Err(format!("system audio: {e}")));
                        return;
                    }
                };
                let _ = ready_tx.send(Ok((mic_stream.1, loop_stream.as_ref().map(|s| s.1))));
                // Block until stop (or the Capture is dropped).
                let _ = stop_rx.recv();
                drop(loop_stream);
                drop(mic_stream);
                log::info!("capture devices closed");
                let _ = stream_tx.send(Msg::End);
            })
            .map_err(|e| ProtoError::new(ErrorCode::Internal, e.to_string()))?;
        drop(tx);

        let (mic_rate, loop_rate) = match ready_rx.recv() {
            Ok(Ok(rates)) => rates,
            Ok(Err(e)) => {
                let _ = stream_thread.join();
                return Err(device_unavailable(e));
            }
            Err(_) => {
                let _ = stream_thread.join();
                return Err(device_unavailable("capture thread exited"));
            }
        };
        log::info!(
            "{} capture started: mic {mic_rate} Hz, loopback {loop_rate:?}",
            req.kind.as_str()
        );

        let vad = match req.kind {
            CaptureKind::Meeting => Some(
                Silero::new()
                    .map(|s| Vad::new(s, SegmenterConfig::default()))
                    .map_err(|e| {
                        ProtoError::new(ErrorCode::Internal, format!("VAD init failed: {e}"))
                    }),
            ),
            CaptureKind::Dictation => None,
        };
        let vad = match vad.transpose() {
            Ok(v) => v,
            Err(e) => {
                let _ = stop_tx.send(());
                let _ = stream_thread.join();
                return Err(e);
            }
        };
        let session_id = req.meeting.as_ref().map(|m| m.session_id.clone());
        let proc_stop = stop_tx.clone();
        let kind = req.kind;
        let proc = Processor {
            kind,
            out,
            mic_rs: StreamResampler::new(mic_rate)
                .map_err(|e| ProtoError::new(ErrorCode::Internal, e))?,
            loop_rs: match loop_rate {
                Some(r) => Some(
                    StreamResampler::new(r).map_err(|e| ProtoError::new(ErrorCode::Internal, e))?,
                ),
                None => None,
            },
            loop_queue: VecDeque::new(),
            scratch: Vec::new(),
            buffer: Vec::new(),
            total: 0,
            level_acc: Vec::new(),
            last_level: Instant::now(),
            vad,
            meeting: req.meeting,
            segment_index: 0,
        };
        let proc_thread = std::thread::Builder::new()
            .name("capture-proc".into())
            .spawn(move || proc.run(rx, proc_stop))
            .map_err(|e| ProtoError::new(ErrorCode::Internal, e.to_string()))?;

        Ok(Capture {
            kind,
            session_id,
            stop_tx,
            stream_thread: Some(stream_thread),
            proc_thread: Some(proc_thread),
        })
    }

    /// True once the processing thread has finished (device error).
    pub fn has_ended(&self) -> bool {
        self.proc_thread.as_ref().is_none_or(|h| h.is_finished())
    }

    /// Closes the devices, then waits for the processing thread to drain.
    pub fn stop(mut self) -> Outcome {
        let _ = self.stop_tx.send(());
        if let Some(h) = self.stream_thread.take() {
            let _ = h.join();
        }
        self.proc_thread
            .take()
            .and_then(|h| h.join().ok())
            .unwrap_or(Outcome {
                samples: Vec::new(),
                total_samples: 0,
                failed: true,
            })
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        let _ = self.stop_tx.send(());
        if let Some(h) = self.stream_thread.take() {
            let _ = h.join();
        }
        if let Some(h) = self.proc_thread.take() {
            let _ = h.join();
        }
    }
}

struct Processor {
    kind: CaptureKind,
    out: Out,
    mic_rs: StreamResampler,
    loop_rs: Option<StreamResampler>,
    loop_queue: VecDeque<f32>,
    scratch: Vec<f32>,
    buffer: Vec<f32>,
    total: usize,
    level_acc: Vec<f32>,
    last_level: Instant,
    vad: Option<Vad<Silero>>,
    meeting: Option<MeetingContext>,
    segment_index: u64,
}

impl Processor {
    fn run(mut self, rx: mpsc::Receiver<Msg>, stop: mpsc::Sender<()>) -> Outcome {
        let mut failed = false;
        while let Ok(msg) = rx.recv() {
            match msg {
                Msg::Mic(samples) => {
                    self.scratch.clear();
                    if let Err(e) = self.mic_rs.push(&samples, &mut self.scratch) {
                        log::error!("{e}");
                    }
                    self.consume_mic();
                }
                Msg::Loop(samples) => {
                    if let Some(rs) = self.loop_rs.as_mut() {
                        self.scratch.clear();
                        if let Err(e) = rs.push(&samples, &mut self.scratch) {
                            log::error!("{e}");
                        }
                        self.loop_queue.extend(self.scratch.drain(..));
                        while self.loop_queue.len() > MAX_LOOP_BACKLOG {
                            self.loop_queue.pop_front();
                        }
                    }
                }
                Msg::Error(message) => {
                    if !failed {
                        failed = true;
                        log::error!("{} capture error: {message}", self.kind.as_str());
                        let _ = self.out.send(event(
                            "capture.error",
                            json!({ "capture": self.kind.as_str(), "code": "device-unavailable", "message": message }),
                        ));
                        let _ = stop.send(());
                    }
                }
                Msg::End => break,
            }
        }
        self.scratch.clear();
        if let Err(e) = self.mic_rs.flush(&mut self.scratch) {
            log::error!("{e}");
        }
        self.consume_mic();
        if let Some(seg) = self.vad.as_mut().and_then(|v| v.finish()) {
            self.submit_segment(seg);
        }
        Outcome {
            samples: std::mem::take(&mut self.buffer),
            total_samples: self.total,
            failed,
        }
    }

    /// Mixes freshly resampled mic audio (in `scratch`) with queued loopback
    /// audio. The mic is the clock: loopback that is silent (WASAPI sends
    /// nothing) simply contributes zeros.
    fn consume_mic(&mut self) {
        if self.scratch.is_empty() {
            return;
        }
        let mut mixed = std::mem::take(&mut self.scratch);
        if self.loop_rs.is_some() {
            for s in mixed.iter_mut() {
                let l = self.loop_queue.pop_front().unwrap_or(0.0);
                *s = (*s + l).clamp(-1.0, 1.0);
            }
        }
        self.total += mixed.len();
        self.level_acc.extend_from_slice(&mixed);
        if self.last_level.elapsed() >= LEVEL_INTERVAL {
            let level = rms(&self.level_acc);
            self.level_acc.clear();
            self.last_level = Instant::now();
            let _ = self.out.send(event(
                "level",
                json!({ "capture": self.kind.as_str(), "rms": level }),
            ));
        }
        match self.kind {
            CaptureKind::Dictation => self.buffer.extend_from_slice(&mixed),
            CaptureKind::Meeting => {
                let mut segs = Vec::new();
                if let Some(vad) = self.vad.as_mut() {
                    vad.push(&mixed, &mut segs);
                }
                for seg in segs {
                    self.submit_segment(seg);
                }
            }
        }
        mixed.clear();
        self.scratch = mixed;
    }

    fn submit_segment(&mut self, seg: crate::vad::Segment) {
        let Some(meeting) = self.meeting.as_ref() else {
            return;
        };
        let index = self.segment_index;
        self.segment_index += 1;
        let out = self.out.clone();
        let session_id = meeting.session_id.clone();
        let (start_ms, end_ms) = (seg.start_ms, seg.end_ms);
        meeting.engine.transcribe(seg.samples, move |result| {
            let text = match result {
                Ok(t) => t.text,
                Err(e) => {
                    log::error!("segment {index} transcription failed: {e}");
                    String::new()
                }
            };
            let _ = out.send(event(
                "meeting.segment",
                json!({ "sessionId": session_id, "index": index, "startMs": start_ms, "endMs": end_ms, "text": text }),
            ));
        });
    }
}
