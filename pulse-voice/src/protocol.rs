//! Wire types for the pulse-voice JSON Lines protocol (see PROTOCOL.md).
//!
//! Parsing is pure so request validation and error codes are unit-testable.

use serde::Deserialize;
use serde_json::{json, Value};

pub const PROTOCOL_VERSION: u32 = 1;

/// Sender for stdout lines. A single writer thread owns stdout.
pub type Out = std::sync::mpsc::Sender<Value>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    InvalidRequest,
    UnknownMethod,
    ModelNotLoaded,
    ModelLoadFailed,
    Busy,
    NotRecording,
    DeviceUnavailable,
    TranscriptionFailed,
    InsertFailed,
    #[cfg_attr(windows, allow(dead_code))]
    UnsupportedPlatform,
    Internal,
}

impl ErrorCode {
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorCode::InvalidRequest => "invalid-request",
            ErrorCode::UnknownMethod => "unknown-method",
            ErrorCode::ModelNotLoaded => "model-not-loaded",
            ErrorCode::ModelLoadFailed => "model-load-failed",
            ErrorCode::Busy => "busy",
            ErrorCode::NotRecording => "not-recording",
            ErrorCode::DeviceUnavailable => "device-unavailable",
            ErrorCode::TranscriptionFailed => "transcription-failed",
            ErrorCode::InsertFailed => "insert-failed",
            ErrorCode::UnsupportedPlatform => "unsupported-platform",
            ErrorCode::Internal => "internal",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProtoError {
    pub code: ErrorCode,
    pub message: String,
}

impl ProtoError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ProtoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code.as_str(), self.message)
    }
}

pub type ProtoResult<T> = Result<T, ProtoError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HotkeyMode {
    Hold,
    Toggle,
}

/// A key in a hotkey chord, by its protocol name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Key {
    Ctrl,
    Alt,
    Shift,
    Win,
    Space,
    /// `A`..`Z`, stored as the ASCII uppercase letter.
    Letter(u8),
    /// `F1`..`F24`.
    F(u8),
}

impl Key {
    pub fn parse(name: &str) -> Option<Key> {
        match name {
            "Ctrl" => Some(Key::Ctrl),
            "Alt" => Some(Key::Alt),
            "Shift" => Some(Key::Shift),
            "Win" => Some(Key::Win),
            "Space" => Some(Key::Space),
            _ => {
                let bytes = name.as_bytes();
                if bytes.len() == 1 && bytes[0].is_ascii_uppercase() {
                    return Some(Key::Letter(bytes[0]));
                }
                let n = name.strip_prefix('F')?;
                if n.starts_with('0') {
                    return None;
                }
                let n: u8 = n.parse().ok()?;
                (1..=24).contains(&n).then_some(Key::F(n))
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HotkeySpec {
    pub keys: Vec<Key>,
    pub mode: HotkeyMode,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Method {
    Hello,
    Status,
    ModelLoad {
        model_dir: String,
    },
    DevicesList,
    Transcribe {
        pcm16: String,
        sample_rate: u32,
    },
    DictationStart {
        input_device_id: Option<String>,
    },
    DictationStop,
    DictationCancel,
    MeetingStart {
        input_device_id: Option<String>,
        system_audio: bool,
        output_device_id: Option<String>,
    },
    MeetingStop,
    ForegroundMonitor,
    TextInsert {
        text: String,
        restore_clipboard: bool,
    },
    HotkeySet(Option<HotkeySpec>),
    Shutdown,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    pub id: u64,
    pub method: Method,
}

/// A request that could not be parsed. `id` is `None` when the line carried no
/// usable id; the response then uses `"id": null`.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseFailure {
    pub id: Option<u64>,
    pub error: ProtoError,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ModelLoadParams {
    model_dir: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct TranscribeParams {
    pcm16: String,
    sample_rate: u32,
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct DictationStartParams {
    input_device_id: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct MeetingStartParams {
    input_device_id: Option<String>,
    system_audio: bool,
    output_device_id: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct TextInsertParams {
    text: String,
    restore_clipboard: bool,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct HotkeyParams {
    keys: Vec<String>,
    mode: String,
}

fn invalid(message: impl Into<String>) -> ProtoError {
    ProtoError::new(ErrorCode::InvalidRequest, message)
}

fn params<T: for<'de> Deserialize<'de>>(method: &str, raw: Option<Value>) -> ProtoResult<T> {
    let raw = raw.unwrap_or(Value::Object(Default::default()));
    serde_json::from_value(raw).map_err(|e| invalid(format!("invalid params for {method}: {e}")))
}

fn parse_hotkey(raw: Option<Value>) -> ProtoResult<Option<HotkeySpec>> {
    let raw = match raw {
        None | Some(Value::Null) => return Ok(None),
        Some(v) => v,
    };
    let p: HotkeyParams = serde_json::from_value(raw)
        .map_err(|e| invalid(format!("invalid params for hotkey.set: {e}")))?;
    let mode = match p.mode.as_str() {
        "hold" => HotkeyMode::Hold,
        "toggle" => HotkeyMode::Toggle,
        other => return Err(invalid(format!("unknown hotkey mode: {other}"))),
    };
    if p.keys.is_empty() {
        return Err(invalid("hotkey keys must not be empty"));
    }
    let mut keys = Vec::with_capacity(p.keys.len());
    for name in &p.keys {
        let key = Key::parse(name).ok_or_else(|| invalid(format!("unknown key name: {name}")))?;
        if keys.contains(&key) {
            return Err(invalid(format!("duplicate key: {name}")));
        }
        keys.push(key);
    }
    Ok(Some(HotkeySpec { keys, mode }))
}

/// Parses one stdin line into a request.
pub fn parse_request(line: &str) -> Result<Request, ParseFailure> {
    let fail = |id, error| ParseFailure { id, error };
    let value: Value = serde_json::from_str(line)
        .map_err(|e| fail(None, invalid(format!("malformed JSON: {e}"))))?;
    let Value::Object(mut obj) = value else {
        return Err(fail(None, invalid("request must be a JSON object")));
    };
    let id = match obj.get("id") {
        Some(Value::Number(n)) => match n.as_u64() {
            Some(id) if id > 0 => id,
            _ => return Err(fail(None, invalid("id must be a positive integer"))),
        },
        _ => return Err(fail(None, invalid("id must be a positive integer"))),
    };
    let method = match obj.remove("method") {
        Some(Value::String(m)) => m,
        _ => return Err(fail(Some(id), invalid("method must be a string"))),
    };
    let raw = obj.remove("params");
    let raw = match raw {
        Some(Value::Null) if method != "hotkey.set" => None,
        other => other,
    };
    let parsed = match method.as_str() {
        "hello" => Ok(Method::Hello),
        "status" => Ok(Method::Status),
        "model.load" => params::<ModelLoadParams>(&method, raw).and_then(|p| {
            if p.model_dir.is_empty() {
                Err(invalid("modelDir must not be empty"))
            } else {
                Ok(Method::ModelLoad {
                    model_dir: p.model_dir,
                })
            }
        }),
        "devices.list" => Ok(Method::DevicesList),
        "transcribe" => params::<TranscribeParams>(&method, raw).and_then(|p| {
            if p.sample_rate == 0 {
                Err(invalid("sampleRate must be positive"))
            } else {
                Ok(Method::Transcribe {
                    pcm16: p.pcm16,
                    sample_rate: p.sample_rate,
                })
            }
        }),
        "dictation.start" => {
            params::<DictationStartParams>(&method, raw).map(|p| Method::DictationStart {
                input_device_id: p.input_device_id,
            })
        }
        "dictation.stop" => Ok(Method::DictationStop),
        "dictation.cancel" => Ok(Method::DictationCancel),
        "meeting.start" => {
            params::<MeetingStartParams>(&method, raw).map(|p| Method::MeetingStart {
                input_device_id: p.input_device_id,
                system_audio: p.system_audio,
                output_device_id: p.output_device_id,
            })
        }
        "meeting.stop" => Ok(Method::MeetingStop),
        "foreground.monitor" => Ok(Method::ForegroundMonitor),
        "text.insert" => params::<TextInsertParams>(&method, raw).map(|p| Method::TextInsert {
            text: p.text,
            restore_clipboard: p.restore_clipboard,
        }),
        "hotkey.set" => parse_hotkey(raw).map(Method::HotkeySet),
        "shutdown" => Ok(Method::Shutdown),
        other => Err(ProtoError::new(
            ErrorCode::UnknownMethod,
            format!("unknown method: {other}"),
        )),
    };
    parsed
        .map(|method| Request { id, method })
        .map_err(|e| fail(Some(id), e))
}

pub fn response_ok(id: u64, result: Value) -> Value {
    json!({ "id": id, "result": result })
}

pub fn response_err(id: Option<u64>, error: &ProtoError) -> Value {
    json!({ "id": id, "error": { "code": error.code.as_str(), "message": error.message } })
}

pub fn event(name: &str, data: Value) -> Value {
    json!({ "event": name, "data": data })
}

pub fn platform() -> &'static str {
    if cfg!(target_os = "windows") {
        "windows"
    } else if cfg!(target_os = "macos") {
        "macos"
    } else {
        "linux"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err_code(line: &str) -> (Option<u64>, &'static str) {
        let f = parse_request(line).expect_err("expected failure");
        (f.id, f.error.code.as_str())
    }

    #[test]
    fn parses_simple_methods_with_and_without_params() {
        let r = parse_request(r#"{"id":1,"method":"hello"}"#).unwrap();
        assert_eq!(
            r,
            Request {
                id: 1,
                method: Method::Hello
            }
        );
        let r = parse_request(r#"{"id":2,"method":"status","params":{}}"#).unwrap();
        assert_eq!(r.method, Method::Status);
        let r = parse_request(r#"{"id":3,"method":"dictation.start"}"#).unwrap();
        assert_eq!(
            r.method,
            Method::DictationStart {
                input_device_id: None
            }
        );
        let r = parse_request(r#"{"id":3,"method":"dictation.start","params":null}"#).unwrap();
        assert_eq!(
            r.method,
            Method::DictationStart {
                input_device_id: None
            }
        );
    }

    #[test]
    fn parses_typed_params() {
        let r = parse_request(r#"{"id":7,"method":"model.load","params":{"modelDir":"C:\\m"}}"#)
            .unwrap();
        assert_eq!(
            r.method,
            Method::ModelLoad {
                model_dir: "C:\\m".into()
            }
        );
        let r = parse_request(
            r#"{"id":8,"method":"transcribe","params":{"pcm16":"AAA=","sampleRate":48000}}"#,
        )
        .unwrap();
        assert_eq!(
            r.method,
            Method::Transcribe {
                pcm16: "AAA=".into(),
                sample_rate: 48000
            }
        );
        let r = parse_request(
            r#"{"id":9,"method":"meeting.start","params":{"systemAudio":true,"outputDeviceId":"Speakers"}}"#,
        )
        .unwrap();
        assert_eq!(
            r.method,
            Method::MeetingStart {
                input_device_id: None,
                system_audio: true,
                output_device_id: Some("Speakers".into())
            }
        );
        let r = parse_request(
            r#"{"id":10,"method":"text.insert","params":{"text":"hi","restoreClipboard":false}}"#,
        )
        .unwrap();
        assert_eq!(
            r.method,
            Method::TextInsert {
                text: "hi".into(),
                restore_clipboard: false
            }
        );
    }

    #[test]
    fn parses_hotkey_set_and_removal() {
        let r = parse_request(r#"{"id":1,"method":"hotkey.set","params":{"keys":["Ctrl","Win","F13","Q"],"mode":"hold"}}"#)
            .unwrap();
        assert_eq!(
            r.method,
            Method::HotkeySet(Some(HotkeySpec {
                keys: vec![Key::Ctrl, Key::Win, Key::F(13), Key::Letter(b'Q')],
                mode: HotkeyMode::Hold
            }))
        );
        let r = parse_request(r#"{"id":2,"method":"hotkey.set","params":null}"#).unwrap();
        assert_eq!(r.method, Method::HotkeySet(None));
    }

    #[test]
    fn rejects_bad_envelopes_with_invalid_request() {
        assert_eq!(err_code("not json"), (None, "invalid-request"));
        assert_eq!(err_code("[1,2]"), (None, "invalid-request"));
        assert_eq!(err_code(r#"{"method":"hello"}"#), (None, "invalid-request"));
        assert_eq!(
            err_code(r#"{"id":0,"method":"hello"}"#),
            (None, "invalid-request")
        );
        assert_eq!(
            err_code(r#"{"id":-3,"method":"hello"}"#),
            (None, "invalid-request")
        );
        assert_eq!(
            err_code(r#"{"id":1.5,"method":"hello"}"#),
            (None, "invalid-request")
        );
        assert_eq!(
            err_code(r#"{"id":"4","method":"hello"}"#),
            (None, "invalid-request")
        );
        assert_eq!(err_code(r#"{"id":4}"#), (Some(4), "invalid-request"));
    }

    #[test]
    fn rejects_unknown_method() {
        assert_eq!(
            err_code(r#"{"id":5,"method":"nope"}"#),
            (Some(5), "unknown-method")
        );
    }

    #[test]
    fn rejects_bad_params_with_invalid_request() {
        assert_eq!(
            err_code(r#"{"id":6,"method":"model.load"}"#),
            (Some(6), "invalid-request")
        );
        assert_eq!(
            err_code(r#"{"id":6,"method":"model.load","params":{"modelDir":""}}"#),
            (Some(6), "invalid-request")
        );
        assert_eq!(
            err_code(r#"{"id":6,"method":"transcribe","params":{"pcm16":"","sampleRate":0}}"#),
            (Some(6), "invalid-request")
        );
        assert_eq!(
            err_code(r#"{"id":6,"method":"transcribe","params":{"pcm16":1,"sampleRate":16000}}"#),
            (Some(6), "invalid-request")
        );
        assert_eq!(
            err_code(r#"{"id":6,"method":"meeting.start","params":{}}"#),
            (Some(6), "invalid-request")
        );
        assert_eq!(
            err_code(r#"{"id":6,"method":"text.insert","params":{"text":"x"}}"#),
            (Some(6), "invalid-request")
        );
        assert_eq!(
            err_code(r#"{"id":6,"method":"dictation.start","params":{"bogus":1}}"#),
            (Some(6), "invalid-request")
        );
    }

    #[test]
    fn rejects_bad_hotkeys() {
        let bad = [
            r#"{"keys":[],"mode":"hold"}"#,
            r#"{"keys":["Ctrl"],"mode":"press"}"#,
            r#"{"keys":["Control"],"mode":"hold"}"#,
            r#"{"keys":["F25"],"mode":"hold"}"#,
            r#"{"keys":["F0"],"mode":"hold"}"#,
            r#"{"keys":["F01"],"mode":"hold"}"#,
            r#"{"keys":["a"],"mode":"hold"}"#,
            r#"{"keys":["Ctrl","Ctrl"],"mode":"hold"}"#,
        ];
        for p in bad {
            let line = format!(r#"{{"id":9,"method":"hotkey.set","params":{p}}}"#);
            assert_eq!(err_code(&line), (Some(9), "invalid-request"), "{p}");
        }
    }

    #[test]
    fn key_names_cover_protocol_set() {
        for c in b'A'..=b'Z' {
            assert_eq!(Key::parse(&(c as char).to_string()), Some(Key::Letter(c)));
        }
        for n in 1..=24u8 {
            assert_eq!(Key::parse(&format!("F{n}")), Some(Key::F(n)));
        }
    }

    #[test]
    fn serializes_responses_and_events() {
        let ok = response_ok(3, json!({}));
        assert_eq!(ok.to_string(), r#"{"id":3,"result":{}}"#);
        let err = response_err(None, &ProtoError::new(ErrorCode::Busy, "x"));
        assert_eq!(
            err.to_string(),
            r#"{"error":{"code":"busy","message":"x"},"id":null}"#
        );
        let ev = event("hotkey", json!({"phase":"down"}));
        assert_eq!(ev["event"], "hotkey");
        assert_eq!(ev["data"]["phase"], "down");
    }
}
