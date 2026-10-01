# pulse-voice sidecar protocol (v1)

`pulse-voice` is a headless process spawned by the Pulse Code desktop app. It
reuses Meetily's native Parakeet engine and audio capture. The desktop app owns
all UI; the sidecar owns audio, transcription, the global keyboard hook, the
foreground-monitor lookup and text insertion.

## Transport

- JSON Lines over stdio, UTF-8, one object per line.
- stdin: requests from the desktop app.
- stdout: responses and events only. Nothing else may be written to stdout.
- stderr: human-readable logs (`RUST_LOG` style). The desktop app forwards them
  to its log file.
- The sidecar exits when stdin closes (parent died) or on `shutdown`.
- When idle (no capture running) the sidecar must use no measurable CPU: block
  on stdin, block the hook thread in `GetMessageW`, no polling timers.

### Request

```json
{ "id": 7, "method": "dictation.start", "params": { } }
```

`id` is a positive integer chosen by the caller. `params` may be omitted.

### Response (exactly one per request)

```json
{ "id": 7, "result": { } }
{ "id": 7, "error": { "code": "model-not-loaded", "message": "..." } }
```

### Event (unsolicited)

```json
{ "event": "hotkey", "data": { "phase": "down" } }
```

## Error codes

`invalid-request`, `unknown-method`, `model-not-loaded`, `model-load-failed`,
`busy` (a capture is already running), `not-recording`, `device-unavailable`,
`transcription-failed`, `insert-failed`, `unsupported-platform`, `internal`.

## Methods

### `hello`
Result: `{ "protocol": 1, "version": "<crate version>", "platform": "windows" | "macos" | "linux" }`.
Sent first by the desktop app; the desktop app refuses a sidecar whose
`protocol` differs.

### `status`
Result: `{ "modelLoaded": bool, "modelDir": string | null, "capture": null | "dictation" | "meeting" }`.

### `model.load`
Params: `{ "modelDir": string }`. The directory holds
`encoder-model.int8.onnx`, `decoder_joint-model.int8.onnx`, `nemo128.onnx`,
`vocab.txt` (the files Pulse already bundles under `parakeet-model/`).
Result: `{ "loadMs": number }`. Idempotent for the same directory. Loading runs
off the stdin thread; other requests keep being answered.

### `devices.list`
Result: `{ "inputs": [{ "id": string, "name": string, "isDefault": bool }], "outputs": [same] }`.
`id` is stable across restarts (use the device name if nothing better exists).

### `transcribe`
Params: `{ "pcm16": string, "sampleRate": number }`. `pcm16` is base64 of
little-endian signed 16-bit mono samples. The sidecar resamples to 16 kHz.
Result: `{ "text": string, "audioMs": number, "transcribeMs": number }`.
Used by the composer, which already records audio in the renderer.

### `dictation.start`
Params: `{ "inputDeviceId"?: string }` (default input when omitted).
Opens the microphone and buffers audio in memory. Result: `{}`.
Error `busy` if any capture is running.
While recording, emits `level` events at most 20 times per second.

### `dictation.stop`
Stops capture and transcribes the buffer. Result:
`{ "text": string, "audioMs": number, "transcribeMs": number }`.
The microphone must be closed before the transcription starts.

### `dictation.cancel`
Stops capture and discards audio. Result: `{}`. No error if nothing is recording.

### `meeting.start`
Params: `{ "inputDeviceId"?: string, "systemAudio": bool, "outputDeviceId"?: string }`.
Captures the microphone and, when `systemAudio` is true, WASAPI loopback of the
output device. Both sources are resampled to 16 kHz mono and mixed. A VAD splits
speech into segments (silence of 600 ms ends a segment, segments are capped at
30 s). Each segment is transcribed off the capture thread and emitted as a
`meeting.segment` event in order. Result: `{ "sessionId": string }`.

### `meeting.stop`
Stops capture, flushes and transcribes the final segment, emits its
`meeting.segment`, then responds `{ "sessionId": string, "durationMs": number }`.
Every segment of the session is emitted before this response.

### `foreground.monitor`
Result describes the monitor holding the current foreground window, falling
back to the monitor under the cursor when there is no foreground window:
`{ "source": "foreground" | "cursor", "bounds": Rect, "workArea": Rect, "dpi": number }`
where `Rect` is `{ "x", "y", "width", "height" }` in physical pixels
(`MonitorFromWindow` + `GetMonitorInfoW` + `GetDpiForMonitor`). The process is
per-monitor DPI aware (v2).

### `text.insert`
Params: `{ "text": string, "restoreClipboard": bool }`.
Inserts text into the focused control of the foreground window:
1. Snapshot every clipboard format that holds an `HGLOBAL` (skip others).
2. Put `text` on the clipboard as `CF_UNICODETEXT`.
3. Send Ctrl+V with `SendInput`, releasing any modifiers the user still holds
   first so the chord is clean.
4. Wait 150 ms, then restore the snapshot when `restoreClipboard` is true.
Result: `{}`. Error `insert-failed` with a reason when the clipboard cannot be
opened after 10 retries of 20 ms.

### `hotkey.set`
Params: `{ "keys": string[], "mode": "hold" | "toggle" } | null`.
`keys` uses names `Ctrl`, `Alt`, `Shift`, `Win`, `Space`, `A`..`Z`, `F1`..`F24`.
Installs (or replaces, or with `null` removes) a `WH_KEYBOARD_LL` hook on a
dedicated thread. Result: `{}`.
- The chord is active when all `keys` are down. Emits `hotkey` with
  `phase: "down"` once when it becomes active and `phase: "up"` once when any of
  its keys is released. The desktop app interprets `mode`; the sidecar only
  reports edges.
- When the chord includes `Win` and fired, the sidecar injects an unassigned
  virtual key (`0xE8`) before the `Win` release so Windows does not open Start.
- The hook callback must return within a few milliseconds: it only updates key
  state and posts to a channel.
- Key repeat does not re-emit `down`.

### `shutdown`
Result `{}`, then the process exits 0 after closing devices.

## Events

| event | data |
|-------|------|
| `ready` | `{}` emitted once at startup, before any response |
| `hotkey` | `{ "phase": "down" \| "up" }` |
| `level` | `{ "capture": "dictation" \| "meeting", "rms": number }` 0..1 |
| `meeting.segment` | `{ "sessionId", "index": number, "startMs": number, "endMs": number, "text": string }` (times relative to `meeting.start`) |
| `capture.error` | `{ "capture": "dictation" \| "meeting", "code": string, "message": string }` capture ended unexpectedly (device removed) |
| `model.loaded` | `{ "modelDir": string, "loadMs": number }` |
