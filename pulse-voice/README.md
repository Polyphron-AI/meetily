# pulse-voice

Headless voice sidecar for Pulse Code: Parakeet TDT transcription, microphone
and system-audio capture with Silero VAD, a global `WH_KEYBOARD_LL` hotkey,
foreground-monitor lookup and clipboard text insertion. It speaks JSON Lines
over stdio; the contract is [PROTOCOL.md](PROTOCOL.md).

## Build

Requires Rust (MSVC toolchain on Windows). This crate is standalone and is not
part of the Meetily workspace.

```sh
cargo build --release
cargo test
```

ONNX Runtime is downloaded at build time by the `ort` crate and linked
statically, so `target/release/pulse-voice.exe` runs on its own with no DLLs
beside it. Inference uses the CPU execution provider with
`min(physical cores, 4)` intra-op threads.

## Run

The desktop app spawns the executable and talks to it over stdin/stdout.
stdout carries protocol lines only; logs go to stderr (`RUST_LOG=debug` for more).

Smoke test against a Parakeet model directory (`encoder-model.int8.onnx`,
`decoder_joint-model.int8.onnx`, `nemo128.onnx`, `vocab.txt`) and a 16-bit
mono WAV:

```sh
cargo run --release --example smoke -- target/release/pulse-voice.exe <modelDir> <speech.wav>
```

Add `--capture` to also exercise the microphone and a meeting with system
audio (it plays the WAV through the default output device).

Windows only: `hotkey.set`, `foreground.monitor` and `text.insert`. Other
platforms build and answer them with `unsupported-platform`.

## Attribution

Derived from [Meetily](https://github.com/Zackriya-Solutions/meeting-minutes)
(MIT), Zackriya Solutions: the Parakeet inference in `src/engine.rs`, and
the capture and VAD approach in `src/capture.rs` and `src/vad.rs`. See
[LICENSE](LICENSE). The Silero VAD model is embedded through the `silero`
crate (MIT, Emotech Ltd).
