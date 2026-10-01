// Derived from Meetily (MIT), Zackriya Solutions
// Source: frontend/src-tauri/src/parakeet_engine/model.rs (ParakeetModel). The
// download/model-manager machinery is dropped; the model dir comes from `model.load`.

//! Parakeet TDT inference plus the engine thread that owns the model.
//!
//! All model access is serialized on one thread fed by a job queue, so
//! meeting segments are transcribed and emitted in order, and a barrier job
//! queued after the last segment runs only once every segment is out.

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Instant;

use ndarray::{Array, Array1, Array2, Array3, ArrayD, ArrayViewD, IxDyn};
use ort::execution_providers::CPUExecutionProvider;
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use regex::Regex;
use serde_json::json;

use crate::audio::{samples_to_ms, TARGET_RATE};
use crate::protocol::{event, ErrorCode, Out, ProtoError, ProtoResult};

type DecoderState = (Array3<f32>, Array3<f32>);

const MAX_TOKENS_PER_STEP: usize = 3;
const TDT_DURATIONS: [usize; 5] = [0, 1, 2, 3, 4];
/// Very short clips are zero padded so the preprocessor always has frames.
const MIN_SAMPLES: usize = TARGET_RATE as usize / 2;

pub const MODEL_FILES: [&str; 4] = [
    "encoder-model.int8.onnx",
    "decoder_joint-model.int8.onnx",
    "nemo128.onnx",
    "vocab.txt",
];

static DECODE_SPACE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\A\s|\s\B|(\s)\b").expect("static regex"));

#[derive(thiserror::Error, Debug)]
pub enum ParakeetError {
    #[error("ORT error: {0}")]
    Ort(#[from] ort::Error),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("ndarray shape error: {0}")]
    Shape(#[from] ndarray::ShapeError),
    #[error("model input not found: {0}")]
    InputNotFound(String),
    #[error("model output not found: {0}")]
    OutputNotFound(String),
    #[error("failed to get tensor shape for input: {0}")]
    TensorShape(String),
}

pub struct ParakeetModel {
    encoder: Session,
    decoder_joint: Session,
    preprocessor: Session,
    vocab: Vec<String>,
    blank_idx: i32,
    vocab_size: usize,
}

/// Intra-op threads for the big sessions: physical cores, capped at 4.
pub fn intra_threads() -> usize {
    num_cpus::get_physical().clamp(1, 4)
}

impl ParakeetModel {
    pub fn new(model_dir: &Path) -> Result<Self, ParakeetError> {
        let threads = intra_threads();
        let encoder = Self::init_session(&model_dir.join(MODEL_FILES[0]), threads)?;
        let decoder_joint = Self::init_session(&model_dir.join(MODEL_FILES[1]), 1)?;
        let preprocessor = Self::init_session(&model_dir.join(MODEL_FILES[2]), threads)?;
        let (vocab, blank_idx) = Self::load_vocab(&model_dir.join(MODEL_FILES[3]))?;
        let vocab_size = vocab.len();
        log::info!(
            "loaded Parakeet: {vocab_size} tokens, blank_idx={blank_idx}, intra_threads={threads}"
        );
        Ok(Self {
            encoder,
            decoder_joint,
            preprocessor,
            vocab,
            blank_idx,
            vocab_size,
        })
    }

    fn init_session(path: &Path, intra_threads: usize) -> Result<Session, ParakeetError> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([CPUExecutionProvider::default().build()])?
            .with_parallel_execution(false)?
            .with_intra_threads(intra_threads)?
            .with_inter_threads(1)?
            .commit_from_file(path)?;
        Ok(session)
    }

    fn load_vocab(path: &Path) -> Result<(Vec<String>, i32), ParakeetError> {
        let content = std::fs::read_to_string(path)?;
        let mut max_id = 0;
        let mut tokens_with_ids: Vec<(String, usize)> = Vec::new();
        let mut blank_idx: Option<usize> = None;
        for line in content.lines() {
            let parts: Vec<&str> = line.trim_end().split(' ').collect();
            if parts.len() >= 2 {
                if let Ok(id) = parts[1].parse::<usize>() {
                    if parts[0] == "<blk>" {
                        blank_idx = Some(id);
                    }
                    tokens_with_ids.push((parts[0].to_string(), id));
                    max_id = max_id.max(id);
                }
            }
        }
        let mut vocab = vec![String::new(); max_id + 1];
        for (token, id) in tokens_with_ids {
            vocab[id] = token.replace('\u{2581}', " ");
        }
        let blank_idx = blank_idx.ok_or_else(|| {
            ParakeetError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "missing <blk> token in vocabulary",
            ))
        })? as i32;
        Ok((vocab, blank_idx))
    }

    fn preprocess(
        &mut self,
        waveforms: &ArrayViewD<f32>,
        waveforms_lens: &ArrayViewD<i64>,
    ) -> Result<(ArrayD<f32>, ArrayD<i64>), ParakeetError> {
        let outputs = self.preprocessor.run(inputs![
            "waveforms" => TensorRef::from_array_view(waveforms.view())?,
            "waveforms_lens" => TensorRef::from_array_view(waveforms_lens.view())?,
        ])?;
        let features = outputs
            .get("features")
            .ok_or_else(|| ParakeetError::OutputNotFound("features".into()))?
            .try_extract_array()?;
        let features_lens = outputs
            .get("features_lens")
            .ok_or_else(|| ParakeetError::OutputNotFound("features_lens".into()))?
            .try_extract_array()?;
        Ok((features.to_owned(), features_lens.to_owned()))
    }

    fn encode(
        &mut self,
        audio_signal: &ArrayViewD<f32>,
        length: &ArrayViewD<i64>,
    ) -> Result<(ArrayD<f32>, ArrayD<i64>), ParakeetError> {
        let outputs = self.encoder.run(inputs![
            "audio_signal" => TensorRef::from_array_view(audio_signal.view())?,
            "length" => TensorRef::from_array_view(length.view())?,
        ])?;
        let encoder_output = outputs
            .get("outputs")
            .ok_or_else(|| ParakeetError::OutputNotFound("outputs".into()))?
            .try_extract_array()?;
        let encoded_lengths = outputs
            .get("encoded_lengths")
            .ok_or_else(|| ParakeetError::OutputNotFound("encoded_lengths".into()))?
            .try_extract_array()?;
        let encoder_output = encoder_output.permuted_axes(IxDyn(&[0, 2, 1]));
        Ok((encoder_output.to_owned(), encoded_lengths.to_owned()))
    }

    fn create_decoder_state(&self) -> Result<DecoderState, ParakeetError> {
        let shape_of = |name: &str| -> Result<Vec<i64>, ParakeetError> {
            let input = self
                .decoder_joint
                .inputs
                .iter()
                .find(|input| input.name == name)
                .ok_or_else(|| ParakeetError::InputNotFound(name.into()))?;
            input
                .input_type
                .tensor_shape()
                .map(|s| s.iter().copied().collect())
                .ok_or_else(|| ParakeetError::TensorShape(name.into()))
        };
        // Shapes are [2, -1, 640]; use batch size 1.
        let s1 = shape_of("input_states_1")?;
        let s2 = shape_of("input_states_2")?;
        Ok((
            Array::zeros((s1[0] as usize, 1, s1[2] as usize)),
            Array::zeros((s2[0] as usize, 1, s2[2] as usize)),
        ))
    }

    fn decode_step(
        &mut self,
        prev_tokens: &[i32],
        prev_state: &DecoderState,
        encoder_out: &ArrayViewD<f32>,
    ) -> Result<(ArrayD<f32>, DecoderState), ParakeetError> {
        let target_token = prev_tokens.last().copied().unwrap_or(self.blank_idx);
        let encoder_outputs = encoder_out
            .to_owned()
            .insert_axis(ndarray::Axis(0))
            .insert_axis(ndarray::Axis(2));
        let targets = Array2::from_shape_vec((1, 1), vec![target_token])?;
        let target_length = Array1::from_vec(vec![1]);
        let outputs = self.decoder_joint.run(inputs![
            "encoder_outputs" => TensorRef::from_array_view(encoder_outputs.view())?,
            "targets" => TensorRef::from_array_view(targets.view())?,
            "target_length" => TensorRef::from_array_view(target_length.view())?,
            "input_states_1" => TensorRef::from_array_view(prev_state.0.view())?,
            "input_states_2" => TensorRef::from_array_view(prev_state.1.view())?,
        ])?;
        let logits = outputs
            .get("outputs")
            .ok_or_else(|| ParakeetError::OutputNotFound("outputs".into()))?
            .try_extract_array()?;
        let state1 = outputs
            .get("output_states_1")
            .ok_or_else(|| ParakeetError::OutputNotFound("output_states_1".into()))?
            .try_extract_array()?;
        let state2 = outputs
            .get("output_states_2")
            .ok_or_else(|| ParakeetError::OutputNotFound("output_states_2".into()))?
            .try_extract_array()?;
        let logits = logits.remove_axis(ndarray::Axis(0));
        let state1 = state1.to_owned().into_dimensionality::<ndarray::Ix3>()?;
        let state2 = state2.to_owned().into_dimensionality::<ndarray::Ix3>()?;
        Ok((logits.to_owned(), (state1, state2)))
    }

    fn decode_sequence(
        &mut self,
        encodings: &ArrayViewD<f32>,
        encodings_len: usize,
    ) -> Result<Vec<i32>, ParakeetError> {
        let mut prev_state = self.create_decoder_state()?;
        let mut tokens = Vec::new();
        let mut t = 0;
        let mut emitted_tokens = 0;

        while t < encodings_len {
            let encoder_step = encodings.slice(ndarray::s![t, ..]).to_owned().into_dyn();
            let (probs, new_state) =
                self.decode_step(&tokens, &prev_state, &encoder_step.view())?;
            let logits = probs.as_slice().ok_or_else(|| {
                ParakeetError::Shape(ndarray::ShapeError::from_kind(
                    ndarray::ErrorKind::IncompatibleShape,
                ))
            })?;
            // TDT output: [vocab logits | duration logits].
            let (vocab_logits, duration_logits) = if logits.len() > self.vocab_size {
                let (v, d) = logits.split_at(self.vocab_size);
                (v, Some(d))
            } else {
                (logits, None)
            };
            let token = argmax(vocab_logits)
                .map(|i| i as i32)
                .unwrap_or(self.blank_idx);

            if token != self.blank_idx {
                prev_state = new_state;
                tokens.push(token);
                emitted_tokens += 1;
            }

            if let Some(duration_logits) = duration_logits {
                let mut skip = argmax(duration_logits)
                    .and_then(|i| TDT_DURATIONS.get(i).copied())
                    .unwrap_or(1);
                // Guarantee progress on blank-with-zero-duration and cap same-frame emissions.
                if skip == 0 && (token == self.blank_idx || emitted_tokens >= MAX_TOKENS_PER_STEP) {
                    skip = 1;
                }
                if skip > 0 {
                    t += skip;
                    emitted_tokens = 0;
                }
            } else if token == self.blank_idx || emitted_tokens >= MAX_TOKENS_PER_STEP {
                t += 1;
                emitted_tokens = 0;
            }
        }
        Ok(tokens)
    }

    fn decode_tokens(&self, ids: &[i32]) -> String {
        let joined: String = ids
            .iter()
            .filter_map(|&id| self.vocab.get(id as usize).map(String::as_str))
            .collect();
        DECODE_SPACE_RE
            .replace_all(
                &joined,
                |caps: &regex::Captures| if caps.get(1).is_some() { " " } else { "" },
            )
            .trim()
            .to_string()
    }

    /// Transcribes 16 kHz mono samples.
    pub fn transcribe(&mut self, mut samples: Vec<f32>) -> Result<String, ParakeetError> {
        if samples.len() < MIN_SAMPLES {
            samples.resize(MIN_SAMPLES, 0.0);
        }
        let len = samples.len();
        let waveforms = Array2::from_shape_vec((1, len), samples)?.into_dyn();
        let waveforms_lens = Array1::from_vec(vec![len as i64]).into_dyn();
        let (features, features_lens) =
            self.preprocess(&waveforms.view(), &waveforms_lens.view())?;
        let (encoder_out, encoder_out_lens) =
            self.encode(&features.view(), &features_lens.view())?;
        let (Some(encodings), Some(&enc_len)) = (
            encoder_out.outer_iter().next(),
            encoder_out_lens.iter().next(),
        ) else {
            return Ok(String::new());
        };
        let tokens = self.decode_sequence(&encodings, enc_len as usize)?;
        Ok(self.decode_tokens(&tokens))
    }
}

fn argmax(values: &[f32]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
}

#[derive(Debug, Clone)]
pub struct Transcript {
    pub text: String,
    pub audio_ms: u64,
    pub transcribe_ms: u64,
}

impl Transcript {
    pub fn to_json(&self) -> serde_json::Value {
        json!({ "text": self.text, "audioMs": self.audio_ms, "transcribeMs": self.transcribe_ms })
    }
}

#[derive(Default)]
struct Status {
    model_dir: Option<String>,
    load_ms: u64,
}

type Reply<T> = Box<dyn FnOnce(ProtoResult<T>) + Send>;

enum Job {
    Load {
        dir: String,
        reply: Reply<u64>,
    },
    Transcribe {
        samples: Vec<f32>,
        reply: Reply<Transcript>,
    },
    /// Runs after every previously queued job (used as an ordering barrier).
    Run(Box<dyn FnOnce() + Send>),
}

/// Handle to the engine thread. Cheap to clone.
#[derive(Clone)]
pub struct Engine {
    tx: mpsc::Sender<Job>,
    status: Arc<Mutex<Status>>,
}

impl Engine {
    pub fn spawn(out: Out) -> Engine {
        let (tx, rx) = mpsc::channel::<Job>();
        let status = Arc::new(Mutex::new(Status::default()));
        let thread_status = status.clone();
        std::thread::Builder::new()
            .name("engine".into())
            .spawn(move || engine_thread(rx, thread_status, out))
            .expect("spawn engine thread");
        Engine { tx, status }
    }

    pub fn model_dir(&self) -> Option<String> {
        self.status.lock().unwrap().model_dir.clone()
    }

    pub fn is_loaded(&self) -> bool {
        self.status.lock().unwrap().model_dir.is_some()
    }

    pub fn load(&self, dir: String, reply: impl FnOnce(ProtoResult<u64>) + Send + 'static) {
        self.send(Job::Load {
            dir,
            reply: Box::new(reply),
        });
    }

    pub fn transcribe(
        &self,
        samples: Vec<f32>,
        reply: impl FnOnce(ProtoResult<Transcript>) + Send + 'static,
    ) {
        self.send(Job::Transcribe {
            samples,
            reply: Box::new(reply),
        });
    }

    pub fn after_pending(&self, f: impl FnOnce() + Send + 'static) {
        self.send(Job::Run(Box::new(f)));
    }

    fn send(&self, job: Job) {
        if self.tx.send(job).is_err() {
            log::error!("engine thread is gone");
        }
    }
}

fn engine_thread(rx: mpsc::Receiver<Job>, status: Arc<Mutex<Status>>, out: Out) {
    let mut model: Option<(PathBuf, ParakeetModel)> = None;
    // Blocks in recv: no work, no CPU.
    while let Ok(job) = rx.recv() {
        match job {
            Job::Load { dir, reply } => {
                let path = PathBuf::from(&dir);
                if model.as_ref().is_some_and(|(p, _)| *p == path) {
                    reply(Ok(status.lock().unwrap().load_ms));
                    continue;
                }
                if let Some(missing) = MODEL_FILES.iter().find(|f| !path.join(f).is_file()) {
                    reply(Err(ProtoError::new(
                        ErrorCode::ModelLoadFailed,
                        format!("missing {} in {dir}", missing),
                    )));
                    continue;
                }
                let started = Instant::now();
                // Drop the old model first so two models never share memory.
                model = None;
                *status.lock().unwrap() = Status::default();
                match ParakeetModel::new(&path) {
                    Ok(m) => {
                        let load_ms = started.elapsed().as_millis() as u64;
                        model = Some((path, m));
                        *status.lock().unwrap() = Status {
                            model_dir: Some(dir.clone()),
                            load_ms,
                        };
                        log::info!("model loaded from {dir} in {load_ms} ms");
                        let _ = out.send(event(
                            "model.loaded",
                            json!({ "modelDir": dir, "loadMs": load_ms }),
                        ));
                        reply(Ok(load_ms));
                    }
                    Err(e) => {
                        log::error!("model load failed: {e}");
                        reply(Err(ProtoError::new(
                            ErrorCode::ModelLoadFailed,
                            e.to_string(),
                        )));
                    }
                }
            }
            Job::Transcribe { samples, reply } => {
                let Some((_, m)) = model.as_mut() else {
                    reply(Err(ProtoError::new(
                        ErrorCode::ModelNotLoaded,
                        "call model.load first",
                    )));
                    continue;
                };
                let audio_ms = samples_to_ms(samples.len());
                let started = Instant::now();
                let result = m.transcribe(samples);
                let transcribe_ms = started.elapsed().as_millis() as u64;
                reply(match result {
                    Ok(text) => {
                        log::debug!("transcribed {audio_ms} ms in {transcribe_ms} ms");
                        Ok(Transcript {
                            text,
                            audio_ms,
                            transcribe_ms,
                        })
                    }
                    Err(e) => Err(ProtoError::new(
                        ErrorCode::TranscriptionFailed,
                        e.to_string(),
                    )),
                });
            }
            Job::Run(f) => f(),
        }
    }
}
