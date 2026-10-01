// Derived from Meetily (MIT), Zackriya Solutions
// Source: frontend/src-tauri/src/audio/vad.rs (Silero thresholds, 30 ms frames,
// pre-speech padding, minimum speech time). Endpointing is reimplemented as a
// pure state machine so segment boundaries are testable without a model.

//! Voice activity detection and meeting segmentation at 16 kHz.

use std::collections::VecDeque;

use crate::audio::TARGET_RATE;

/// 30 ms at 16 kHz, the Silero frame size Meetily uses.
pub const FRAME: usize = 480;

#[derive(Debug, Clone, Copy)]
pub struct SegmenterConfig {
    pub positive_threshold: f32,
    pub negative_threshold: f32,
    /// Trailing silence that ends a segment.
    pub silence_ms: u32,
    /// Hard cap on segment length.
    pub max_segment_ms: u32,
    /// Audio kept before the first speech frame.
    pub pre_pad_ms: u32,
    /// Segments with less speech than this are dropped.
    pub min_speech_ms: u32,
}

impl Default for SegmenterConfig {
    fn default() -> Self {
        Self {
            positive_threshold: 0.50,
            negative_threshold: 0.35,
            silence_ms: 600,
            max_segment_ms: 30_000,
            pre_pad_ms: 300,
            min_speech_ms: 250,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub samples: Vec<f32>,
    /// Milliseconds since the stream started.
    pub start_ms: u64,
    pub end_ms: u64,
}

fn ms_to_samples(ms: u32) -> usize {
    ms as usize * TARGET_RATE as usize / 1000
}

fn samples_to_ms(samples: usize) -> u64 {
    samples as u64 * 1000 / TARGET_RATE as u64
}

/// Turns per-frame speech probabilities into speech segments.
pub struct Segmenter {
    cfg: SegmenterConfig,
    /// Rolling pre-speech audio while silent.
    pre: VecDeque<f32>,
    current: Vec<f32>,
    in_speech: bool,
    start_sample: usize,
    speech_samples: usize,
    silent_samples: usize,
    total_samples: usize,
}

impl Segmenter {
    pub fn new(cfg: SegmenterConfig) -> Self {
        Self {
            cfg,
            pre: VecDeque::with_capacity(ms_to_samples(cfg.pre_pad_ms) + FRAME),
            current: Vec::new(),
            in_speech: false,
            start_sample: 0,
            speech_samples: 0,
            silent_samples: 0,
            total_samples: 0,
        }
    }

    /// Feeds one frame and its speech probability. Returns a segment when one ends.
    pub fn push_frame(&mut self, frame: &[f32], prob: f32) -> Option<Segment> {
        let frame_start = self.total_samples;
        self.total_samples += frame.len();

        if !self.in_speech {
            if prob > self.cfg.positive_threshold {
                self.in_speech = true;
                self.start_sample = frame_start - self.pre.len();
                self.current.clear();
                self.current.extend(self.pre.drain(..));
                self.current.extend_from_slice(frame);
                self.speech_samples = frame.len();
                self.silent_samples = 0;
            } else {
                self.pre.extend(frame.iter().copied());
                let keep = ms_to_samples(self.cfg.pre_pad_ms);
                while self.pre.len() > keep {
                    self.pre.pop_front();
                }
            }
            return None;
        }

        self.current.extend_from_slice(frame);
        if prob < self.cfg.negative_threshold {
            self.silent_samples += frame.len();
        } else {
            self.silent_samples = 0;
            self.speech_samples += frame.len();
        }

        if self.silent_samples >= ms_to_samples(self.cfg.silence_ms) {
            self.in_speech = false;
            return self.take();
        }
        if self.current.len() >= ms_to_samples(self.cfg.max_segment_ms) {
            // Cap reached while still speaking: cut here and keep going in speech.
            let seg = self.take_raw();
            self.start_sample = self.total_samples;
            self.speech_samples = 0;
            self.silent_samples = 0;
            return Some(seg);
        }
        None
    }

    /// Ends the stream, returning the in-progress segment if it holds enough speech.
    pub fn finish(&mut self) -> Option<Segment> {
        if !self.in_speech {
            return None;
        }
        self.in_speech = false;
        self.take()
    }

    fn take(&mut self) -> Option<Segment> {
        let enough = self.speech_samples >= ms_to_samples(self.cfg.min_speech_ms);
        let seg = self.take_raw();
        self.speech_samples = 0;
        self.silent_samples = 0;
        enough.then_some(seg)
    }

    fn take_raw(&mut self) -> Segment {
        let samples = std::mem::take(&mut self.current);
        let start_ms = samples_to_ms(self.start_sample);
        let end_ms = samples_to_ms(self.start_sample + samples.len());
        Segment {
            samples,
            start_ms,
            end_ms,
        }
    }
}

/// Source of per-frame speech probability.
pub trait SpeechProb {
    fn prob(&mut self, frame: &[f32]) -> f32;
}

/// Silero VAD (model embedded by the `silero` crate).
pub struct Silero {
    session: silero::VadSession,
}

impl Silero {
    pub fn new() -> anyhow::Result<Self> {
        let session = silero::VadSession::new(silero::VadConfig::default())?;
        Ok(Self { session })
    }
}

impl SpeechProb for Silero {
    fn prob(&mut self, frame: &[f32]) -> f32 {
        match self.session.forward(frame.to_vec()) {
            Ok(v) => v
                .try_extract_array::<f32>()
                .ok()
                .and_then(|a| a.first().copied())
                .unwrap_or(0.0),
            Err(e) => {
                log::warn!("silero inference failed: {e}");
                0.0
            }
        }
    }
}

/// Frames a 16 kHz stream and runs it through a detector and segmenter.
pub struct Vad<P: SpeechProb> {
    detector: P,
    segmenter: Segmenter,
    pending: Vec<f32>,
}

impl<P: SpeechProb> Vad<P> {
    pub fn new(detector: P, cfg: SegmenterConfig) -> Self {
        Self {
            detector,
            segmenter: Segmenter::new(cfg),
            pending: Vec::with_capacity(FRAME * 2),
        }
    }

    pub fn push(&mut self, samples: &[f32], out: &mut Vec<Segment>) {
        self.pending.extend_from_slice(samples);
        let mut start = 0;
        while self.pending.len() - start >= FRAME {
            let frame = &self.pending[start..start + FRAME];
            let p = self.detector.prob(frame);
            if let Some(seg) = self.segmenter.push_frame(frame, p) {
                out.push(seg);
            }
            start += FRAME;
        }
        self.pending.drain(..start);
    }

    pub fn finish(&mut self) -> Option<Segment> {
        let rest = std::mem::take(&mut self.pending);
        if !rest.is_empty() {
            let p = self.detector.prob(&{
                let mut f = rest.clone();
                f.resize(FRAME, 0.0);
                f
            });
            if let Some(seg) = self.segmenter.push_frame(&rest, p) {
                // A segment that ended on the last partial frame; nothing more can follow.
                return Some(seg);
            }
        }
        self.segmenter.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Energy detector over synthetic audio: tone frames read as speech.
    struct Energy;
    impl SpeechProb for Energy {
        fn prob(&mut self, frame: &[f32]) -> f32 {
            if crate::audio::rms(frame) > 0.05 {
                0.9
            } else {
                0.0
            }
        }
    }

    fn tone(ms: u32) -> Vec<f32> {
        (0..ms_to_samples(ms))
            .map(|i| (i as f32 * 0.2).sin() * 0.5)
            .collect()
    }

    fn silence(ms: u32) -> Vec<f32> {
        vec![0.0; ms_to_samples(ms)]
    }

    fn run(audio: &[f32]) -> Vec<Segment> {
        let mut vad = Vad::new(Energy, SegmenterConfig::default());
        let mut segs = Vec::new();
        // Odd block size to exercise framing.
        for block in audio.chunks(777) {
            vad.push(block, &mut segs);
        }
        segs.extend(vad.finish());
        segs
    }

    #[test]
    fn silence_ends_segment() {
        let audio = [
            silence(990),
            tone(1500),
            silence(900),
            tone(990),
            silence(990),
        ]
        .concat();
        let segs = run(&audio);
        assert_eq!(
            segs.len(),
            2,
            "{:?}",
            segs.iter()
                .map(|s| (s.start_ms, s.end_ms))
                .collect::<Vec<_>>()
        );
        // First: 300 ms pre-pad before speech at 990 ms, ends after 600 ms of silence.
        assert_eq!(segs[0].start_ms, 690);
        assert_eq!(segs[0].end_ms, 990 + 1500 + 600);
        assert_eq!(segs[0].samples.len(), ms_to_samples(300 + 1500 + 600));
        // Second starts 300 ms before the second tone.
        assert_eq!(segs[1].start_ms, 990 + 1500 + 900 - 300);
        assert_eq!(segs[1].end_ms, 990 + 1500 + 900 + 990 + 600);
    }

    #[test]
    fn short_pause_does_not_end_segment() {
        let audio = [tone(990), silence(510), tone(990), silence(990)].concat();
        let segs = run(&audio);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].start_ms, 0);
        assert_eq!(segs[0].end_ms, 990 + 510 + 990 + 600);
    }

    #[test]
    fn segments_are_capped_at_30_seconds() {
        let audio = [tone(70_020), silence(990)].concat();
        let segs = run(&audio);
        assert_eq!(segs.len(), 3);
        assert_eq!((segs[0].start_ms, segs[0].end_ms), (0, 30_000));
        assert_eq!((segs[1].start_ms, segs[1].end_ms), (30_000, 60_000));
        assert_eq!(segs[2].start_ms, 60_000);
        assert_eq!(segs[2].end_ms, 70_020 + 600);
        assert!(segs
            .iter()
            .all(|s| s.samples.len() <= ms_to_samples(30_000)));
        // Contiguous: no audio lost at the cut.
        assert_eq!(segs[0].end_ms, segs[1].start_ms);
    }

    #[test]
    fn blips_shorter_than_min_speech_are_dropped() {
        let audio = [silence(600), tone(90), silence(1500)].concat();
        assert!(run(&audio).is_empty());
    }

    #[test]
    fn finish_flushes_open_segment() {
        let audio = [silence(300), tone(1200)].concat();
        let segs = run(&audio);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].start_ms, 0);
        assert_eq!(segs[0].end_ms, 1500);
    }

    #[test]
    fn pure_silence_yields_nothing() {
        assert!(run(&silence(5000)).is_empty());
    }
}
