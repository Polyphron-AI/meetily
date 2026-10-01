//! Sample helpers shared by `transcribe` and live capture: PCM16 decoding,
//! resampling to 16 kHz mono and RMS levels.

use base64::Engine as _;
use rubato::{FftFixedIn, Resampler};

pub const TARGET_RATE: u32 = 16_000;
const CHUNK: usize = 1024;

/// Decodes base64 little-endian signed 16-bit mono PCM into f32 samples in [-1, 1).
pub fn decode_pcm16_base64(b64: &str) -> Result<Vec<f32>, String> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .map_err(|e| format!("pcm16 is not valid base64: {e}"))?;
    if bytes.len() % 2 != 0 {
        return Err(format!(
            "pcm16 byte length {} is not a multiple of 2",
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
        .collect())
}

/// Number of 16 kHz samples that correspond to `len` samples at `rate`.
pub fn resampled_len(len: usize, rate: u32) -> usize {
    ((len as u64 * TARGET_RATE as u64 + rate as u64 / 2) / rate as u64) as usize
}

pub fn samples_to_ms(samples: usize) -> u64 {
    samples as u64 * 1000 / TARGET_RATE as u64
}

/// One-shot resample of a whole buffer to 16 kHz. The output is aligned (the
/// filter delay is removed) and has exactly `resampled_len(input.len(), rate)` samples.
pub fn resample_to_16k(input: &[f32], rate: u32) -> Result<Vec<f32>, String> {
    if rate == TARGET_RATE || input.is_empty() {
        return Ok(input.to_vec());
    }
    let expected = resampled_len(input.len(), rate);
    let mut stream = StreamResampler::new(rate)?;
    let mut out = Vec::with_capacity(expected + CHUNK);
    stream.push(input, &mut out)?;
    stream.flush(&mut out)?;
    let delay = stream.delay();
    // Flush emits at least one chunk of padding; keep feeding silence until the
    // delayed tail is fully out.
    while out.len() < delay + expected {
        stream.push(&vec![0.0; CHUNK], &mut out)?;
    }
    out.drain(..delay);
    out.truncate(expected);
    Ok(out)
}

/// Streaming resampler for live capture. Accepts arbitrary block sizes.
pub struct StreamResampler {
    inner: Option<FftFixedIn<f32>>,
    pending: Vec<f32>,
}

impl StreamResampler {
    pub fn new(rate: u32) -> Result<Self, String> {
        let inner = if rate == TARGET_RATE {
            None
        } else {
            Some(
                FftFixedIn::<f32>::new(rate as usize, TARGET_RATE as usize, CHUNK, 2, 1)
                    .map_err(|e| format!("resampler init failed for {rate} Hz: {e}"))?,
            )
        };
        Ok(Self {
            inner,
            pending: Vec::with_capacity(CHUNK * 2),
        })
    }

    pub fn delay(&self) -> usize {
        self.inner.as_ref().map_or(0, |r| r.output_delay())
    }

    pub fn push(&mut self, input: &[f32], out: &mut Vec<f32>) -> Result<(), String> {
        let Some(inner) = self.inner.as_mut() else {
            out.extend_from_slice(input);
            return Ok(());
        };
        self.pending.extend_from_slice(input);
        let mut start = 0;
        while self.pending.len() - start >= inner.input_frames_next() {
            let n = inner.input_frames_next();
            let res = inner
                .process(&[&self.pending[start..start + n]], None)
                .map_err(|e| format!("resample failed: {e}"))?;
            out.extend_from_slice(&res[0]);
            start += n;
        }
        self.pending.drain(..start);
        Ok(())
    }

    /// Pushes out the remaining partial block (zero padded).
    pub fn flush(&mut self, out: &mut Vec<f32>) -> Result<(), String> {
        let Some(inner) = self.inner.as_mut() else {
            return Ok(());
        };
        let res = inner
            .process_partial(Some(&[&self.pending[..]]), None)
            .map_err(|e| format!("resample failed: {e}"))?;
        out.extend_from_slice(&res[0]);
        self.pending.clear();
        Ok(())
    }
}

pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f32 = samples.iter().map(|s| s * s).sum();
    (sum / samples.len() as f32).sqrt().clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b64(samples: &[i16]) -> String {
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }

    #[test]
    fn decodes_pcm16_little_endian() {
        let decoded = decode_pcm16_base64(&b64(&[0, 16384, -32768, 32767])).unwrap();
        assert_eq!(decoded.len(), 4);
        assert_eq!(decoded[0], 0.0);
        assert_eq!(decoded[1], 0.5);
        assert_eq!(decoded[2], -1.0);
        assert!((decoded[3] - 0.99997).abs() < 1e-4);
    }

    #[test]
    fn rejects_bad_pcm16() {
        assert!(decode_pcm16_base64("!!!").is_err());
        // three bytes: odd length
        assert!(decode_pcm16_base64("AAAA").is_err());
        assert_eq!(decode_pcm16_base64("").unwrap().len(), 0);
    }

    #[test]
    fn resample_produces_expected_length() {
        for &(rate, len) in &[
            (48_000u32, 48_000usize),
            (44_100, 44_100 * 3 + 17),
            (22_050, 1234),
            (8_000, 8_000),
            (16_000, 999),
        ] {
            let input: Vec<f32> = (0..len).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
            let out = resample_to_16k(&input, rate).unwrap();
            assert_eq!(out.len(), resampled_len(len, rate), "rate {rate} len {len}");
        }
        assert_eq!(resampled_len(48_000, 48_000), 16_000);
        assert_eq!(resampled_len(44_100, 44_100), 16_000);
    }

    #[test]
    fn resample_preserves_tone_alignment() {
        // A 440 Hz tone at 48 kHz should still be a 440 Hz tone at 16 kHz, aligned to
        // within one output sample (one sample of phase is ~0.17 here).
        let input: Vec<f32> = (0..48_000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin())
            .collect();
        let out = resample_to_16k(&input, 48_000).unwrap();
        for i in (2000..14_000).step_by(997) {
            let expected = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin();
            assert!(
                (out[i] - expected).abs() < 0.1,
                "sample {i}: {} vs {expected}",
                out[i]
            );
        }
    }

    #[test]
    fn streaming_resample_handles_odd_blocks() {
        let mut r = StreamResampler::new(48_000).unwrap();
        let mut out = Vec::new();
        for _ in 0..100 {
            r.push(&[0.1; 333], &mut out).unwrap();
        }
        r.flush(&mut out).unwrap();
        let expected = resampled_len(33_300, 48_000);
        assert!(out.len() >= expected, "{} < {expected}", out.len());
    }

    #[test]
    fn rms_of_full_scale_square_is_one() {
        assert_eq!(rms(&[]), 0.0);
        assert!((rms(&[1.0, -1.0, 1.0, -1.0]) - 1.0).abs() < 1e-6);
    }
}
