//! Greedy (argmax) decoding.

use super::Sampler;

/// Always takes the most likely token. Deterministic; ideal for tests.
pub struct Greedy;

impl Sampler for Greedy {
    fn sample(&mut self, logits: &[f32]) -> u32 {
        crate::ops::argmax(logits) as u32
    }
}
