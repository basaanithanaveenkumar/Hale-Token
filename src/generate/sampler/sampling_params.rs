//! Temperature / top-k / top-p / repetition settings.

use super::{Greedy, Sampler, TopPSampler};

/// User-facing sampling settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
    /// 0 means greedy; higher values flatten the distribution.
    pub temperature: f32,
    /// Nucleus sampling: keep the smallest set of tokens whose probability
    /// mass reaches `top_p` (1.0 disables it).
    pub top_p: f32,
    /// Keep only the `top_k` most likely tokens (0 disables it).
    pub top_k: usize,
    /// Random seed, so runs are reproducible.
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            seed: 42,
        }
    }
}

impl SamplingParams {
    /// Greedy decoding.
    pub fn greedy() -> Self {
        SamplingParams {
            temperature: 0.0,
            ..Self::default()
        }
    }

    /// Builds the matching sampler (Factory).
    pub fn build(&self) -> Box<dyn Sampler> {
        if self.temperature <= 0.0 {
            Box::new(Greedy)
        } else {
            Box::new(TopPSampler::new(*self))
        }
    }
}
