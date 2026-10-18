//! Timing and cache counters of one generation.

use crate::expert::CacheStats;
use serde::Serialize;

/// Timing and cache figures for one generation call.
#[derive(Debug, Clone, Serialize)]
pub struct GenerationStats {
    /// Prompt length in tokens.
    pub prompt_tokens: usize,
    /// Number of tokens generated.
    pub generated_tokens: usize,
    /// Time spent on the prompt (prefill) in seconds.
    pub prefill_seconds: f64,
    /// Time spent generating (decode) in seconds.
    pub decode_seconds: f64,
    /// Expert cache statistics at the end of the run.
    pub cache: CacheStats,
}

impl GenerationStats {
    /// Prompt processing speed.
    pub fn prefill_tokens_per_second(&self) -> f64 {
        rate(self.prompt_tokens, self.prefill_seconds)
    }

    /// Generation speed (excluding the first token, which comes from prefill).
    pub fn decode_tokens_per_second(&self) -> f64 {
        rate(self.generated_tokens.saturating_sub(1), self.decode_seconds)
    }
}

fn rate(tokens: usize, seconds: f64) -> f64 {
    if seconds > 0.0 {
        tokens as f64 / seconds
    } else {
        0.0
    }
}
