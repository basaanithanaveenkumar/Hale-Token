//! Timing and cache counters of one generation.

use crate::expert::CacheStats;
use serde::Serialize;

/// Timing and cache figures for one generation call.
#[derive(Debug, Clone, Serialize)]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
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
