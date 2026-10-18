//! Decode-loop settings.

use super::super::SamplingParams;

/// Settings for one generation call.
#[derive(Debug, Clone)]
pub struct GenerationOptions {
    /// Maximum number of tokens to generate.
    pub max_new_tokens: usize,
    /// Sampling settings.
    pub sampling: SamplingParams,
    /// Generation stops after emitting any of these ids.
    pub stop_tokens: Vec<u32>,
}
