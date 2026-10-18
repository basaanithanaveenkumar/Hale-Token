//! Generated text plus run statistics.

use crate::generate::GenerationStats;

/// Result of [`Engine::generate_text`](super::Engine::generate_text).
#[derive(Debug, Clone)]
pub struct TextOutput {
    /// Decoded text of the generated tokens.
    pub text: String,
    /// Generated token ids (excluding the prompt).
    pub tokens: Vec<u32>,
    /// Timing and cache statistics of the run.
    pub stats: GenerationStats,
}
