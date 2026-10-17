//! Generated text plus run statistics.

use crate::generate::GenerationStats;

/// Result of [`Engine::generate_text`].
#[derive(Debug, Clone)]
pub struct TextOutput {
    pub text: String,
    pub tokens: Vec<u32>,
    pub stats: GenerationStats,
}
