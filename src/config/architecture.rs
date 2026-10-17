//! The model families Hale-Token can execute.

use crate::error::{HaleTokenError, Result};
use serde::{Deserialize, Serialize};

/// The model families Hale-Token can execute.
///
/// Both are "classic" sparse MoE decoders: GQA attention followed by a
/// router that picks the top-k SwiGLU experts per token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Architecture {
    /// Qwen3-MoE (e.g. Qwen3-30B-A3B, Qwen3-235B-A22B). Adds per-head
    /// RMSNorm on queries and keys ("QK-norm").
    Qwen3Moe,
    /// Mixtral (e.g. Mixtral-8x7B, Mixtral-8x22B).
    Mixtral,
}

impl Architecture {
    /// Maps the `architectures[0]` string from `config.json` to a family.
    pub fn from_hf_name(name: &str) -> Result<Self> {
        match name {
            "Qwen3MoeForCausalLM" => Ok(Architecture::Qwen3Moe),
            "MixtralForCausalLM" => Ok(Architecture::Mixtral),
            other => Err(HaleTokenError::UnsupportedArchitecture(other.to_string())),
        }
    }

    /// Whether attention applies RMSNorm to each query/key head.
    pub fn uses_qk_norm(self) -> bool {
        matches!(self, Architecture::Qwen3Moe)
    }
}
