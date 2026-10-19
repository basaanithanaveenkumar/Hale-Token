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

/// Registry of supported Hugging Face `architectures[0]` names.
///
/// Supporting another family with the same maths means one row here plus one
/// [`TensorNaming`](crate::model::arch::TensorNaming) impl. The tensor-naming
/// factory stays an exhaustive `match` on [`Architecture`] on purpose: the
/// compiler then rejects a variant without a naming scheme, which a runtime
/// registry could not.
pub const HF_ARCHITECTURES: &[(&str, Architecture)] = &[
    ("Qwen3MoeForCausalLM", Architecture::Qwen3Moe),
    ("MixtralForCausalLM", Architecture::Mixtral),
];

impl Architecture {
    /// Maps the `architectures[0]` string from `config.json` to a family.
    ///
    /// Linear scan of [`HF_ARCHITECTURES`] (a handful of entries).
    ///
    /// # Errors
    /// [`HaleTokenError::UnsupportedArchitecture`] for a name not in the registry.
    pub fn from_hf_name(name: &str) -> Result<Self> {
        HF_ARCHITECTURES
            .iter()
            .find(|(hf, _)| *hf == name)
            .map(|(_, arch)| *arch)
            .ok_or_else(|| HaleTokenError::UnsupportedArchitecture(name.to_string()))
    }

    /// The Hugging Face name of this family (the inverse of [`Self::from_hf_name`]).
    pub fn hf_name(self) -> &'static str {
        HF_ARCHITECTURES
            .iter()
            .find(|(_, arch)| *arch == self)
            .map_or("unknown", |(hf, _)| hf)
    }

    /// Whether attention applies RMSNorm to each query/key head.
    pub fn uses_qk_norm(self) -> bool {
        matches!(self, Architecture::Qwen3Moe)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_round_trips_every_family() {
        for (hf, arch) in HF_ARCHITECTURES {
            assert_eq!(Architecture::from_hf_name(hf).unwrap(), *arch);
            assert_eq!(arch.hf_name(), *hf);
        }
    }

    #[test]
    fn unknown_name_is_rejected_and_the_message_lists_supported_families() {
        let err = Architecture::from_hf_name("LlamaForCausalLM").unwrap_err();
        let message = err.to_string();
        assert!(message.contains("LlamaForCausalLM"));
        // The error text is static; keep it in sync with the registry.
        for (hf, _) in HF_ARCHITECTURES {
            assert!(message.contains(hf), "{message} does not mention {hf}");
        }
    }

    #[test]
    fn only_qwen3_uses_qk_norm() {
        assert!(Architecture::Qwen3Moe.uses_qk_norm());
        assert!(!Architecture::Mixtral.uses_qk_norm());
    }
}
