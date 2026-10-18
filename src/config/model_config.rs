//! Family-independent model hyper-parameters.

use super::{Architecture, HfConfig, OneOrMany};
use crate::error::{HaleTokenError, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Family-independent model hyper-parameters used by the engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model family, which selects the tensor naming scheme.
    pub architecture: Architecture,
    /// Width of the residual stream (`d_model`).
    pub hidden_size: usize,
    /// Hidden width of one routed expert's SwiGLU MLP.
    pub expert_intermediate_size: usize,
    /// Hidden width of a dense (non-MoE) MLP layer, if the model has any.
    pub dense_intermediate_size: usize,
    /// Number of transformer decoder layers.
    pub num_layers: usize,
    /// Number of attention (query) heads.
    pub num_heads: usize,
    /// Number of key/value heads (grouped-query attention).
    pub num_kv_heads: usize,
    /// Dimension of each attention head.
    pub head_dim: usize,
    /// Size of the token vocabulary.
    pub vocab_size: usize,
    /// Epsilon used by RMSNorm.
    pub rms_norm_eps: f32,
    /// Base frequency of the rotary position embedding.
    pub rope_theta: f32,
    /// Maximum context length the model was trained for.
    pub max_position_embeddings: usize,
    /// Routed experts per MoE layer.
    pub num_experts: usize,
    /// Experts activated per token (the "k" in top-k).
    pub experts_per_token: usize,
    /// Re-normalise the top-k router probabilities so they sum to one.
    pub normalize_topk: bool,
    /// Whether the LM head shares weights with the token embeddings.
    pub tie_word_embeddings: bool,
    /// Indices of layers that use a dense MLP instead of MoE.
    pub dense_layers: Vec<usize>,
    /// Beginning-of-sequence token id, if the model defines one.
    pub bos_token_id: Option<u32>,
    /// Token ids that terminate generation.
    pub eos_token_ids: Vec<u32>,
}

impl ModelConfig {
    /// Reads and normalises `<dir>/config.json`.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path).map_err(|e| HaleTokenError::io(&path, e))?;
        Self::from_json_str(&text).map_err(|e| match e {
            HaleTokenError::InvalidArgument(msg) => {
                HaleTokenError::InvalidArgument(format!("{}: {msg}", path.display()))
            }
            other => other,
        })
    }

    /// Parses a `config.json` document held in memory.
    ///
    /// # Errors
    ///
    /// [`HaleTokenError::InvalidArgument`] for malformed or inconsistent
    /// documents and [`HaleTokenError::UnsupportedArchitecture`] for unknown
    /// model families.
    ///
    /// # Examples
    ///
    /// ```
    /// use hale::{Architecture, ModelConfig};
    ///
    /// let config = ModelConfig::from_json_str(r#"{
    ///     "architectures": ["MixtralForCausalLM"], "hidden_size": 64,
    ///     "intermediate_size": 32, "num_hidden_layers": 2,
    ///     "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 16,
    ///     "vocab_size": 272, "rms_norm_eps": 1e-5, "rope_theta": 1000000.0,
    ///     "max_position_embeddings": 512, "num_local_experts": 4,
    ///     "num_experts_per_tok": 2, "tie_word_embeddings": false,
    ///     "eos_token_id": 2
    /// }"#).unwrap();
    /// assert_eq!(config.architecture, Architecture::Mixtral);
    /// assert_eq!(config.num_experts, 4);
    /// ```
    pub fn from_json_str(text: &str) -> Result<Self> {
        let hf: HfConfig = serde_json::from_str(text)
            .map_err(|e| HaleTokenError::InvalidArgument(format!("config.json: {e}")))?;
        Self::from_hf(hf)
    }

    fn from_hf(hf: HfConfig) -> Result<Self> {
        let arch_name = hf.architectures.first().ok_or_else(|| {
            HaleTokenError::InvalidArgument("config.json has no architectures".into())
        })?;
        let architecture = Architecture::from_hf_name(arch_name)?;

        let num_experts = hf.num_experts.or(hf.num_local_experts).ok_or_else(|| {
            HaleTokenError::InvalidArgument("config.json has no expert count".into())
        })?;
        let dense_intermediate_size = hf.intermediate_size.unwrap_or(0);
        // Mixtral experts use `intermediate_size`; Qwen3 has a dedicated field.
        let expert_intermediate_size = hf.moe_intermediate_size.unwrap_or(dense_intermediate_size);

        let dense_layers = match architecture {
            Architecture::Mixtral => Vec::new(),
            Architecture::Qwen3Moe => {
                let step = hf.decoder_sparse_step.unwrap_or(1).max(1);
                (0..hf.num_hidden_layers)
                    .filter(|i| hf.mlp_only_layers.contains(i) || (i + 1) % step != 0)
                    .collect()
            }
        };

        let eos_token_ids = match hf.eos_token_id {
            Some(OneOrMany::One(id)) => vec![id],
            Some(OneOrMany::Many(ids)) => ids,
            None => Vec::new(),
        };

        let config = ModelConfig {
            architecture,
            hidden_size: hf.hidden_size,
            expert_intermediate_size,
            dense_intermediate_size,
            num_layers: hf.num_hidden_layers,
            num_heads: hf.num_attention_heads,
            num_kv_heads: hf.num_key_value_heads.unwrap_or(hf.num_attention_heads),
            head_dim: hf
                .head_dim
                .unwrap_or(hf.hidden_size / hf.num_attention_heads),
            vocab_size: hf.vocab_size,
            rms_norm_eps: hf.rms_norm_eps,
            rope_theta: hf.rope_theta,
            max_position_embeddings: hf.max_position_embeddings,
            num_experts,
            experts_per_token: hf.num_experts_per_tok,
            // Mixtral always renormalises; Qwen3 makes it configurable.
            normalize_topk: match architecture {
                Architecture::Mixtral => true,
                Architecture::Qwen3Moe => hf.norm_topk_prob.unwrap_or(false),
            },
            tie_word_embeddings: hf.tie_word_embeddings,
            dense_layers,
            bos_token_id: hf.bos_token_id,
            eos_token_ids,
        };
        config.validate()?;
        Ok(config)
    }

    /// Rejects configurations the engine cannot run.
    pub fn validate(&self) -> Result<()> {
        let bad = |m: String| Err(HaleTokenError::InvalidArgument(m));
        if self.num_kv_heads == 0 || self.num_heads % self.num_kv_heads != 0 {
            return bad(format!(
                "num_attention_heads ({}) must be a multiple of num_key_value_heads ({})",
                self.num_heads, self.num_kv_heads
            ));
        }
        if self.head_dim % 2 != 0 {
            return bad(format!(
                "head_dim ({}) must be even for RoPE",
                self.head_dim
            ));
        }
        if self.experts_per_token == 0 || self.experts_per_token > self.num_experts {
            return bad(format!(
                "num_experts_per_tok ({}) must be in 1..={}",
                self.experts_per_token, self.num_experts
            ));
        }
        Ok(())
    }

    /// Whether `layer` routes through experts (as opposed to a dense MLP).
    pub fn is_moe_layer(&self, layer: usize) -> bool {
        !self.dense_layers.contains(&layer)
    }

    /// Number of layers that use MoE.
    pub fn num_moe_layers(&self) -> usize {
        self.num_layers - self.dense_layers.len()
    }

    /// Parameters in a single routed expert (gate + up + down projections).
    pub fn params_per_expert(&self) -> usize {
        3 * self.hidden_size * self.expert_intermediate_size
    }

    /// Parameters in everything that is *not* a routed expert: embeddings,
    /// attention, norms, routers, dense MLPs and the LM head.
    pub fn dense_params(&self) -> usize {
        let h = self.hidden_size;
        let q = self.num_heads * self.head_dim;
        let kv = self.num_kv_heads * self.head_dim;
        let attention = h * q + 2 * h * kv + q * h;
        let norms = 2 * h
            + if self.architecture.uses_qk_norm() {
                2 * self.head_dim
            } else {
                0
            };
        let per_layer = attention + norms;
        let routers = self.num_moe_layers() * self.num_experts * h;
        let dense_mlps = self.dense_layers.len() * 3 * h * self.dense_intermediate_size;
        let embeddings = self.vocab_size * h * if self.tie_word_embeddings { 1 } else { 2 };
        self.num_layers * per_layer + routers + dense_mlps + embeddings + h
    }

    /// Total parameter count of the model.
    pub fn total_params(&self) -> usize {
        self.dense_params() + self.num_moe_layers() * self.num_experts * self.params_per_expert()
    }

    /// Parameters touched per generated token (dense part + k experts per layer).
    pub fn active_params(&self) -> usize {
        self.dense_params()
            + self.num_moe_layers() * self.experts_per_token * self.params_per_expert()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mixtral_json(extra: &str) -> String {
        format!(
            r#"{{"architectures":["MixtralForCausalLM"],"hidden_size":64,
            "intermediate_size":32,"num_hidden_layers":2,"num_attention_heads":4,
            "num_key_value_heads":1,"head_dim":16,"vocab_size":272,
            "rms_norm_eps":1e-5,"rope_theta":1000000.0,"max_position_embeddings":512,
            "num_local_experts":4,"num_experts_per_tok":2,"tie_word_embeddings":false
            {extra}}}"#
        )
    }

    #[test]
    fn parses_mixtral_and_normalises_fields() {
        let c = ModelConfig::from_json_str(&mixtral_json(r#","eos_token_id":[2,3]"#)).unwrap();
        assert_eq!(c.expert_intermediate_size, 32);
        assert!(c.normalize_topk);
        assert_eq!(c.eos_token_ids, vec![2, 3]);
        assert!(c.dense_layers.is_empty());
    }

    #[test]
    fn rejects_unknown_architecture() {
        let text = mixtral_json("").replace("MixtralForCausalLM", "LlamaForCausalLM");
        assert!(matches!(
            ModelConfig::from_json_str(&text),
            Err(HaleTokenError::UnsupportedArchitecture(_))
        ));
    }

    #[test]
    fn rejects_inconsistent_heads_and_topk() {
        let heads =
            mixtral_json("").replace("\"num_key_value_heads\":1", "\"num_key_value_heads\":3");
        assert!(ModelConfig::from_json_str(&heads).is_err());
        let topk =
            mixtral_json("").replace("\"num_experts_per_tok\":2", "\"num_experts_per_tok\":9");
        assert!(ModelConfig::from_json_str(&topk).is_err());
    }

    #[test]
    fn rejects_malformed_json() {
        assert!(ModelConfig::from_json_str("{").is_err());
    }
}
