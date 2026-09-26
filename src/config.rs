//! Model hyper-parameters.
//!
//! Hugging Face checkpoints ship a `config.json`. Different model families
//! spell the same idea differently (`num_experts` vs `num_local_experts`), so
//! we parse the raw file into [`HfConfig`] and then normalise it into one
//! family-independent [`ModelConfig`] that the rest of the engine uses.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{HaleError, Result};

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
            other => Err(HaleError::UnsupportedArchitecture(other.to_string())),
        }
    }

    /// Whether attention applies RMSNorm to each query/key head.
    pub fn uses_qk_norm(self) -> bool {
        matches!(self, Architecture::Qwen3Moe)
    }
}

/// `eos_token_id` may be a single integer or a list in `config.json`.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum OneOrMany {
    One(u32),
    Many(Vec<u32>),
}

/// The subset of a Hugging Face `config.json` that we read.
///
/// Every field that differs between families is optional; normalisation
/// happens in [`ModelConfig::from_hf`].
#[derive(Debug, Clone, Deserialize)]
struct HfConfig {
    architectures: Vec<String>,
    hidden_size: usize,
    #[serde(default)]
    intermediate_size: Option<usize>,
    #[serde(default)]
    moe_intermediate_size: Option<usize>,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    #[serde(default)]
    head_dim: Option<usize>,
    vocab_size: usize,
    #[serde(default = "default_eps")]
    rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    #[serde(default = "default_max_pos")]
    max_position_embeddings: usize,
    #[serde(default)]
    num_experts: Option<usize>,
    #[serde(default)]
    num_local_experts: Option<usize>,
    num_experts_per_tok: usize,
    #[serde(default)]
    norm_topk_prob: Option<bool>,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default)]
    decoder_sparse_step: Option<usize>,
    #[serde(default)]
    mlp_only_layers: Vec<usize>,
    #[serde(default)]
    bos_token_id: Option<u32>,
    #[serde(default)]
    eos_token_id: Option<OneOrMany>,
}

fn default_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    10_000.0
}
fn default_max_pos() -> usize {
    4096
}

/// Family-independent model hyper-parameters used by the engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub architecture: Architecture,
    /// Width of the residual stream (`d_model`).
    pub hidden_size: usize,
    /// Hidden width of one routed expert's SwiGLU MLP.
    pub expert_intermediate_size: usize,
    /// Hidden width of a dense (non-MoE) MLP layer, if the model has any.
    pub dense_intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    /// Routed experts per MoE layer.
    pub num_experts: usize,
    /// Experts activated per token (the "k" in top-k).
    pub experts_per_token: usize,
    /// Re-normalise the top-k router probabilities so they sum to one.
    pub normalize_topk: bool,
    pub tie_word_embeddings: bool,
    /// Indices of layers that use a dense MLP instead of MoE.
    pub dense_layers: Vec<usize>,
    pub bos_token_id: Option<u32>,
    pub eos_token_ids: Vec<u32>,
}

impl ModelConfig {
    /// Reads and normalises `<dir>/config.json`.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path).map_err(|e| HaleError::io(&path, e))?;
        Self::from_json_str(&text).map_err(|e| match e {
            HaleError::InvalidArgument(msg) => {
                HaleError::InvalidArgument(format!("{}: {msg}", path.display()))
            }
            other => other,
        })
    }

    /// Parses a `config.json` document held in memory.
    pub fn from_json_str(text: &str) -> Result<Self> {
        let hf: HfConfig = serde_json::from_str(text)
            .map_err(|e| HaleError::InvalidArgument(format!("config.json: {e}")))?;
        Self::from_hf(hf)
    }

    fn from_hf(hf: HfConfig) -> Result<Self> {
        let arch_name = hf
            .architectures
            .first()
            .ok_or_else(|| HaleError::InvalidArgument("config.json has no architectures".into()))?;
        let architecture = Architecture::from_hf_name(arch_name)?;

        let num_experts = hf
            .num_experts
            .or(hf.num_local_experts)
            .ok_or_else(|| HaleError::InvalidArgument("config.json has no expert count".into()))?;
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
        let bad = |m: String| Err(HaleError::InvalidArgument(m));
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

    const QWEN3_30B: &str = r#"{
        "architectures": ["Qwen3MoeForCausalLM"],
        "hidden_size": 2048, "intermediate_size": 6144, "moe_intermediate_size": 768,
        "num_hidden_layers": 48, "num_attention_heads": 32, "num_key_value_heads": 4,
        "head_dim": 128, "vocab_size": 151936, "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0, "max_position_embeddings": 40960,
        "num_experts": 128, "num_experts_per_tok": 8, "norm_topk_prob": true,
        "decoder_sparse_step": 1, "mlp_only_layers": [], "tie_word_embeddings": false,
        "bos_token_id": 151643, "eos_token_id": 151645
    }"#;

    #[test]
    fn parses_qwen3_30b_and_counts_parameters() {
        let c = ModelConfig::from_json_str(QWEN3_30B).unwrap();
        assert_eq!(c.architecture, Architecture::Qwen3Moe);
        assert_eq!(c.num_experts, 128);
        assert_eq!(c.num_moe_layers(), 48);
        assert!(c.normalize_topk);
        // Published numbers: 30.5B total, 3.3B active.
        let total = c.total_params() as f64 / 1e9;
        let active = c.active_params() as f64 / 1e9;
        assert!((30.0..31.0).contains(&total), "total = {total}");
        assert!((3.0..3.6).contains(&active), "active = {active}");
    }

    #[test]
    fn mixtral_uses_num_local_experts_and_always_normalizes() {
        let json = r#"{
            "architectures": ["MixtralForCausalLM"], "hidden_size": 4096,
            "intermediate_size": 14336, "num_hidden_layers": 32,
            "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 32000,
            "num_local_experts": 8, "num_experts_per_tok": 2, "eos_token_id": [2]
        }"#;
        let c = ModelConfig::from_json_str(json).unwrap();
        assert_eq!(c.num_experts, 8);
        assert_eq!(c.head_dim, 128);
        assert_eq!(c.expert_intermediate_size, 14336);
        assert!(c.normalize_topk);
        assert_eq!(c.eos_token_ids, vec![2]);
        let total = c.total_params() as f64 / 1e9;
        assert!(
            (46.0..47.5).contains(&total),
            "Mixtral-8x7B is 46.7B, got {total}"
        );
    }

    #[test]
    fn sparse_step_marks_dense_layers() {
        let json = QWEN3_30B.replace("\"decoder_sparse_step\": 1", "\"decoder_sparse_step\": 2");
        let c = ModelConfig::from_json_str(&json).unwrap();
        assert!(!c.is_moe_layer(0));
        assert!(c.is_moe_layer(1));
        assert_eq!(c.num_moe_layers(), 24);
    }

    #[test]
    fn rejects_unknown_architecture() {
        let json = QWEN3_30B.replace("Qwen3MoeForCausalLM", "LlamaForCausalLM");
        assert!(matches!(
            ModelConfig::from_json_str(&json),
            Err(HaleError::UnsupportedArchitecture(_))
        ));
    }
}
