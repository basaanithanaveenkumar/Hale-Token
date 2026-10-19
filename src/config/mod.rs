//! Model hyper-parameters.
//!
//! Hugging Face checkpoints ship a `config.json`. Different model families
//! spell the same idea differently (`num_experts` vs `num_local_experts`), so
//! we parse the raw file into `HfConfig` and then normalise it into one
//! family-independent [`ModelConfig`] that the rest of the engine uses.

mod architecture;
mod model_config;

pub use self::architecture::{Architecture, HF_ARCHITECTURES};
pub use self::model_config::ModelConfig;

use serde::Deserialize;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::HaleTokenError;

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
            Err(HaleTokenError::UnsupportedArchitecture(_))
        ));
    }
}
