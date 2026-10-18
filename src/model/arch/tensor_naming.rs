//! The [`TensorNaming`] trait: logical weight -> checkpoint tensor name.

use super::{AttnTensor, MlpProj};

/// Maps logical weights to checkpoint tensor names.
pub trait TensorNaming: Send + Sync {
    /// Routed expert projection weight.
    fn expert(&self, layer: usize, expert: usize, proj: MlpProj) -> String;
    /// Router ("gate") weight `[num_experts, hidden]`.
    fn router(&self, layer: usize) -> String;
    /// Dense (non-MoE) MLP projection weight.
    fn dense_mlp(&self, layer: usize, proj: MlpProj) -> String;

    /// Token embedding matrix name.
    fn embeddings(&self) -> String {
        "model.embed_tokens.weight".into()
    }
    /// Final RMSNorm weight name.
    fn final_norm(&self) -> String {
        "model.norm.weight".into()
    }
    /// LM head weight name.
    fn lm_head(&self) -> String {
        "lm_head.weight".into()
    }
    /// Pre-attention RMSNorm weight name for `layer`.
    fn input_norm(&self, layer: usize) -> String {
        format!("model.layers.{layer}.input_layernorm.weight")
    }
    /// Pre-MLP RMSNorm weight name for `layer`.
    fn post_attention_norm(&self, layer: usize) -> String {
        format!("model.layers.{layer}.post_attention_layernorm.weight")
    }
    /// Attention tensor name for `layer`.
    fn attention(&self, layer: usize, tensor: AttnTensor) -> String {
        let part = match tensor {
            AttnTensor::Q => "q_proj",
            AttnTensor::K => "k_proj",
            AttnTensor::V => "v_proj",
            AttnTensor::O => "o_proj",
            AttnTensor::QNorm => "q_norm",
            AttnTensor::KNorm => "k_norm",
        };
        format!("model.layers.{layer}.self_attn.{part}.weight")
    }
}
