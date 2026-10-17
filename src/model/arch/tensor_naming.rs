//! The [`TensorNaming`] trait: logical weight -> checkpoint tensor name.

/// The three projections of a SwiGLU MLP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlpProj {
    Gate,
    Up,
    Down,
}

/// The tensors of one attention block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnTensor {
    Q,
    K,
    V,
    O,
    QNorm,
    KNorm,
}

/// Maps logical weights to checkpoint tensor names.
pub trait TensorNaming: Send + Sync {
    /// Routed expert projection weight.
    fn expert(&self, layer: usize, expert: usize, proj: MlpProj) -> String;
    /// Router ("gate") weight `[num_experts, hidden]`.
    fn router(&self, layer: usize) -> String;
    /// Dense (non-MoE) MLP projection weight.
    fn dense_mlp(&self, layer: usize, proj: MlpProj) -> String;

    fn embeddings(&self) -> String {
        "model.embed_tokens.weight".into()
    }
    fn final_norm(&self) -> String {
        "model.norm.weight".into()
    }
    fn lm_head(&self) -> String {
        "lm_head.weight".into()
    }
    fn input_norm(&self, layer: usize) -> String {
        format!("model.layers.{layer}.input_layernorm.weight")
    }
    fn post_attention_norm(&self, layer: usize) -> String {
        format!("model.layers.{layer}.post_attention_layernorm.weight")
    }
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
