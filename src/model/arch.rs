//! Per-family tensor names.
//!
//! Qwen3-MoE and Mixtral compute the same thing but name their tensors
//! differently (`mlp.experts.3.gate_proj` vs `block_sparse_moe.experts.3.w1`).
//! [`TensorNaming`] captures only that difference. Supporting a new family
//! with the same maths means adding one small `impl` here and nothing else.

use crate::config::Architecture;

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

/// Qwen3-MoE naming (`mlp.experts.N.gate_proj`).
pub struct Qwen3MoeNaming;

impl TensorNaming for Qwen3MoeNaming {
    fn expert(&self, layer: usize, expert: usize, proj: MlpProj) -> String {
        format!(
            "model.layers.{layer}.mlp.experts.{expert}.{}.weight",
            hf_proj(proj)
        )
    }
    fn router(&self, layer: usize) -> String {
        format!("model.layers.{layer}.mlp.gate.weight")
    }
    fn dense_mlp(&self, layer: usize, proj: MlpProj) -> String {
        format!("model.layers.{layer}.mlp.{}.weight", hf_proj(proj))
    }
}

/// Mixtral naming (`block_sparse_moe.experts.N.w1/w3/w2`).
pub struct MixtralNaming;

impl TensorNaming for MixtralNaming {
    fn expert(&self, layer: usize, expert: usize, proj: MlpProj) -> String {
        let w = match proj {
            MlpProj::Gate => "w1",
            MlpProj::Down => "w2",
            MlpProj::Up => "w3",
        };
        format!("model.layers.{layer}.block_sparse_moe.experts.{expert}.{w}.weight")
    }
    fn router(&self, layer: usize) -> String {
        format!("model.layers.{layer}.block_sparse_moe.gate.weight")
    }
    fn dense_mlp(&self, layer: usize, proj: MlpProj) -> String {
        format!("model.layers.{layer}.mlp.{}.weight", hf_proj(proj))
    }
}

fn hf_proj(proj: MlpProj) -> &'static str {
    match proj {
        MlpProj::Gate => "gate_proj",
        MlpProj::Up => "up_proj",
        MlpProj::Down => "down_proj",
    }
}

/// Factory: the naming scheme for an architecture.
pub fn naming_for(arch: Architecture) -> Box<dyn TensorNaming> {
    match arch {
        Architecture::Qwen3Moe => Box::new(Qwen3MoeNaming),
        Architecture::Mixtral => Box::new(MixtralNaming),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_match_hugging_face_checkpoints() {
        let q = naming_for(Architecture::Qwen3Moe);
        assert_eq!(
            q.expert(3, 7, MlpProj::Up),
            "model.layers.3.mlp.experts.7.up_proj.weight"
        );
        assert_eq!(
            q.attention(0, AttnTensor::KNorm),
            "model.layers.0.self_attn.k_norm.weight"
        );
        let m = naming_for(Architecture::Mixtral);
        assert_eq!(
            m.expert(1, 0, MlpProj::Down),
            "model.layers.1.block_sparse_moe.experts.0.w2.weight"
        );
        assert_eq!(m.router(5), "model.layers.5.block_sparse_moe.gate.weight");
    }
}
