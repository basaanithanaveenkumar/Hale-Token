//! Tensor names of Qwen3-MoE checkpoints.

use super::{hf_proj, MlpProj, TensorNaming};

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
