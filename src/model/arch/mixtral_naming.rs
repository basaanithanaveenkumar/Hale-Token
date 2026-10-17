//! Tensor names of Mixtral checkpoints.

use super::{hf_proj, MlpProj, TensorNaming};

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
