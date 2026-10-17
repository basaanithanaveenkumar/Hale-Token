//! Per-family tensor names.
//!
//! Qwen3-MoE and Mixtral compute the same thing but name their tensors
//! differently (`mlp.experts.3.gate_proj` vs `block_sparse_moe.experts.3.w1`).
//! [`TensorNaming`] captures only that difference. Supporting a new family
//! with the same maths means adding one small `impl` here and nothing else.

mod mixtral_naming;
mod qwen3_moe_naming;
mod tensor_naming;

pub use self::mixtral_naming::MixtralNaming;
pub use self::qwen3_moe_naming::Qwen3MoeNaming;
pub use self::tensor_naming::AttnTensor;
pub use self::tensor_naming::MlpProj;
pub use self::tensor_naming::TensorNaming;

use crate::config::Architecture;

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
