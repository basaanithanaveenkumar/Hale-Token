//! Per-family tensor names.
//!
//! Qwen3-MoE and Mixtral compute the same thing but name their tensors
//! differently (`mlp.experts.3.gate_proj` vs `block_sparse_moe.experts.3.w1`).
//! [`TensorNaming`] captures only that difference. Supporting a new family
//! with the same maths means adding one small `impl` here and nothing else.

mod attn_tensor;
mod mixtral_naming;
mod mlp_proj;
mod qwen3_moe_naming;
mod tensor_naming;

pub use self::attn_tensor::AttnTensor;
pub use self::mixtral_naming::MixtralNaming;
pub use self::mlp_proj::MlpProj;
pub use self::qwen3_moe_naming::Qwen3MoeNaming;
pub use self::tensor_naming::TensorNaming;

use std::sync::OnceLock;

use crate::config::Architecture;
use crate::registry::Registry;

fn hf_proj(proj: MlpProj) -> &'static str {
    match proj {
        MlpProj::Gate => "gate_proj",
        MlpProj::Up => "up_proj",
        MlpProj::Down => "down_proj",
    }
}

/// Registry of tensor-naming schemes, keyed by Hugging Face architecture name.
///
/// Supporting a new family means adding one `register` call here (plus the
/// [`TensorNaming`] impl); no `match` elsewhere changes.
///
/// # Examples
///
/// ```
/// use hale::model::arch::{naming_registry, MlpProj};
///
/// let naming = naming_registry().build("MixtralForCausalLM", &()).unwrap();
/// assert_eq!(naming.router(2), "model.layers.2.block_sparse_moe.gate.weight");
/// assert!(naming_registry().build("MixtralForCausalLm", &()).is_err());
/// let _ = MlpProj::Gate;
/// ```
pub fn naming_registry() -> &'static Registry<dyn TensorNaming, ()> {
    static REGISTRY: OnceLock<Registry<dyn TensorNaming, ()>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let mut registry: Registry<dyn TensorNaming, ()> = Registry::new("tensor naming");
        registry
            .register(Architecture::Qwen3Moe.hf_name(), |_| {
                Box::new(Qwen3MoeNaming)
            })
            .and_then(|()| {
                registry.register(Architecture::Mixtral.hf_name(), |_| Box::new(MixtralNaming))
            })
            .expect("built-in naming keys are distinct");
        registry
    })
}

/// The naming scheme for an architecture, looked up in [`naming_registry`].
pub fn naming_for(arch: Architecture) -> Box<dyn TensorNaming> {
    naming_registry()
        .build(arch.hf_name(), &())
        .expect("every Architecture variant is registered")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_architecture_is_registered() {
        for arch in [Architecture::Qwen3Moe, Architecture::Mixtral] {
            assert!(naming_registry().contains(arch.hf_name()));
        }
    }

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
