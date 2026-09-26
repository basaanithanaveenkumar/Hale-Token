//! Loads experts straight from a Hugging Face checkpoint.

use std::sync::Arc;

use super::{Expert, ExpertKey, ExpertSource};
use crate::config::ModelConfig;
use crate::error::Result;
use crate::loader::TensorSource;
use crate::model::arch::{naming_for, MlpProj, TensorNaming};

/// Reads experts from an (unconverted) checkpoint through a [`TensorSource`].
///
/// With safetensors this is zero-copy: `load` just creates views into the
/// memory map and the OS pages bytes in from the SSD when they are first
/// multiplied. Convenient for trying a model, but the OS cannot know which
/// experts matter, so for models larger than RAM prefer `hale convert` and
/// [`super::PackExpertSource`].
pub struct CheckpointExpertSource {
    tensors: Arc<dyn TensorSource>,
    naming: Box<dyn TensorNaming>,
    expert_bytes: usize,
}

impl CheckpointExpertSource {
    /// Creates a source; loads expert (first MoE layer, 0) once to learn its size.
    pub fn new(tensors: Arc<dyn TensorSource>, config: &ModelConfig) -> Result<Self> {
        let naming = naming_for(config.architecture);
        let first_moe = (0..config.num_layers)
            .find(|l| config.is_moe_layer(*l))
            .unwrap_or(0);
        let mut source = CheckpointExpertSource {
            tensors,
            naming,
            expert_bytes: 0,
        };
        source.expert_bytes = source.load(ExpertKey::new(first_moe, 0))?.size_bytes();
        Ok(source)
    }
}

impl ExpertSource for CheckpointExpertSource {
    fn load(&self, key: ExpertKey) -> Result<Expert> {
        let (l, e) = (key.layer as usize, key.expert as usize);
        Ok(Expert {
            gate: self
                .tensors
                .matrix(&self.naming.expert(l, e, MlpProj::Gate))?,
            up: self
                .tensors
                .matrix(&self.naming.expert(l, e, MlpProj::Up))?,
            down: self
                .tensors
                .matrix(&self.naming.expert(l, e, MlpProj::Down))?,
        })
    }

    fn expert_bytes(&self) -> usize {
        self.expert_bytes
    }

    fn loads_into_ram(&self) -> bool {
        false
    }

    fn describe(&self) -> String {
        "checkpoint (memory-mapped safetensors)".into()
    }
}
