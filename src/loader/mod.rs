//! Reading (and writing) model weights on disk.
//!
//! The engine never talks to a file format directly; it asks a
//! [`TensorSource`] for named tensors. Today the only implementation reads
//! Hugging Face *safetensors* shards, but a GGUF reader could be added
//! without touching the model code (Open/Closed + Dependency Inversion).

mod safetensors;

pub use self::safetensors::{write_safetensors, SafetensorsCheckpoint, TensorInfo};

use crate::error::Result;
use crate::tensor::WeightMatrix;

/// Anything that can hand out named weight tensors.
pub trait TensorSource: Send + Sync {
    /// Whether a tensor called `name` exists.
    fn contains(&self, name: &str) -> bool;

    /// Loads a 2-D tensor as a matrix (zero-copy when the source is mmapped).
    fn matrix(&self, name: &str) -> Result<WeightMatrix>;

    /// Loads a 1-D tensor (e.g. a norm weight) decoded to `f32`.
    fn vector(&self, name: &str) -> Result<Vec<f32>>;
}
