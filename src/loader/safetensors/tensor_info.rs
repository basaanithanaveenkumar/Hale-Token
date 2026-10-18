//! Header entry of one safetensors tensor.

use serde::{Deserialize, Serialize};

/// Header entry for one tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    /// Safetensors dtype string (e.g. `BF16`).
    pub dtype: String,
    /// Tensor shape.
    pub shape: Vec<usize>,
    /// `[start, end)` byte offsets into the data section.
    pub data_offsets: [usize; 2],
}
