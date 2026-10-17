//! Header entry of one safetensors tensor.

use serde::{Deserialize, Serialize};

/// Header entry for one tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}
