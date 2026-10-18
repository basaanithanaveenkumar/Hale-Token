//! Inputs of one planning run.

use super::{HardwareProfile, ModelShape};
use crate::tensor::DType;

/// Planner inputs.
#[derive(Debug, Clone)]
pub struct PlanRequest {
    /// The model to plan for.
    pub model: ModelShape,
    /// The machine to plan for.
    pub hardware: HardwareProfile,
    /// Storage format of the dense weights (usually the checkpoint's bf16).
    pub dense_dtype: DType,
    /// Storage format of the experts (e.g. q4_0 after `hale convert`).
    pub expert_dtype: DType,
    /// Share of RAM we may use (see [`DEFAULT_RAM_FRACTION`](super::DEFAULT_RAM_FRACTION)).
    pub ram_fraction: f64,
    /// Whether routing statistics exist to choose a pinned set.
    pub have_routing_stats: bool,
}
