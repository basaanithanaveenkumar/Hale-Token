//! One token routing decision: chosen experts and weights.

use crate::expert::ExpertKey;

/// One routing decision: which expert, with what weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Route {
    pub key: ExpertKey,
    pub weight: f32,
}
