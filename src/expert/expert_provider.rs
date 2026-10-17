//! [`ExpertProvider`]: what the MoE layer depends on to obtain experts.

use std::sync::Arc;

use super::{CacheStats, Expert, ExpertKey, RoutingStats};
use crate::error::Result;

/// What the MoE layer depends on to obtain expert weights
/// (Dependency Inversion: the model never knows about tiers or files).
pub trait ExpertProvider: Send + Sync {
    /// Returns the experts for `keys`, in the same order.
    fn fetch(&self, keys: &[ExpertKey]) -> Result<Vec<Arc<Expert>>>;

    /// Cache counters, if the provider keeps any.
    fn stats(&self) -> CacheStats;

    /// Routing counters observed so far.
    fn routing_stats(&self) -> RoutingStats;
}
