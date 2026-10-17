//! How the cache splits RAM between the pinned and LRU tiers.

use super::super::ExpertKey;

/// How much RAM each tier may use.
#[derive(Debug, Clone, Default)]
pub struct CachePolicy {
    /// Experts to load at start-up and never evict (tier 0).
    pub pinned: Vec<ExpertKey>,
    /// Byte budget of the LRU tier (tier 1).
    pub lru_bytes: usize,
}
