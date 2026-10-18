//! [`TierSplit`]: how RAM-resident experts divide between tiers.

/// How many RAM-resident experts go to each tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TierSplit {
    /// Experts pinned to the hottest set.
    pub pinned: usize,
    /// Experts managed by the LRU cache.
    pub lru: usize,
}
