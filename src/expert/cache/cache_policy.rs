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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_policy_pins_nothing_and_has_no_lru() {
        let policy = CachePolicy::default();
        assert!(policy.pinned.is_empty());
        assert_eq!(policy.lru_bytes, 0);
    }
}
