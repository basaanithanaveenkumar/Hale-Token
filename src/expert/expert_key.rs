//! [`ExpertKey`]: the `(layer, expert)` address of a routed expert.

/// Identifies one routed expert: `(layer, expert index within the layer)`.
///
/// # Examples
///
/// ```
/// use hale::expert::ExpertKey;
///
/// let key = ExpertKey::new(3, 17);
/// assert_eq!(key.to_string(), "L3E17");
/// assert!(ExpertKey::new(3, 17) < ExpertKey::new(4, 0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExpertKey {
    /// Decoder layer index.
    pub layer: u32,
    /// Expert index within the layer.
    pub expert: u32,
}

impl ExpertKey {
    /// Builds a key from `usize` indices (as produced by loops over the config).
    pub fn new(layer: usize, expert: usize) -> Self {
        ExpertKey {
            layer: layer as u32,
            expert: expert as u32,
        }
    }
}

impl std::fmt::Display for ExpertKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "L{}E{}", self.layer, self.expert)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn new_stores_both_indices() {
        let key = ExpertKey::new(2, 5);
        assert_eq!((key.layer, key.expert), (2, 5));
    }

    #[test]
    fn keys_order_by_layer_then_expert_and_hash_by_value() {
        assert!(ExpertKey::new(1, 9) < ExpertKey::new(2, 0));
        assert!(ExpertKey::new(1, 1) < ExpertKey::new(1, 2));
        let set: HashSet<_> = [ExpertKey::new(1, 1), ExpertKey::new(1, 1)].into();
        assert_eq!(set.len(), 1);
    }
}
