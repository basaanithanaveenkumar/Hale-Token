//! [`ExpertKey`]: the `(layer, expert)` address of a routed expert.

/// Identifies one routed expert: `(layer, expert index within the layer)`.
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
