//! [`MlpProj`]: which MLP projection a weight is.

/// The three projections of a SwiGLU MLP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlpProj {
    /// Gate projection (`w1` in Mixtral).
    Gate,
    /// Up projection (`w3` in Mixtral).
    Up,
    /// Down projection (`w2` in Mixtral).
    Down,
}
