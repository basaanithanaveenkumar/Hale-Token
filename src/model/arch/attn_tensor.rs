//! [`AttnTensor`]: which attention tensor a weight is.

/// The tensors of one attention block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnTensor {
    /// Query projection.
    Q,
    /// Key projection.
    K,
    /// Value projection.
    V,
    /// Output projection.
    O,
    /// Per-head query RMSNorm weight.
    QNorm,
    /// Per-head key RMSNorm weight.
    KNorm,
}
