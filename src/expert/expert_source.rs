//! [`ExpertSource`]: cold storage (tier 2) that can load any expert.

use super::{Expert, ExpertKey};
use crate::error::Result;

/// Cold storage able to load any expert on demand (tier 2).
///
/// Implementations must be thread-safe: the cache loads missing experts in
/// parallel to keep the SSD's command queue full.
pub trait ExpertSource: Send + Sync {
    /// Loads expert `key`.
    fn load(&self, key: ExpertKey) -> Result<Expert>;

    /// Size in bytes of one loaded expert (all experts are the same size).
    fn expert_bytes(&self) -> usize;

    /// Whether `load` returns owned bytes (true) or zero-copy mmap views
    /// (false). Caching mmap views gains nothing: the OS page cache already
    /// is the cache, so the tiered cache is bypassed for such sources.
    fn loads_into_ram(&self) -> bool;

    /// Short human-readable description for logs.
    fn describe(&self) -> String;
}
