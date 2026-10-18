//! Key/value cache for autoregressive decoding.
//!
//! Attention at position `t` needs the keys and values of every earlier
//! position. Recomputing them each step would make generation quadratic, so
//! each layer appends its new key/value vectors here and reads the history
//! back. Vectors are stored contiguously per layer (`[position][kv_dim]`)
//! for cache-friendly scans.

mod layer_cache;

pub use self::layer_cache::LayerCache;

/// The KV cache of the whole model for one sequence.
#[derive(Debug, Clone)]
pub struct KvCache {
    layers: Vec<LayerCache>,
    len: usize,
    max_len: usize,
    kv_dim: usize,
}

impl KvCache {
    /// Empty cache for `num_layers` layers and at most `max_len` positions.
    pub fn new(num_layers: usize, kv_dim: usize, max_len: usize) -> Self {
        KvCache {
            layers: vec![LayerCache::default(); num_layers],
            len: 0,
            max_len,
            kv_dim,
        }
    }

    /// Positions stored so far (= position of the next token).
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when no positions have been cached yet.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Maximum positions this cache accepts.
    pub fn max_len(&self) -> usize {
        self.max_len
    }

    /// Width of one key or value vector.
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    /// Mutable access to one layer.
    pub fn layer_mut(&mut self, layer: usize) -> &mut LayerCache {
        &mut self.layers[layer]
    }

    /// Marks one more position as complete (called after all layers ran).
    pub fn advance(&mut self) {
        self.len += 1;
    }

    /// Forgets everything (start a new conversation).
    pub fn clear(&mut self) {
        self.layers
            .iter_mut()
            .for_each(|l| *l = LayerCache::default());
        self.len = 0;
    }

    /// Bytes currently used by keys and values.
    pub fn size_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| (l.keys.len() + l.values.len()) * 4)
            .sum()
    }
}
