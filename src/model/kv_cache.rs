//! Key/value cache for autoregressive decoding.
//!
//! Attention at position `t` needs the keys and values of every earlier
//! position. Recomputing them each step would make generation quadratic, so
//! each layer appends its new key/value vectors here and reads the history
//! back. Vectors are stored contiguously per layer (`[position][kv_dim]`)
//! for cache-friendly scans.

/// Keys and values of one layer.
#[derive(Debug, Clone, Default)]
pub struct LayerCache {
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl LayerCache {
    /// Appends one position's key and value vectors.
    pub fn push(&mut self, key: &[f32], value: &[f32]) {
        self.keys.extend_from_slice(key);
        self.values.extend_from_slice(value);
    }

    /// Key vector at position `t` (length `kv_dim`).
    pub fn key(&self, t: usize, kv_dim: usize) -> &[f32] {
        &self.keys[t * kv_dim..(t + 1) * kv_dim]
    }

    /// Value vector at position `t` (length `kv_dim`).
    pub fn value(&self, t: usize, kv_dim: usize) -> &[f32] {
        &self.values[t * kv_dim..(t + 1) * kv_dim]
    }
}

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
