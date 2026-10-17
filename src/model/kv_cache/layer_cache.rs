//! Keys and values of one attention layer.

/// Keys and values of one layer.
#[derive(Debug, Clone, Default)]
pub struct LayerCache {
    pub(super) keys: Vec<f32>,
    pub(super) values: Vec<f32>,
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
