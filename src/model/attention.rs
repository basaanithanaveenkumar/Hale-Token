//! Grouped-query self-attention with rotary embeddings.

use rayon::prelude::*;

use super::kv_cache::LayerCache;
use crate::ops::{rms_norm_in_place, softmax_in_place, Rope};
use crate::tensor::WeightMatrix;

/// One attention block.
///
/// *Grouped-query attention* (GQA): `num_heads` query heads share
/// `num_kv_heads` key/value heads (`num_heads / num_kv_heads` queries per
/// KV head), which shrinks the KV cache without hurting quality much.
pub struct Attention {
    pub q_proj: WeightMatrix,
    pub k_proj: WeightMatrix,
    pub v_proj: WeightMatrix,
    pub o_proj: WeightMatrix,
    /// Per-head RMSNorm weights (Qwen3 "QK-norm"); `None` for Mixtral.
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub norm_eps: f32,
}

impl Attention {
    /// Runs attention for the token at `position`, appending its key/value
    /// to `cache`. `x` is the normalised hidden state; the result goes to `out`.
    pub fn forward(
        &self,
        x: &[f32],
        position: usize,
        cos: &[f32],
        sin: &[f32],
        cache: &mut LayerCache,
        out: &mut [f32],
    ) {
        let hd = self.head_dim;
        let kv_dim = self.num_kv_heads * hd;
        let mut q = vec![0.0; self.num_heads * hd];
        let mut k = vec![0.0; kv_dim];
        let mut v = vec![0.0; kv_dim];
        self.q_proj.matvec(x, &mut q);
        self.k_proj.matvec(x, &mut k);
        self.v_proj.matvec(x, &mut v);

        for (heads, norm) in [(&mut q, &self.q_norm), (&mut k, &self.k_norm)] {
            for head in heads.chunks_exact_mut(hd) {
                if let Some(w) = norm {
                    rms_norm_in_place(head, w, self.norm_eps);
                }
                Rope::apply(head, cos, sin);
            }
        }
        cache.push(&k, &v);

        let cache: &LayerCache = cache;
        // Each query head is independent: compute them in parallel.
        let seq_len = position + 1;
        let group = self.num_heads / self.num_kv_heads;
        let scale = 1.0 / (hd as f32).sqrt();
        let mut context = vec![0.0; self.num_heads * hd];
        context
            .par_chunks_mut(hd)
            .zip(q.par_chunks(hd))
            .enumerate()
            .for_each(|(h, (ctx, q_head))| {
                let kv_offset = (h / group) * hd;
                let mut scores: Vec<f32> = (0..seq_len)
                    .map(|t| {
                        let key = &cache.key(t, kv_dim)[kv_offset..kv_offset + hd];
                        q_head.iter().zip(key).map(|(a, b)| a * b).sum::<f32>() * scale
                    })
                    .collect();
                softmax_in_place(&mut scores);
                for (t, p) in scores.iter().enumerate() {
                    let value = &cache.value(t, kv_dim)[kv_offset..kv_offset + hd];
                    for (c, v) in ctx.iter_mut().zip(value) {
                        *c += p * v;
                    }
                }
            });
        self.o_proj.matvec(&context, out);
    }
}
