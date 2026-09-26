//! The sparse Mixture-of-Experts feed-forward layer.

use std::sync::Arc;

use crate::error::Result;
use crate::expert::{ExpertKey, ExpertProvider};
use crate::ops::{softmax_in_place, top_k};
use crate::tensor::WeightMatrix;

/// One routing decision: which expert, with what weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Route {
    pub key: ExpertKey,
    pub weight: f32,
}

/// Router plus a handle to the (tiered) expert store.
///
/// ```text
/// probs   = softmax(router · x)            over all experts
/// chosen  = top_k(probs, k)                 e.g. 8 of 128
/// weights = chosen / sum(chosen)            if normalize_topk
/// y       = Σ weight_i · Expert_i(x)
/// ```
pub struct MoeLayer {
    pub layer: usize,
    /// `[num_experts, hidden]`
    pub router: WeightMatrix,
    pub experts_per_token: usize,
    pub normalize_topk: bool,
    pub experts: Arc<dyn ExpertProvider>,
}

impl MoeLayer {
    /// Picks the experts for hidden state `x`.
    pub fn route(&self, x: &[f32]) -> Vec<Route> {
        let mut probs = vec![0.0; self.router.rows()];
        self.router.matvec(x, &mut probs);
        softmax_in_place(&mut probs);
        let chosen = top_k(&probs, self.experts_per_token);
        let norm = if self.normalize_topk {
            chosen.iter().map(|c| c.score).sum::<f32>()
        } else {
            1.0
        };
        chosen
            .into_iter()
            .map(|c| Route {
                key: ExpertKey::new(self.layer, c.index),
                weight: c.score / norm,
            })
            .collect()
    }

    /// Computes the layer output for `x` into `out`.
    pub fn forward(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        let routes = self.route(x);
        let keys: Vec<ExpertKey> = routes.iter().map(|r| r.key).collect();
        let experts = self.experts.fetch(&keys)?;

        out.fill(0.0);
        let mut tmp = vec![0.0; out.len()];
        for (route, expert) in routes.iter().zip(&experts) {
            expert.forward(x, &mut tmp);
            for (o, t) in out.iter_mut().zip(&tmp) {
                *o += route.weight * t;
            }
        }
        Ok(())
    }
}
