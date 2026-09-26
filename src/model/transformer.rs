//! The complete decoder-only transformer.

use std::sync::Arc;

use super::arch::{naming_for, AttnTensor, MlpProj};
use super::{Attention, KvCache, MoeLayer};
use crate::config::ModelConfig;
use crate::error::{HaleError, Result};
use crate::expert::{Expert, ExpertProvider};
use crate::loader::TensorSource;
use crate::ops::{rms_norm, Rope};
use crate::tensor::WeightMatrix;

/// The feed-forward half of a decoder layer.
pub enum FeedForward {
    /// A plain SwiGLU MLP that every token uses.
    Dense(Expert),
    /// A routed Mixture-of-Experts block.
    Moe(MoeLayer),
}

impl FeedForward {
    fn forward(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        match self {
            FeedForward::Dense(mlp) => {
                mlp.forward(x, out);
                Ok(())
            }
            FeedForward::Moe(moe) => moe.forward(x, out),
        }
    }
}

/// Attention + feed-forward with pre-normalisation and residuals.
pub struct DecoderLayer {
    pub input_norm: Vec<f32>,
    pub attention: Attention,
    pub post_attention_norm: Vec<f32>,
    pub feed_forward: FeedForward,
}

/// A loaded model ready to run.
pub struct Transformer {
    pub config: ModelConfig,
    pub embeddings: WeightMatrix,
    pub layers: Vec<DecoderLayer>,
    pub final_norm: Vec<f32>,
    pub lm_head: WeightMatrix,
    rope: Rope,
    experts: Arc<dyn ExpertProvider>,
}

impl Transformer {
    /// Builds the model from dense tensors plus an expert provider.
    ///
    /// Dense weights stay zero-copy views into the memory-mapped checkpoint;
    /// experts are obtained on demand through `experts`.
    pub fn load(
        config: ModelConfig,
        tensors: &dyn TensorSource,
        experts: Arc<dyn ExpertProvider>,
    ) -> Result<Self> {
        let names = naming_for(config.architecture);
        let c = &config;

        let embeddings = tensors.matrix(&names.embeddings())?;
        check_shape(
            &names.embeddings(),
            &embeddings,
            c.vocab_size,
            c.hidden_size,
        )?;
        let lm_head = if c.tie_word_embeddings || !tensors.contains(&names.lm_head()) {
            embeddings.clone()
        } else {
            tensors.matrix(&names.lm_head())?
        };

        let mut layers = Vec::with_capacity(c.num_layers);
        for l in 0..c.num_layers {
            let attn = |t| tensors.matrix(&names.attention(l, t));
            let qk_norm = |t| -> Result<Option<Vec<f32>>> {
                if c.architecture.uses_qk_norm() {
                    tensors.vector(&names.attention(l, t)).map(Some)
                } else {
                    Ok(None)
                }
            };
            let attention = Attention {
                q_proj: attn(AttnTensor::Q)?,
                k_proj: attn(AttnTensor::K)?,
                v_proj: attn(AttnTensor::V)?,
                o_proj: attn(AttnTensor::O)?,
                q_norm: qk_norm(AttnTensor::QNorm)?,
                k_norm: qk_norm(AttnTensor::KNorm)?,
                num_heads: c.num_heads,
                num_kv_heads: c.num_kv_heads,
                head_dim: c.head_dim,
                norm_eps: c.rms_norm_eps,
            };
            check_shape(
                &names.attention(l, AttnTensor::Q),
                &attention.q_proj,
                c.num_heads * c.head_dim,
                c.hidden_size,
            )?;
            check_shape(
                &names.attention(l, AttnTensor::K),
                &attention.k_proj,
                c.num_kv_heads * c.head_dim,
                c.hidden_size,
            )?;

            let feed_forward = if c.is_moe_layer(l) {
                let router = tensors.matrix(&names.router(l))?;
                check_shape(&names.router(l), &router, c.num_experts, c.hidden_size)?;
                FeedForward::Moe(MoeLayer {
                    layer: l,
                    router,
                    experts_per_token: c.experts_per_token,
                    normalize_topk: c.normalize_topk,
                    experts: Arc::clone(&experts),
                })
            } else {
                let mlp = |p| tensors.matrix(&names.dense_mlp(l, p));
                FeedForward::Dense(Expert {
                    gate: mlp(MlpProj::Gate)?,
                    up: mlp(MlpProj::Up)?,
                    down: mlp(MlpProj::Down)?,
                })
            };

            layers.push(DecoderLayer {
                input_norm: tensors.vector(&names.input_norm(l))?,
                attention,
                post_attention_norm: tensors.vector(&names.post_attention_norm(l))?,
                feed_forward,
            });
        }

        Ok(Transformer {
            rope: Rope::new(c.head_dim, c.rope_theta),
            embeddings,
            layers,
            final_norm: tensors.vector(&names.final_norm())?,
            lm_head,
            experts,
            config,
        })
    }

    /// The expert provider (for statistics).
    pub fn experts(&self) -> &Arc<dyn ExpertProvider> {
        &self.experts
    }

    /// A fresh KV cache sized for this model.
    pub fn new_cache(&self, max_len: usize) -> KvCache {
        let c = &self.config;
        KvCache::new(
            c.num_layers,
            c.num_kv_heads * c.head_dim,
            max_len.min(c.max_position_embeddings),
        )
    }

    /// Feeds one token and returns the final hidden state (before the LM head).
    pub fn forward_hidden(&self, token: u32, cache: &mut KvCache) -> Result<Vec<f32>> {
        let c = &self.config;
        let position = cache.len();
        if position >= cache.max_len() {
            return Err(HaleError::ContextOverflow {
                needed: position + 1,
                max: cache.max_len(),
            });
        }
        if token as usize >= c.vocab_size {
            return Err(HaleError::InvalidArgument(format!(
                "token id {token} >= vocab size {}",
                c.vocab_size
            )));
        }

        let mut x = vec![0.0; c.hidden_size];
        self.embeddings.row_to_f32(token as usize, &mut x);

        let half = c.head_dim / 2;
        let (mut cos, mut sin) = (vec![0.0; half], vec![0.0; half]);
        self.rope.angles(position, &mut cos, &mut sin);

        let mut normed = vec![0.0; c.hidden_size];
        let mut delta = vec![0.0; c.hidden_size];
        for (l, layer) in self.layers.iter().enumerate() {
            rms_norm(&x, &layer.input_norm, c.rms_norm_eps, &mut normed);
            layer.attention.forward(
                &normed,
                position,
                &cos,
                &sin,
                cache.layer_mut(l),
                &mut delta,
            );
            add_in_place(&mut x, &delta);

            rms_norm(&x, &layer.post_attention_norm, c.rms_norm_eps, &mut normed);
            layer.feed_forward.forward(&normed, &mut delta)?;
            add_in_place(&mut x, &delta);
        }
        cache.advance();

        let mut out = vec![0.0; c.hidden_size];
        rms_norm(&x, &self.final_norm, c.rms_norm_eps, &mut out);
        Ok(out)
    }

    /// Projects a final hidden state to vocabulary logits.
    pub fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        let mut logits = vec![0.0; self.config.vocab_size];
        self.lm_head.matvec(hidden, &mut logits);
        logits
    }

    /// Feeds one token and returns next-token logits.
    pub fn forward(&self, token: u32, cache: &mut KvCache) -> Result<Vec<f32>> {
        let hidden = self.forward_hidden(token, cache)?;
        Ok(self.logits(&hidden))
    }

    /// Feeds a whole prompt; returns the logits after its last token.
    /// Skips the (large) LM head for every token but the last.
    pub fn prefill(&self, tokens: &[u32], cache: &mut KvCache) -> Result<Vec<f32>> {
        let (last, rest) = tokens
            .split_last()
            .ok_or_else(|| HaleError::InvalidArgument("empty prompt".into()))?;
        for &t in rest {
            self.forward_hidden(t, cache)?;
        }
        self.forward(*last, cache)
    }
}

fn add_in_place(x: &mut [f32], delta: &[f32]) {
    x.iter_mut().zip(delta).for_each(|(a, b)| *a += b);
}

fn check_shape(name: &str, m: &WeightMatrix, rows: usize, cols: usize) -> Result<()> {
    if (m.rows(), m.cols()) != (rows, cols) {
        return Err(HaleError::bad_tensor(
            name,
            format!(
                "expected shape [{rows}, {cols}], got [{}, {}]",
                m.rows(),
                m.cols()
            ),
        ));
    }
    Ok(())
}
