---
name: hale-token-add-model
description: Add a new MoE model family (e.g. a new Qwen/Mixtral-like architecture, or DeepSeek-V3/Kimi-K2 with MLA and shared experts) or a planner preset to Hale-Token. Use when asked to support another Hugging Face MoE checkpoint or to make `hale plan` know a new model.
---

# Adding a model family

## Same maths, different tensor names (easy path)

1. `src/config.rs`: map the new `architectures` string to an `Architecture` variant, and
   read any extra `config.json` fields (e.g. `num_experts_per_tok`, `norm_topk_prob`).
2. `src/model/arch.rs`: implement `TensorNaming` (embedding, per-layer attention, norms,
   router, `expert(layer, e, MlpProj)`, lm_head), and add a match arm in `naming_for`.
3. Router semantics: check whether top-k weights are renormalised (Qwen3 `norm_topk_prob`)
   and whether softmax comes before or after top-k. Mixtral applies softmax over the top-k
   logits. Encode this in `model/moe.rs`, not in the naming.
4. Fixture: add `tests/fixtures/tiny-<family>/` produced by
   `scripts/reference/make_fixtures.py` (a tiny random config, `model.safetensors`,
   `tokenizer.json`, `expected.json` from the float64 reference). Add it to
   `tests/litmus.rs` and `tests/tiered_cache.rs`.
5. Cross-check against `transformers` with `scripts/reference/compare_with_transformers.py`.
6. Update `docs/models.md` and the README table.

## New maths (MLA, shared experts: DeepSeek-V3, Kimi-K2)

These are listed as "planned" in `docs/roadmap.md`. You need:
- an `Attention` variant with low-rank compressed KV (MLA), including its RoPE split
  and a KV cache that stores the latent;
- `MoeLayer` support for **shared experts** (always active; they are good candidates to
  keep dense rather than in the tiered cache);
- the planner's `active_params` to include shared experts.

Land it behind the litmus test with a tiny fixture first.

## Planner preset

`hale plan --preset <name>` reads shapes from the preset table in
`src/expert/planner.rs` (matched by case-insensitive prefix). Add total/active parameters, layer
count, experts, top-k and expert size, then check `hale plan --compare` output and update
the README / `docs/models.md` tables. Estimates are floors that assume uniform routing, so
say so.
