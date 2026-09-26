# Testing and verification

Hale-Token uses four layers of tests. Each one catches a different kind of
bug.

```text
               ┌────────────────────────┐
               │  real model (CI, Mac)  │  transformers cross-check on trained weights
             ┌─┴────────────────────────┴─┐
             │   smoke (cargo install)    │  the installed binary works end to end
           ┌─┴────────────────────────────┴─┐
           │ litmus / integration (tests/)  │  whole-model maths vs independent reference
         ┌─┴────────────────────────────────┴─┐
         │       unit (#[cfg(test)] mod)      │  every kernel and data structure
         └────────────────────────────────────┘
```

## Running

```bash
cargo test                            # unit + integration + litmus + doc tests
cargo test --test litmus              # only the reference comparisons
./scripts/smoke_test.sh               # install + every CLI command
./scripts/verify_real_model.sh <hf-repo-id>   # needs Python with torch + transformers
```

## Unit tests (`src/**`)

These live next to the code they test. Highlights:

- `tensor::quant`: every dtype round-trips within its precision; quantized
  dot products agree with decoding followed by a dot product.
- `tensor::matrix`: parallel and serial mat-vec agree bit for bit.
- `expert::lru`: eviction order, byte weights, slot recycling, zero
  capacity.
- `expert::cache`: tier order, request order is preserved, mmap
  pass-through.
- `expert::pack`: out-of-order writes, and rejection of truncated or foreign
  files.
- `expert::planner`: preset sizes match published totals; monotonic in RAM;
  infeasible cases.
- `config`: Qwen3-30B-A3B and Mixtral-8x7B parameter counts match the
  published numbers.

## Litmus tests (`tests/litmus.rs`, `tests/tiered_cache.rs`)

A litmus test answers one question with no room for doubt.

**Does the engine compute the model correctly?**
`scripts/reference/make_fixtures.py` is a separate NumPy float64
implementation of Qwen3-MoE and Mixtral that uses *full-sequence* causal
attention. It generates two tiny random models (one f32, one bf16; one with a
dense layer between MoE layers, one with GQA 4:1) and their logits. The Rust
engine uses an f32 incremental KV cache and must match **every logit at every
position** to within 2e-3, and produce the same greedy tokens. The fixture
generator checks that the top-1/top-2 margin exceeds 1e-2, so greedy
comparisons are stable.

**Does tiering change the answer?**
Each fixture is converted to a pack and run twice: fully resident, and with
room for a single expert so that almost every request is an SSD read. The
logits must be **bit-identical**.

**Is quantization sane?**
q8_0 must stay under 1% relative RMS error with 95%+ top-1 agreement; q4_0
under 25% and 60%. These bounds are for random Gaussian weights, the worst
case for quantization. Trained models do better.

**Does warm start work?**
Routing statistics from one run must pin experts that are then hit on the
next.

## Regenerating fixtures

```bash
pip install numpy tokenizers
python3 scripts/reference/make_fixtures.py
```

Commit the regenerated `tests/fixtures/**` files together with the change
that needed them.

## Smoke test (`scripts/smoke_test.sh`)

This builds and installs with `cargo install --locked`, then runs
`sysinfo`, `info`, `convert`, `run` (streaming from SSD), `logits` (RAM and
SSD placements must give identical output), `plan` and `bench` on both
fixtures.

## Cross-check with Hugging Face transformers

`scripts/reference/compare_with_transformers.py` runs a token sequence
through `transformers` (float32) and through `hale logits`, then compares
relative RMS error and top-1 agreement. CI runs it on both fixtures, which
also validates the NumPy reference against transformers.

`scripts/verify_real_model.sh` does the same on a **real, trained** MoE
downloaded from the Hub. It checks both the unconverted checkpoint and a
q8_0 pack, then generates text with experts streamed through a cache holding
25% of them.

## Continuous integration (`.github/workflows/ci.yml`)

| Job | Runner | Does |
|---|---|---|
| lint | ubuntu | `fmt --check`, `clippy -D warnings`, `doc -D warnings` |
| apple-silicon | macos-14, macos-15, macos-26 (arm64) | all tests, smoke test, `sysinfo`, memory and kernel benchmarks |
| reference-transformers | macos-14 | fixtures vs transformers |
| real-moe | macos-14 | `verify_real_model.sh` on a trained Mixtral-architecture MoE; report uploaded as an artifact |
