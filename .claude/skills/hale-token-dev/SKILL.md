---
name: hale-token-dev
description: Build, test and navigate Hale-Token, the Rust MoE inference engine for Apple Silicon (tiered expert cache: pinned + LRU in unified memory, SSD pack). Use when starting work in this repo, before changing model maths, kernels, the expert cache or the pack format, or when CI fails.
---

# Hale-Token development

Default branch: `claude/modest-cerf-mc599d`. Crate `hale-token`, library `hale`, binary
`hale`. Rust ≥ 1.80. The target is Apple Silicon, and it also builds and tests on Linux.

## Build and test

```bash
cargo build --release
cargo fmt --check && cargo clippy -- -D warnings
cargo test          # 49 unit + CLI/litmus/tiered-cache integration tests (all pass on Linux x86_64, 2026-09)
./scripts/smoke_test.sh
./scripts/verify_real_model.sh Isotonic/TinyMixtral-4x248M-MoE    # needs Python + torch + HF access
```

`[profile.test]` uses `opt-level = 2`, because unoptimised maths is about 30× slower.

## Module map (dependencies point downwards only)

| Module | Responsibility |
|---|---|
| `cli/` | `hale` subcommands: `sysinfo`, `plan`, `convert`, `run`, `bench`, `logits`, `info` |
| `engine.rs` | Facade: loads config/tokenizer/weights, plans memory, chooses pinned experts |
| `generate/` | `Sampler` strategies, decode loop, stats |
| `model/` | `attention.rs` (GQA, QK-norm, RoPE), `kv_cache.rs`, `moe.rs` (router → top-k → `ExpertProvider::fetch`), `transformer.rs`, `arch.rs` (`TensorNaming` per family) |
| `expert/` | `ExpertCache` (pinned → LRU → source), `lru.rs` (O(1) byte-weighted slab LRU), `pack.rs` (`experts.hpk`), `checkpoint_source.rs`, `stats.rs` (routing counts → `hale-routing.json`), `planner.rs` (cost model) |
| `loader/` | `TensorSource` trait, zero-copy safetensors reader and writer |
| `tensor/` | `DType` (f32/f16/bf16/q8_0/q4_0), `ByteBuf`, `WeightMatrix::matvec`, quant kernels (NEON `sdot`) |
| `ops/` | RMSNorm, RoPE, softmax/SiLU, heap top-k |
| `hardware/` | M1–M5 chip table, sysctl detection, memory/SSD/kernel benchmarks |

## Non-negotiables

- **Any change to model maths needs a litmus test** (`tests/litmus.rs`): logits at every
  position must match the NumPy float64 reference (`scripts/reference/make_fixtures.py`
  regenerates `tests/fixtures/*/expected.json`).
- **Placement must not change results**: `tests/tiered_cache.rs` asserts that SSD-streamed
  logits are bit-identical to fully resident ones. Keep reductions in a fixed order,
  with no order-dependent parallel sums.
- `unsafe` only in short, commented system-call or mmap blocks.
- Every public item has a doc comment (`cargo doc --open`).
- The pack format is versioned by its JSON header. Bump it and keep a reader for old
  packs if you change the record layout.

## Docs

`docs/` contains architecture, memory tiering, the CLI, models, testing, a Rust primer and
the roadmap (prose with ASCII diagrams). `docs/diagrams.md` has the Mermaid versions used
by the project page and paper.
