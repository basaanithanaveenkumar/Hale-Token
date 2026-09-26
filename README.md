<div align="center">

# Hale-Token

**Run data-center scale Mixture-of-Experts LLMs on your Mac.**

A Rust inference engine for Apple Silicon (M1 - M5) that keeps the hot experts
in unified memory and streams the rest from the SSD.

[![CI](https://github.com/basaanithanaveenkumar/Hale-Token/actions/workflows/ci.yml/badge.svg)](https://github.com/basaanithanaveenkumar/Hale-Token/actions/workflows/ci.yml)
![Rust](https://img.shields.io/badge/rust-1.80%2B-orange)
![Platform](https://img.shields.io/badge/platform-Apple%20Silicon%20M1--M5-black)
![License](https://img.shields.io/badge/license-Apache--2.0-blue)

</div>

---

## Why

Frontier open-weight models are Mixture-of-Experts (MoE): Qwen3-235B-A22B has
235 B parameters but touches only 22 B per token; DeepSeek-V3 has 671 B and
touches 37 B. **Most of the model sits idle on every token.** Yet the usual way
to run a model is to load *all* of it into memory, which puts these models out
of reach of a laptop.

Hale-Token treats a Mac as a three-tier memory system and places each expert
where it pays off most:

```text
            ┌──────────────────────────────────────────────┐
  tier 0    │ pinned   hottest experts, never evicted       │  unified memory
  tier 1    │ LRU      recently used experts                │  (CPU & GPU share it)
            ├──────────────────────────────────────────────┤
  tier 2    │ SSD      every expert, page-aligned pack file │  internal NVMe
            └──────────────────────────────────────────────┘
```

A cost model based on your chip's memory bandwidth, SSD speed and RAM decides
how big each tier is. Routing statistics saved from earlier runs choose which
experts get pinned.

It is a Rust re-imagining of [FreeToken](https://github.com/FlashML-org/FreeToken),
which does the same for NVIDIA GPUs + host RAM, adapted to Apple's unified
memory architecture.

## Features

- **Tiered expert cache**: pinned + O(1) byte-budgeted LRU in RAM, the rest
  streamed from SSD with parallel `pread` and `F_NOCACHE` (no double caching).
- **SSD-optimised expert pack**: one contiguous, 16 KiB-aligned record per
  expert, O(1) addressing, and one read per cache miss.
- **Quantization**: `q8_0` (near-lossless) and `q4_0` experts, with integer
  dot-product kernels that compile to NEON `sdot` on every M-series chip.
- **Planner**: `hale plan` estimates tokens/second for any model on any Mac,
  M1 through M5, before you download 100+ GB.
- **Real models**: Qwen3-MoE (30B-A3B, 235B-A22B) and Mixtral (8x7B, 8x22B),
  loaded straight from Hugging Face safetensors (zero-copy mmap).
- **Verified**: logits match an independent reference implementation at
  every position, and streaming from SSD is bit-identical to running fully
  in RAM (see [Verification](#verification)).
- **Readable**: small, modular Rust. `unsafe` appears only in a handful of
  short, commented blocks (mmap and macOS system calls). Every public item
  has a doc comment, and there is a [Rust primer](docs/rust-primer.md) for
  newcomers.

## Quick start

```bash
# 1. Install (Rust 1.80+; https://rustup.rs)
cargo install --git https://github.com/basaanithanaveenkumar/Hale-Token

# 2. Check your machine and what it can run
hale sysinfo
hale plan --preset qwen3-235b          # uses this Mac's chip and RAM

# 3. Get a model (any Qwen3-MoE or Mixtral checkpoint)
huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir models/Qwen3-30B-A3B

# 4. Convert: quantize experts to 4-bit and pack them for the SSD
hale convert models/Qwen3-30B-A3B models/Qwen3-30B-A3B-q4 --expert-dtype q4_0

# 5. Run
hale run models/Qwen3-30B-A3B-q4 -p "Explain mixture-of-experts in two sentences."
```

```text
# on a 64 GB Mac
[hale] experts from SSD pack models/Qwen3-30B-A3B-q4/experts.hpk (q4_0) | pinned 6144 (16.3 GB) | LRU 0 B | RAM holds 100% of experts
...
[hale] prompt 21 tok @ ... tok/s | generated 256 tok @ ... tok/s | expert hit rate 100.0% | SSD 0 B
```

Step 4 is optional: `hale run` also works directly on a Hugging Face
directory, but then experts are memory-mapped and the OS decides what stays
in RAM.

## What can my Mac run?

`hale plan --compare` prints estimated decode speed (tokens/s, `q4_0` experts,
bf16 dense weights). **RAM** means every expert fits in memory; **SSD** means
experts are streamed; `-` means the dense weights alone do not fit.

| Model | Total / active | M1 16GB | M2 Pro 32GB | M4 Pro 64GB | M4 Max 128GB | M2 Ultra 192GB | M3 Ultra 512GB |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B | 30.5B / 3.3B | 4.9 SSD | 29.4 RAM | 39.5 RAM | 46.7 RAM | 64.6 RAM | 93.4 RAM |
| Mixtral-8x7B | 46.7B / 12.9B | 0.9 SSD | 2.2 SSD | 10.3 RAM | 12.1 RAM | 16.8 RAM | 24.2 RAM |
| Mixtral-8x22B | 141B / 39B | - | 0.3 SSD | 0.4 SSD | 4.0 RAM | 5.5 RAM | 8.0 RAM |
| **Qwen3-235B-A22B** | 235B / 22B | - | 0.6 SSD | 0.7 SSD | 1.3 SSD | 8.1 SSD | 14.1 RAM |
| DeepSeek-V3 * | 671B / 37B | - | - | 0.4 SSD | 0.5 SSD | 0.6 SSD | 4.1 SSD |
| Kimi-K2 * | 1.03T / 33B | - | - | 0.4 SSD | 0.4 SSD | 0.5 SSD | 1.0 SSD |

<sub>These are model-based estimates, not measurements. The SSD numbers are a
floor: they assume uniform routing, while real routers are skewed, so the
measured hit rate that `hale run` prints is higher. * DeepSeek-V3 and Kimi-K2
can be planned but not executed yet (MLA attention is on the
[roadmap](docs/roadmap.md)). See [docs/models.md](docs/models.md) for how the
numbers are derived.</sub>

## How it works

```text
 token ─► embed ─► ┌ DecoderLayer x N ───────────────────────────────┐ ─► norm ─► lm_head ─► sample
                   │ x += Attention(RMSNorm(x))   GQA, RoPE, KV cache │
                   │ x += MoE(RMSNorm(x))                             │
                   │      router ─► top-k ─► ExpertProvider.fetch()   │
                   └──────────────────────────────┬──────────────────┘
                                                  ▼
                           ExpertCache: pinned ─► LRU ─► PackExpertSource (SSD)
```

1. **Convert** once: dense weights (attention, norms, embeddings) keep their
   precision in `dense.safetensors`; experts are quantized into
   `experts.hpk`.
2. **Plan** at start-up: the RAM budget, minus dense weights, is split
   between pinned and LRU tiers; pinned experts are loaded in parallel.
3. **Decode**: each MoE layer asks the cache for its top-k experts; misses
   are read from the SSD concurrently, so the NVMe queue stays full.
4. **Learn**: routing counts are saved to `hale-routing.json`, so the next
   start pins the experts your workload actually uses.

Deep dives: [architecture](docs/architecture.md) ·
[memory tiering & pack format](docs/memory-tiering.md).

## Verification

| Level | What | Where |
|---|---|---|
| Unit | 46 tests: kernels, quantization, LRU, top-k, planner, pack format, ... | `src/**` |
| Litmus | Logits at **every position** match an independent NumPy float64 reference (full-sequence attention vs our KV cache), and greedy tokens are identical | `tests/litmus.rs` |
| Tiering | Experts streamed from SSD through a one-expert cache give **bit-identical** logits to fully resident experts; quantized packs stay within bounds | `tests/tiered_cache.rs` |
| CLI | `convert`, `run`, `plan`, `bench`, `logits` end to end | `tests/cli.rs` |
| Smoke | `cargo install` + every command on both fixture models | `scripts/smoke_test.sh` |
| Cross-check | `hale logits` vs Hugging Face `transformers` on fixtures and a real MoE | `scripts/reference/compare_with_transformers.py` |
| Apple Silicon | All of the above on GitHub's arm64 macOS runners | `.github/workflows/ci.yml` |

```bash
cargo test                       # unit + integration + litmus
./scripts/smoke_test.sh          # install and exercise the CLI
./scripts/verify_real_model.sh Isotonic/TinyMixtral-4x248M-MoE   # needs torch
```

See [docs/testing.md](docs/testing.md).

## Supported platforms

- **Target:** every Apple Silicon Mac: M1, M2, M3, M4, M5 and their Pro,
  Max and Ultra variants. Binaries built for `aarch64-apple-darwin` use the
  `apple-m1` CPU baseline, which every later chip supports, so one build runs
  on all of them.
- **Also builds and tests on** Linux (x86_64 / aarch64), for development and CI.

## Documentation

| Doc | For |
|---|---|
| [Getting started](docs/getting-started.md) | Installing, downloading, converting, running |
| [CLI reference](docs/cli.md) | Every command and flag |
| [Supported models](docs/models.md) | Architectures, comparison table, how estimates work |
| [Architecture](docs/architecture.md) | Module map, data flow, SOLID and data-structure choices |
| [Memory tiering](docs/memory-tiering.md) | Unified memory, the pack format, the cost model |
| [Testing](docs/testing.md) | Unit, litmus, smoke and real-model verification |
| [Rust primer](docs/rust-primer.md) | Rust concepts used in this codebase, for beginners |
| [Roadmap](docs/roadmap.md) | Metal backend, batched prefill, more architectures |

API docs: `cargo doc --open`.

## Using it as a library

```rust
use hale::engine::{Engine, EngineOptions};
use hale::generate::{GenerationOptions, SamplingParams};
use hale::tokenizer::PromptFormat;

let engine = Engine::load("models/Qwen3-30B-A3B-q4".as_ref(), EngineOptions::default())?;
let options = GenerationOptions { max_new_tokens: 128, sampling: SamplingParams::default(), stop_tokens: vec![] };
let out = engine.generate_text("Hello!", PromptFormat::ChatMl, &options, |piece| print!("{piece}"))?;
println!("\n{:.1} tok/s, expert hit rate {:.0}%", out.stats.decode_tokens_per_second(), out.stats.cache.hit_rate() * 100.0);
```

## Project layout

```text
src/
  config.rs        config.json -> ModelConfig
  tensor/          dtypes, zero-copy buffers, quantized kernels, mat-vec
  ops/             rmsnorm, rope, softmax/silu, heap top-k
  loader/          safetensors reader/writer (TensorSource trait)
  expert/          Expert, sources (checkpoint, SSD pack), tiered cache, LRU, stats, planner
  model/           attention, KV cache, MoE layer, transformer, tensor naming
  generate/        samplers, decode loop
  tokenizer.rs     HF tokenizers wrapper, streaming decode, chat formats
  hardware/        M1-M5 chip table, sysctl detection, benchmarks
  convert.rs       hale convert
  engine.rs        Engine facade
  cli/             the `hale` binary
tests/             litmus, tiered cache, CLI tests + fixtures
scripts/           smoke test, real-model verification, reference implementation
docs/              guides
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). In short: `cargo fmt`, `cargo clippy
-- -D warnings`, `cargo test`, and a litmus test for any change to model
maths.

## Acknowledgements

Hale-Token's design follows [FreeToken](https://github.com/FlashML-org/FreeToken)
(Yang et al., *FreeToken: Efficient Edge-Native MoE Serving with
Bandwidth-Adaptive Execution*, 2026). The quantized block layouts follow
[llama.cpp / GGML](https://github.com/ggml-org/llama.cpp); tokenization uses
Hugging Face [tokenizers](https://github.com/huggingface/tokenizers).

## License

[Apache License 2.0](LICENSE).
