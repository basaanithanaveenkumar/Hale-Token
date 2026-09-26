# Supported models

## Runs today

| Family | `architectures` | Examples | Notes |
|---|---|---|---|
| Qwen3-MoE | `Qwen3MoeForCausalLM` | Qwen3-30B-A3B, Qwen3-235B-A22B (and Instruct/Thinking variants) | QK-norm, 128 experts, top-8; dense layers via `mlp_only_layers` / `decoder_sparse_step` are supported |
| Mixtral | `MixtralForCausalLM` | Mixtral-8x7B, Mixtral-8x22B, TinyMixtral | 8 experts, top-2 |

Checkpoints must be safetensors with `f32`, `f16` or `bf16` weights (the
standard Hugging Face releases). Pre-quantized formats (GPTQ, AWQ, FP8,
NVFP4, GGUF) are not read; convert from the bf16 release with `hale convert`.

## Planned and estimated only

`hale plan` knows the shapes of DeepSeek-V3 (671B) and Kimi-K2 (1T) for
capacity planning, but `hale run` cannot execute them yet: they use
Multi-head Latent Attention and shared experts. See the [roadmap](roadmap.md).

## Comparison: the best open MoE models on Macs

Estimated decode tokens/s, `q4_0` experts, bf16 dense weights, SSD at 5 GB/s
(regenerate with `hale plan --compare`):

| Model | Total / active | M1 16GB | M2 Pro 32GB | M3 Max 128GB | M4 Pro 64GB | M4 Max 128GB | M5 32GB | M2 Ultra 192GB | M3 Ultra 512GB |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B | 30.5B / 3.3B | 4.9 SSD | 29.4 RAM | 46.7 RAM | 39.5 RAM | 46.7 RAM | 19.7 RAM | 64.6 RAM | 93.4 RAM |
| Qwen3-235B-A22B | 235B / 22.2B | - | 0.6 SSD | 1.3 SSD | 0.7 SSD | 1.3 SSD | 0.5 SSD | 8.1 SSD | 14.1 RAM |
| Mixtral-8x7B | 46.7B / 12.9B | 0.9 SSD | 2.2 SSD | 12.1 RAM | 10.3 RAM | 12.1 RAM | 1.9 SSD | 16.8 RAM | 24.2 RAM |
| Mixtral-8x22B | 141B / 39.2B | - | 0.3 SSD | 4.0 RAM | 0.4 SSD | 4.0 RAM | 0.3 SSD | 5.5 RAM | 8.0 RAM |
| DeepSeek-V3 | 671B / 37.5B | - | - | 0.5 SSD | 0.4 SSD | 0.5 SSD | - | 0.6 SSD | 4.1 SSD |
| Kimi-K2 | 1.03T / 32.6B | - | - | 0.4 SSD | 0.4 SSD | 0.4 SSD | - | 0.5 SSD | 1.0 SSD |

### How to read it

- **RAM**: every expert fits, so speed is bounded by memory bandwidth or CPU
  compute.
- **SSD**: some experts stream from the SSD each token. The figure is a
  *floor*, because it assumes routing is uniform. Real MoE routers
  concentrate on a subset of experts, and the pinned tier learns that subset
  from `hale-routing.json`, so measured hit rates and speeds are higher.
- **-**: the dense weights plus runtime overhead exceed 75% of RAM.

### Assumptions (all in `src/expert/planner.rs`)

| Constant | Value | Why |
|---|---|---|
| `MEMORY_EFFICIENCY` | 0.6 | CPU cores reach about 60% of advertised bandwidth |
| `GFLOPS_PER_PERFORMANCE_CORE` | 24 | sustained quantized mat-vec throughput |
| `GFLOPS_PER_EFFICIENCY_CORE` | 6 | |
| `DEFAULT_SSD_GBPS` | 5.0 | conservative Apple internal NVMe random read |
| `DEFAULT_RAM_FRACTION` | 0.75 | leave room for macOS and apps |
| `RUNTIME_OVERHEAD_BYTES` | 2 GB | activations, KV cache, allocator slack |

Measure your own machine with `hale bench memory|ssd|kernels` and pass
`--ssd-gbps` to `hale plan`.

### Which model should I compare against?

**Qwen3-235B-A22B** is the reference "data-center" MoE for Hale-Token. It is
a frontier-class open model with 128 experts and top-8 routing (the regime
where tiering matters most), its architecture runs today, and its bf16
weights (470 GB) are far beyond any laptop. At q4_0 with SSD tiering it
becomes runnable on a 64-128 GB MacBook Pro.

**Qwen3-30B-A3B** is the everyday model: the same architecture at 1/8 the
size, fully resident on 32 GB+ Macs.
