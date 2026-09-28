---
name: hale-token-perf
description: Measure and improve Hale-Token performance — tokens/s, expert hit rate, SSD throughput, memory bandwidth, kernel speed — and tune the planner and tiering (RAM budget, pinned vs LRU split, routing statistics, quantization). Use when asked why decoding is slow, to benchmark a Mac, or to change the cache or planner policy.
---

# Performance workflow

## Measure first

```bash
hale sysinfo                                   # chip, cores, RAM, detected bandwidth
hale bench memory                              # achievable GB/s (CPU)
hale bench ssd models/X-q4/experts.hpk         # random 16 KiB-aligned expert reads
hale bench kernels                             # q4_0 / q8_0 / f16 mat-vec
hale plan --preset qwen3-235b --ssd-gbps <measured>
hale run models/X-q4 -p "..." -n 128           # prints tok/s, expert hit rate, SSD bytes
```

GitHub macOS runners are shared 3-vCPU M1 VMs (7–10 GB/s measured memory bandwidth versus
68 GB/s on a physical M1). Treat CI numbers as lower bounds.

## Model of where time goes (`src/expert/planner.rs`)

```
t_token = max(t_memory, t_compute) + t_ssd
t_memory = (dense_bytes + hit · active_expert_bytes) / (mem_bw · 0.6)
t_ssd    = (1 − hit) · active_expert_bytes / ssd_bw
```

SSD time is additive because a layer cannot start before its experts arrive. The biggest
levers, in order:
1. **Quantize experts** (`hale convert --expert-dtype q4_0`): 3.5× fewer bytes than bf16.
2. **Raise the hit rate**: run a representative prompt once so `hale-routing.json` pins the
   hot experts, or give more RAM (`--expert-ram-gb`).
3. **Keep the NVMe queue full**: misses in one layer are read in parallel (`rayon` +
   `pread`). Don't serialise them.

## Placement policy invariants

- Everything fits → pin everything.
- Budget < 2 × `moe_layers × k` experts → pin all slots, **no LRU**. Token access is
  cyclic over layers, so a too-small LRU thrashes. This was measured: 0% hits on
  TinyMixtral at 25% cache under pure LRU, against about 18% when pinned.
- Otherwise → half pinned, half LRU.

Any policy change must keep `tests/tiered_cache.rs` bit-identical, and should report hit
rate and SSD bytes before and after on the TinyMixtral recipe from `docs/testing.md`.

## Kernels

Quantized mat-vec quantizes the activation to 8 bits once per call and uses integer dot
products (NEON `sdot` with the `apple-m1` baseline). A new kernel needs a unit test
against the f32 reference and a `hale bench kernels` entry. The Metal backend is on the
roadmap.
