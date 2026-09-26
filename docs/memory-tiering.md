# Memory tiering on Apple Silicon

## Unified memory changes the problem

On a PC, a GPU has its own VRAM, and weights must cross PCIe to reach it.
FreeToken's core problem is scheduling that PCIe traffic. On an Apple
Silicon Mac, the CPU, GPU and Neural Engine share **one pool of unified
memory**, so there is no PCIe hop: once an expert is in RAM, every processor
can read it at full memory bandwidth (68 GB/s on M1 up to 819 GB/s on M3
Ultra).

The remaining boundary is **RAM ↔ SSD**. Apple's internal NVMe SSDs read at
roughly 3-7 GB/s, which is 20-100x slower than unified memory but far faster
than the SATA disks older offloading schemes were built for. This is what
makes "bigger than RAM" MoE models practical on a laptop: per token, only
the *missing* active experts have to come from the SSD.

## The three tiers

| Tier | Contents | Filled | Evicted |
|---|---|---|---|
| 0: pinned | hottest experts according to `hale-routing.json` | at start-up, in parallel | never |
| 1: LRU | experts used recently | on a miss | least-recently-used, by bytes |
| 2: SSD | every expert (`experts.hpk`) | by `hale convert` | - |

`ExpertCache::fetch(keys)` (in `src/expert/cache.rs`):

1. records the routing decision (for next time);
2. looks every key up in tier 0, then tier 1;
3. reads all misses **concurrently** from tier 2 (a full NVMe queue is
   several times faster than one read at a time);
4. inserts the loaded experts into tier 1, evicting as needed.

## Why not just `mmap` the whole checkpoint?

That works, and `hale run` does exactly this on an unconverted Hugging Face
directory. But:

- the OS evicts by *page* recency and knows nothing about experts;
- bf16 experts are 3.5x the bytes of q4_0 experts, both in RAM and on SSD;
- safetensors keeps `gate`, `up` and `down` of one expert in different
  places, so one miss becomes several scattered reads;
- pages served through the page cache *and* copied into our cache would be
  held twice.

## The expert pack (`experts.hpk`)

```text
0        "HALEPACK"
8        u64 header length
16       JSON header (dtype, sizes, list of MoE layers, record size, stride)
...      padding
data     [L0 E0][L0 E1]...[L0 En][L1 E0]...     each record = gate | up | down
```

- **Records are 16 KiB aligned** (Apple Silicon's page size), so every read
  is page-aligned.
- **All records are the same size**, so the offset is
  `data_offset + (moe_layer_ordinal × num_experts + expert) × stride`: O(1)
  and no index to load.
- **One `pread` per miss**: the three matrices of an expert are contiguous.
- **`F_NOCACHE`** is set on the file descriptor on macOS, so the kernel does
  not keep a second copy in the unified buffer cache; our cache decides what
  stays in memory.
- Writers use `write_all_at`, so the converter fills records in parallel and
  in any order.

## The planner

`src/expert/planner.rs` is a pure function from (model shape, hardware,
dtypes) to a placement plan and a speed estimate:

```text
dense_bytes   = dense_params  × bits(dense_dtype)  / 8
expert_bytes  = params/expert × bits(expert_dtype) / 8
ram_budget    = RAM × 0.75
expert_ram    = ram_budget − dense_bytes − 2 GB overhead
in_ram        = min(expert_ram / expert_bytes, total_experts)
hit_rate      ≥ in_ram / total_experts          (uniform-routing floor)

t_memory  = (dense_bytes + hit_rate × active_expert_bytes) / (mem_bw × 0.6)
t_compute = 2 × active_params / cpu_flops
t_ssd     = (1 − hit_rate) × active_expert_bytes / ssd_bw
t_token   = max(t_memory, t_compute) + t_ssd
```

SSD time is added rather than overlapped because a layer cannot start
until its experts have arrived.

Placement policy (`split_budget` and `Engine::choose_pinned`):

1. everything fits → pin everything (no SSD traffic);
2. the RAM budget is smaller than **two tokens' working set**
   (`2 × moe_layers × k` experts) → pin every slot, no LRU. Each token visits
   the layers in the same order, so the access pattern is cyclic, and an
   LRU smaller than the cycle evicts each expert just before it is needed
   again. On Apple Silicon CI, TinyMixtral with a cache of 25% of its experts
   had a **0% hit rate under pure LRU**. A pinned set of the same size hits
   about 25% of the time even with uniform routing;
3. otherwise → half pinned, half LRU.

Pinned experts are the hottest ones from `hale-routing.json`. If there is
no history yet, they are spread evenly across layers, so every layer gets
the same share of hits.

Measure your machine and feed the planner real numbers:

```bash
hale bench memory
hale bench ssd models/Qwen3-235B-A22B-q4/experts.hpk
hale plan --preset qwen3-235b --ssd-gbps 6.1
```

## Tuning tips

- **Quantize experts** (`--expert-dtype q4_0`). It is the single biggest win:
  3.5x fewer bytes than bf16 in RAM *and* on SSD.
- **Run a representative prompt once.** Routing statistics make the pinned
  tier match your workload.
- **Close memory-hungry apps.** macOS compresses and swaps before it
  refuses allocations; the default 75% RAM share leaves room for the system.
- `--expert-ram-gb` overrides the planner if you want a specific budget.
