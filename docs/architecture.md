# Architecture

This page explains how the code is organised and why. Read it before
changing the code; read [memory-tiering.md](memory-tiering.md) for the
details of the expert cache.

## Design goals

1. **Correct first.** Every numeric path is covered by tests that compare
   against an independent implementation.
2. **Memory is the bottleneck, not arithmetic.** Decoding one token reads
   every active weight exactly once, so everything is organised around
   moving fewer bytes: zero-copy loading, quantized experts, tiered caching.
3. **Readable by a Rust beginner.** Plain structs, small traits, no macros
   of our own, no lifetimes in public APIs beyond the obvious, `unsafe`
   only in a few commented system calls.

## Module map

```text
                        ┌──────────────┐
                        │   cli/ (bin) │  argument parsing, printing
                        └──────┬───────┘
                               ▼
                        ┌──────────────┐
                        │   engine     │  Facade: wires everything together
                        └──┬───┬───┬───┘
            ┌──────────────┘   │   └───────────────┐
            ▼                  ▼                   ▼
     ┌────────────┐     ┌────────────┐      ┌────────────┐
     │  generate  │     │   model    │      │  tokenizer │
     │ sampler,   │────►│ attention, │      └────────────┘
     │ decode loop│     │ moe, kv    │
     └────────────┘     └─────┬──────┘
                              │ depends on traits only
                 ┌────────────┴─────────────┐
                 ▼                          ▼
         ┌──────────────┐          ┌─────────────────┐
         │ loader       │          │ expert          │
         │ TensorSource │          │ ExpertProvider  │◄── planner ◄── hardware
         │ (safetensors)│          │ ExpertCache/LRU │
         └──────┬───────┘          │ Pack/Checkpoint │
                │                  └────────┬────────┘
                ▼                           ▼
         ┌─────────────────────────────────────────┐
         │ tensor: DType, ByteBuf, WeightMatrix,   │
         │         quant kernels      ops: norm,   │
         │                            rope, top-k  │
         └─────────────────────────────────────────┘
```

Dependencies only point downwards. `tensor` and `ops` know nothing about
models; `model` knows nothing about files or caches.

## One token, end to end

```text
Transformer::forward(token, cache)
 ├─ embeddings.row_to_f32(token)                     tensor/matrix.rs
 ├─ Rope::angles(position)                           ops/rope.rs (once per token)
 ├─ for each DecoderLayer                            model/transformer.rs
 │   ├─ rms_norm                                     ops/norm.rs
 │   ├─ Attention::forward                           model/attention.rs
 │   │   ├─ q/k/v = W · x          (WeightMatrix::matvec, parallel over rows)
 │   │   ├─ QK-norm + RoPE per head
 │   │   ├─ KvCache.push(k, v)                       model/kv_cache.rs
 │   │   └─ softmax(q·K) · V per head (parallel over heads), o_proj
 │   ├─ rms_norm
 │   └─ FeedForward
 │       ├─ Dense(Expert)  -> SwiGLU MLP
 │       └─ Moe(MoeLayer)                            model/moe.rs
 │           ├─ router · x -> softmax -> top_k (heap)
 │           ├─ ExpertProvider::fetch(keys)          expert/cache.rs
 │           │    pinned? -> LRU? -> SSD pread (parallel)
 │           └─ Σ weight_i · Expert_i(x)
 └─ final rms_norm -> lm_head · x -> logits
```

## SOLID in practice

| Principle | Where |
|---|---|
| **S**ingle responsibility | `LruCache` only orders keys; `ExpertCache` only decides tiers; `PackExpertSource` only reads bytes; `planner` only does arithmetic. |
| **O**pen/closed | New file format: implement `TensorSource`. New model family with the same maths: implement `TensorNaming`. New sampling strategy: implement `Sampler`. Nothing else changes. |
| **L**iskov substitution | Any `ExpertSource` (mmap checkpoint, SSD pack, a test double) can be put behind `ExpertCache`; the litmus tests rely on this to prove SSD == RAM. |
| **I**nterface segregation | The model sees only `ExpertProvider::fetch`; it cannot touch tiers, files or statistics. |
| **D**ependency inversion | `Transformer` holds an `Arc<dyn ExpertProvider>` and loads through `&dyn TensorSource`; the `Engine` decides the concrete types. |

Object-oriented patterns used: **Facade** (`Engine`), **Strategy**
(`Sampler`, `ExpertSource`), **Factory** (`SamplingParams::build`,
`naming_for`), **Proxy/Decorator** (`ExpertCache` wraps an `ExpertSource`
and adds caching).

## Data structures and algorithms

| Problem | Choice | Cost | File |
|---|---|---|---|
| Keep recently used experts, evict the oldest | Hash map + doubly linked list stored in a `Vec` slab, byte-weighted | O(1) get/insert/evict, no allocation in steady state | `expert/lru.rs` |
| Router picks k of n experts | Bounded min-heap of size k | O(n log k) | `ops/topk.rs` |
| Pick the hottest experts to pin | Quickselect (`select_nth_unstable_by`) and then sort the winners | O(n + p log p) | `expert/stats.rs` |
| Find an expert on disk | Fixed-stride records, offset = base + index × stride | O(1), no index | `expert/pack.rs` |
| Tensor lookup across shards | Hash map name → (shard, offset) | O(1) | `loader/safetensors.rs` |
| Share weights without copying | Reference-counted windows into mmaps (`ByteBuf`) | O(1) clone/slice | `tensor/buffer.rs` |
| Sampling from a large vocabulary | Quickselect top-k, then sort only the survivors | O(V + k log k) | `generate/sampler.rs` |
| Attention history | Append-only contiguous KV arrays per layer | O(1) append, sequential scan | `model/kv_cache.rs` |

## Concurrency

- **rayon** provides data parallelism: mat-vec rows, attention heads,
  parallel SSD reads of cache misses, parallel pinning at start-up, and
  parallel expert conversion.
- The cache uses two short `Mutex` critical sections per `fetch`: one
  lookup pass and one insert pass. SSD reads happen *outside* the locks.
- `pread` (`FileExt::read_exact_at`) takes an explicit offset, so reader
  threads share one file descriptor without a shared cursor.

## Numerics

- Activations and accumulations are `f32`.
- Weights are stored as f32, f16, bf16, q8_0 or q4_0, and decoded inside
  the dot product (they are never expanded in memory).
- For quantized weights the activation vector is quantized to 8 bits once
  per mat-vec and dotted with integer arithmetic (llama.cpp's scheme).
- The code is deterministic: the same inputs give bit-identical outputs
  regardless of cache placement or thread count, because every row and head
  is reduced in a fixed order.
