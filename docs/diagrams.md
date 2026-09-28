# Diagrams

Mermaid versions of the architecture (they render on GitHub). The prose explanations
are in [architecture.md](architecture.md) and [memory-tiering.md](memory-tiering.md).
These figures also appear on the [project page](../project-page/index.html) and in the
[paper](../paper/main.tex).

## 1. Three memory tiers

```mermaid
flowchart TB
  subgraph UM["Unified memory (CPU + GPU share it; 68–819 GB/s)"]
    T0["Tier 0: pinned<br/>hottest experts from hale-routing.json<br/>loaded in parallel at start-up · never evicted"]
    T1["Tier 1: LRU<br/>recently used experts · byte-budgeted · O(1)"]
  end
  subgraph SSD["Internal NVMe (≈3–7 GB/s)"]
    T2["Tier 2: experts.hpk<br/>every expert · 16 KiB-aligned fixed-stride records"]
  end
  REQ["MoE layer asks for top-k experts"] --> T0
  T0 -->|miss| T1
  T1 -->|miss| T2
  T2 -->|"parallel pread, F_NOCACHE"| T1
```

## 2. One token, end to end

```mermaid
flowchart LR
  TOK["token"] --> EMB["embedding row"]
  EMB --> L{"× N decoder layers"}
  L --> A["RMSNorm → GQA attention<br/>QK-norm · RoPE · KV cache"]
  A --> M["RMSNorm → MoE"]
  M --> R["router → softmax → heap top-k"]
  R --> F["ExpertProvider::fetch(keys)"]
  F --> E["Σ wᵢ · SwiGLU expertᵢ(x)"]
  E --> L
  L --> O["final RMSNorm → lm_head → logits → sampler"]
```

## 3. Cache lookup (`ExpertCache::fetch`)

```mermaid
sequenceDiagram
  participant MoE as MoeLayer
  participant C as ExpertCache
  participant S as RoutingStats
  participant P as Pinned tier
  participant L as LRU tier
  participant D as PackExpertSource (SSD)
  MoE->>C: fetch([k experts])
  C->>S: record routing decision
  C->>P: lookup (lock 1)
  C->>L: lookup misses
  par all misses concurrently, outside locks
    C->>D: pread record e1
    C->>D: pread record e2
  end
  D-->>C: gate / up / down (quantized)
  C->>L: insert, evict LRU by bytes (lock 2)
  C-->>MoE: Arc<Expert> × k
```

## 4. Planner (cost model and placement)

```mermaid
flowchart TB
  IN["model shape · chip (bandwidth, cores) · RAM · SSD GB/s · dtypes"] --> B["ram_budget = 0.75 · RAM<br/>expert_ram = budget − dense − 2 GB"]
  B --> H["in_ram = expert_ram / expert_bytes<br/>hit ≥ in_ram / total (uniform floor)"]
  H --> T["t_token = max(t_mem, t_compute) + t_ssd"]
  B --> P{"placement"}
  P -->|"all experts fit"| PA["pin everything"]
  P -->|"budget < 2 × layers × k"| PB["pin all slots, no LRU<br/>(cyclic access defeats a small LRU)"]
  P -->|otherwise| PC["half pinned, half LRU"]
```

## 5. Conversion

```mermaid
flowchart LR
  HF["Hugging Face checkpoint<br/>safetensors bf16 (mmap, zero-copy)"] --> SPLIT{"tensor kind"}
  SPLIT -->|"attention, norms, embeddings, router"| DENSE["dense.safetensors<br/>(original precision)"]
  SPLIT -->|"expert gate/up/down"| Q["quantize q8_0 / q4_0<br/>(parallel, rayon)"]
  Q --> PACK["experts.hpk<br/>HALEPACK · JSON header · records at<br/>base + (layer·E + e) · stride"]
```

## 6. Module dependencies

```mermaid
flowchart TB
  CLI["cli (bin)"] --> ENG["engine (Facade)"]
  ENG --> GEN["generate"]
  ENG --> MOD["model"]
  ENG --> TOKN["tokenizer"]
  GEN --> MOD
  MOD -->|"traits only"| LOAD["loader: TensorSource"]
  MOD -->|"traits only"| EXP["expert: ExpertProvider · ExpertCache · LRU · Pack"]
  PLAN["planner"] --> EXP
  HW["hardware"] --> PLAN
  LOAD --> TEN["tensor + ops"]
  EXP --> TEN
```
