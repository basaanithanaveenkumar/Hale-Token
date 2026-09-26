# Roadmap

Ordered by expected impact on tokens/second for large models.

## 1. Metal GPU backend
The M-series GPU reaches more of the unified-memory bandwidth than the CPU
cores do, especially on Max and Ultra chips. The mat-vec kernel is the only
hot spot, and it sits behind `WeightMatrix::matvec`. A Metal implementation
can read the same `ByteBuf` bytes without copies, because memory is unified.
Plan: a `Backend` trait with CPU and Metal implementations, and experts
wrapped in `MTLBuffer`s with `storageModeShared`.

## 2. Expert prefetching
Predict layer *l+1*'s experts from layer *l*'s hidden state (the router of
layer *l+1* applied early is a strong predictor) and start SSD reads while
layer *l* computes, overlapping I/O with compute.

## 3. Batched prefill
Prefill currently runs token by token. Grouping prompt tokens by expert
would read each expert once per chunk instead of once per token.

## 4. More architectures
- DeepSeek-V3 / Kimi-K2: Multi-head Latent Attention, shared experts,
  sigmoid routing with group-limited top-k.
- Qwen2-MoE / OLMoE / gpt-oss.

## 5. Quantization
- Quantize dense weights (attention, LM head) to q8_0 to cut dense bytes 2x.
- Add k-quants (q4_K) for better quality at 4 bits.

## 6. Server
An OpenAI-compatible HTTP API (`hale serve`) with a radix prefix cache for
multi-turn chats, as FreeToken has.
