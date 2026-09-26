#!/usr/bin/env python3
"""Generate tiny MoE checkpoints and golden outputs for Hale-Token's tests.

This is an *independent* reference implementation of Qwen3-MoE and Mixtral
written with NumPy in float64. It deliberately uses a different algorithm
from the Rust engine (full-sequence causal attention instead of an
incremental KV cache) so that agreement between the two is meaningful.

For each architecture it writes, under tests/fixtures/<name>/:

    config.json          Hugging Face style config
    model.safetensors    random weights (written by our own tiny writer)
    tokenizer.json       small byte-level BPE tokenizer (smoke tests)
    expected.json        token ids, per-position logits, greedy continuation

Usage:  python3 scripts/reference/make_fixtures.py
Needs:  numpy, tokenizers (pip install numpy tokenizers)
"""

import json
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
VOCAB = 272
PROMPT = [1, 17, 42, 99, 5, 200, 7, 64, 128, 3]
CONTINUATION = 12


# --------------------------------------------------------------------------
# bf16 helpers (NumPy has no bfloat16)
# --------------------------------------------------------------------------

def to_bf16_bits(x):
    """Round float32 -> bfloat16 (round-to-nearest-even), return uint16 bits."""
    bits = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounding = ((bits >> 16) & 1) + 0x7FFF
    return ((bits + rounding) >> 16).astype(np.uint16)


def bf16_bits_to_f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


# --------------------------------------------------------------------------
# Minimal safetensors writer (independent of the Rust one)
# --------------------------------------------------------------------------

def write_safetensors(path, tensors):
    """tensors: dict name -> (dtype_str, shape, raw_bytes)."""
    header, offset, blobs = {}, 0, []
    for name in sorted(tensors):
        dtype, shape, raw = tensors[name]
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        blobs.append(raw)
    header["__metadata__"] = {"format": "pt"}
    js = json.dumps(header).encode()
    js += b" " * (-len(js) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(js)))
        f.write(js)
        for b in blobs:
            f.write(b)


# --------------------------------------------------------------------------
# Reference model (float64, full-sequence attention)
# --------------------------------------------------------------------------

def rms_norm(x, w, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def rope(x, positions, theta):
    """x: [seq, heads, head_dim]; Hugging Face rotate-half convention."""
    d = x.shape[-1]
    inv_freq = 1.0 / theta ** (np.arange(0, d, 2, dtype=np.float64) / d)
    ang = np.outer(positions, inv_freq)  # [seq, d/2]
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    a, b = x[..., : d // 2], x[..., d // 2 :]
    return np.concatenate([a * cos - b * sin, b * cos + a * sin], axis=-1)


def swiglu(x, gate, up, down):
    g = x @ gate.T
    return ((g / (1.0 + np.exp(-g))) * (x @ up.T)) @ down.T


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def forward(cfg, W, names, tokens):
    """Returns logits [seq, vocab] for the whole token sequence."""
    h, nh, nkv, hd = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    eps, theta = cfg["rms_norm_eps"], cfg["rope_theta"]
    seq = len(tokens)
    pos = np.arange(seq)
    x = W["model.embed_tokens.weight"][tokens]
    mask = np.triu(np.full((seq, seq), -np.inf), 1)
    for l in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{l}."
        a = rms_norm(x, W[p + "input_layernorm.weight"], eps)
        q = (a @ W[p + "self_attn.q_proj.weight"].T).reshape(seq, nh, hd)
        k = (a @ W[p + "self_attn.k_proj.weight"].T).reshape(seq, nkv, hd)
        v = (a @ W[p + "self_attn.v_proj.weight"].T).reshape(seq, nkv, hd)
        if names["qk_norm"]:
            q = rms_norm(q, W[p + "self_attn.q_norm.weight"], eps)
            k = rms_norm(k, W[p + "self_attn.k_norm.weight"], eps)
        q, k = rope(q, pos, theta), rope(k, pos, theta)
        group = nh // nkv
        k, v = np.repeat(k, group, axis=1), np.repeat(v, group, axis=1)
        scores = np.einsum("qhd,khd->hqk", q, k) / np.sqrt(hd) + mask
        ctx = np.einsum("hqk,khd->qhd", softmax(scores), v).reshape(seq, nh * hd)
        x = x + ctx @ W[p + "self_attn.o_proj.weight"].T

        m = rms_norm(x, W[p + "post_attention_layernorm.weight"], eps)
        if l in names["dense_layers"]:
            d = p + "mlp."
            x = x + swiglu(m, W[d + "gate_proj.weight"], W[d + "up_proj.weight"], W[d + "down_proj.weight"])
            continue
        probs = softmax(m @ W[names["router"](l)].T)
        out = np.zeros_like(m)
        for t in range(seq):
            # Stable sort: ties go to the lower expert index, like the Rust top-k.
            order = np.argsort(-probs[t], kind="stable")[: cfg["num_experts_per_tok"]]
            w = probs[t, order]
            if names["normalize"]:
                w = w / w.sum()
            for e, we in zip(order, w):
                g, u, dn = (W[names["expert"](l, e, j)] for j in ("gate", "up", "down"))
                out[t] += we * swiglu(m[t], g, u, dn)
        x = x + out
    x = rms_norm(x, W["model.norm.weight"], eps)
    head = W["model.embed_tokens.weight"] if cfg.get("tie_word_embeddings") else W["lm_head.weight"]
    return x @ head.T


def greedy(cfg, W, names, prompt, n):
    """Greedy continuation; also returns the smallest top-1/top-2 margin."""
    tokens, margin = list(prompt), np.inf
    for _ in range(n):
        logits = forward(cfg, W, names, tokens)[-1]
        top2 = np.sort(logits)[-2:]
        margin = min(margin, top2[1] - top2[0])
        tokens.append(int(np.argmax(logits)))
    return tokens[len(prompt):], margin


# --------------------------------------------------------------------------
# Fixture definitions
# --------------------------------------------------------------------------

def qwen3_moe():
    cfg = {
        "architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe",
        "hidden_size": 64, "intermediate_size": 64, "moe_intermediate_size": 32,
        "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 32, "vocab_size": VOCAB, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
        "max_position_embeddings": 512, "num_experts": 8, "num_experts_per_tok": 3,
        "norm_topk_prob": True, "decoder_sparse_step": 1, "mlp_only_layers": [1],
        "tie_word_embeddings": False, "bos_token_id": 1, "eos_token_id": 2,
    }
    proj = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}
    names = {
        "qk_norm": True, "normalize": True, "dense_layers": {1},
        "router": lambda l: f"model.layers.{l}.mlp.gate.weight",
        "expert": lambda l, e, j: f"model.layers.{l}.mlp.experts.{e}.{proj[j]}.weight",
    }
    return "tiny-qwen3-moe", cfg, names, "F32"


def mixtral():
    cfg = {
        "architectures": ["MixtralForCausalLM"], "model_type": "mixtral",
        "hidden_size": 64, "intermediate_size": 32, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 16,
        "vocab_size": VOCAB, "rms_norm_eps": 1e-5, "rope_theta": 1000000.0,
        "max_position_embeddings": 512, "num_local_experts": 4, "num_experts_per_tok": 2,
        "tie_word_embeddings": False, "bos_token_id": 1, "eos_token_id": 2,
    }
    proj = {"gate": "w1", "down": "w2", "up": "w3"}
    names = {
        "qk_norm": False, "normalize": True, "dense_layers": set(),
        "router": lambda l: f"model.layers.{l}.block_sparse_moe.gate.weight",
        "expert": lambda l, e, j: f"model.layers.{l}.block_sparse_moe.experts.{e}.{proj[j]}.weight",
    }
    return "tiny-mixtral", cfg, names, "BF16"


def random_weights(cfg, names, rng):
    h, nh, nkv, hd = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    n_exp = cfg.get("num_experts", cfg.get("num_local_experts"))
    inter = cfg.get("moe_intermediate_size", cfg["intermediate_size"])
    W = {}

    def mat(name, rows, cols, scale=None):
        W[name] = rng.standard_normal((rows, cols)) * (scale or 1.0 / np.sqrt(cols))

    def vec(name, n):
        W[name] = 1.0 + 0.1 * rng.standard_normal(n)

    mat("model.embed_tokens.weight", VOCAB, h, 1.0)
    mat("lm_head.weight", VOCAB, h, 3.0 / np.sqrt(h))
    vec("model.norm.weight", h)
    for l in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{l}."
        vec(p + "input_layernorm.weight", h)
        vec(p + "post_attention_layernorm.weight", h)
        mat(p + "self_attn.q_proj.weight", nh * hd, h)
        mat(p + "self_attn.k_proj.weight", nkv * hd, h)
        mat(p + "self_attn.v_proj.weight", nkv * hd, h)
        mat(p + "self_attn.o_proj.weight", h, nh * hd)
        if names["qk_norm"]:
            vec(p + "self_attn.q_norm.weight", hd)
            vec(p + "self_attn.k_norm.weight", hd)
        if l in names["dense_layers"]:
            mat(p + "mlp.gate_proj.weight", cfg["intermediate_size"], h)
            mat(p + "mlp.up_proj.weight", cfg["intermediate_size"], h)
            mat(p + "mlp.down_proj.weight", h, cfg["intermediate_size"])
            continue
        mat(names["router"](l), n_exp, h, 2.0 / np.sqrt(h))
        for e in range(n_exp):
            mat(names["expert"](l, e, "gate"), inter, h)
            mat(names["expert"](l, e, "up"), inter, h)
            mat(names["expert"](l, e, "down"), h, inter)
    return W


def encode(W, dtype):
    """Store weights in `dtype`; return (tensors for the file, weights as the file holds them)."""
    stored, exact = {}, {}
    for name, arr in W.items():
        f32 = arr.astype(np.float32)
        if dtype == "BF16":
            bits = to_bf16_bits(f32)
            stored[name] = ("BF16", arr.shape, bits.tobytes())
            exact[name] = bf16_bits_to_f32(bits).astype(np.float64)
        else:
            stored[name] = ("F32", arr.shape, f32.tobytes())
            exact[name] = f32.astype(np.float64)
    return stored, exact


def write_tokenizer(out_dir):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB, special_tokens=specials,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False,
    )
    corpus = ["the quick brown fox jumps over the lazy dog. hello world! mixture of experts on apple silicon."] * 50
    tok.train_from_iterator(corpus, trainer)
    assert tok.get_vocab_size() <= VOCAB, tok.get_vocab_size()
    tok.save(str(out_dir / "tokenizer.json"))


def main():
    for i, make in enumerate([qwen3_moe, mixtral]):
        name, cfg, names, dtype = make()
        out = FIXTURES / name
        out.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(1234 + i)
        stored, W = encode(random_weights(cfg, names, rng), dtype)
        write_safetensors(out / "model.safetensors", stored)
        (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        write_tokenizer(out)

        continuation, margin = greedy(cfg, W, names, PROMPT, CONTINUATION)
        assert margin > 1e-2, f"{name}: greedy margin {margin} too small for a stable test"
        all_tokens = PROMPT + continuation
        logits = forward(cfg, W, names, all_tokens)
        expected = {
            "prompt": PROMPT,
            "greedy_continuation": continuation,
            "min_greedy_margin": float(margin),
            "tokens": all_tokens,
            "logits": [[round(float(v), 6) for v in row] for row in logits],
        }
        (out / "expected.json").write_text(json.dumps(expected) + "\n")
        print(f"{name}: continuation {continuation}, min margin {margin:.4f}")


if __name__ == "__main__":
    main()
