#!/usr/bin/env python3
"""Cross-check Hale-Token against Hugging Face `transformers` on a real model.

Runs the same token sequence through `transformers` (PyTorch, float32) and
through `hale logits`, then compares every position's logits.

Usage:
    python3 scripts/reference/compare_with_transformers.py MODEL_DIR \
        --hale target/release/hale --prompt "The capital of France is"

Exit code 0 when the implementations agree, 1 otherwise.
Needs: torch, transformers (>= 4.51 for Qwen3-MoE), numpy.
"""

import argparse
import json
import subprocess
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--hale", default="target/release/hale")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--tokens", help="comma-separated ids instead of --prompt")
    ap.add_argument("--max-rel-rms", type=float, default=1e-3)
    ap.add_argument("--hale-model-dir", help="compare a converted copy (e.g. q8_0 pack) instead")
    args = ap.parse_args()

    if args.tokens:
        ids = [int(t) for t in args.tokens.split(",")]
    else:
        ids = AutoTokenizer.from_pretrained(args.model_dir)(args.prompt).input_ids

    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        ref = model(torch.tensor([ids])).logits[0].numpy().astype(np.float64)

    hale_dir = args.hale_model_dir or args.model_dir
    out = subprocess.run(
        [args.hale, "logits", hale_dir, "--tokens", ",".join(map(str, ids))],
        check=True, capture_output=True, text=True,
    )
    got = np.array(json.loads(out.stdout)["logits"], dtype=np.float64)

    assert got.shape == ref.shape, f"shape {got.shape} vs {ref.shape}"
    rel_rms = float(np.sqrt(((got - ref) ** 2).sum() / (ref ** 2).sum()))
    max_abs = float(np.abs(got - ref).max())
    top1 = float((got.argmax(-1) == ref.argmax(-1)).mean())
    report = {
        "model": args.model_dir,
        "hale_model": hale_dir,
        "positions": len(ids),
        "relative_rms_error": rel_rms,
        "max_abs_error": max_abs,
        "top1_agreement": top1,
    }
    print(json.dumps(report, indent=2))
    ok = rel_rms <= args.max_rel_rms and top1 == 1.0
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
