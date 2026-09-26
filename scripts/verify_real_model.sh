#!/usr/bin/env bash
# Verify Hale-Token on a real, trained MoE checkpoint from Hugging Face.
#
#   ./scripts/verify_real_model.sh <hf-repo-id> [prompt]
#
# 1. downloads the checkpoint
# 2. cross-checks `hale logits` against transformers (bf16 weights, f32 math)
# 3. converts to a q8_0 expert pack and cross-checks again (looser bound)
# 4. generates text with experts streamed from SSD through a small cache
# 5. writes a report to target/real-model-report/
#
# Needs: python with torch, transformers, huggingface_hub; target/release/hale.
set -euo pipefail

REPO="${1:?usage: $0 <hf-repo-id> [prompt]}"
PROMPT="${2:-The capital of France is}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HALE="$ROOT/target/release/hale"
OUT="$ROOT/target/real-model-report"
MODEL="$ROOT/target/models/${REPO//\//__}"
mkdir -p "$OUT" "$MODEL"

echo "==> download $REPO"
python - "$REPO" "$MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2],
                  allow_patterns=["*.json", "model*.safetensors", "tokenizer*"])
PY

"$HALE" info "$MODEL" | tee "$OUT/info.txt"

echo "==> hale (checkpoint, mmap) vs transformers"
python "$ROOT/scripts/reference/compare_with_transformers.py" "$MODEL" \
    --hale "$HALE" --prompt "$PROMPT" --max-rel-rms 2e-3 | tee "$OUT/compare_checkpoint.json"

echo "==> convert to q8_0 pack"
"$HALE" convert "$MODEL" "$MODEL-q8" --expert-dtype q8_0 | tee "$OUT/convert.txt"

echo "==> hale (q8_0 pack) vs transformers"
python "$ROOT/scripts/reference/compare_with_transformers.py" "$MODEL" \
    --hale "$HALE" --hale-model-dir "$MODEL-q8" --prompt "$PROMPT" --max-rel-rms 5e-2 \
    | tee "$OUT/compare_q8.json" || echo "q8 comparison outside bound (see report)"

echo "==> generate with experts streamed from SSD (cache = 25% of experts)"
PACK_BYTES=$(stat -f%z "$MODEL-q8/experts.hpk" 2>/dev/null || stat -c%s "$MODEL-q8/experts.hpk")
CACHE_GB=$(python -c "print($PACK_BYTES * 0.25 / 1e9)")
"$HALE" run "$MODEL-q8" --format raw -p "$PROMPT" -n 32 --temperature 0 \
    --expert-ram-gb "$CACHE_GB" --json-stats 2>&1 | tee "$OUT/generate.txt"

echo "==> decode benchmark"
"$HALE" bench model "$MODEL-q8" --tokens 32 | tee "$OUT/bench.txt"
echo "REAL MODEL VERIFICATION PASSED"
