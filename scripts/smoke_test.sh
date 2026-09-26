#!/usr/bin/env bash
# Smoke test: install the `hale` binary with cargo and exercise every command
# on the tiny fixture models. Runs in seconds; used locally and in CI.
#
#   ./scripts/smoke_test.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

step() { printf '\n==> %s\n' "$*"; }
fail() { echo "SMOKE TEST FAILED: $*" >&2; exit 1; }

step "cargo install"
cargo install --quiet --locked --path "$ROOT" --root "$WORK/install"
HALE="$WORK/install/bin/hale"
"$HALE" --version

step "sysinfo"
"$HALE" sysinfo

for MODEL in tiny-qwen3-moe tiny-mixtral; do
  SRC="$ROOT/tests/fixtures/$MODEL"

  step "$MODEL: info"
  "$HALE" info "$SRC" | tee "$WORK/info.txt"
  grep -q "per layer" "$WORK/info.txt" || fail "info output"

  step "$MODEL: convert to q4_0 pack"
  "$HALE" convert "$SRC" "$WORK/$MODEL-q4" --expert-dtype q4_0
  test -s "$WORK/$MODEL-q4/experts.hpk" || fail "no expert pack"

  step "$MODEL: run (streaming experts from SSD with a 10 KB cache)"
  "$HALE" run "$WORK/$MODEL-q4" -p "hello world" -n 8 --temperature 0 \
      --expert-ram-gb 0.00001 --json-stats | tee "$WORK/run.txt"
  tail -n 1 "$WORK/run.txt" | grep -q '"generated_tokens"' || fail "no stats json"

  step "$MODEL: logits are deterministic across placements"
  "$HALE" logits "$WORK/$MODEL-q4" --tokens 1,2,3,4 --expert-ram-gb 1 > "$WORK/a.json"
  "$HALE" logits "$WORK/$MODEL-q4" --tokens 1,2,3,4 --expert-ram-gb 0.00001 > "$WORK/b.json"
  cmp -s "$WORK/a.json" "$WORK/b.json" || fail "RAM vs SSD logits differ"
done

step "plan"
"$HALE" plan --preset qwen3-235b --chip "M4 Max" --ram-gb 128
"$HALE" plan --compare

step "bench"
"$HALE" bench kernels
"$HALE" bench memory --size-mb 256

printf '\nSMOKE TEST PASSED\n'
