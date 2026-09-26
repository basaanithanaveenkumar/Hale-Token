# CLI reference

```text
hale <COMMAND> [OPTIONS]
```

| Command | Purpose |
|---|---|
| `hale run` | Generate text |
| `hale convert` | Hugging Face checkpoint → quantized Hale model with an SSD expert pack |
| `hale info` | Architecture, parameter counts, pack details |
| `hale plan` | RAM/SSD placement plan and tokens/s estimate |
| `hale bench` | Memory, SSD, kernel and end-to-end benchmarks |
| `hale logits` | Dump logits as JSON (verification) |
| `hale sysinfo` | Detected chip and memory |

Every command supports `--help`.

## hale run

```bash
hale run <MODEL> -p <PROMPT> [options]
```

| Flag | Default | Meaning |
|---|---|---|
| `-p, --prompt` | required | The user message |
| `-n, --max-tokens` | 256 | Maximum new tokens |
| `--temperature` | 0.7 | 0 = greedy |
| `--top-p` | 0.9 | Nucleus sampling mass |
| `--top-k` | 40 | Keep the k most likely tokens (0 = off) |
| `--seed` | 42 | Sampling seed (runs are reproducible) |
| `--format` | auto | `auto`, `raw`, `chatml` (Qwen), `mistral` |
| `--expert-ram-gb` | planned | RAM for experts (pinned + LRU) |
| `--ram-fraction` | 0.75 | Share of installed RAM the planner may use |
| `--no-save-stats` | off | Do not update `hale-routing.json` |
| `--json-stats` | off | Print final statistics as one JSON line |

Generated text goes to stdout; the placement and speed lines go to stderr.

## hale convert

```bash
hale convert <INPUT_DIR> <OUTPUT_DIR> [--expert-dtype q4_0|q8_0|bf16|f16|f32]
```

Writes `config.json`, the tokenizer files, `dense.safetensors` and
`experts.hpk`. The input is never modified.

## hale info

```bash
hale info <MODEL>
```

## hale plan

```bash
hale plan --preset <NAME> [--chip <CHIP>] [--ram-gb N] [--ssd-gbps X] [--expert-dtype q4_0]
hale plan --model <DIR> ...
hale plan --compare [--expert-dtype q4_0]
```

Presets: `qwen3-30b`, `qwen3-235b`, `mixtral-8x7b`, `mixtral-8x22b`,
`deepseek-v3`, `kimi-k2` (matched by prefix, case-insensitive).
Chips: `M1` … `M5`, with `Pro`, `Max` and `Ultra` variants; spellings
such as `m4-max` or `Apple M4 Max` work. Without `--chip`, the current machine
is used.

## hale bench

```bash
hale bench memory [--size-mb 1024]
hale bench ssd <FILE> [--chunk-mb 8] [--threads 8]
hale bench kernels
hale bench model <MODEL> [--tokens 32] [--expert-ram-gb N]
```

## hale logits

```bash
hale logits <MODEL> --tokens 1,2,3        # or --text "hello"
```

Prints `{"tokens": [...], "logits": [[...], ...]}`, with one logits row per
input position (teacher-forced). Used by
`scripts/reference/compare_with_transformers.py`.

## hale sysinfo

Prints the CPU brand string, the matched Apple chip and its specs, the core
counts and the installed memory.
