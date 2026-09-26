# Getting started

## 1. Requirements

- An Apple Silicon Mac (M1, M2, M3, M4 or M5, any tier) running macOS 13+.
  Linux works too, for development.
- Rust 1.80 or newer: `curl https://sh.rustup.rs -sSf | sh`
- Disk space for the model you want to run (see `hale plan`).

## 2. Install

```bash
cargo install --git https://github.com/basaanithanaveenkumar/Hale-Token
# or, from a checkout:
git clone https://github.com/basaanithanaveenkumar/Hale-Token && cd Hale-Token
cargo install --path .
```

This installs one binary, `hale`, into `~/.cargo/bin`. Release builds use
LTO and `-O3`.

Check it:

```bash
hale --version
hale sysinfo
```

## 3. Pick a model

```bash
hale plan --compare                     # all presets on a range of Macs
hale plan --preset qwen3-30b            # on *this* Mac
hale plan --preset qwen3-235b --chip "M4 Max" --ram-gb 128
```

Good first choices:

| Mac RAM | Try |
|---|---|
| 16 GB | `Qwen/Qwen3-30B-A3B` (streams from SSD) |
| 32-64 GB | `Qwen/Qwen3-30B-A3B` (fully in RAM at q4_0) |
| 128 GB | `mistralai/Mixtral-8x22B-Instruct-v0.1` or `Qwen/Qwen3-235B-A22B` (SSD tiering) |
| 192 GB+ | `Qwen/Qwen3-235B-A22B` |

## 4. Download

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir models/Qwen3-30B-A3B \
    --include "*.json" "model*.safetensors" "tokenizer*"
```

The `--include` filter skips duplicate formats some repositories ship (for
example Mistral's `consolidated.*` files). Hale-Token only reads the shards
that `model.safetensors.index.json` lists.

## 5. Convert (recommended)

```bash
hale convert models/Qwen3-30B-A3B models/Qwen3-30B-A3B-q4 --expert-dtype q4_0
hale info models/Qwen3-30B-A3B-q4
```

- `q4_0`: about 4.5 bits per weight; the smallest and fastest. Good for large
  models.
- `q8_0`: about 8.5 bits per weight; practically lossless.
- `bf16`: the original precision.

Once converted, the original download is no longer needed.

## 6. Run

```bash
hale run models/Qwen3-30B-A3B-q4 -p "Write a haiku about unified memory."
hale run models/Qwen3-30B-A3B-q4 -p "2+2=" --temperature 0 -n 16      # greedy
hale run models/Qwen3-30B-A3B-q4 -p "..." --expert-ram-gb 8           # cap RAM for experts
```

The first line printed shows the placement, and the last line shows speed
and the measured expert hit rate. Each run updates
`models/.../hale-routing.json`, so later starts pin better experts.

## 7. Use it from Rust

```toml
[dependencies]
hale-token = { git = "https://github.com/basaanithanaveenkumar/Hale-Token" }
```

See the example in the [README](../README.md#using-it-as-a-library) and
`cargo doc --open`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `unsupported model architecture` | Only Qwen3-MoE and Mixtral run today; see [models.md](models.md). |
| Very slow, hit rate near 0% | The RAM budget is too small. Check `hale plan`, close apps, or use q4_0. |
| `context overflow` | Prompt + `-n` exceed the model's `max_position_embeddings`. |
| `no tokenizer.json` | Download the tokenizer files too (`huggingface-cli` gets them by default). |
