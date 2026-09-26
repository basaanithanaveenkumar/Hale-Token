# Contributing

Thanks for helping! This is how changes get in.

## Before you start
- Read [docs/architecture.md](docs/architecture.md).
- Open an issue for anything bigger than a bug fix.

## Checks (CI runs the same)
```bash
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo test
./scripts/smoke_test.sh
```

## Rules
- **Model maths changes need a litmus test.** Extend
  `scripts/reference/make_fixtures.py` if the reference needs a new feature,
  regenerate the fixtures, and make `tests/litmus.rs` cover it.
- **Performance changes need numbers.** Post `hale bench kernels` or
  `hale bench model` before and after, including the chip you ran on.
- **Keep it readable.** Doc comments on public items; comments explain *why*,
  not *what*. Prefer a plain struct and a small trait over generics
  gymnastics.
- **No new `unsafe`** without a `// SAFETY:` comment and a reviewer's
  explicit OK.

## Commits
[Conventional Commits](https://www.conventionalcommits.org/): for example
`feat(expert): prefetch next-layer experts`, `fix(pack): ...`, `docs: ...`.
