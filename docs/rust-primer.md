# Rust primer for this codebase

New to Rust? This page covers every language feature Hale-Token relies on,
with a pointer to where it is used. Read it alongside the code.

## Ownership and borrowing

Every value has one owner; other code *borrows* it with `&T` (read-only,
many at once) or `&mut T` (read-write, exclusive).

```rust
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32])
```

`x` and `weight` are borrowed for reading and `out` for writing. Nothing is
copied. (`src/ops/norm.rs`)

## Slices

`&[f32]` is a view of contiguous floats: a pointer plus a length. Most
kernels take slices, so they work for `Vec`s, arrays and memory maps alike.
`chunks_exact(n)` walks a slice in fixed-size pieces, which is how rows,
blocks and heads are processed.

## `Arc`: shared ownership across threads

`Arc<T>` is a reference-counted pointer. Cloning it is cheap (it bumps a
counter), and the value is freed when the last clone is dropped.
Hale-Token uses it for:

- expert weights handed out by the cache (`Arc<Expert>`), so an expert
  evicted from the LRU stays alive while a layer is still using it;
- memory maps shared by many tensors (`ByteBuf`).

## `Mutex`: shared mutable state

`ExpertCache` is used from many threads but must update its LRU. A `Mutex`
gives one thread at a time `&mut` access; `lock()` returns a guard that
unlocks when it goes out of scope. The cache keeps these sections short and
never holds a lock during SSD I/O.

## Traits: interfaces

A trait lists methods a type promises to provide:

```rust
pub trait ExpertSource: Send + Sync {
    fn load(&self, key: ExpertKey) -> Result<Expert>;
    ...
}
```

`Send + Sync` means "safe to move to and share between threads".
`Box<dyn ExpertSource>` and `Arc<dyn ExpertSource>` hold *some* type that
implements the trait, chosen at run time (dynamic dispatch). This is how the
engine swaps an SSD pack for a checkpoint without the model knowing.

## Enums and `match`

Rust enums can carry data, and `match` must handle every case:

```rust
pub enum FeedForward {
    Dense(Expert),
    Moe(MoeLayer),
}
```

(`src/model/transformer.rs`). Adding a variant makes the compiler point at
every `match` that needs updating.

## Errors: `Result` and `?`

Fallible functions return `Result<T, HaleError>`. The `?` operator returns
early with the error if there is one. `HaleError` (`src/error.rs`) uses the
`thiserror` crate to generate readable messages.

## Iterators

Chains such as

```rust
row.chunks_exact(34).zip(x.chunks_exact(32)).map(...).sum()
```

compile to tight loops with no allocation. `enumerate()` adds an index, and
`zip` walks two sequences in lockstep.

## Parallelism with rayon

Replacing `iter()` with `par_iter()` spreads work over all cores:

```rust
out.par_iter_mut().enumerate().for_each(|(r, o)| *o = dot(row(r), x));
```

rayon's work stealing balances load across Apple's performance and
efficiency cores automatically.

## `unsafe`

`unsafe` marks code the compiler cannot check. Hale-Token uses it only for:

- `Mmap::map`: mapping a file into memory (it is only unsound if another
  process modifies the file);
- `libc::fcntl(F_NOCACHE)` and `libc::sysctlbyname`: macOS system calls.

Each block has a `// SAFETY:` comment explaining why it is sound.

## `#[cfg(...)]`: conditional compilation

```rust
#[cfg(target_os = "macos")]
fn disable_os_cache(file: &File) { ... }
```

Only one version is compiled for each platform. This keeps the code building
on Linux CI while using macOS-only features on a Mac.

## Tests

`#[cfg(test)] mod tests { #[test] fn ... }` at the bottom of a file holds
its unit tests. Files in `tests/` are integration tests that use the crate
the way a user would. Run everything with `cargo test`.

## Where to start reading

1. `src/lib.rs`: the module map.
2. `src/engine.rs`: how the pieces fit together.
3. `src/model/transformer.rs`: the forward pass.
4. `src/expert/cache.rs`: the tiered cache.
5. `src/tensor/quant.rs`: the kernels.
