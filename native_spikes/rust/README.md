# Rust P1 batch-kernel comparison

This dependency-free Rust crate implements the same versioned C ABI and P1
diffusion output contract as the C++20 kernel. It exists to make the production
language decision from measured evidence rather than preference.

```text
cargo test --manifest-path native_spikes/rust/Cargo.toml
cargo build --release --manifest-path native_spikes/rust/Cargo.toml
```

The `cdylib` artifact can be supplied to
`benchmarks/language_kernel_comparison.py`. Promotion requires complete COO and
load equality against the NumPy oracle, sanitizer/interpreter checks, all
Tier-1 targets, packaging cost, and an ecosystem decision—not timing alone.
