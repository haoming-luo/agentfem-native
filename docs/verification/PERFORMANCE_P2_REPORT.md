# Performance P2 verification report

Date: 2026-09-04
Status: locally verified candidate; hosted three-platform evidence pending

## Outcome

AgentFEM Native now packages its dependency-free C++20 P1 volume-assembly
kernel in a CPython 3.11+ stable-ABI wheel. `auto` selects it for supported
static fields, falls back to bounded vectorized NumPy when it is unavailable,
and retains the readable element oracle for callable semantics. The C ABI is
versioned and separately specified.

A dependency-free Rust implementation executed the same ABI workload. C++20,
Rust, and vectorized NumPy emitted exactly identical COO row/column/data and
load arrays for the admitted structured cases. C++20 was faster than Rust at
all three recorded scales, so ADR-0016 selects C++20 as the single production
compiled-kernel direction.

## Recorded local evidence

Environment: macOS arm64, Apple Clang 21.0.0, rustc 1.98.1, Python 3.12.13,
NumPy 2.3.5, and SciPy 1.18.1.

| Workload | Median result | Equivalence |
|---|---:|---|
| Raw 524,288-cell C++ vs Rust | C++ 1.26x faster | exact full outputs |
| Raw 524,288-cell C++ vs NumPy | C++ 31.21x faster | exact full outputs |
| Packaged 524,288-cell native vs vectorized | native 13.85x faster | exact full outputs |
| 32,768-cell assembly + SciPy solve | native 1.28x faster | exact solution, reaction, energy |

The locally built `cp311-abi3` macOS arm64 wheel was produced with Python 3.11,
installed under Python 3.12, imported its bundled `_p1_native.abi3` extension,
and passed all 83 tests. C++ optimized/sanitizer tests, C-header smoke tests,
Rust formatting/lint/tests, ABI equivalence, package checks, and SciPy-provider
equivalence also pass locally.

## Claim boundary

These data demonstrate local correctness, packaging feasibility, Python-minor
ABI reuse, and material speedup. They do not claim native Windows, Linux, or
cross-machine superiority until hosted workflows produce artifacts and
evidence on those systems. Raw repetitions, tool versions, digests, and
caveats are in
`benchmarks/results/2026-09-04-darwin-arm64-performance-p2.json`.
