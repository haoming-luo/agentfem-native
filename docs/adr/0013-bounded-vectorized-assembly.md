# ADR-0013: bounded vectorized assembly beside the mathematical oracle

Status: accepted — 2026-09-04

AgentFEM Native keeps two serial assembly implementations with different jobs:

- `reference` evaluates each element through the readable geometry,
  quadrature, basis, and material functions and remains the executable oracle;
- `vectorized` batches static P1 cells, computes the affine inverse analytically,
  preallocates deterministic COO arrays, and limits temporary storage by a
  fixed chunk size;
- `auto` selects vectorized assembly only for constant scalar/tensor source and
  conductivity data whose call semantics cannot change. Callable fields
  conservatively remain on the reference path.

Both paths must emit identical COO row/column ordering. Matrix, load, solution,
energy, reaction, and null-mode differences have fixed test tolerances. Chunk
size must not change numerical meaning. A performance result is admissible only
after these checks pass.

This design lets optimized C++/Rust/GPU kernels replace the fast path later
without replacing the oracle or changing the Kernel Contract.
