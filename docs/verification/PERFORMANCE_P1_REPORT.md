# Performance milestone P1 evidence

Date: 2026-09-04. Version: `0.3.0a1`. Status: locally verified performance
architecture; cross-platform and end-to-end claims remain open.

## What changed

- Structured triangle generation, cell geometry checks, and declared boundary
  adjacency validation now operate in bounded vectorized batches.
- Static scalar/tensor diffusion assembly now preallocates deterministic COO
  storage and processes cells in bounded chunks using analytical affine
  gradients.
- `auto` selects the fast path only when doing so cannot change callable-field
  semantics. The original element/quadrature implementation remains directly
  executable as `reference`.
- A standard-library-only C++20 shared-library spike exposes one stable C
  function for static P1 assembly. CMake and numerical tests target native
  Windows, macOS, and Linux.
- The exported header compiles as C11, and the implementation passes local
  optimized plus Address/Undefined Behavior Sanitizer builds.

## Numerical gates

- Reference and vectorized COO rows/columns are exactly identical.
- Scalar, anisotropic, material override, boundary load, orientation, and
  chunk-size cases compare complete matrix/load arrays.
- The recorded 8,192-cell comparison has maximum matrix difference
  `6.66134e-16` and load difference `1.62630e-19`.
- The 524,288-cell scale case has exact constant null mode and total load 1.
- The 131,072-cell C++20 comparison has exact COO, matrix, and load identity on
  the structured anisotropic case.

## Local performance

Environment: macOS 26.6.2 arm64, Python 3.12.13, NumPy 2.3.5, Apple Clang
21.0.0, optimized C++ flags recorded in the JSON evidence.

- 8,192 cells: reference `0.92866 s`, vectorized `0.002410 s`, speedup
  `385.40x`.
- 524,288 cells: mesh generation/validation `0.11300 s`; vectorized assembly
  `0.19912 s`; 2.63 million cells/s and 23.70 million COO entries/s; observed
  `tracemalloc` peak 260,036,911 bytes.
- 131,072 cells: C++20 `0.001196 s`, vectorized NumPy `0.039903 s`, C++20
  speedup `33.36x` with exact data identity.

Raw repetitions and limitations are stored in
`benchmarks/results/2026-09-04-darwin-arm64-performance-p1.json`.

## Interpretation

The measurements prove local headroom, not universal performance. The
reference/vectorized comparison includes Python object and COO construction;
the C++ comparison includes output allocation and C ABI invocation but not
sparse duplicate reduction or solving; the NumPy path additionally constructs
and validates an owned `COOMatrix`. Hardware, compiler, allocator, and provider
differences can change ratios.

C++20 therefore leads the next production experiment, but cannot enter the
release path until hosted three-platform builds, sanitizers, versioned ABI
ownership, material semantics, compiled-wheel packaging, and end-to-end sparse
solve benchmarks pass. Local sanitizer evidence exists; CI must reproduce it.
Rust remains required comparison evidence.

## Release-candidate verification

- Repository `.venv`: 75 tests run, 74 passed and one optional-SciPy test
  skipped; editable AgentFEM Native 0.3.0a1 and AgentFEM 0.3.1 coexist;
  dependency check and contract CLI smoke test pass.
- SciPy-enabled comparison environment: 74 pytest cases plus 34 parameterized
  subtests passed and one environment-specific optional-provider case skipped.
- A wheel rebuilt from the source distribution was installed into an isolated
  target and the complete 75-test suite passed against that installed wheel.

The local candidate artifacts under `work/` are not a published release or
proof of Windows/Linux execution.
