# Mechanics Alpha 0.1 implementation report

**Date:** 2026-09-07

**Revision maturity:** implemented; hosted Tier-1 foundation accepted

## Delivered vertical slices

- immutable BSR conversion, block matvec, diagonal blocks, CSR round trip, and
  two/three-component block-Jacobi CG use;
- C++20 ABI 1.1 T3 volume assembly with per-cell constitutive matrices;
- reference/vectorized/native T3 equivalence and elastic material regions;
- 2D pure shear, uniform dilation, uniaxial, constant-strain, orientation,
  numbering, provider, balance, energy, and second-order manufactured evidence;
- owned tetrahedral mesh, T4 stiffness/body/face loads, 3D rigid-mode
  constraints, sparse solve, reactions, energy, stress/strain recovery, and
  vector/tensor VTK output;
- 3D constant-strain, uniaxial cube, orientation, provider, balance, and
  second-order manufactured evidence;
- T3 consistent/lumped mass, centered explicit integration, Newmark average
  acceleration, energy histories, and digest-bound exact restart;
- structured resource budgets, accepted-state progress, cancellation, and
  retry guidance;
- experimental Neo-Hookean objectivity/energy derivative and J2 radial-return
  commit/rollback material-point evidence.

## Local performance record

The exact JSON record is
`benchmarks/results/2026-09-07-darwin-arm64-mechanics-alpha.json`. On Darwin
arm64 with Python 3.12.13 and NumPy 2.3.5, five warm-process repetitions gave:

| Work | Size | Median / result |
|---|---:|---:|
| readable T3 volume assembly | 8,192 cells / 8,450 DOFs | 0.37982 s |
| vectorized T3 volume assembly | same | 0.00310 s |
| C++20 T3 volume assembly | same | 0.000786 s |
| readable T4 end-to-end solve | 162 cells / 192 DOFs | 0.01310 s |
| explicit SDOF integration | 10,000 steps | 0.14981 s |

The C++20 T3 assembly was 483× faster than the readable oracle in this local
case. Its maximum relative canonical-matrix difference was `7.65e-16`. Native
BSR storage was 1,196,088 bytes versus 1,927,256 bytes for scalar CSR; BSR and
CSR matvec differed by `2.51e-14` relatively due to reduction ordering.

## Claim boundary

- Timings are local kernel evidence, not native Windows/Linux performance.
- T3 compiled acceleration currently covers volume assembly, not boundary
  integration, sparse matvec/solve, or threading.
- T4 and dynamics are readable implementations, not production-optimized paths.
- Gate 2 remains open pending the remaining engineering benchmark corpus and
  compiled T4/sparse execution; hosted acceptance of this revision passed.
- Gate 3 remains open pending constrained FEM dynamics, wave evidence,
  external-work ledgers, and broader restart/platform evidence.
- Nonlinear materials are experimental material points only.

## Local acceptance record

- 151 source tests passed.
- The `0.6.0a1` source distribution and macOS arm64 `cp311-abi3` wheel built;
  an isolated installation loaded C++ ABI 1.1 and passed the same 151 tests.
- Ruff lint and formatting, the clean-room independence scanner, strict C++20
  compilation, AddressSanitizer, and UndefinedBehaviorSanitizer passed.
- Rust was not locally compiled because Cargo is unavailable on this host. The
  Rust spike truthfully remains an ABI 1.0 diffusion comparator; hosted CI
  exercised its formatting, lints, tests, and three-platform ABI comparison.
- Commit `db33958` passed all 20 jobs in hosted run `34099397789`: native
  Windows, Linux, macOS x86_64/arm64 installation wheels, CPython 3.11/3.13,
  C++20 contracts/sanitizers, optional SciPy, ABI comparison, and independence.
- No public package was published.
