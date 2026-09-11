# Mechanics Alpha 0.2 P0-closure candidate report

**Date:** 2026-09-11

**Revision maturity:** implemented; Tier-1 hosted platform acceptance passed

## Delivered chain

- C++20 Stable ABI 1.2 T4 volume assembly with per-cell 6-by-6 constitutive
  matrices and exact constant body loads;
- C++20 canonical CSR SpMV with readable NumPy differential oracle and
  conservative automatic dispatch;
- T4 reference/native canonical matrix and load equivalence;
- symmetric-mesh 2D cantilever evidence and exact 3D pure-shear/hydrostatic
  response evidence;
- fixed-DOF sparse linear dynamics, full-state and reaction recovery,
  trapezoidal external work, and energy-balance history;
- constrained T3 mesh-to-transient evidence;
- deterministic internal triangle/tetrahedron mesh, set, point-field, cell-
  field, and metadata interchange;
- Agent-native plan/execute/explain receipts with stale-plan rejection, resource
  budgets, progress/cancellation reuse, runtime identities, numerical evidence,
  and explicit maturity boundaries.

## Local performance record

The exact record is
`benchmarks/results/2026-09-11-darwin-arm64-mechanics-alpha-0.2.json`. On Darwin
arm64 with Python 3.12.14 and NumPy 2.3.5, five warm-process repetitions gave:

| Work | Size | Readable median | Native median | Local speedup |
|---|---:|---:|---:|---:|
| T3 volume assembly | 8,192 cells / 8,450 DOFs | 0.42688 s | 0.000864 s | 494× |
| canonical CSR SpMV | 116,228 stored values | 0.000138 s | 0.0000565 s | 2.45× |
| T4 volume assembly | 162 cells / 192 DOFs | 0.007845 s | 0.000531 s | 14.8× |

T3 and T4 maximum canonical-matrix differences were `4.77e-6`
(`7.65e-16` relative) and `5.68e-14`, respectively. T4 load arrays were exact.
CSR SpMV maximum absolute difference was `2.00e-6`; the different result is at
floating-point reduction scale for matrix entries near `1e11`. The T4 solve
free residual was `2.08e-13` and resultant-balance norm was `1.78e-14`.

## Local acceptance

- 169 source tests pass.
- Ruff lint and format checks and the clean-room independence scan pass.
- Direct C++20 and C11 contracts compile with warnings-as-errors; the C++
  numerical suite passes both normally and with AddressSanitizer plus
  UndefinedBehaviorSanitizer.
- The `0.7.0a1` source distribution and macOS arm64 `cp311-abi3` wheel build.
  A separate Python 3.12 environment installed the wheel, loaded C++ ABI 1.2,
  and passed diffusion plan/execute/explain plus interchange smoke checks.
- CMake is unavailable on the local host, so the equivalent direct compiler
  commands were used. CI remains responsible for CMake and Tier-1 platform
  evidence.

## Claim boundary and remaining work

- These performance numbers apply only to the recorded local environment.
- Commit `4f3da30` passed the complete Tier-1 acceptance in GitHub Actions run
  `34566488847`: 14 acceptance jobs succeeded across native Windows, Linux,
  macOS x86_64, and macOS arm64. The ordinary Linux fast job was intentionally
  skipped in the explicit acceptance run. ABI 1.2 wheels, Python 3.13 Stable
  ABI reuse, C++20 contracts, sanitizers, the Rust comparator, and the artifact
  manifest all passed.
- Gate 2 remains open for a broader engineering and convergence corpus,
  contract admission, and production CPU parallel work.
- The linear Gate 3 slice remains open for wave propagation, implicit restart,
  additional time-refinement evidence, and dedicated transient platform
  evidence beyond the shared installed scientific suite.
- CPU threading/SIMD, stronger preconditioning/provider comparison, million-
  DOF solve evidence, global nonlinear procedures, MPI, and GPU remain work,
  not implied capabilities.
- No public package was published.
