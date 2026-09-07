# Changelog

## 0.6.0a1 — compressed Mechanics Alpha implementation

- Added deterministic BSR, block matvec/diagonals/CSR conversion, and
  block-Jacobi CG for vector mechanics.
- Extended the C++20 Stable ABI to 1.1 with per-cell-constitutive T3 volume
  assembly; added reference/vectorized/native equivalence, elastic material
  regions, response/convergence evidence, and vector/tensor VTK.
- Added owned tetrahedral meshes and a readable T4/3D small-strain elasticity
  mesh-to-result path with loads, constraints, reactions, energy, recovery,
  patch, uniaxial, orientation, provider, and manufactured-convergence tests.
- Added T3 consistent/lumped mass, centered explicit and Newmark linear
  dynamics, energy histories, digest-bound checkpoint/restart, resource plans,
  budgets, progress, cancellation, and structured failures.
- Added experimental Neo-Hookean and J2 material-point candidates with
  objectivity, energy derivative, yield consistency, and transaction evidence.
- Recorded a local Mechanics Alpha benchmark. No public package release,
  cross-platform performance claim, Gate 4, MPI, or GPU claim is made.

## 0.5.0a1 — sparse foundation and T3 reference candidate

- Added immutable deterministic scalar CSR, canonical COO reduction, SpMV,
  residual/diagonal/storage operations, and symmetric sparse Dirichlet handling.
- Added dependency-free CG/Jacobi with machine-readable convergence, iteration-
  limit, and breakdown outcomes; made `native_sparse` the default provider and
  retained dense NumPy as an explicitly selected small-problem oracle.
- Added vector DOF numbering and a readable T3 plane-stress/plane-strain
  elasticity vertical slice with loads, constraints, reactions, energy, and
  cell stress/strain recovery.
- Added execution-free resource plans, deterministic plan digests, installed
  capability reporting, analytical/patch/failure/provider tests, and local
  sparse memory/performance evidence.
- Added the one-month acceleration overlay. T3 is `implemented`; Gate 2 and P3
  remain open until their wider scientific, optimized, and hosted evidence pass.

## Unreleased — Gate 2 preparation

- Closed Gate 1 from its native three-platform scientific and installability
  evidence. Reclassified public AF-IR promotion as an optional, non-blocking
  AgentFEM integration track while it is not an upstream development priority.
- Accepted a large-team six-month Mechanics Alpha roadmap: owned end-to-end
  sparse execution, verified 2D/3D linear mechanics, linear dynamics/state,
  high-performance CPU work, evidence-gated MPI/GPU research, and Agent-native
  planning/diagnostic/evidence interfaces.
- Made the Sparse Foundation Sprint and performance milestone P3 the immediate
  Gate 2 prerequisite under ADR-0018.

## 0.4.0a1 — packaged native kernel candidate

- Shipped the standard-library-only C++20 P1 volume-assembly kernel inside a
  CPython 3.11+ stable-ABI wheel with no binding-framework dependency.
- Added an ABI-versioned C boundary, ownership-safe buffer validation,
  material-region tensors, GIL release, explicit `native` assembly selection,
  and conservative `auto` fallback to vectorized/reference paths.
- Built the same kernel in Rust, compared complete COO/load outputs and raw
  timings, and selected C++20 as the production compiled-kernel direction.
- Installed and verified the optional BSD-licensed SciPy provider without
  transferring any finite-element semantics to SciPy.
- Added native/runtime/end-to-end sparse benchmarks, cross-language ABI tests,
  three-OS wheel CI, and Python-minor stable-ABI installation evidence.
- Added CI installability checks for Linux x86_64, Windows x86_64, macOS
  x86_64, and macOS arm64, with ephemeral private artifacts, `abi3audit`,
  wheel-installed tests, and a deterministic SHA-256 manifest. No public
  package release is made during rapid iteration.

## Unreleased — Gate 0 foundation

- Established the project charter, architecture, Kernel Contract draft,
  provenance policy, verification policy, and gate-based roadmap.
- Established the initial cross-platform CI verification matrix.
- Added an independent NumPy reference implementation for the P1 triangle,
  affine geometry mapping, and degree-1/degree-2 triangle quadrature.
- Added mathematical, orientation, independence, and packaging tests.

## 0.3.0a1 — performance architecture candidate

- Added a bounded-memory, vectorized P1 diffusion assembly path with safe
  automatic dispatch and the element-by-element implementation retained as
  the executable mathematical oracle.
- Vectorized structured-mesh generation, geometric validation, and declared
  boundary adjacency checks.
- Added reference/vectorized equivalence, chunk invariance, material-region,
  failure, scale, memory, and machine-readable performance evidence.
- Added a standard-library-only C++20/C ABI assembly spike with numerical
  tests and native Windows/macOS/Linux CMake CI configuration.
- Selected C++20 as the leading production-kernel experiment while retaining
  Rust as an evidence-driven second candidate.

## 0.2.0a1 — Gate 1 release candidate

- Added named cell material regions, spatially varying scalar conductivity,
  and symmetric positive-definite anisotropic conductivity.
- Added replaceable NumPy dense and optional SciPy sparse providers with
  runtime provider identity.
- Implemented Kernel Contract 0.1, bundled JSON Schema 2020-12 description,
  structured failures, portable artifact digests, and cross-platform CLI.
- Added a public AgentFEM AF-IR lowering prototype with an explicit portable
  extension and no AgentFEM/FEniCSx runtime dependency.
- Added one-command repository-local environment bootstrapping for native
  Windows, macOS, and Linux.

## 0.1.0a1 — Gate 1 diffusion slice

- Accepted PolyForm Noncommercial 1.0.0 plus a separate commercial-license
  requirement.
- Defined native Windows, macOS, and Linux as equal first-class targets.
- Added owned triangle meshes, named boundary sets, P1 diffusion/source/flux
  integration, deterministic COO assembly, Dirichlet elimination, a dense
  NumPy solve provider, reaction/balance evidence, and portable VTK output.
- Added analytical, patch, reordering, manufactured-convergence, failure, and
  output tests.
