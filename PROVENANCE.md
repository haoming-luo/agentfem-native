# Provenance policy and initial record

Every scientific feature must connect requirement, mathematical source,
implementation, tests, and acceptance evidence. Unattributed or
license-incompatible code is rejected.

## Rules

- Do not inspect or reproduce FEniCSx implementation source for Native work.
- Public API documentation and black-box numerical results may inform
  compatibility and verification, but are not implementation specifications.
- Record task origin, theory sources, AI-generated scope, reviewer, third-party
  exposure, test evidence, and unresolved discrepancies.
- All autonomous Python/C/C++ implementation files use
  `SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0`.
- A successful run is not scientific validation.

## Initial provenance record — 2026-09-04

- Task origin: project-owner charter supplied to Codex.
- AI-assisted scope: repository documents, independent API sketches, P1
  reference implementation, tests, and CI configuration.
- Theory sources: formulas documented in
  `docs/specifications/P1_TRIANGLE.md`; no third-party source code used.
- AgentFEM audit boundary: public module declarations, public signatures,
  public docstrings, package metadata, Git status, and commit metadata only.
  `agentfem.backends.fenicsx` and implementation internals were not read.
- Third-party implementation exposure: none intentionally used.
- Verification: 23 local mathematical/independence tests, a clean wheel build,
  and artifact import without forbidden packages; exact evidence is recorded
  in `docs/verification/GATE_0_REPORT.md`.
- License decision: project owner selected PolyForm Noncommercial 1.0.0 with a
  separate written commercial license requirement on 2026-09-04.

## Gate 1 diffusion record — 2026-09-04

- Task origin: project owner requested continued development to the next
  milestone and clarified native Windows, macOS, and Linux equality.
- AI-assisted scope: owned triangle mesh, named sets, element/source/boundary
  integration, COO assembly, constrained dense solve provider, result evidence,
  VTK writer, mathematical tests, and associated specifications.
- Mathematical source: weak-form and P1 formulas recorded in
  `docs/specifications/STEADY_DIFFUSION.md`; no third-party FEM source used.
- Verification: analytical `u=x`, independent element formulas, linear patch,
  manufactured convergence, orientation and node-renumbering invariance,
  balance, failure, artifact, and forbidden-import tests.
- Cross-platform status: source and CI are platform-neutral; only macOS local
  execution evidence exists until the remote three-OS workflow runs.

## Gate 1 release-candidate extension — 2026-09-04

- Task origin: project owner requested an intensive construction milestone,
  explicit freedom to evaluate faster implementation languages and permissive
  libraries, and a shared development environment containing AgentFEM.
- AI-assisted scope: material cell sets, variable/tensor conductivity,
  provider protocol, optional SciPy sparse solve, Kernel Contract 0.1 schema
  and executor, CLI, public AF-IR lowering prototype, environment bootstrap,
  tests, examples, CI, architecture records, and roadmap.
- External source boundary: only official public license/documentation facts
  and AgentFEM public AF-IR declarations were consulted. No FEniCSx, DOLFINx,
  Basix, UFL, or FFCx implementation source was read or used.
- Local integration environment: repository `.venv`, Python 3.12.13,
  NumPy 2.3.5, editable AgentFEM Native 0.2.0a1, editable AgentFEM 0.3.1
  from clean commit `5dbeee0`, h5py 3.16.0, and mpi4py 4.1.2 installed
  without changing either repository; `pip check` reports a consistent
  environment.
- Platform status: macOS arm64 execution is local evidence. Native Windows and
  Linux remain CI claims pending hosted runs; WSL is not a substitute.

## Performance milestone P1 — 2026-09-04

- Task origin: project owner requested sustained development emphasizing
  future-facing architecture, numerical rigor, computation speed, confidence,
  and the long-term independent-kernel vision.
- AI-assisted scope: bounded vectorized assembly, vectorized mesh generation
  and validation, conservative dispatch, equivalence/scale benchmarks,
  standard-library C++20/C ABI spike, CMake tests, CI jobs, ADRs, evidence, and
  2035 technical vision.
- Mathematical source: the existing independently documented P1 affine
  diffusion formulation. The optimized implementations were derived from that
  specification and tested against the executable oracle.
- External research: official CMake platform/toolchain information, Rust
  platform tiers, PETSc/Kokkos/nanobind/SciPy/OpenBLAS license and platform
  information. No third-party finite-element implementation source was read.
- Local compiler: Apple Clang 21.0.0 targeting arm64 Darwin. C++ source uses
  only the standard library; no third-party C++ code or runtime was added.
- Local optimized and Address/Undefined Behavior Sanitizer builds pass the
  numerical contract, including reversed orientation and non-finite input.
  A C11 smoke test verifies that the exported header is a real C ABI rather
  than a C++-only declaration.
- Local integration environment: editable AgentFEM Native 0.3.0a1 and editable
  AgentFEM 0.3.1 coexist in `.venv`; `pip check` and a contract CLI execution
  using automatic vectorized assembly pass.
- Exact raw results and caveats:
  `benchmarks/results/2026-09-04-darwin-arm64-performance-p1.json`.

## Performance milestone P2 — 2026-09-04

- Task origin: project owner requested that the C++ kernel enter wheels, that
  optional SciPy be installed when compatible with autonomy, and that C++20
  and Rust be compared before one production direction was selected.
- AI-assisted scope: ABI-versioned C++ core, hand-written CPython Limited-API
  adapter, native dispatch/fallback, Rust equivalent, cross-language/runtime/
  sparse benchmarks, wheel tests, CI, specifications, ADRs, and roadmap.
- Third-party source boundary: official Python packaging/C API, Rust platform
  and linkage, and SciPy package/license metadata were consulted. No external
  finite-element implementation source was read or incorporated.
- Local toolchains: Apple Clang 21.0.0 and rustc 1.98.1 on macOS arm64. The
  C++ runtime kernel and Rust comparator contain no third-party source/library
  dependency; Rust is development-only and not in the wheel.
- Local integration environment: Python 3.12.13, NumPy 2.3.5, verified SciPy
  1.18.1, editable AgentFEM Native 0.4.0a1, and editable AgentFEM 0.3.1;
  `pip check` succeeds and the full suite has no optional-provider skip.
- Packaging evidence: a `cp311-abi3` macOS arm64 wheel built with Python 3.11
  imports and passes the full suite under Python 3.12, exercising the bundled
  `_p1_native.abi3` extension. Hosted native Windows/Linux/macOS evidence is
  pending and is not inferred from this local result.
- Language decision: full-output-equivalent Rust was 1.25–1.27 times the C++
  time across the recorded scales. C++20 is selected under ADR-0016; Rust is a
  non-shipping comparator.
- Exact raw results and caveats:
  `benchmarks/results/2026-09-04-darwin-arm64-performance-p2.json`.

## Cross-platform release automation extension — 2026-09-04

- Task origin: the project owner assigned local macOS validation to the
  development workstation and future native Windows/Linux validation to
  GitHub CI.
- AI-assisted scope: manually dispatchable four-target release-wheel matrix,
  wheel-installed scientific tests, strict stable-ABI auditing, downloadable
  artifacts, deterministic SHA-256 artifact manifest, and manifest tests.
- External research: official GitHub Actions artifact/matrix documentation and
  the official cibuildwheel 4.1.1 platform example. No numerical source or
  third-party finite-element implementation was consulted.
- Diagnostic finding: an emulated manylinux x86_64 compile exposed a missing
  direct `<cstdint>` include that Apple Clang had tolerated indirectly. The
  include was corrected, but the emulated run is not claimed as native Linux
  acceptance; GitHub's Linux runner remains authoritative.
