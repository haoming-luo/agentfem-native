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
- All autonomous Python source files use
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
