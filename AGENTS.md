# AgentFEM Native repository instructions

This repository is the clean-room, independently developed finite-element
engine for AgentFEM. Read `docs/charter/PROJECT_CHARTER.md`, `ROADMAP.md`,
`ARCHITECTURE.md`, `PROVENANCE.md`, and the relevant ADR/specification before
changing scientific code.

## Independence

- Never read, copy, translate, rearrange, or AI-paraphrase DOLFINx, Basix, UFL,
  or FFCx implementation source for this project.
- FEniCSx may be used only as a black-box verification comparator in a
  separately scoped verification task. It is never the only oracle.
- Do not add code with unclear authorship or license provenance.
- Run `python tools/check_independence.py` for every scientific change.

## Scientific workflow

- Start from a written mathematical specification and capability maturity.
- Keep the reference NumPy layer small, readable, and independent of future
  optimized kernels.
- Add algebraic identities, failure cases, analytical/manufactured evidence,
  and convergence evidence in that order.
- A passing execution is not automatically `verified` or `validated`.
- Do not widen tolerances merely to hide a platform or provider discrepancy.

## Architecture

- Dependencies flow from the neutral Kernel Contract through owned finite-
  element concepts to replaceable linear-algebra/hardware providers.
- AgentFEM owns engineering language; Native owns topology, elements,
  quadrature, DOFs, assembly, state, procedures, and numerical results.
- Keep materials, solvers, output, verification, agents, and GUIs decoupled.
- Record consequential changes in `docs/adr/` and update provenance/evidence.

## Platforms

- Native Windows x86_64, macOS arm64/x86_64, and Linux x86_64 are first-class,
  release-blocking targets. WSL is additional, not a replacement for Windows.
- Use Python APIs and `pathlib`; do not require a POSIX shell for install,
  testing, or core behavior.
- Keep optional PETSc, MPI, GPU, and vendor providers out of reference-layer
  imports. Report provider/platform capability explicitly.

## Development and CI economy

- Follow `docs/development/CI_GOVERNANCE.md`: focused tests per increment,
  complete local verification before a milestone, one Linux fast job for
  ordinary pushes, and manual Tier-1 acceptance for gate/platform candidates.
- Do not use a full remote matrix as an inner development loop. Batch pushes by
  reviewable capability and let newer pushes cancel superseded fast CI.
- Documentation-only changes must cite the already tested source revision and
  run; they do not repeat compilation merely to validate the evidence prose.
- Treat billing-limit jobs as not executed. Do not rerun unchanged work; retain
  the pending Windows/Linux/macOS acceptance boundary and continue locally.
- CI artifacts are temporary test evidence, never a public package release.
- The repository is temporarily public through September 2026. Visibility does
  not change its noncommercial license. Never commit secrets, private models,
  customer data, or proprietary benchmark inputs; review visibility on
  2026-10-01.

## Scope discipline

Work gate by gate. The next milestone is the Gate 1 steady-diffusion vertical
slice. Do not start nonlinear mechanics, MPI, or GPU implementation before its
mesh-to-result verification chain passes.
