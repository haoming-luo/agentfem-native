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
  `SPDX-License-Identifier: LicenseRef-AgentFEM-Native-Draft` until licensing
  is finalized.
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
- Human/legal review: still required for licensing, API adoption, and release.
