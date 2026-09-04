# Gate 0 evidence report

Report date: 2026-09-04. Status: local reference evidence passed; remote
cross-platform CI remains pending.

## Designed

- Project identity, clean-room boundary, architecture, Kernel Contract draft,
  provider boundary, cross-platform policy, governance, and gate roadmap.

## Implemented

- NumPy reference P1 triangle basis and gradients.
- Degree-1 and degree-2 reference triangle quadrature.
- Affine triangle mapping, area, coordinate mapping, and gradient transform.
- Independence/SPDX scanner and three-OS CI matrix.

## Tested

- Environment: bundled isolated Python 3.12.13, NumPy 2.3.5, macOS arm64;
  `dolfinx`, `ufl`, `basix`, and `ffcx` were all absent.
- `python -m unittest discover -v`: 23 tests passed.
- `python tools/check_independence.py`: passed with no forbidden imports.
- Built `agentfem_native-0.0.0-py3-none-any.whl` without downloading runtime
  dependencies; SHA-256:
  `ee8ea7727fc165e27932dacd1b9434973bc278c4ecc5ffedeba04794cd42f7f3`.
- Installed the wheel into a separate target directory and imported version
  0.0.0 from that artifact while all four forbidden packages remained absent.

## Cross-validated

At the Gate 0 snapshot no PDE solve existed and remote platform CI had not run.
The subsequent Gate 1 diffusion evidence is recorded separately.

## Unresolved

- Owner acceptance of Kernel Contract direction.
- Remote native Windows, macOS, and Linux CI evidence (workflow is configured
  but has not run because no remote repository was created or pushed).
- AgentFEM external lowering prototype and first steady-diffusion vertical slice.

## Gate decision

The local Gate 0 foundation is accepted. Three-platform release acceptance
remains open until remote native Windows, macOS, and Linux jobs pass.
