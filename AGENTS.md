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

- Follow `docs/verification/SCRUM_VERIFICATION.md` during rapid development:
  use the smallest risk-matched loop for each increment and reserve convergence,
  broad comparison, performance sweeps, and platform matrices for milestones.
- Start from a written mathematical specification and capability maturity.
- Keep the reference NumPy layer small, readable, and independent of future
  optimized kernels.
- Add algebraic identities, failure cases, analytical/manufactured evidence,
  and convergence evidence in that order.
- A passing execution is not automatically `verified` or `validated`.
- Do not widen tolerances merely to hide a platform or provider discrepancy.
- Prefer 3–8 high-information tests per increment. Do not build a general test
  framework when a small deterministic case, identity, and failure test are
  sufficient.

## Architecture

- Treat AgentFEM Native as a domestically controlled, independently mastered
  solver kernel for AgentFEM. Ownership must be demonstrated by specifications,
  executable reference mathematics, maintained production paths, platform
  builds, diagnostics, and verification evidence—not merely by wrapping an API.
- AgentFEM is the primary and long-term product consumer. Prioritize future
  AgentFEM engineering workflows and agent-safe computation, not an unrelated
  standalone frontend or feature catalog.
- Face AgentFEM through a versioned Kernel Contract. Do not couple Native to
  AgentFEM's mutable internal objects, current AF-IR shape, agents, or GUI.
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
- Permissive general-purpose libraries are allowed leverage, but must remain
  replaceable and may not own finite-element semantics or disable the Tier-1
  serial baseline when absent.

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

## 中文工程表达

- 遵循 `docs/development/CHINESE_DOCUMENTATION_POLICY.md`。重要代码注释、
  文档字符串、项目介绍、ADR、规格、路线图、验证结论和面向人的诊断以中文
  为主。
- Python/C/C++ 标识符、JSON 字段、ABI、稳定错误码和第三方专有名称保留
  英文；它们的用途、数学含义和修复建议使用中文解释。
- 注释说明为什么这样实现、数学与状态不变量、性能取舍和证据边界，不逐行
  复述代码。已有英文内容采用“触及即迁移”，不得机械翻译科学记录。

## Scope discipline

Work gate by gate. Gates 0 and 1 are complete; Gate 2 is verified for its
bounded T3/T4 basic linear-static scope. The next milestones are closing
Performance P3 and verifying the linear-dynamics portion of Gate 3. Do not move
global nonlinear mechanics, MPI, or GPU work into the production path before
their own prerequisites pass; isolated experiments retain explicit maturity labels.
