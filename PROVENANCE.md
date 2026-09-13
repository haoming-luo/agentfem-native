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

## Cross-platform installability automation extension — 2026-09-04

- Task origin: the project owner assigned local macOS validation to the
  development workstation and future native Windows/Linux validation to
  GitHub CI.
- AI-assisted scope: manually dispatchable four-target CI installation-wheel matrix,
  wheel-installed scientific tests, strict stable-ABI auditing, downloadable
  artifacts, deterministic SHA-256 artifact manifest, and manifest tests.
- External research: official GitHub Actions artifact/matrix documentation and
  the official cibuildwheel 4.1.1 platform example. No numerical source or
  third-party finite-element implementation was consulted.
- Diagnostic finding: an emulated manylinux x86_64 compile exposed a missing
  direct `<cstdint>` include that Apple Clang had tolerated indirectly. The
  include was corrected, but the emulated run is not claimed as native Linux
  acceptance; GitHub's Linux runner remains authoritative.
- Hosted acceptance: GitHub Actions run `33834316684` at commit `f005c8d`
  passed all 21 jobs. Native Windows x86_64, Linux x86_64, macOS x86_64, and
  macOS arm64 CI wheels passed strict Stable ABI audit, installation, and the
  scientific suite. Windows/macOS/Linux SciPy, CMake, and C++/Rust/NumPy
  equivalence jobs also passed.
- Distribution boundary: these wheels and their SHA-256 manifest are private,
  ephemeral CI evidence retained for 14 days. No PyPI upload, GitHub Release,
  public binary publication, or release compatibility commitment was made.

## Gate 1 closure and AF-IR reprioritization — 2026-09-04

- Task origin: the project owner observed that AF-IR is not a current AgentFEM
  development priority.
- Decision: Native Gate 1 closes on its owned scientific, independence,
  performance, contract, and native three-platform evidence. Promotion of the
  existing external lowering prototype into AgentFEM's public AF-IR export is
  an optional integration track, not a Native gate blocker.
- Rationale: Native milestone progress must not depend on an upstream interface
  that is neither required for standalone kernel execution nor currently
  prioritized. The versioned Kernel Contract remains the stable ownership
  boundary; integration can resume when AgentFEM exposes a prioritized,
  reconstructable public export.
- Consequence: Gate 2 basic solid mechanics becomes the next active scientific
  milestone. No numerical code or evidence threshold was changed.

## Six-month Mechanics Alpha product plan — 2026-09-07

- Task origin: the project owner requested a bold estimate of what a large
  development team should achieve in six months, emphasizing both serious
  high-performance computing and fitness for AgentFEM.
- Planning assumption: 28–36 experienced contributors, approximately 15
  person-years during the window. This is a capacity model, not a staffing or
  delivery claim.
- Decision: target an internal Mechanics Alpha with an owned sparse spine,
  verified Gate 2, verified linear Gate 3 capability, high-performance CPU
  execution, Agent-native planning/evidence, and maturity-labelled nonlinear,
  MPI, matrix-free, and GPU research tracks. No public package release is
  implied.
- Official research consulted: PETSc feature/install/license documentation;
  Ginkgo documentation and license/platform metadata; Kokkos/Kokkos Kernels
  documentation and release/license information; hypre documentation; RAJA
  repository/documentation; and libCEED public user documentation.
- Clean-room boundary: this research was limited to public architecture,
  capability, platform, and license information for general-purpose
  infrastructure. No DOLFINx, Basix, UFL, FFCx, or other third-party FEM
  implementation source was read or incorporated. No scientific code changed
  in this planning step.
- Durable artifacts: `docs/charter/SIX_MONTH_PRODUCT_ROADMAP.md`, ADR-0018, and
  synchronized roadmap, vision, changelog, and ADR index entries.

## Compressed Sparse Foundation and T3 reference slice — 2026-09-07

- Task origin: the project owner compressed the six-month program into one
  month, its first month into one week, and requested that the nominal first
  week be completed in one sustained development run.
- Independent mathematical sources: standard CSR definitions, symmetric
  strong-Dirichlet elimination, preconditioned conjugate gradient, small-strain
  kinematics, and isotropic plane-stress/plane-strain formulas were written into
  `SPARSE_ALGEBRA_V0.md` and `T3_LINEAR_ELASTICITY.md` before implementation.
- AI-assisted scope: Native scalar CSR, COO canonicalization, sparse operations,
  CG/Jacobi, default-provider migration, vector DOFs, T3 elasticity, loads,
  constraints, recovery, result evidence, execution-free planning, capability
  manifest, tests, benchmark, documentation, and acceleration roadmap.
- Third-party exposure: NumPy public array operations and the already admitted
  optional SciPy provider interface. No third-party finite-element source was
  read, copied, translated, or used; no new runtime dependency was added.
- Local evidence: 115 tests and 47 subtests passed before documentation
  finalization; the independence scan passed. Exact sparse/mechanics benchmark
  repetitions and environment are recorded in
  `benchmarks/results/2026-09-07-darwin-arm64-sparse-foundation.json`.
- Packaging evidence: the `0.5.0a1` sdist and macOS arm64 `cp311-abi3` wheel
  built successfully. A clean Python 3.11 environment installed the wheel with
  no broken requirements, loaded every new module from `site-packages`, exposed
  the packaged C++ kernel, and passed all 115 repository tests plus installed
  sparse-diffusion and T3-mechanics smoke solves.
- Maturity boundary: the sparse scalar baseline and T3 reference slice are
  implemented and locally evidenced. P3, Gate 2, native Windows/Linux evidence,
  BSR, compiled sparse/T3 kernels, 3D, dynamics, MPI, and GPU remain open.

## Compressed Mechanics Alpha 0.1 implementation — 2026-09-07

- Task origin: the project owner requested immediate execution of the complete
  one-month acceleration plan in one sustained development pass.
- Prior hosted evidence: commit `5230970` passed all 19 Windows, Linux, and
  macOS jobs in GitHub Actions run `34091224291`, closing the hosted boundary
  for the preceding sparse/T3 reference revision.
- Independent specifications: BSR extended the owned sparse specification;
  T4/3D elasticity, linear dynamics, and nonlinear material-point formulas are
  recorded in `docs/specifications/` alongside their maturity boundaries.
- AI-assisted implementation: deterministic BSR and block-Jacobi; C++20 ABI
  1.1 T3 volume assembly; T3 material regions and three-way assembly
  equivalence; owned T4 mesh/element/solve/recovery; mechanics VTK; T3 mass;
  explicit/Newmark dynamics; checkpoint digest and restart; budgets, progress,
  cancellation, and structured failures; experimental Neo-Hookean and J2
  material points; tests, benchmarks, roadmap, ADR, and evidence report.
- Mathematical sources: standard affine simplex interpolation, isotropic
  Hooke elasticity, consistent P1 mass, centered/velocity-Verlet integration,
  Newmark average acceleration, compressible Neo-Hookean energy, and radial-
  return J2 equations as written in the repository specifications. No external
  FEM implementation source was consulted or used.
- Third-party exposure: NumPy array and linear-algebra primitives only in the
  reference layer; the C++20 path uses the standard library and the CPython
  Stable ABI. No runtime dependency was added.
- Local evidence: the final source and isolated wheel-installed suites each
  passed 151 tests. Strict C++20 warnings, AddressSanitizer,
  UndefinedBehaviorSanitizer, Ruff, and the independence scanner pass. The
  `0.6.0a1` sdist and macOS arm64 CPython Stable-ABI wheel build locally.
- Hosted acceptance: commit `db33958` passed all 21 jobs in GitHub Actions run
  `34099397789`, including native Windows, Linux, macOS x86_64/arm64 wheels,
  CPython 3.11/3.13 tests, C++20 contracts/sanitizers, optional SciPy, Rust
  formatting/lints/tests, three-platform ABI equivalence, and independence.
- Maturity boundary: this pass implements every tranche's central vertical
  slice but does not claim every tranche exit. Gate 2, Gate 3, global nonlinear
  mechanics, production threading/SIMD, MPI, and GPU remain open until their
  stated evidence passes.

## Mechanics Alpha 0.2 P0 closure candidate — 2026-09-11

- Task origin: the project owner requested a sustained push toward the complete
  six-month Mechanics Alpha target while retaining scientific honesty and
  cross-platform discipline.
- Independent specifications: the existing affine T4, canonical CSR, and
  linear second-order dynamics mathematics were extended before acceptance to
  cover compiled T4/SpMV behavior, fixed-DOF reduction, reactions, external
  work, internal interchange, and Agent execution receipts.
- AI-assisted implementation: C++20 ABI 1.2 T4 volume assembly and CSR SpMV;
  stable-ABI Python binding; reference/native dispatch and equivalence; 2D
  cantilever and 3D shear/bulk evidence; constrained dynamics and work-energy
  ledger; deterministic mesh/field interchange; plan/execute/explain API;
  tests, benchmark, ADR, roadmap, limitations, and verification report.
- Mathematical sources: standard affine simplex gradients, small-strain
  `B^T D B`, canonical CSR row products, principal-submatrix constraint
  elimination, dynamic reaction residuals, and trapezoidal force-displacement
  work as written in repository specifications. No third-party finite-element
  implementation source was consulted or used.
- Third-party exposure: NumPy array primitives and build tooling already
  admitted by the project. The optimized kernel uses only standard-library
  C++20 and the CPython Stable ABI. No new runtime dependency was added.
- Local evidence: 169 source tests, Ruff, the independence scanner, direct
  warnings-as-errors C++20/C11 builds, ASan/UBSan, `0.7.0a1` sdist/wheel build,
  and an isolated wheel-installed smoke workflow pass. Exact performance data
  are in `benchmarks/results/2026-09-11-darwin-arm64-mechanics-alpha-0.2.json`.
- Hosted acceptance: commit `4f3da30` passed all 14 explicit Tier-1 acceptance
  jobs in GitHub Actions run `34566488847`. Native Windows, Linux, macOS
  x86_64, and macOS arm64 wheels passed their installed scientific tests;
  Python 3.13 Stable ABI reuse passed on Windows, Linux, and macOS arm64; all
  three C++20 platform contracts, Linux sanitizers, the Rust comparator, and
  the artifact manifest passed. The ordinary Linux fast job was intentionally
  skipped for this explicit full run.
- Maturity boundary: Gate 2 and the linear portion of Gate 3 remain
  `implemented`, not `verified`. Production CPU threading/SIMD, million-DOF
  evidence, global nonlinear mechanics, MPI, and GPU remain open.

## CI economy and temporary visibility — 2026-09-11

- Task origin: the owner supplied a GitHub Actions billing audit and explicitly
  authorized making `haoming-luo/agentfem-native` public for September 2026 to
  continue development.
- Observed cause: the prior workflow launched 21 jobs for every push, including
  a documentation-only evidence update. macOS dominated the private-runner
  cost. Billing-limited jobs had not executed and were not test failures.
- Decision: ordinary source pushes use one cancellable Linux job; documentation-
  only changes do not start CI; native Tier-1 wheels, Stable ABI reuse, C/C++
  contracts, sanitizers, Rust comparison, and manifests move to explicit gate/
  platform acceptance or version tags. Artifact retention changes from 14 to
  five days.
- Visibility boundary: the repository was verified as `PRIVATE`, changed to
  `PUBLIC`, and verified again through the GitHub repository API. Review is due
  2026-10-01 and is not automatic. License, package publication, scientific
  maturity, and commercial-use permissions did not change.
- Durable policy: `docs/development/CI_GOVERNANCE.md`, repository instructions,
  cross-platform policy, workflow definitions, README, and changelog.
- First policy evidence: the ordinary push run `34566483839` used one Linux
  fast job and passed. The explicitly dispatched Tier-1 run `34566488847`
  completed successfully with 14 acceptance jobs at commit `4f3da30`; this
  replaces the pending hosted boundary for ABI 1.2 without publishing a
  package.

## 后续架构规划与中文工程表达 — 2026-09-13

- 任务来源：项目负责人要求结合优秀有限元内核与 AgentFEM 需求规划后续架构、
  功能和工作计划，并明确要求重要代码注释与项目介绍使用中文。
- AI 辅助范围：检查本项目宪章、路线图、架构、规格、ADR、测试结构、能力输出
  和已记录验证证据；编写后续架构计划与中文工程表达规范；修正项目指令中已经
  过期的 Gate 1 下一里程碑表述。没有修改科学代码。
- 外部资料边界：仅查阅 FEniCSx、DOLFINx、Basix、UFL、FFCx、Akantu 和
  PETSc 的官方公开文档，用于理解公开能力划分、产品结构和依赖边界。没有查看、
  复制、翻译、重排或 AI 改写任何第三方有限元实现源码。
- 规划决策：下一主线是 `KernelPlan`、可复用稀疏图、C++20 CPU 线程化、Gate 2
  验证、Gate 3 线性验证和 Kernel Contract 0.2；全局非线性、MPI、GPU、接触与
  断裂继续保持后置或实验成熟度。
- 语言决策：重要注释、文档字符串、项目介绍、ADR、规格、验证与诊断以中文为
  主；稳定标识符、JSON 字段、ABI 和错误码保留英文并配中文解释。已有记录采用
  触及即迁移，不做破坏证据含义的机械翻译。
- 当前复核：仓库 `.venv` 中 AgentFEM Native 版本为 `0.7.0a1`，C++ ABI 1.2
  可用，169 项测试通过。误用未安装项目依赖的系统 Python 会产生导入错误，
  该结果不属于科学回归；自动化和开发命令应显式使用仓库环境。

## AgentFEM 首要产品方向确认 — 2026-09-13

- 任务来源：项目负责人确认 AgentFEM Native 的开发目的就是面向 AgentFEM
  的未来使用。
- 决策：AgentFEM 是 Native 唯一首要和长期的产品使用者；独立 Python/CLI
  入口服务开发、验证和集成，不构成第二个终端产品路线。
- 架构边界：面向 AgentFEM 不等于耦合其当前内部实现。双方继续通过版本化
  Kernel Contract 连接，Native 不依赖易变的 AF-IR 形状、Agent 或 GUI 对象。
- 优先级规则：每项新增能力必须说明 AgentFEM 工程消费场景、意图降低路径、
  Native 数值所有权、Agent 预检方式、结果返回和证据保存方式；仅为模仿其他
  内核功能目录而提出的能力不进入生产主线。
