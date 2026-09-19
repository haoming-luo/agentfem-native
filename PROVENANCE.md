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

## 国产替代与自主掌握使命确认 — 2026-09-13

- 任务来源：项目负责人明确 AgentFEM Native 的开发使命是国产替代和自主掌握
  有限元求解内核。
- 定义：国产替代不是复制国外有限元项目，而是让 AgentFEM 不需要第三方有限元
  内核承担关键数值语义；自主掌握必须由数学规格、可读参考实现、生产热路径、
  跨平台构建、诊断、验证和持续维护能力共同证明。
- 依赖边界：许可和来源清楚的 NumPy、SciPy、PETSc、MPI、BLAS 等通用基础设施
  可以通过窄提供者接口使用，但必须可替换，不得拥有 Native 的离散语义，也不得
  使无可选提供者的 Tier-1 串行核心失效。
- 本次范围：更新项目宪章、愿景、路线图、后续计划、README 和开发指令；没有
  修改科学实现，也没有引入新的第三方代码或依赖。

## Scrum 轻量验证体系 — 2026-09-13

- 任务来源：项目负责人要求设计未来的验证、测试与对照方式，并明确快速 Scrum
  阶段不应建设庞大的验证工程。
- 决策：采用 L0 增量、L1 本地提交前、L2 里程碑本地和 L3 Tier-1 验收四层环；
  按改动风险选择最小配方，不让每个提交承担完整 Gate 证据。
- 对照顺序：数学真值优先，其次是自有 NumPy 基准层、自有优化路径差分、通用
  求解提供者，最后才是隔离环境中的第三方 FEM 黑盒。FEniCSx 不进入日常环，也
  永远不是唯一正确性依据。
- 成本约束：单个增量通常只增加 3–8 个高信息测试；不立即更换 `unittest`、重构
  全部测试目录或建设通用验证平台。完整报告、性能 JSON 和跨平台运行只在里程碑
  生成一次。
- 本次范围：重写验证政策为中文，新增 Scrum 轻量方案并连接开发、CI、贡献和
  路线图规则；没有修改科学代码或引入新依赖。

## 五小时 Gate 2 冲刺规划与 NumPy 边界澄清 — 2026-09-13

- 任务来源：项目负责人要求澄清“自研 NumPy/自研 Python-C++ 对照”的歧义，
  并规划一个可快速推进的五小时工作包。
- 边界澄清：NumPy 是第三方通用数组计算基础，不属于项目自研对象。项目自主
  掌握的是有限元公式、数据模型、装配、求解程序、验证判断与 C++ 高性能热路径；
  替代对象是 FEniCSx 等承担有限元核心语义的内核。
- 冲刺决策：五小时只形成 Gate 2 本地科学验收候选，集中补齐 T3/T4 实际求解
  收敛、畸变稳健性、Agent 可消费证据和轻量性能记录；不建设通用验证平台。
- 后续顺序：Gate 2 科学候选稳定后，独立推进 P3 CPU 并行，再运行一次 Tier-1
  验收；FEniCSx 对照保留为隔离、少案例、非唯一真值的里程碑任务。
- 本次范围：修改架构与验证措辞，新增时间盒计划；没有修改科学代码、运行外部
  对照或引入依赖。

## Gate 2 本地科学验收候选 — 2026-09-13

- 任务来源：项目负责人授权开始三小时快速开发，继续执行五小时 Gate 2 冲刺的
  高价值部分。
- 数学来源：沿用项目独立规格中的小应变线弹性、T3/T4 仿射单元、制造解、刚体
  模态、载荷—反力平衡和能量定义；没有读取第三方有限元实现源码。
- 代码变化：强化 T3/T4 实际求解收敛与尺度化平衡断言，增加合法畸变 T4 patch，
  并为执行收据增加 Agent 可读的维数、单元、路径、提供者和成熟度摘要。
- 性能证据：在 macOS arm64 上记录三档 T3 完整装配差分和中位时间，并复核 T4、
  CSR SpMV 与动力学记录；性能声明不跨平台外推。
- 成熟度：G2-01 至 G2-10 形成本地科学验收候选，但 Contract 0.2、P3 CPU 并行
  和最终 Tier-1 验收仍未完成，Gate 2 继续为 `implemented`。
- 远端证据：源提交 `d9c4ced` 的普通 Linux 快检运行 `34750325307` 通过；完整
  Tier-1 任务按治理规则跳过，没有据此声明 Windows/macOS 新验收。
- 外部代码：没有查看、复制、翻译、重排或 AI 改写 FEniCSx、Basix、UFL、FFCx、
  Akantu 或其他第三方有限元实现源码；没有新增依赖。

## P3 确定性 CPU 并行装配候选 — 2026-09-13

- 任务来源：项目负责人要求持续大力推进，并指定最终验收使用 CI。
- 数学与架构来源：沿用项目自有 T3/T4 规格和 ABI 1.2 串行实现，新增 ABI 1.3
  并行入口；使用 C++20 标准线程、静态连续单元分块、独占 COO 写入、线程私有
  载荷和固定线程顺序归并。没有引入新的第三方代码或运行时依赖。
- AgentFEM 适配：线程数由请求对象显式给出，默认一个线程；执行计划、摘要、
  能力清单和结果收据报告线程数及额外工作区，使 Agent 可在执行前预算资源。
- 本地证据：172 项科学/契约测试、严格 C/C++ 编译、ASan/UBSan、Ruff 和独立性
  扫描通过。macOS arm64 七次中位数记录显示 T3 四线程 2.53×、T4 八线程 4.00×；
  COO 完全相同，载荷差为舍入量级。该速度不向其他平台外推。
- 验收边界：ABI 1.3 的 Windows x86_64、Linux x86_64、macOS x86_64/arm64
  wheel、Stable ABI、C/C++、sanitizer 与科学测试必须由显式 Tier-1 CI 决定；
  CI 通过也不自动提升 Gate 2/P3 整体成熟度。
- 独立性声明：未查看、复制、翻译、重排或 AI 改写 DOLFINx、Basix、UFL、FFCx、
  Akantu 或其他第三方有限元实现源码。
- 远端证据：源提交 `da92dbd` 的普通 Linux 快检运行 `34752678699` 通过；显式
  Tier-1 运行 `34752687326` 的 14 项验收全部通过。文档后续提交复用该源提交
  证据，不重复消耗编译矩阵。

## P3 可复用 CSR 稀疏图参考候选 — 2026-09-13

- 任务来源：在 ABI 1.3 通过验收后，继续推进 P3 的下一瓶颈，服务多载荷工况、
  隐式动力和未来 Newton 的符号图复用。
- 设计来源：依据项目自有规范 COO/CSR 语义，把一次性行列排序和重复项识别保存
  为不可变 `CSRPattern`，后续按原贡献顺序只回填有限数值。未查阅或改写第三方
  有限元实现源码。
- 代码与证据：新增一般 COO/单元 DOF 构图、只读共享、结构摘要和三项高信息
  测试；175 项完整测试、Ruff、格式和独立性扫描本地通过。
- 性能边界：macOS arm64 的局部结构基准显示回填比重新规范化快 133×–147×；
  该测量不含单元装配、约束或求解，不构成端到端性能声明。
- 依赖：只使用既有 Python 标准库和 NumPy，没有新增第三方依赖。
- 远端证据：源提交 `9e33b6c` 的普通 Linux 快检运行 `34753072074` 通过；完整
  Tier-1 矩阵保留给后续 C++ ABI 数值回填候选，不重复验证未变化的平台代码。

## P3 ABI 1.4 确定性 CSR 数值回填 — 2026-09-13

- 任务来源：项目负责人要求继续快速开发，并明确最终验收交给 CI。
- 数学与实现来源：依据项目自有 `CSRPattern` 规格，将固定映射的数值归并定义为
  按原贡献序号串行执行 `data[mapping[k]] += value[k]`；独立实现 C++20 C ABI、
  手写 CPython Stable ABI 边界和 NumPy 参考差分，没有查阅第三方 FEM 源码。
- 代码与证据：ABI 提升到 1.4，新增空图、正常归并、映射越界、非有限输入、
  Python/参考逐位一致与能力清单测试；177 项本地回归、严格 C/C++ 直编、Ruff、
  格式和独立性扫描通过。
- 性能边界：macOS arm64 局部记录中，生产回填相对冷 COO 规范化为 134×–146×，
  相对 NumPy 参考回填仅为 1.02×–1.07×；不据此声称端到端求解加速。
- 依赖：只使用既有 Python、NumPy 和 C++ 标准库，没有新增第三方依赖。
- 远端证据：源提交 `d172cf6` 的 Linux 快检 `34757574303` 通过；完整 Tier-1
  运行 `34757610403` 的 14 项 Windows/Linux/macOS 验收全部通过。

## P3 预备力学装配本地候选 — 2026-09-14

- 任务来源：项目负责人要求设计五小时计划后直接执行，并在结束时提供高管可读
  的产品总结与展望。
- 数学与架构来源：依据项目自有 `CSRPattern`、T3/T4 单元 DOF 顺序和 ABI 1.4
  回填规格，建立显式 `CSRAssemblyPlan`；未查看或改写第三方 FEM 实现源码。
- 产品接入：T3/T4 装配与求解、T3 线性动力系统构建可接收预备计划；自有、
  NumPy 与可选 SciPy 提供者接收 COO 或规范 CSR，不改变有限元所有权边界。
- 本地证据：182 项测试、Ruff、格式和独立性扫描通过；冷/预备装配、拓扑失配、
  三类提供者与动力响应均覆盖。
- 性能边界：macOS arm64 的装配到 CSR 阶段，T3 为 17.4×–39.8×，T4 为
  7.5×–27.5×；所有数据逐位一致。仍生成 COO 索引，不包含约束或求解，不能
  外推为完整工作流倍率。
- 依赖：没有新增第三方依赖；仅复用既有 Python、NumPy 与 C++ ABI 1.4。
- 远端证据：源提交 `628d9a4` 的 Linux 快检 `34775672146` 通过。本增量没有
  修改 C ABI，按治理规则复用 ABI 1.4 三平台运行 `34757610403`，未重复执行
  未变化的平台矩阵。

## P3 预备约束求解生命周期候选 — 2026-09-14

- 任务来源：项目负责人要求连续五小时大力开发，并要求结束时提供高管可读的
  产品汇报与展望。
- 数学与架构来源：依据本项目强制 Dirichlet、规范 CSR、CG 与块 Jacobi 规格，
  独立设计约束图、约束后矩阵及数值预条件器的显式生命周期；没有查阅或改写
  第三方有限元实现源码。
- 代码变化：新增 `CSRConstraintPlan`、`PreparedPreconditioner`、
  `NativeSparseSolvePlan` 和能力清单；CG 增加真实残差复核与 binary64 后向误差
  底线。没有新增第三方依赖或 C ABI。
- 本地证据：188 项测试、53 个子用例、Ruff、格式和独立性扫描均通过。
  T3/T4 六载荷解与应变能逐项一致，完整约束求解稳态本地
  加速约 2.5×–4.4×，含一次准备约为 2.0×–2.8×。
- 平台与成熟度：性能只适用于记录的 macOS arm64 环境。源提交 `52b1e02` 的
  首次运行 `34776729496` 在科学测试前因 Ruff 格式检查失败；格式修复提交
  `00b7085` 的 Linux 快检 `34776794443` 随后通过全部门禁。既有 C++ ABI 1.4
  三平台验收继续有效，本增量未重复执行未变化的完整矩阵。
- 外部代码：未查看、复制、翻译、重排或 AI 改写 DOLFINx、Basix、UFL、FFCx、
  Akantu 或其他第三方有限元实现源码。

## Kernel Contract 0.2 本地候选 — 2026-09-20

- 任务来源：项目负责人要求遵循 AgentFEM 软件理念持续推进自主有限元内核。
- 设计来源：复用项目自有 T3/T4 问题类、资源计划、执行收据与版本化 0.1 契约，
  独立设计 0.2 的力学 JSON 表达、预检和证据绑定。
- 产品边界：AgentFEM 通过中立契约规划和执行；Native 不依赖 AgentFEM 私有
  对象、当前 AF-IR 形状、GUI 或智能体状态。动力学契约延后到 Gate 3。
- 代码与证据：0.1 兼容、T3/T4 端到端、精确失败路径、力学 VTK、版本 Schema
  与 CLI 预检进入测试；完整本地套件 194 项、53 个子用例通过。
- 依赖：没有新增依赖；NumPy 继续作为可替换通用数组基础，有限元语义由项目
  自主拥有。
- 成熟度：G2-11 本地关闭；本源提交的 Tier-1 验收尚未运行，Gate 2 继续为
  `implemented`。
- 外部代码：未查看、复制、翻译、重排或 AI 改写 DOLFINx、Basix、UFL、FFCx、
  Akantu 或其他第三方有限元实现源码。

## Gate 2 三平台验收与成熟度晋级 — 2026-09-20

- 证据源：提交 `654fe02` 本地通过 194 项测试、53 个子用例、Ruff、格式与独立性
  扫描；普通 Linux 快检 `35461701061` 通过。
- 平台证据：完整 Tier-1 运行 `35461748299` 通过 Windows x86_64、Linux x86_64、
  macOS x86_64/arm64 wheel，Python 3.13 Stable ABI 复装科学套件，三平台 C++20
  合同、sanitizer、Rust/C++/NumPy 对照、产物清单和最终汇总门禁。
- 数值处置：Intel macOS 揭示两个独立求解的应力恢复存在 binary64 尾差；CSR
  矩阵和载荷保持逐位相同，测试以 `‖DB‖∞` 传播已接受位移误差界，没有按平台
  扩大经验容差。修正后同一科学断言在全部 Tier-1 平台通过。
- 成熟度：G2-01 至 G2-12 关闭，T3/T4 基础线性静力提升为 `verified`。该结论不
  覆盖外部实验 `validated`、整体内核 `production`、Gate 3、全局非线性、MPI
  或 GPU。
- 晋级提交：只改变成熟度元数据、中文说明和证据文档的提交 `6b54a49` 通过 Linux
  快检 `35462064592`；数值与平台实现未变，按 CI 治理不重复完整 Tier-1。
- CI 风险：GitHub 提示 `ubuntu-latest` 将于 2026-10-19 起迁移到 Ubuntu 26；
  此提示不影响本次验收，但需在迁移前执行一次独立的 runner 预检。

## P3 ABI 1.5 无重复 COO 索引预备装配候选 — 2026-09-20

- 任务来源：Gate 2 闭环后继续遵循 AgentFEM 显式计划、可审计资源和高性能自主
  内核方向，推进 P3 剩余内存瓶颈。
- 设计来源：依据项目自有 `CSRAssemblyPlan`、T3/T4 局部贡献顺序和 ABI 1.4
  规格，允许串行 C++ 入口成对省略已由计划拥有的 COO 行列输出；没有读取或
  改写第三方有限元源码。
- 实现边界：ABI 1.5 与 Stable ABI 增加仅数值模式；预备串行 native T3/T4
  自动使用，普通 COO 和多线程入口保持不变。仍保留贡献数组与映射，不宣称最终
  直接 CSR 或 P3 完成。
- 本地证据：旧/新 `data/load` 与最终 CSR 逐位一致；194 项测试、严格 C++20、
  C11 头文件、ASan/UBSan、Ruff、格式和独立性扫描通过；macOS arm64 abi3 wheel
  构建与隔离安装冒烟通过。
- 性能与内存：七次中位记录中，T3 阶段倍率为 30.9×–43.7×、T4 为
  11.0×–16.0×；最大档明确省去索引分别为 4.72 MB 和 7.08 MB。倍率包含图复用，
  内存仅为数组载荷，不外推为端到端、RSS 或跨平台声明。
- 远端证据：源提交 `28b2ce7` 的 Linux 快检 `35462720452` 通过；显式 Tier-1
  运行 `35462761452` 随后通过 Windows x86_64、Linux x86_64、macOS
  x86_64/arm64 安装态 wheels、三平台 C/C++ 契约、sanitizer、Python 3.13
  Stable ABI 复用、Rust/C++/NumPy 对照、工件清单和最终汇总，ABI 1.5 平台
  边界关闭。
- CI 风险：运行继续提示 `ubuntu-latest` 将于 2026-10-19 起迁移到 Ubuntu 26；
  该提示不影响本次通过结论，迁移前仍须按既定计划单独预检。

## Gate 3 线性动力稳定预检与时间细化增量 — 2026-09-20

- 任务来源：ABI 1.5 平台闭环后继续向 AgentFEM 的安全计划—执行主干推进，不在
  Gate 3 科学证据未闭合时扩张非线性、MPI 或 GPU。
- 数学来源：对正对角质量使用项目自行规格化的 `M^-1 K` 诱导无穷范数谱上界，
  得到中心差分的充分保守步长；没有读取、复制或改写第三方有限元源码。
- 实现边界：计划层报告显式步长安全比，执行层在状态推进前拒绝超限；约束系统
  对实际自由自由度主子系统复核。Newmark 不受显式限制。
- 科学证据：受约束 T3 最低离散模态在 20→40 步细化时，中心差分与 Newmark
  误差倍率分别为 4.0021× 和 3.9972×；Newmark 检查点重启与不中断轨迹逐位
  一致。小型 NumPy 特征值只作为时间积分对照，不拥有有限元语义。
- 远端证据：源提交 `fd869b1` 的 Linux 快检 `35463268931` 通过 197 项数学与
  提供者测试、格式、可移植契约示例和独立性检查；C ABI 与平台构建链未变，
  按 CI 治理不重复完整 Tier-1。
- 后续科学增量：固定—自由 T3 纵向杆在 4/8/16 划分下的最低频率误差倍率为
  4.0194× 和 4.0049×；集中质量最低模态显式积分十周期的最大相对能量偏移为
  1.5422e-3。该记录只接纳单模态空间趋势与长时能量，不冒充局部脉冲或宽频
  色散验证。源提交 `39eb09c` 的 Linux 快检 `35463412190` 通过 199 项测试和
  全部快速门禁。
- P3 交叉增量：固定时间步 Newmark 在循环外准备有效矩阵的 Jacobi
  `PreparedPreconditioner`，后续所有步共享同一只读对象；五步结构测试证明
  质量求解一次加有效矩阵准备一次，不改变积分语义。没有引入全局缓存或新依赖。
- 成熟度边界：该增量保持 `implemented`；波传播、长时色散、隐式有效矩阵复用
  和三平台 Gate 3 验收仍未关闭。
