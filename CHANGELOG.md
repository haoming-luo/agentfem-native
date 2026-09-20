# Changelog

## 0.7.0a1 — Mechanics Alpha P0 closure candidate

- 新增面向 AgentFEM 的 Kernel Contract 0.2：兼容 0.1 热传导，接纳 T3/T4
  线性静力、执行前资源预检、计划一致性执行、位移/应变/应力、力学 VTK 和
  请求—计划—执行摘要证据。源提交 `654fe02` 已在 Tier-1 运行 `35461748299`
  通过完整 Windows/Linux/macOS 验收，T3/T4 基础线性静力提升为 `verified`。
- 新增 0.2 请求/结果 JSON Schema、T3 可运行示例与 CLI `--plan-only`；不引入
  新依赖，也不把当前 AF-IR 或第三方有限元对象耦合进内核。
- 新增 C++ ABI 1.3 确定性 T3/T4 CPU 并行装配：显式线程数、静态连续分块、
  独占 COO 写入和固定顺序载荷归并；旧串行 ABI 保持不变。
- 线程数与线程私有载荷内存进入 Agent 执行计划、摘要、能力清单和结果证据；
  默认仍为单线程，并行资源创建失败具有稳定状态码。
- 新增可重跑的 macOS arm64 并行性能记录；本地结果不外推为跨平台性能声明。
  ABI 1.3 已在运行 `34752687326` 中通过 14 项 Windows/Linux/macOS Tier-1 验收。
- 新增不可变 `CSRPattern`，把规范稀疏图与重复数值回填分离；支持一般 COO 和
  单元 DOF 构图、只读图共享、稳定摘要与明确失败语义。局部记录显示数值回填
  比每次重新规范化 COO 快 133×–147×，但不把它解释为端到端求解加速。
- C++ Stable ABI 提升到 1.4，新增按原 COO 贡献顺序的确定性 CSR 数值回填；
  保留独立 `fill_reference`，生产/参考输出逐位一致。源提交 `d172cf6` 已在运行
  `34757610403` 中通过 14 项 Windows/Linux/macOS Tier-1 验收。
- 新增 `CSRAssemblyPlan` 及 T3/T4 预备装配入口；固定拓扑的重复装配可以绕过
  CSR 图重建。自有、NumPy、SciPy 提供者接受规范 CSR，T3 动力构建可复用刚度
  图；`native_sparse` 提供者契约版本提升为 0.3。
- 新增 `CSRConstraintPlan`、`PreparedPreconditioner` 与
  `NativeSparseSolvePlan`：固定刚度的多载荷可复用约束图、约束后 CSR 和
  Jacobi/块 Jacobi 数值预条件器；`native_sparse` 提供者版本提升为 0.4。
- CG 收敛前复算真实残差，并把用户容差与 binary64 可达后向误差底线合并，
  避免递推残差漂移造成虚假收敛或不可达到的无意义迭代。
- C++ Stable ABI 提升到 1.5：串行预备 T3/T4 可只输出局部数值贡献与载荷，
  不再重复生成已由 `CSRAssemblyPlan` 拥有的 COO 行列索引；普通 COO 与并行
  入口保持兼容。源提交 `28b2ce7` 已在运行 `35462761452` 中通过完整
  Windows/Linux/macOS Tier-1 验收。
- 新增中心差分保守稳定步长预检：计划层报告请求值、上限和比值，执行层在状态
  推进前拒绝超限时间步；新增受约束 T3 显式/隐式二阶时间细化与 Newmark 逐位
  重启证据。源提交 `fd869b1` 的 Linux 快检 `35463268931` 通过；不据此提前
  提升 Gate 3。
- 新增固定—自由 T3 纵向波模态频率二阶空间趋势，以及集中质量显式积分十周期
  的有界能量证据；源提交 `39eb09c` 的 Linux 快检 `35463412190` 通过。局部
  脉冲传播与宽频色散仍明确留在后续 Gate 3 增量。
- Newmark 固定时间步的有效矩阵与 Jacobi 预条件器改为循环外一次准备、逐步
  复用；`N` 步预条件器构造从 `N+1` 次降到 2 次，积分公式、容差、结果和重启
  格式保持不变。源提交 `7fac770` 的 Linux 快检 `35463558192` 通过。
- 新增 T3 局部高斯位移脉冲的反射前传播验证；32→64 划分时两传感点测得波速
  相对误差从 2.87% 降至 1.33%，能量偏移同步下降。边界反射和宽频色散仍未
  关闭。源提交 `c1b3c9f` 的 Linux 快检 `35463705055` 通过。
- 新增自由端同相反射与脉冲宽度分组色散证据：64→128 划分时反射到时误差由
  0.80% 降至 0.04%，反射/入射峰值由 0.9756 趋近 0.9973；窄脉冲误差高于
  宽脉冲并随细化下降。源提交 `7e67cda` 的 Linux 快检 `35481948296` 通过。
- 冻结 Gate 3 限定线性动力学的 12 项验收矩阵：G3L-01 至 G3L-10 已有本地
  证据，版本化动力学 Kernel Contract 与最新候选三平台验收是仅余门禁；阻尼、
  非零/时变位移约束和非线性不混入当前声明。
- 新增向后兼容的 Kernel Contract 0.3 本地候选：AgentFEM 可对限定 T3 线性
  动力进行无装配资源预检、预算/取消控制、显式或 Newmark 执行、能量/反力
  审查和摘要绑定检查点续算；0.1 热传导和 0.2 静力兼容测试保持通过。
- 源提交 `a630793` 的 Linux 快检 `35482650495` 与完整 Tier-1 `35482705887`
  通过 Windows/Linux/macOS 安装态验收，G3L-01 至 G3L-12 关闭；验收矩阵限定
  的无阻尼 T3 线性动力范围提升为 `verified`。
- Extended the packaged C++20 Stable ABI to 1.2 with T4 volume assembly and
  canonical CSR SpMV, retaining readable differential oracles and automatic
  fallback when the extension is unavailable.
- Added T4 pure-shear/hydrostatic response and a symmetric engineering
  cantilever, plus local assembly/operator performance evidence.
- Added fixed-DOF linear dynamics, reaction recovery, external-work and energy-
  balance ledgers, and a constrained T3 mesh-to-transient test.
- Added deterministic internal mesh/field interchange and an Agent-native
  plan/execute/explain receipt surface with budgets and honest maturity labels.
- 不声明 Gate 3 非线性范围、整体 P3、MPI、GPU 或生产级成熟度；Gate 2 静力与
  Gate 3 线性 `verified` 都仅覆盖各自验收矩阵中的明确范围。
- Split CI into cancellable Linux fast checks and explicit Tier-1 acceptance,
  reduced acceptance duplication, shortened artifact retention, and recorded
  the time-boxed September 2026 public-repository policy without changing the
  noncommercial license or publishing a package.

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
