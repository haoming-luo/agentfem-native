# Project charter

## Mission

AgentFEM Native Engine is the independently developed finite-element engine
for AgentFEM. Its success means one AgentFEM engineering model can be lowered
through a versioned contract and solved without FEniCSx performing mesh,
element, DOF, quadrature, assembly, state evolution, or result generation.

AgentFEM Native 的战略使命，是建设面向 AgentFEM 的国产替代、自主掌握的
有限元求解内核。这里的“替代”不是复刻某个国外软件，而是使 AgentFEM 在不
依赖第三方有限元内核承担关键数值语义的情况下，仍能完成可信、高性能、可持续
演进的工程计算。

“自主掌握”至少意味着项目拥有并能独立维护：数学规格、网格与单元语义、DOF
与装配、材料与状态、求解程序、核心 C++ 热路径、稳定 ABI、跨平台构建、错误
诊断、验证体系和结果证据。任何一项只有调用入口而没有可解释语义、替代路径和
验证证据，都不能宣称已经自主掌握。

## 产品归属与主要使用者

AgentFEM Native 的首要且长期产品目标，是服务 AgentFEM 未来的工程建模与
自主计算。它不是另一个与 AgentFEM 竞争的终端建模产品，也不以独立扩张通用
用户界面为目标。功能优先级由 AgentFEM 可表达的工程问题、Agent 自动执行的
安全性以及计算可信度共同决定。

这一产品归属不意味着源码级耦合。AgentFEM 可以持续演进工程语言和用户体验；
Native 通过版本化 Kernel Contract 接收已降低的计算意图，并独立拥有网格、
单元、装配、求解、状态和结果语义。双方不得共享易变的内部对象作为公共边界。

判断新功能是否进入主线时，依次回答：

1. AgentFEM 是否有明确、真实的未来工程使用场景；
2. Native 是否应当拥有该能力的数值语义；
3. Agent 是否能够在执行前判断能力、资源和风险；
4. 结果是否能以 AgentFEM 可消费的结构返回，并保留完整证据；
5. 是否值得占用当前 Gate 的验证与性能预算。

不能通过这些问题的功能保留为研究候选，不进入生产主干。

## Identity

- Repository and distribution: `agentfem-native`
- Python import: `agentfem_native`
- AgentFEM backend identifier: `native`
- Public description: “AgentFEM Native is the independently developed
  finite-element engine for AgentFEM.”

AgentFEM owns the user-facing engineering language. Native owns finite-element
discretization and computation. General linear algebra and hardware services
remain replaceable providers.

## Non-negotiable boundaries

- Native is not a FEniCSx fork, translation, rearrangement, or wrapper.
- DOLFINx, Basix, UFL, and FFCx source is not an implementation input.
- FEniCSx may be a production backend and black-box comparator, never the sole
  correctness oracle.
- General libraries such as NumPy, SciPy, PETSc, MPI, BLAS/LAPACK, and HDF5
  may be used behind approved, documented boundaries.
- 自主掌握不等于拒绝开源生态。许可证和来源清楚的通用计算库可以作为可替换
  提供者，但不得拥有 Native 的有限元离散语义，也不得成为可读参考层或 Tier-1
  串行核心的强制依赖。
- Native Windows, macOS, and Linux are equal product targets. No OS-specific fork of
  the mathematical kernel is permitted.
- Correctness, provenance, and maintainability take priority over code volume
  or premature optimization.

## Development roles

The Kernel role works from mathematical specifications and owns independent
implementation. A separately scoped Verification role runs analytical,
manufactured, benchmark, and cross-backend comparisons without passing
third-party implementation source to the Kernel role.

## Scientific admission rule

Every admitted capability has a real consumer, formulas, typed inputs and
outputs, unit tests, analytical/manufactured evidence, golden data, failure
cases, convergence evidence, and a maturity label: `experimental`,
`implemented`, `verified`, `validated`, or `production`.

## Governance

Key decisions live in ADRs. Each release records commit, dependencies,
compiler/runtime, platform, tests, benchmarks, performance, failures,
capabilities, license inventory, and AI/provenance statements. The source-
available boundary is PolyForm Noncommercial 1.0.0 plus a separately granted
AgentFEM commercial license.
