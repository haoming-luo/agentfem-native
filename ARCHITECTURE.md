# 架构

## Dependency direction

```text
AgentFEM engineering language
          |
          v
backend-neutral lowering
          |
          v
Kernel Model + Solve Request
          |
          v
Mesh -> Space -> Element -> Operator
          |
          v
Assembly -> State -> Procedure
          |
          v
LinearAlgebraProvider
          |
          v
Simulation Result + Evidence
```

Dependencies point downward. Materials do not know solver implementations;
solvers do not know agents or GUIs; output and verification do not mutate the
numerical solve. Public contracts expose owned arrays and typed records rather
than FEniCSx objects.

## Layers

- `reference`：本项目独立维护、使用 NumPy 数组运算表达的可读有限元数学参考
  实现，用于校验后续优化内核；NumPy 本身是第三方通用库，不是自研对象。
- `kernel`: triangle/tetrahedron topology, scalar/vector DOFs, P1 diffusion,
  T3/T4 linear-elasticity operators, deterministic assembly, and linear state
  owned by AgentFEM Native.
- `fast path`：打包的标准库 C++20 内核负责已支持的静态体积数据；显式线程数、
  静态连续分块和确定性载荷归并用于 T3/T4 CPU 并行。受限向量化 NumPy 和可读
  参考路径永久保留；优化路径复现数学参考，但不反过来定义数学语义。
- `providers`: an owned dependency-free CSR/BSR/CG/Jacobi/block-Jacobi baseline,
  C++20 CSR SpMV, replaceable dense/sparse linear algebra, and future parallel runtime and hardware
  implementations. Dense NumPy is a bounded oracle, not the production default.
- `prepared assembly`：由调用方显式拥有 `CSRAssemblyPlan`，绑定单元 DOF 顺序
  与规范图；T3/T4 和动力学可复用它，COO/CSR 都通过同一提供者边界。
- `prepared solve`：`CSRConstraintPlan` 固定约束图分析，
  `NativeSparseSolvePlan` 固定约束后矩阵和数值预条件器；AgentFEM 可显式持有该
  资源执行固定刚度多载荷，矩阵数值变化时必须重建预条件器。
- `contract`: versioned JSON request/result execution and public AgentFEM
  AF-IR lowering boundary.
- `verification`: tests and evidence that consume public results without
  becoming part of the solver.

The procedure layer now owns centered explicit and Newmark average-acceleration
linear dynamics, fixed-DOF elimination, reaction and external-work/energy
ledgers, immutable histories, digest-bound restart, progress, cancellation, and
budgets. The Agent control surface revalidates plans before execution and emits
compact evidence receipts without coupling Native to AF-IR. A deterministic
internal JSON bundle carries owned meshes, named sets, and result fields.
Experimental nonlinear material points remain
separate from admitted element and procedure capabilities.

The current implementation deliberately keeps these small enough to audit.
C++20 is the selected production compiled language; Rust remains a
non-shipping comparison track, and both must reproduce the same reference.

## 跨平台规则

新增隔离 `IncompleteCholesky` 候选拥有缩放、显式平移、IC0 分解与矩阵身份。
外层仍是自有 CG 的真残差/停止规则；三角应用可显式选择自有参考、可选 SciPy
或 C++20。ABI 1.6 新增受检的严格下三角分解/应用，旧函数保持兼容；不把 IC0
设为默认、不进入 JSON，也不把非正枢轴失败静默替换为另一个方程。

新增候选层以 `SimplexGeometry` 复用参考梯度、体积、连通分量和稀疏图：
`ThermalOperator` 拥有热导/热容，`ThermoelasticOperator` 拥有温度到热应变载荷，
`FiniteStrainOperator` 拥有总拉格朗日内力和解析切线，`AT2FractureOperator`
拥有退化能、损伤界限约束和交替接受状态。它们不依赖 AgentFEM 内部对象，也不
把通用 NumPy 数组库宣称为自研。最初多物理增量未改变 C++ ABI 1.5；当前 IC0
独立增量扩展到 1.6。新的有限变形切线
主要为 NumPy 向量化候选，复用自有 CSR 和 C++ SpMV，不冒充已完成 C++ 迁移。

单独候选 JSON 边界在构建算子前校验物理参数、连通分量、能力和预算；通过
`multiphysics_capabilities()` 暴露限制。非线性真实残差和 AT2 KKT 是接受条件，
失败和取消仅返回已接受历史，持久化重启仍待设计。Windows 安装态诊断与 JSON
替换属于外围运行服务，不改变有限元语义，见 ADR-0032。

平台差异隔离在提供者或打包层之后。科学语义、编号规则、文件模式、容差策略和
Kernel Contract 在原生 Windows、macOS、Linux 上保持一致。代码使用 Python API
与 `pathlib`，不要求 POSIX shell，显式处理文件系统大小写差异，也不在工件中
嵌入开发者绝对路径。

基础包只依赖 Python、NumPy 和平台 C/C++ 运行时。C++20 扩展通过 C ABI 1.6
提供 P1 扩散、T3/T4 串行与并行体积装配、CSR SpMV 和确定性数值回填，并使用 CPython 3.11+
Stable ABI；已准备的串行 T3/T4 可省略重复 COO 行列索引。扩展不使用第三方绑定
框架。PETSc、SciPy、MPI、GPU 和厂商求解器只能是
可选提供者，不能成为参考路径的导入时依赖。

材料扫描的 IC0 结构复用仅由调用方持有 `IncompleteCholeskyPattern`；结构计划
无数值因子。不同值必须产生新分解快照，异图拒绝复用，失败保留旧快照。

See `docs/adr/` for decisions and `docs/specifications/KERNEL_CONTRACT_V0.md`
for the boundary presented to AgentFEM.
