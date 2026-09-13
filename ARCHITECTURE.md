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

平台差异隔离在提供者或打包层之后。科学语义、编号规则、文件模式、容差策略和
Kernel Contract 在原生 Windows、macOS、Linux 上保持一致。代码使用 Python API
与 `pathlib`，不要求 POSIX shell，显式处理文件系统大小写差异，也不在工件中
嵌入开发者绝对路径。

基础包只依赖 Python、NumPy 和平台 C/C++ 运行时。C++20 扩展通过 C ABI 1.4
提供 P1 扩散、T3/T4 串行与并行体积装配、CSR SpMV 和确定性数值回填，并使用 CPython 3.11+
Stable ABI，不使用第三方绑定框架。PETSc、SciPy、MPI、GPU 和厂商求解器只能是
可选提供者，不能成为参考路径的导入时依赖。

See `docs/adr/` for decisions and `docs/specifications/KERNEL_CONTRACT_V0.md`
for the boundary presented to AgentFEM.
