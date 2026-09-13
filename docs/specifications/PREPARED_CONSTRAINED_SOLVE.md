# 预备约束求解生命周期规格 0.1

**状态：** `implemented`；本地数学、T3/T4 与性能证据通过，Linux 快检待运行

## AgentFEM 使用场景

AgentFEM 的载荷组合、线性参数扫描和同一切线矩阵上的多右端项求解，会在固定
DOF 图、固定约束集合和固定矩阵数值上重复求解。Native 应让程序层显式持有可复用
求解计划，而不是依赖隐式全局缓存或对象身份猜测。

本规格只覆盖对称正定线性系统。非线性切线更新、非对称系统、分布式所有权和
设备端预条件器仍保留其独立成熟度，不借本接口进入生产路径。

## 数学定义

给定规范 CSR 矩阵 $A$、右端项 $b$、互不重复的约束自由度集合 $C$ 和对应值
$g$，强制 Dirichlet 变换生成

\[
\hat A_{ij}=\begin{cases}
1,&i=j\in C,\\
0,&i\in C\ \text{或}\ j\in C,\\
A_{ij},&\text{其他},
\end{cases}
\qquad
\hat b_i=\begin{cases}
g_i,&i\in C,\\
b_i-\sum_{j\in C}A_{ij}g_j,&i\notin C.
\end{cases}
\]

约束计划预先分析源图到目标图的槽位映射、自由—自由保留项、自由—约束提升项
和约束对角槽位。若源图缺少约束对角项，目标图在准备阶段一次性补入。

## 生命周期与契约

1. `CSRConstraintPlan.from_matrix(matrix, constrained)` 固定源 CSR 图和约束索引顺序；
2. `transform_matrix(matrix)` 只接受完全相同的规范图，生成与约束值无关的
   $\hat A$；
3. `transform_rhs(matrix, rhs, values)` 按准备时的约束索引顺序解释 `values`，只
   计算 $\hat b$；
4. `PreparedPreconditioner.from_matrix(\hat A, ...)` 固定矩阵数值并构造 Jacobi 或
   块 Jacobi 数值预条件器；
5. `NativeSparseSolvePlan.solve(rhs, values)` 在固定 $A$ 上重复更新右端项并调用 CG。

图、矩阵数值和约束集合均为计划身份的一部分。图失配、非有限数值、越界或重复
约束必须在求解前失败。计划持有只读快照，不修改调用方矩阵或载荷。

## 能力边界

0.1 优化的是固定矩阵的多右端项生命周期。矩阵数值变化时必须重新构造数值
预条件器；仅图不变时可以复用 `CSRConstraintPlan`，但不得复用旧数值预条件器。
本规格不承诺直接法符号分解复用，也不把有限元材料、单元或工程边界条件语义交给
线性代数提供者。

## 验证要求

- 预备与冷路径的约束矩阵、右端项和解一致；
- 非零约束值和缺失对角项有独立确定性用例；
- 旧计划面对图或数值失配时明确拒绝；
- T3/T4 多载荷完整求解记录残差、反力/平衡量和应变能等价性；
- 性能结论分开报告准备成本与稳态重复求解成本。
