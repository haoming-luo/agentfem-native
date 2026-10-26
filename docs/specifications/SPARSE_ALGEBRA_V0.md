# Native sparse algebra specification 0.1

Status: implementation target for performance milestone P3.

## Purpose and ownership

Native sparse algebra is the dependency-free production baseline between
finite-element assembly and replaceable high-performance providers. It owns
enough storage, constraint, residual, and iterative-solve behavior to execute
real problems without SciPy and without dense conversion. It is deliberately
not an attempt to reproduce industrial direct solvers or algebraic multigrid.

NumPy supplies array primitives only. It does not own the sparse data model or
solver semantics.

## Scalar CSR contract

For a matrix with shape `(m, n)`, CSR is represented by:

```text
indptr:  int64, shape (m + 1,)
indices: int64, shape (nnz,)
data:    float64, shape (nnz,)
```

The invariants are:

- dimensions are nonnegative and indexable by signed 64-bit integers;
- `indptr[0] == 0`, `indptr[-1] == nnz`, and `indptr` is nondecreasing;
- every column index is in `[0, n)`;
- column indices are strictly increasing within every row;
- every stored value is finite;
- arrays are copied on construction and exposed read-only;
- explicit numerical zeros are permitted because structural identity is
  separate from numerical cancellation.

## Deterministic COO canonicalization

COO entries are ordered lexicographically by `(row, column)` using a stable
ordering. Entries with equal coordinates are reduced in that order using
float64 addition. The canonical result contains one stored entry per coordinate
and preserves an exactly zero result as an explicit structural entry.

Input ordering therefore cannot change the CSR graph. Floating values may
differ at roundoff level when the summation order inside an equal-coordinate
group differs; finite-element assembly supplies deterministic cell/local order
and provider comparisons specify tolerances accordingly.

## Operations

For CSR matrix `A` and finite vector `x`, matrix-vector multiplication returns
`y_i = sum_j A_ij x_j`. Residual is `r = b - A x`; Euclidean norm uses float64.
Diagonal extraction requires every requested diagonal location to be present.

The readable NumPy reduction is the permanent differential oracle. Canonical
non-empty CSR matrices with at least 2,048 stored values automatically use the
C++20 ABI 1.2 SpMV when installed; smaller or extension-free cases use the
oracle. Both accumulate each row in stored-column order and must agree within
documented floating-point tolerances without densification.

Memory reporting is exact for owned CSR arrays:

```text
storage_bytes = indptr.nbytes + indices.nbytes + data.nbytes
```

It excludes Python object headers and temporary solver work vectors, which are
reported separately by benchmarks.

## Strong Dirichlet transformation

Given distinct constrained indices `C` and values `g`, form an algebraically
equivalent symmetric system:

```text
b' = b - A[:, C] g
A'[C, :] = 0
A'[:, C] = 0
A'[c, c] = 1 and b'[c] = g_c for every c in C.
```

Missing diagonal entries are inserted structurally before transformation. The
operation returns a new CSR matrix and right-hand side; the input is immutable.
For symmetric positive-definite `A` on the free subspace, `A'` is symmetric
positive definite.

Native 预备求解在非零约束时用 `x0_C=g, x0_F=0` 初始化 CG；单位约束行与
自由行已解耦，Jacobi/块 Jacobi 保持约束搜索方向为零。全零约束维持默认零
初猜。已知边界值不受全局残差近似误差支配；原方程、真残差和停止门槛不变。
详见 ADR-0035；不以事后改写结果或报告替代这个状态不变量。

## Conjugate gradient and Jacobi

The baseline CG algorithm solves symmetric positive-definite systems. Optional
Jacobi preconditioning uses `M^-1 r = r / diag(A)` and requires finite strictly
positive diagonal entries.

给定右端项 `b`，用户要求的停止阈值为：

```text
max(atol, rtol * ||b||_2).
```

生产报告还把该值与 binary64 的保守可达后向误差底线取最大值。底线按
`eps sqrt(n) (||A||_inf ||x||_inf + ||b||_inf)` 计算，防止条件较差的问题因
递推残差舍入漂移而虚假收敛，也避免在用户阈值低于当前浮点问题可达精度时无意义
迭代。CG 命中阈值后重新计算 `b-Ax`；只有真实残差不超过报告中的实际阈值才
返回收敛，否则替换残差并重启搜索方向。

The report records convergence, reason, iteration count, initial residual,
final residual, threshold, and configured tolerances. The following paths are
distinct and machine-readable:

- initial residual already satisfies the threshold;
- converged after one or more iterations;
- iteration limit reached;
- non-positive/non-finite curvature indicates breakdown or an inadmissible
  matrix;
- non-finite input, iterate, or residual is rejected.

The production provider raises on nonconvergence and includes the report in a
successful outcome. Lower-level CG may return a nonconverged report so tests and
future procedures can inspect controlled failure.

## BSR contract

Mechanics uses node-major, component-minor interleaved DOFs. BSR groups a
canonical scalar CSR matrix into square dense blocks with ordered block columns,
immutable float64 block data, and int64 pointers/indices. Scalar dimensions must
be divisible by block size. BSR matvec, diagonal-block extraction, scalar-CSR
round trip, exact storage reporting, and missing-diagonal failures are owned.

Block-Jacobi inverts only the small symmetric positive-definite diagonal blocks
and applies them inside CG. A singular or non-positive diagonal block fails
before iteration. Scalar CSR remains the canonical constraint representation;
BSR is not claimed merely from interleaved numbering.

## Acceptance evidence

### IC0 隔离候选

令 S=diag(A_ii^-1/2)，B=S A S + sigma I，sigma>=0 为显式预条件平移。
对源矩阵的严格下三角结构位置依次计算单位下三角 L 和正对角 D：
`L_ij=(B_ij-sum_{k<j} L_ik D_k L_jk)/D_j`，
`D_i=B_ii-sum_{j<i} L_ij² D_j`。仅遍历两行结构交集，不添加填充；显式结构
零仍属于已声明的稀疏图。不采用排序、丢弃阈值、自动平移或不明确的回退。
每个枢轴须有限且严格为正；失败返回行号，不留下可用的部分对象。

预条件应用为 `z=S L^-T D^-1 L^-1 S r`。B 的对称性按 64*eps 的逐项相对尺度
准入，仅对容许的舍入差取对称均值；原 CG 的 A 不变。矩阵须非空、方阵、正
对角、结构对称；拒绝非有限平移、非法后端、陈旧图/数值和不合法 RHS。
准备对象绑定原 CSR 数值快照，缓存原 A 的范数，保持 CG 的真残差与停止规则。

`reference` 后端只使用 NumPy 数组与自有前/后代入；`scipy` 后端延迟导入
通用稀疏三角求解，缺席时明确不可用。两者数学完全相同，科学容差一致。
最低证据：三对角精确分解、一般稀疏应用对照、SPD 但 IC0 失败的反例、显式
平移、单位缩放、缓存失配、无稠密化、T3/T4 平衡/能量以及完整求解成本。
该实验不进入默认提供者或契约能力清单。

ABI 1.6 实验增量只接受已经规范化的严格下三角 CSR 和独立对角：
`afn_ic0_factor` 按升序两指针交集执行相同 LDLT 递推；
`afn_ldlt_solve` 执行单位下三角前代入、正 D 除法及反向散射。
输入指针、行界限、升序且列号小于行号、有限数值及正 D 每次检查；不以不安全
的 trusted 标志省略边界验证。零严格下三角条目允许，行数须大于零。
分解失败返回独立的非正枢轴状态及失败行号，输出只在成功时可用；调用方不得
把部分分解当作缓存。输出缓冲不得与结构或因子输入重叠。Python 绑定拥有独立
输出，并以 Stable ABI 的 buffer 协议检查长度、类型与可写性。

结构复用实验只缓存规范 CSR 的形状/行指针/列号、镜像位置、严格下三角位置与
对角位置。`IncompleteCholeskyPattern` 不缓存材料数值、缩放或分解：每次
`factor(matrix, ...)` 先精确匹配结构，再重新检查数值对称、正对角、范数并完成
数值分解，返回独立不可变快照。显式零也属于图；图变化必须重新分析。失败不
修改 pattern 或已有分解。不添加全局缓存或隐藏自动重试，ABI 和默认提供者不变。
验收覆盖同图变值/完整应用差分、异图拒绝、失败后原快照仍有效、只分析一次和
含首轮结构准备的多矩阵成本；仅省去结构分析不能宣称省去数值重分解。

- malformed shapes, pointers, columns, ordering, and non-finite data fail;
- COO duplicates and empty rows canonicalize exactly;
- matvec, diagonal, residual, and constraint transformation match small dense
  calculations without using dense conversion in implementation;
- CG solves independently constructed SPD systems and reports initial,
  convergence, limit, and breakdown paths;
- diffusion solution, reactions, energy, and residual match dense NumPy and
  optional SciPy providers;
- a no-densification guard proves Native production solve does not call
  `COOMatrix.to_dense`;
- memory growth is O(nnz), with raw evidence for representative meshes.
