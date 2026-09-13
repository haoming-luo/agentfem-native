# Native P1/T3/T4 与稀疏 C ABI 1.3

**状态：** ABI 1.3 本地实现候选；ABI 1.2 已通过三平台验收。Rust 对照实验仅
实现 ABI 1.0 扩散子集。

## 版本与所有权

`afn_p1_abi_version()` 返回 `0x00010003`，编码为主版本 1、次版本 3。主版本不
兼容时不得调用。调用方在整个调用期间拥有所有输入和输出缓冲区；内核不保留
指针、不分配调用方输出、不调用回调，也不能让 C++ 异常越过 C ABI。

所有标量为 IEEE-754 binary64，索引为有符号 64 位整数，计数使用平台 C
`size_t`。缓冲区必须自然对齐、连续、互不重叠，并满足函数声明所要求的长度。

## 已有串行入口

- `afn_p1_diffusion_assemble` 与 `afn_p1_diffusion_assemble_cells`：二维 P1 三角形
  扩散装配；
- `afn_t3_elasticity_assemble_cells`：二维 T3 小应变线弹性体积装配；
- `afn_t4_elasticity_assemble_cells`：三维 T4 小应变线弹性体积装配；
- `afn_csr_spmv`：规范 CSR 矩阵向量乘。

ABI 1.3 不改变这些函数的签名、布局、顺序或数值语义。

### 扩散布局

- points：`node_count * 2`，节点主序 `(x, y)`；
- cells：`cell_count * 3`，从零开始的节点索引；
- conductivity：全局四个行主序值，或逐单元 `cell_count * 4`；
- rows/columns/data：`cell_count * 9`；
- load：`node_count`。

### T3 布局

- points/cells 与扩散相同；
- constitutive：逐单元 `cell_count * 9` 行主序值；
- body force：两个全局分量；
- rows/columns/data：`cell_count * 36`，每单元六个节点主序向量 DOF；
- load：`node_count * 2`；
- thickness：一个有限正 binary64 标量。

### T4 布局

- points：`node_count * 3`；cells：`cell_count * 4`；
- constitutive：逐单元 `cell_count * 36` 行主序值；
- body force：三个全局分量；
- rows/columns/data：`cell_count * 144`，每单元十二个节点主序向量 DOF；
- load：`node_count * 3`。

### CSR 布局

`afn_csr_spmv` 接收 `row_count + 1` 个行指针、`nonzero_count` 个有符号 64 位
列索引与 binary64 数值、`column_count` 个向量值，并输出 `row_count` 个结果。
空矩阵继续由可读层处理；ABI 入口要求非空缓冲区。

## ABI 1.3 确定性 CPU 并行入口

ABI 1.3 新增：

```text
afn_t3_elasticity_assemble_cells_parallel(..., size_t thread_count, ...)
afn_t4_elasticity_assemble_cells_parallel(..., size_t thread_count, ...)
```

`thread_count` 必须大于零，并在内部限制为不超过 `cell_count`。调用方显式选择
线程数；内核不得读取环境变量或静默使用全部硬件线程。

并行契约：

1. 单元按输入顺序划分为连续、确定的静态区间；
2. 每个线程只写自己的 COO 输出区间，因此 rows、columns、data 的顺序与串行
   入口完全相同；
3. 每个线程使用私有节点载荷缓冲区，完成后按线程编号从小到大归并；
4. 同一输入和线程数必须逐次可重复；不同线程数的 COO 输出必须完全相同，载荷
   允许由归约分组导致的舍入级差异；
5. 私有载荷内存上界为
   `thread_count * node_count * vector_dimension * sizeof(double)`，计划层必须报告
   这一额外资源；
6. 任一工作线程返回非零状态时，所有输出均无效；调用方不得消费部分结果；
7. 线程创建、内部缓冲分配或其他 C++ 异常必须在 ABI 内捕获并转换成状态码。

第一阶段只并行化 T3/T4 单元体积装配。扩散与 CSR SpMV 保留串行入口，直到独立
测量证明并行收益覆盖线程启动成本。

## 数值契约

二维 P1 扩散对每个仿射三角形计算解析物理梯度、面积、`area * G K G^T` 和
常源精确载荷。T3 计算 `thickness * area * B^T D B` 与常体力精确载荷。T4
按照 `T4_LINEAR_ELASTICITY.md` 的应变和 DOF 顺序计算
`volume * B^T D B` 与常体力精确载荷。

每个电导或本构矩阵必须有限、在规定浮点容差内对称且正定。坐标、源和体力必须
有限；连通性必须在范围内且单元不得退化。COO 始终按单元、局部行、局部列排序。
CSR SpMV 在每行内按存储列顺序归约。项目维护的 Python 有限元参考实现继续作为
差分依据；NumPy 本身不是自研对象。

## 失败契约

| 状态码 | 含义 |
|---:|---|
| 0 | 成功 |
| 1 | 空指针、空网格或空单元集合 |
| 2 | 无效电导或本构矩阵 |
| 3 | 无效连通性、尺寸范围或退化几何 |
| 4 | 非有限坐标、源、体力或计算结果 |
| 5 | `thread_count` 为零 |
| 6 | 线程或内部临时资源创建失败 |

任何非零状态都使所有输出缓冲区无效，包括失败前已经写入的条目。Python 边界在
构造公开矩阵结果前把状态转换为稳定异常。

## 兼容规则

次版本可以新增函数或诊断，但不能改变已有布局或含义。旧调用方可以继续使用
对应次版本中已经存在的函数。缓冲布局、标量宽度、排序、状态或数学语义发生不
兼容变化时必须提升主版本。

ABI 1.3 必须经过 Windows x86_64、Linux x86_64、macOS x86_64/arm64 wheel、
C/C++ 契约、sanitizer 和 Python Stable ABI 复用验收后，才能成为正式平台证据。
