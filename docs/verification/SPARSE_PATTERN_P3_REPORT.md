# P3 可复用 CSR 稀疏图参考候选报告

**日期：** 2026-09-13

**成熟度：** 参考数据层已实现并通过 Linux 快检；C++ 生产回填待实现

## 已完成

- 新增不可变 `CSRPattern`，一次保存规范 `indptr/indices` 和原始 COO 贡献到
  CSR 数值槽位的映射；
- 支持从一般 COO 结构或二维单元 DOF 表构图；
- `fill(values)` 只更新数值，不重新排序图，并让结果矩阵安全共享只读图数组；
- `CSRMatrix.from_coo` 复用同一契约，不保留两套重复项归并语义；
- 图提供跨平台字节序稳定的 SHA-256 结构摘要、存储量和贡献项计数；
- 长度、非有限值、越界 DOF、无效形状与无效映射均显式失败。

175 项完整测试、Ruff、格式和独立性扫描在 macOS arm64 本地通过。新增三项测试
覆盖重复贡献的多次回填、单元局部矩阵顺序、只读共享、摘要和失败路径。
源提交 `9e33b6c` 的
[GitHub Actions Linux 快检 `34753072074`](https://github.com/haoming-luo/agentfem-native/actions/runs/34753072074)
也已通过。该纯 Python/NumPy 参考增量没有重复运行刚完成的完整平台矩阵；当
C++ ABI 数值回填形成平台候选时，再执行下一次 Tier-1 验收。

## 局部成本信号

Python 3.12.14、NumPy 2.3.5 热进程中，每项取九次中位数：

| T3 网格 | COO 贡献 | CSR nnz | 每次重新规范化 | 复用图数值回填 | 局部加速 |
|---:|---:|---:|---:|---:|---:|
| 32² | 73,728 | 29,444 | 8.148 ms | 0.061 ms | 133.48× |
| 64² | 294,912 | 116,228 | 35.162 ms | 0.247 ms | 142.36× |
| 128² | 1,179,648 | 461,828 | 148.811 ms | 1.015 ms | 146.65× |

全部图和数值与冷 `from_coo` 逐位一致。原始记录见
[`benchmarks/results/2026-09-13-darwin-arm64-sparse-pattern-reuse.json`](../../benchmarks/results/2026-09-13-darwin-arm64-sparse-pattern-reuse.json)，
重跑入口见 [`benchmarks/sparse_pattern_reuse.py`](../../benchmarks/sparse_pattern_reuse.py)。

## 声明边界

这些数字只比较“已有 COO 的重新排序/归并”和“已有图的数值回填”，不包含单元
计算、约束、预条件器或求解，不能宣称端到端快 100 倍。下一生产增量应让 C++
T3/T4 直接写入映射后的预分配数值区，并测量完整装配与多载荷重复求解；在那之前
本能力保持参考数据层成熟度。
