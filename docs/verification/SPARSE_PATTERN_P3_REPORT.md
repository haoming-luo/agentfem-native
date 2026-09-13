# P3 可复用 CSR 稀疏图与 ABI 1.4 验收报告

**日期：** 2026-09-13

**成熟度：** 参考数据层与 ABI 1.4 生产回填已实现；Tier-1 三平台验收通过

## 已完成

- 新增不可变 `CSRPattern`，一次保存规范 `indptr/indices` 和原始 COO 贡献到
  CSR 数值槽位的映射；
- 支持从一般 COO 结构或二维单元 DOF 表构图；
- `fill(values)` 只更新数值，不重新排序图，并让结果矩阵安全共享只读图数组；
- `CSRMatrix.from_coo` 复用同一契约，不保留两套重复项归并语义；
- 图提供跨平台字节序稳定的 SHA-256 结构摘要、存储量和贡献项计数；
- 长度、非有限值、越界 DOF、无效形状与无效映射均显式失败。
- 新增独立 `fill_reference` 与 C++20 生产 `fill`；两者严格按原 COO 贡献顺序
  归并并逐位一致，编译内核不可用时确定性回退；
- ABI 1.4 在调用前校验全部映射与贡献，空图、越界和非有限归并具有稳定语义。

177 项完整测试、Ruff、格式、独立性扫描和直接 C/C++ 合约编译在 macOS arm64
本地通过。源提交 `d172cf6` 的
[Linux 快检 `34757574303`](https://github.com/haoming-luo/agentfem-native/actions/runs/34757574303)
通过；[Tier-1 运行 `34757610403`](https://github.com/haoming-luo/agentfem-native/actions/runs/34757610403)
的 14 项验收全部通过，包括四类平台 wheel、Windows/Linux/macOS C/C++ 合约、
sanitizer、三平台 Python 3.13 Stable ABI 复用、Rust 对照和工件清单。

## 局部成本信号

Python 3.12.14、NumPy 2.3.5 热进程中，每项取九次中位数：

| T3 网格 | COO 贡献 | 每次重新规范化 | C++ 回填 | NumPy 参考回填 | 冷转换/C++ |
|---:|---:|---:|---:|---:|---:|
| 32² | 73,728 | 8.018 ms | 0.060 ms | 0.064 ms | 134.18× |
| 64² | 294,912 | 33.213 ms | 0.227 ms | 0.233 ms | 146.34× |
| 128² | 1,179,648 | 138.443 ms | 0.971 ms | 0.994 ms | 142.60× |

全部图和数值与冷 `from_coo` 逐位一致。原始记录见
[`benchmarks/results/2026-09-13-darwin-arm64-sparse-pattern-reuse-abi14.json`](../../benchmarks/results/2026-09-13-darwin-arm64-sparse-pattern-reuse-abi14.json)，
重跑入口见 [`benchmarks/sparse_pattern_reuse.py`](../../benchmarks/sparse_pattern_reuse.py)。

## 声明边界

这些数字只比较“已有 COO 的重新排序/归并”和“已有图的数值回填”，不包含单元
计算、约束、预条件器或求解，不能宣称端到端快 100 倍。C++ 相对 NumPy 回填只
快 1.02×–1.07×；它当前的主要价值是自主生产边界、稳定 ABI 与释放 GIL 的演进
起点。下一增量应接入真实 T3/T4 多载荷或隐式动力生命周期，测量完整装配、复制
和峰值内存后，再决定批量回填、线程私有缓冲或直接元素到 CSR 的方案。
