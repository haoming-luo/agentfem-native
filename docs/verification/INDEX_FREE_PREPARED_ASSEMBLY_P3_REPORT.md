# P3 ABI 1.5 无重复 COO 索引预备装配验收报告

**日期：** 2026-09-20

**成熟度：** `implemented`；ABI 1.5 三平台实现边界已验收，P3 整体仍未关闭

- **源提交：** `28b2ce7`
- **Linux 快检：** [GitHub Actions `35462720452`](https://github.com/haoming-luo/agentfem-native/actions/runs/35462720452)
- **完整 Tier-1：** [GitHub Actions `35462761452`](https://github.com/haoming-luo/agentfem-native/actions/runs/35462761452)

## 产品结论

固定 T3/T4 拓扑的 `CSRAssemblyPlan` 已经拥有全部行列结构，因此串行 native
预备装配不再重复生成 COO `rows/columns`。ABI 1.5 只返回按原贡献顺序排列的
局部数值与载荷，再交给既有确定性映射回填 CSR。普通 COO API、并行装配和
AgentFEM 的计划所有权边界均保持不变。

这个增量是“元素到 CSR”内存路线的第一步：它精确消除了两份大索引数组，但仍
保留局部贡献数组与 `coo_to_csr` 映射，所以不宣称已经完成最终直接 CSR 装配。

## 数值与构建证据

- T3/T4 仅数值入口与旧完整 COO 的 `data/load` 逐位一致；
- 冷路径与预备路径的最终 CSR `indptr/indices/data` 逐位一致；
- 194 项 Python 测试、53 个子用例、Ruff、格式和独立性扫描通过；
- C++20 使用 `-Wall -Wextra -Wpedantic -Werror` 直编合同通过；
- AddressSanitizer/UndefinedBehaviorSanitizer 与 C11 头文件编译通过；
- macOS arm64 `cp311-abi3` wheel 严格构建并在隔离环境安装，安装态 ABI 1.5
  T3 预备装配冒烟通过。
- Windows x86_64、Linux x86_64、macOS x86_64/arm64 安装态 wheels、三平台
  C/C++ 契约、sanitizer、Python 3.13 Stable ABI 复用、Rust/C++/NumPy 对照、
  工件清单与验收汇总在运行 `35462761452` 中全部通过。

## macOS arm64 本机记录

环境：Python 3.12.14、NumPy 2.3.5、ABI 1.5；热进程七次中位数。原始记录见
`benchmarks/results/2026-09-20-darwin-arm64-prepared-values-abi15.json`。

| 单元 | 网格 | 单元数 | DOF | 每次省去索引 | 冷 COO→CSR | 预备数值→CSR | 阶段倍率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| T3 | 16² | 512 | 578 | 0.295 MB | 2.085 ms | 0.067 ms | 30.90× |
| T3 | 32² | 2,048 | 2,178 | 1.180 MB | 8.302 ms | 0.212 ms | 39.16× |
| T3 | 64² | 8,192 | 8,450 | 4.719 MB | 34.066 ms | 0.780 ms | 43.66× |
| T4 | 2³ | 48 | 81 | 0.111 MB | 0.688 ms | 0.063 ms | 10.98× |
| T4 | 4³ | 384 | 375 | 0.885 MB | 4.937 ms | 0.352 ms | 14.03× |
| T4 | 8³ | 3,072 | 2,187 | 7.078 MB | 41.985 ms | 2.624 ms | 16.00× |

“每次省去索引”严格等于 `16 × contribution_count`，只计算两份不再分配的
`int64` 数组，不是进程 RSS。阶段倍率同时包含已验证的图复用和本次索引消除，
不能单独归因为 ABI 1.5，也不能外推为完整求解或其他平台速度。

## 剩余边界

- 多线程预备装配仍生成完整 COO，避免未经规格化的共享 CSR 归并；
- 贡献数组、映射、材料数据、CSR 结果、载荷和求解器工作区仍存在；
- 最终元素直接 CSR、SIMD 和更强预条件器仍是 P3 后续项；
- ABI 1.5 平台边界已经关闭；性能只在 macOS arm64 测量，因此没有跨平台速度
  或 RSS 结论。
