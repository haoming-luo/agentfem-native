# Gate 2 验收矩阵

**日期：** 2026-09-20
**整体成熟度：** `verified`
**证据源提交：** `654fe02`
**完整 Tier-1：** [GitHub Actions 运行 `35461748299`](https://github.com/haoming-luo/agentfem-native/actions/runs/35461748299)

**用途：** 把 T3/T4 基础线性静力范围压缩成有限、可审计、已关闭的验收清单。

| 编号 | 验收声明 | 最小证据 | 当前状态 |
|---|---|---|---|
| G2-01 | T3 材料与单元数学正确 | 平面应力/应变闭式矩阵、对称性、3 个刚体模态、仿射应变恢复 | 本地通过 |
| G2-02 | T3 载荷和约束保持物理合力 | 体力/边界力精确合力、欠约束与冲突约束失败 | 本地通过 |
| G2-03 | T3 工程响应与恢复正确 | 常应变 patch、单轴、纯剪、均匀膨胀、悬臂对称性 | 本地通过 |
| G2-04 | T3 实际求解随网格收敛 | 三层网格制造解、自由残差与全局平衡 | 本地通过 |
| G2-05 | T3 优化路径不改变语义 | 参考、向量化、C++ 装配及稀疏/稠密小问题全输出差分 | 本地通过 |
| G2-06 | T4 拓扑、几何和单元数学正确 | 体积、边界面、6 个刚体模态、方向置换、仿射应变 | 本地通过 |
| G2-07 | T4 载荷、约束和恢复闭合 | 体力/面力合力、欠约束失败、反力、能量和应力恢复 | 本地通过 |
| G2-08 | T4 工程响应正确 | 立方体 patch、单轴、纯剪和静水响应 | 本地通过 |
| G2-09 | T4 求解收敛并容许合法畸变 | 三层网格制造解、畸变网格 patch、方向/平衡不变量 | 本地通过 |
| G2-10 | T4 优化路径不改变语义 | Python 参考与 C++20 ABI 1.2 的矩阵、载荷及求解差分 | 本地通过 |
| G2-11 | AgentFEM 能审查一次力学执行 | 计划摘要、维数、单元、路径、提供者、残差/平衡、成熟度 | Contract 0.2 本地与安装态通过 |
| G2-12 | 生产性能和三平台 Gate 证据完整 | P3 CPU 并行记录与 Gate 候选 Windows/Linux/macOS 验收 | 通过：ABI 1.3/1.4 与 Contract 0.2 三平台证据齐全 |

## 判定规则

- G2-01 至 G2-12 已关闭，T3/T4 基础线性静力声明提升为 `verified`；
- `verified` 只覆盖本表声明，不代表外部实验 `validated`，也不代表整体内核已经
  `production`；
- 任一收敛、残差、平衡或差分断言失败时记录最小反例，不扩大容差；
- FEniCSx 等第三方仅在隔离的里程碑验证中提供少量黑盒交叉证据，不能替代本表
  中的解析、守恒和收敛依据。

## 直接证据入口

- `tests/test_elasticity.py`：G2-01 至 G2-05；
- `tests/test_solid.py`：G2-06 至 G2-10；
- `tests/test_planning.py`、`tests/test_execution.py` 与 `tests/test_contract.py`：
  G2-11；
- `docs/verification/KERNEL_CONTRACT_0_2_LOCAL_CANDIDATE.md`：0.1 兼容、T3/T4
  契约、预检、证据和失败边界；
- `docs/verification/MECHANICS_ALPHA_0_2_REPORT.md`：已通过的 ABI 1.2 本地与平台
  证据；
- `docs/verification/PERFORMANCE_P3_REPORT.md`：G2-12 当前性能边界。

## 平台验收摘要

源提交 `654fe02` 先通过普通 Linux 快检 `35461701061`，随后完整 Tier-1 运行
`35461748299` 通过 Windows x86_64、Linux x86_64、macOS x86_64/arm64 wheel，
Python 3.13 Stable ABI 复装科学套件，三平台 C++20 合同、sanitizer、Rust/C++/NumPy
对照、产物清单与最终汇总门禁。运行中的 Ubuntu 26 迁移提示不影响本次结果，需在
2026-10-19 前单独做 runner 迁移预检。
