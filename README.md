# AgentFEM Native 自主有限元内核

AgentFEM Native 是面向 AgentFEM 独立开发的有限元求解内核。它是正式的自主
计算后端，不是 FEniCSx 分支，也不是第二个面向最终用户的产品。

本项目唯一首要的产品目标，是成为 AgentFEM 面向未来工程建模、Agent 自动
求解与可信解释的自主计算底座。Native 保持独立的数学与运行时边界，但不发展
成与 AgentFEM 平行的终端产品；双方通过版本化 Kernel Contract 连接，而不是
共享易变的内部对象。

AgentFEM Native 同时承担国产替代和自主掌握有限元求解内核的使命。项目自主
拥有有限元语义、核心算法、C++ 计算路径、跨平台构建和验证证据，而不是给国外
有限元内核再包一层接口。NumPy、SciPy、PETSc、MPI、BLAS 等通用且许可清晰的
基础设施可以作为可替换能力使用，但不能控制 Native 的数学定义或成为串行核心
唯一可运行的条件。

仓库已经完成 **Gate 1**，并实现 T3/T4 Gate 2 垂直切片和 Gate 3 线性候选。
已验证串行内核覆盖二维 P1 三角形稳态扩散、命名集合、材料区域、各向异性电导、
确定性 COO、自有 CSR/CG/Jacobi、可选 NumPy/SciPy 对照与提供者、平衡证据和
VTK 输出。力学切片覆盖向量 DOF、T3 平面应力/应变、T4 三维弹性、体力与面力、
位移约束、反力、能量和应力/应变恢复。C++20 ABI 1.4 在 T3/T4 与 CSR SpMV
基础上，拥有显式线程数的确定性 T3/T4 CPU 并行装配和可复用图的确定性 CSR
数值回填；`CSRAssemblyPlan` 已形成 T3/T4 与动力学的本地接入候选，使 AgentFEM
可显式拥有固定拓扑的重复装配计划。默认仍为单线程，避免 Agent 工作流过度订阅。
固定 DOF 线性动力学、功—能量证据、
内部网格/结果交换和 Agent 的计划—执行—解释收据也已实现，且不把 AF-IR 作为
核心依赖。API 仍处于 1.0 前实验阶段，Gate 2 尚未因实现完成而自动提升为已验证。

## 稳定架构方向

```text
AgentFEM engineering model
        -> versioned Kernel Contract
        -> AgentFEM Native
        -> replaceable linear-algebra and hardware providers
```

The same AgentFEM engineering model is intended eventually to support:

```python
study.solve()
study.solve(backend="fenicsx")
study.solve(backend="native")
```

Windows、macOS 和 Linux 都是一等发布目标；三者共享同一公开契约、科学测试、
构建和独立性检查，不维护不同物理分支。本地开发负责 macOS，GitHub CI 负责
Windows、Linux 以及最终 Tier-1 权威验收。当前远端基线是提交 `d172cf6` 的
GitHub Actions 运行 `34757610403`：ABI 1.4 的 14 项 Windows/Linux/macOS 验收
全部通过。CI wheel 只是短期安装测试工件，不是公开发布包；项目仍快速迭代，
尚未发布到 PyPI 或 GitHub Releases。

The GitHub repository is temporarily public through September 2026 to avoid
private Actions-minute pressure during intensive development. This does not
change the PolyForm Noncommercial license or permit unrestricted commercial
use. Ordinary pushes run one Linux fast job; full native Tier-1 acceptance is
manual or tag-triggered. See
[CI governance](docs/development/CI_GOVERNANCE.md).

项目的重要注释、介绍、规格和验证说明从 2026-09-13 起以中文为主要工程表达
语言；稳定代码标识符和机器契约继续使用英文。参见
[中文工程表达规范](docs/development/CHINESE_DOCUMENTATION_POLICY.md)和
[后续架构与开发计划](docs/charter/NEXT_DEVELOPMENT_PLAN.md)。

## 当前证据与环境

Create the repository-local environment on Windows, macOS, or Linux:

```text
python tools/bootstrap_env.py
python tools/bootstrap_env.py --with-scipy  # optional sparse provider
python tools/bootstrap_env.py --agentfem /path/to/agentfem
```

For this development checkout, the active `.venv` contains editable
AgentFEM 0.3.1 from the clean local `agentfem-main-worktree` and editable
AgentFEM Native 0.7.0a1 with its C++20 extension and optional SciPy provider.
Installing both does not make AgentFEM or FEniCSx a Native runtime dependency;
their communication remains the public contract.

Then run the reference tests without FEniCSx:

```text
python -m unittest discover -v
python tools/check_independence.py
```

执行一个可移植的 Kernel Contract 请求：

```text
agentfem-native examples/linear_elasticity_t3_request.json \
  --artifact-directory work/example --result work/example-result.json
```

AgentFEM 或智能体在正式求解前，可以先做不装配、不求解的资源预检：

```text
agentfem-native examples/linear_elasticity_t3_request.json --plan-only
```

Windows PowerShell 使用同一条单行命令。命令行与 Python API 都返回版本化、
JSON 安全的结果；失败包含稳定错误码与字段路径。Contract 0.2 覆盖 T3/T4
线性静力，并继续兼容 0.1 热传导请求；详见
[Kernel Contract 0.2 规格](docs/specifications/KERNEL_CONTRACT_0_2.md)。

See [ROADMAP.md](ROADMAP.md), [ARCHITECTURE.md](ARCHITECTURE.md), the
[project charter](docs/charter/PROJECT_CHARTER.md), and the
[steady-diffusion specification](docs/specifications/STEADY_DIFFUSION.md).
环境说明见[开发环境](docs/development/ENVIRONMENT.md)，编译边界见
[Native ABI 规格](docs/specifications/NATIVE_P1_ABI.md)。性能与力学证据分别见
[P2 报告](docs/verification/PERFORMANCE_P2_REPORT.md)、
[P3 稀疏报告](docs/verification/PERFORMANCE_P3_REPORT.md)、
[P3 CPU 并行候选报告](docs/verification/CPU_PARALLEL_P3_REPORT.md)、
[P3 可复用稀疏图候选报告](docs/verification/SPARSE_PATTERN_P3_REPORT.md)、
[Gate 2 T3 参考报告](docs/verification/GATE_2_T3_REFERENCE_REPORT.md)和
[Mechanics Alpha 0.2 报告](docs/verification/MECHANICS_ALPHA_0_2_REPORT.md)。
长期技术判断记录在[2035 愿景](docs/charter/NATIVE_VISION_2035.md)中。

## Licensing

The kernel is source-available under PolyForm Noncommercial 1.0.0. Others may
use it only for purposes permitted by that license; commercial use requires a
separate written AgentFEM commercial license. See [LICENSE](LICENSE) and
[LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md). It is not OSI-approved open
source.
