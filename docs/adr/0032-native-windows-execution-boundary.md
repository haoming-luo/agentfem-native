# ADR-0032：原生 Windows 安装态与自主串行执行边界

状态：接受；日期：2026-10-27。

## 决策

Windows x86_64 使用与 macOS/Linux 相同的 Python、C++20、单元和稀疏求解实现，
不建立 Windows 专属数学分支，也不转写第三方有限元实现。MSVC 明确按 UTF-8
编译，避免中文注释依赖开发机代码页。运行时不要求编译器、POSIX shell、WSL、
PETSc、MPI 或 SciPy；源码开发构建需要平台 C++20 工具链。

增加安装态诊断入口，报告当前 OS/架构、WSL 身份、Python/NumPy 版本、实际
C++ ABI 与能力成熟度。检查加速器的真实数值调用，不以“模块能够导入”替代。
可选依赖仅查询分发元数据，不导入、不安装。诊断失败返回机器错误码与中文建议。
本机自检是运行环境证据，不是数值验证等级，也不是另一个平台的验收。

JSON 文件使用 UTF-8、同目录独占临时文件、关闭句柄后替换目标；失败只清理本次
创建的临时文件，不破坏原目标。并发写保证每次可见内容完整，最后一次成功替换
生效；不承诺多写者合并或断电持久性。文件被 Windows 程序锁定时明确报告失败，
不强制解除其他程序的锁、不自动重试覆盖。中文、空格路径进入安装态子进程测试。

## CI 验收

开发继续仅在本地 macOS；普通推送一个 Linux 快检。Tier-1 安装包验收在已有
Windows/Linux/macOS jobs 内加入上述诊断，不新增常驻矩阵。Windows 必须由
`windows-latest` 原生 Python/MSVC 执行；WSL 结果不得计入 Windows 通过。
安装产物只供临时测试，不是公开发布。

## 竞争判断与来源

“其他内核只有 Linux”不是准确的产品前提。FEniCSx v0.9.0 公开发布说明已经列出
原生 Windows 支持；PETSc 官方文档列出 Microsoft/Intel 以及 MSYS2 等构建路径。
本项目的差异是面向 AgentFEM 的自主有限元语义、轻量无 MPI 串行主干、明确的
Agent 预算/能力/失败边界，而不是宣称独占 Windows 支持。

- https://fenicsproject.org/blog/v0.9.0/ （公开发布说明；不读取实现源码）
- https://petsc.org/main/install/windows/ （公开安装要求）

新多物理候选仍分别保持 `implemented`/`experimental`，不借平台诊断升级。
