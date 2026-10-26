# 原生 Windows 开发与 AgentFEM 集成

本页是开发代码包的说明，不代表已公开发布安装包。Windows、macOS、Linux
使用相同源码、数学规格和版本契约；Windows 验收不允许用 WSL 代替。

## 环境准备

Windows x86_64 使用 64 位 CPython 3.11 或更新的受测版本。源码构建需要支持
C++20 的 Visual Studio Build Tools，安装 C++ 桌面工作负载，在 x64 开发者终端
运行以下命令。脚本不要求 PowerShell 激活环境，不修改系统 PATH，不安装 WSL。

```text
python tools/bootstrap_env.py
.venv\Scripts\python.exe -m agentfem_native.cli --diagnose --require-native
.venv\Scripts\python.exe -m unittest discover -q
```

macOS/Linux 对应解释器为 `.venv/bin/python`。`--environment "工作环境 FEM"`
可选择其他目录。中文和空格路径属于测试范围，不要求用户改成纯英文目录。
源码采用 UTF-8，MSVC 显式启用 `/utf-8`。

可选环境：

```text
python tools/bootstrap_env.py --with-scipy
python tools/bootstrap_env.py --agentfem "已有 AgentFEM 源码目录"
```

Native 的必需通用运行依赖只有 NumPy；SciPy、PETSc、MPI 不拥有有限元语义，
缺席时不关闭自有串行 CSR 求解。此批多物理候选不要求它们，也没有新增 PETSc
适配器。已编译开发 wheel 的使用端不需要 C++ 编译器，测试 wheel 不公开发布。

## AgentFEM 的使用边界

1. 查询 `native_capabilities()` 中的稳定 Gate 能力；
2. 对新能力另查 `multiphysics_capabilities()` 和 `multiphysics_request_schema()`；
3. 先 `plan_multiphysics_request()`，确认维数、预算、成熟度和限制；
4. 使用同一请求调用 `run_multiphysics_request()`；保留请求、计划和结果摘要。

候选 JSON 版本为 `0.4.0-experimental.1`，不是稳定 0.1/0.2/0.3 契约的替代。
有限变形和 AT2 必须显式 `experimental=true`。`examples/multiphysics_candidate.py`
生成二维/三维热、热力、大变形和断裂的十个合成请求，供开发重放。

```text
.venv\Scripts\python.exe examples/multiphysics_candidate.py --directory "work/多物理 验收"
.venv\Scripts\python.exe -m agentfem_native.cli "work/多物理 验收/finite_strain-3d-request.json" --plan-only
```

诊断输出区分本机原生 Windows 与 WSL，并真正调用 C++ ABI 检查一个手算稀疏
乘法。它不外推 Windows 性能，也不证明所有模型正确。可选包的元数据存在只说明
安装信息，不保证提供者可用。

## 文件、失败与取消

- 结果以 UTF-8 写入同目录独占临时文件，关闭后替换；锁定/权限失败返回
  `io.result_write_failed`，不会强制覆盖旧结果。
- 结果路径不能覆盖输入请求。多写者只保证完整文件可见，最后成功替换者生效；
  不合并结果，也不承诺断电后持久性。
- 非线性取消保留已接受历史，不返回被拒绝的 Newton/损伤试探。该历史尚不是
  持久化、可跨版本继续计算的检查点协议。
- 新模型仍是候选。AT2 无张压分裂，不得用于压缩裂纹闭合；有限变形无接触、
  弧长跟踪、壳梁或近不可压缩混合公式。

## 平台证据

本地只负责 macOS；新增测试进入既有 GitHub Tier-1 安装态矩阵，在真正 Windows
和 Linux runner 上验证。普通推送仍只有 Linux 快检。未运行的新矩阵必须写成
待验收，不得复用旧提交的绿色结果作为新代码的证据。
