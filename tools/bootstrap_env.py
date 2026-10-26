# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""用同一 Python 脚本创建 Windows/macOS/Linux 开发环境，无需激活脚本。"""

from __future__ import annotations

import argparse
import subprocess
import sys
import venv
from pathlib import Path


def environment_python(environment: Path) -> Path:
    if sys.platform == "win32":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-scipy", action="store_true", help="安装可选 SciPy 稀疏提供者"
    )
    parser.add_argument(
        "--agentfem",
        metavar="SOURCE",
        help="同时从本地源码目录或包要求安装 AgentFEM（不是内核必需依赖）",
    )
    parser.add_argument(
        "--environment",
        type=Path,
        help="环境目录，默认仓库内 .venv；允许中文及空格路径",
    )
    parser.add_argument(
        "--no-build-isolation",
        action="store_true",
        help="使用环境中已有构建依赖，适合已准备的离线开发环境",
    )
    arguments = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    environment = (arguments.environment or repository / ".venv").resolve()
    if not environment_python(environment).exists():
        venv.EnvBuilder(with_pip=True).create(environment)
    package = ".[scipy]" if arguments.with_scipy else "."
    subprocess.run(
        [
            str(environment_python(environment)),
            "-m",
            "pip",
            "install",
            *(["--no-build-isolation"] if arguments.no_build_isolation else []),
            "--editable",
            package,
        ],
        cwd=repository,
        check=True,
    )
    if arguments.agentfem:
        source = Path(arguments.agentfem).expanduser()
        target = str(source.resolve()) if source.exists() else arguments.agentfem
        command = [str(environment_python(environment)), "-m", "pip", "install"]
        if source.exists():
            command.append("--editable")
        command.append(target)
        subprocess.run(command, cwd=repository, check=True)
    # 显式使用该环境解释器，不依赖 PowerShell 执行策略或 POSIX activate。
    subprocess.run(
        [
            str(environment_python(environment)),
            "-m",
            "agentfem_native.cli",
            "--diagnose",
            "--require-native",
        ],
        check=True,
    )
    print(f"AgentFEM Native 开发环境已就绪：{environment}")
    return 0


if __name__ == "__main__":
    try:
        status = main()
    except subprocess.CalledProcessError as error:
        print(
            "开发环境安装/诊断失败；请查看上述具体错误。原有环境未被删除。",
            file=sys.stderr,
        )
        if sys.platform == "win32":
            print(
                "Windows 源码构建需要 Visual Studio Build Tools 的 C++ 桌面工具链。请在 x64 开发者终端重试；运行已构建的包不需要 WSL 或编译器。",
                file=sys.stderr,
            )
        status = error.returncode
    raise SystemExit(status)
