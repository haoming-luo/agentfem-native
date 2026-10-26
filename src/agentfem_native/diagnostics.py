# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""安装态诊断：本机身份、真实 ABI 调用与可选依赖边界。"""

from __future__ import annotations

import platform
import struct
import sys
from importlib.metadata import PackageNotFoundError, version

import numpy as np

from .multiphysics import multiphysics_capabilities
from .native import csr_spmv, ic0_factor, ldlt_solve, native_kernel_identity


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def runtime_diagnostics(*, require_native: bool = False) -> dict:
    """只诊断当前进程，不安装依赖、不调用 shell、不外推平台或科学成熟度。"""
    system, machine = platform.system(), platform.machine().lower()
    architecture = {"amd64": "x86_64", "aarch64": "arm64"}.get(machine, machine)
    wsl = system == "Linux" and "microsoft" in platform.release().lower()
    tier_one = (system, architecture) in {
        ("Windows", "x86_64"),
        ("Linux", "x86_64"),
        ("Darwin", "x86_64"),
        ("Darwin", "arm64"),
    } and struct.calcsize("P") == 8
    identity = native_kernel_identity()
    checks = []
    if identity["available"]:
        try:
            # 实际经过自有 C ABI；只用手算 CSR，不让 NumPy 替代加速器调用。
            actual = csr_spmv(
                (2, 2),
                np.array([0, 2, 4], dtype=np.int64),
                np.array([0, 1, 0, 1], dtype=np.int64),
                np.array([2.0, -1.0, -1.0, 2.0]),
                np.array([3.0, 4.0]),
            )
            if not np.array_equal(actual, [2.0, 5.0]):
                raise ValueError("CSR 手算结果不一致。")
            factors, diagonal = ic0_factor([0, 0, 1], [0], [-1.0], [2.0, 2.0])
            applied = ldlt_solve([0, 0, 1], [0], factors, diagonal, [1.0, 1.0])
            if not np.array_equal(applied, [1.0, 1.0]):
                raise ValueError("IC0/LDLT 手算结果不一致。")
        except (ValueError, RuntimeError, AttributeError) as error:
            checks.append(
                {
                    "code": "native.smoke_failed",
                    "status": "failed",
                    "message": str(error),
                }
            )
        else:
            checks.append(
                {
                    "code": "native.smoke_passed",
                    "status": "passed",
                    "message": "C++20 ABI 稀疏乘法与 IC0/LDLT 手算检查通过（不代表实验算法已晋级）。",
                }
            )
    else:
        checks.append(
            {
                "code": "native.unavailable",
                "status": "failed" if require_native else "warning",
                "message": "C++20 扩展缺失或 ABI 不匹配；参考路径不等于编译内核验收。请在当前 Python 环境重新构建/安装匹配平台的开发包。",
            }
        )
    if not tier_one:
        checks.append(
            {
                "code": "platform.outside_tier1",
                "status": "warning",
                "message": "当前 OS/架构不在已声明 Tier-1 范围；不得声称通过 Tier-1 验收。",
            }
        )
    return {
        "diagnostic_version": "1.0.0",
        "status": "failed"
        if any(check["status"] == "failed" for check in checks)
        else "success",
        "runtime": {
            "os": system,
            "architecture": architecture,
            "pointer_bits": struct.calcsize("P") * 8,
            "python": platform.python_version(),
            "implementation": sys.implementation.name,
            "numpy": np.__version__,
            "native_windows": system == "Windows",
            "wsl": wsl,
            "tier1_target": tier_one,
        },
        "native_kernel": identity,
        "candidate_capabilities": multiphysics_capabilities(),
        "optional_distributions": {
            name: _version(name) for name in ("scipy", "petsc4py", "mpi4py")
        },
        "required_runtime_dependencies": ["numpy"],
        "checks": checks,
        "scope": "仅为当前安装态自检；不替代科学验证或其他平台的 CI 证据。",
    }
