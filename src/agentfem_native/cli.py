# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""AgentFEM Native 的跨平台 JSON 命令行边界。"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from .contract import (
    CONTRACT_NAME,
    CONTRACT_VERSION,
    plan_kernel_request,
    run_kernel_request,
)
from .diagnostics import runtime_diagnostics
from .jsonio import write_json_atomic
from .multiphysics import (
    MULTIPHYSICS_VERSION,
    plan_multiphysics_request,
    run_multiphysics_request,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentfem-native",
        description="执行或预检 AgentFEM Native Kernel Contract 请求。",
    )
    parser.add_argument("request", type=Path, nargs="?", help="UTF-8 JSON 请求文件")
    parser.add_argument(
        "--diagnose", action="store_true", help="诊断当前安装与原生 ABI，不运行用户模型"
    )
    parser.add_argument(
        "--require-native",
        action="store_true",
        help="诊断时要求 C++20 内核可用，供安装态验收",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="只返回能力与资源计划，不装配也不求解",
    )
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        help="请求结果产物的根目录",
    )
    parser.add_argument(
        "--result",
        type=Path,
        help="把 JSON 结果写入文件，而不是标准输出",
    )
    return parser


def _write_result(
    result: dict[str, object], destination: Path | None, stream: TextIO
) -> None:
    encoded = json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if destination is None:
        stream.write(encoded)
        return
    write_json_atomic(result, destination)


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    """运行 CLI，并返回跨平台进程退出码。"""

    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.diagnose:
        if arguments.request or arguments.plan_only or arguments.artifact_directory:
            parser.error("--diagnose 不能与计算请求、预检或产物目录混用。")
    elif arguments.request is None or arguments.require_native:
        parser.error("请提供请求文件；--require-native 仅用于 --diagnose。")
    stream = sys.stdout if stdout is None else stdout
    try:
        if arguments.diagnose:
            request = None
            result = runtime_diagnostics(require_native=arguments.require_native)
        else:
            if (
                arguments.result
                and arguments.request.resolve() == arguments.result.resolve()
            ):
                raise ValueError("结果文件不得覆盖原始请求文件。")
            request = json.loads(arguments.request.read_text(encoding="utf-8"))
        if not arguments.diagnose and not isinstance(request, dict):
            raise TypeError("请求根节点必须是 JSON 对象。")
        if arguments.diagnose:
            pass
        elif request.get("contract_version") == MULTIPHYSICS_VERSION:
            if arguments.artifact_directory is not None:
                raise ValueError(
                    "多物理候选当前通过 JSON 返回结果，不支持产物目录选项。"
                )
            result = (
                plan_multiphysics_request(request)
                if arguments.plan_only
                else run_multiphysics_request(request)
            )
        elif arguments.plan_only:
            result = plan_kernel_request(request)
        else:
            result = run_kernel_request(
                request,
                artifact_directory=arguments.artifact_directory,
            )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as error:
        result = {
            "contract": CONTRACT_NAME,
            "contract_version": CONTRACT_VERSION,
            "request_id": "unknown",
            "status": "failed",
            "backend": {"name": "native"},
            "error": {
                "code": "invalid_input_file",
                "message": str(error),
                "path": "$",
            },
        }
    # 输入和输出相同的错误不能再写回该路径，否则错误处理本身会破坏输入。
    destination = arguments.result
    if (
        arguments.request
        and destination
        and arguments.request.resolve() == destination.resolve()
    ):
        destination = None
    try:
        _write_result(result, destination, stream)
    except OSError as error:
        _write_result(
            {
                "status": "failed",
                "error": {
                    "code": "io.result_write_failed",
                    "message": f"结果写入失败，未强制覆盖目标：{error}",
                    "retryable": True,
                },
            },
            None,
            stream,
        )
        return 2
    return 0 if result["status"] in {"success", "planned"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
