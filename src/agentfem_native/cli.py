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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentfem-native",
        description="执行或预检 AgentFEM Native Kernel Contract 请求。",
    )
    parser.add_argument("request", type=Path, help="UTF-8 JSON 请求文件")
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(destination)


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    """运行 CLI，并返回跨平台进程退出码。"""

    arguments = _parser().parse_args(argv)
    stream = sys.stdout if stdout is None else stdout
    try:
        request = json.loads(arguments.request.read_text(encoding="utf-8"))
        if not isinstance(request, dict):
            raise TypeError("请求根节点必须是 JSON 对象。")
        if arguments.plan_only:
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
    _write_result(result, arguments.result, stream)
    return 0 if result["status"] in {"success", "planned"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
