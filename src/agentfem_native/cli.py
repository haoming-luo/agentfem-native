# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Cross-platform JSON command-line boundary for AgentFEM Native."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from .contract import run_kernel_request


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentfem-native",
        description="Execute an AgentFEM Native Kernel Contract request.",
    )
    parser.add_argument("request", type=Path, help="UTF-8 JSON request file")
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        help="Root directory for requested result artifacts",
    )
    parser.add_argument(
        "--result",
        type=Path,
        help="Write the JSON result to this file instead of standard output",
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
    """Run the CLI and return an operating-system exit status."""

    arguments = _parser().parse_args(argv)
    stream = sys.stdout if stdout is None else stdout
    try:
        request = json.loads(arguments.request.read_text(encoding="utf-8"))
        if not isinstance(request, dict):
            raise TypeError("The request root must be a JSON object.")
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
            "contract": "agentfem.native-kernel-request",
            "contract_version": "0.1.0",
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
    return 0 if result["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
