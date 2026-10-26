# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""跨平台完整 JSON 替换：只拥有本次创建的临时文件。"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def write_json_atomic(value: object, destination: Path) -> None:
    """同目录关闭句柄后替换；失败保留旧目标，不承诺断电持久性。"""
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(filename)
    try:
        # Windows 上不能保持写句柄打开再 rename/replace。
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
