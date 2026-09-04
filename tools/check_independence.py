# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Fail when Native source imports a forbidden finite-element implementation."""

from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN = {"dolfinx", "ufl", "basix", "ffcx"}
ROOT = Path(__file__).resolve().parents[1]


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.partition(".")[0])
    return roots


def main() -> int:
    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        forbidden = sorted(imported_roots(path) & FORBIDDEN)
        if forbidden:
            offenders.append(f"{path.relative_to(ROOT)}: {', '.join(forbidden)}")
    if offenders:
        print("Forbidden finite-element imports found:")
        print("\n".join(offenders))
        return 1

    missing_spdx = []
    for directory in ("src", "tests", "tools", "examples", "benchmarks"):
        for path in (ROOT / directory).rglob("*.py"):
            if "SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0" not in "\n".join(
                path.read_text(encoding="utf-8").splitlines()[:3]
            ):
                missing_spdx.append(str(path.relative_to(ROOT)))
    for pattern in ("*.c", "*.cpp", "*.h"):
        for path in (ROOT / "native_spikes").rglob(pattern):
            if "SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0" not in "\n".join(
                path.read_text(encoding="utf-8").splitlines()[:3]
            ):
                missing_spdx.append(str(path.relative_to(ROOT)))
    if missing_spdx:
        print("Files missing the selected SPDX identifier:")
        print("\n".join(missing_spdx))
        return 1

    print(
        "Independence scan passed: no forbidden finite-element imports and "
        "all implementation files carry the selected SPDX identifier."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
