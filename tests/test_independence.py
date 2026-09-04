# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import ast
from pathlib import Path
import unittest


FORBIDDEN = {"dolfinx", "ufl", "basix", "ffcx"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.partition(".")[0])
    return roots


class IndependenceTests(unittest.TestCase):
    def test_source_has_no_forbidden_finite_element_imports(self) -> None:
        offenders = {
            str(path.relative_to(PROJECT_ROOT)): sorted(imported_roots(path) & FORBIDDEN)
            for path in (PROJECT_ROOT / "src").rglob("*.py")
            if imported_roots(path) & FORBIDDEN
        }
        self.assertEqual(offenders, {})

    def test_all_python_files_have_selected_spdx_identifier(self) -> None:
        missing = []
        for directory in ("src", "tests", "tools", "examples", "benchmarks"):
            for path in (PROJECT_ROOT / directory).rglob("*.py"):
                first_lines = path.read_text(encoding="utf-8").splitlines()[:3]
                if not any(
                    "SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0" in line
                    for line in first_lines
                ):
                    missing.append(str(path.relative_to(PROJECT_ROOT)))
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
