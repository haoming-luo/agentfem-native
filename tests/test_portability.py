# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Windows 同样执行：中文路径、已关闭句柄、安装态诊断和可选依赖隔离。"""

import json
import subprocess
import sys
import unittest
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from agentfem_native.cli import main
from agentfem_native.diagnostics import runtime_diagnostics
from agentfem_native.jsonio import write_json_atomic
from tests.test_contract import request_record


class PortabilityTests(unittest.TestCase):
    def test_unicode_space_path_subprocess_and_replacement(self):
        with TemporaryDirectory(prefix="原生 FEM ") as directory:
            root = Path(directory)
            request, result = root / "请求 文件.json", root / "结果 文件.json"
            write_json_atomic(request_record(), request)
            result.write_text("旧结果", encoding="utf-8")
            # 不使用 shell、不设置 PYTHONPATH；和 CI 一样测试当前安装包。
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "agentfem_native.cli",
                    str(request),
                    "--result",
                    str(result),
                ],
                cwd=root,
                capture_output=True,
                check=False,
                timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                json.loads(result.read_text(encoding="utf-8"))["status"], "success"
            )
            self.assertEqual(list(root.glob("*.tmp")), [])

    def test_failed_replace_keeps_previous_and_unrelated_temporary(self):
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "结果.json"
            unrelated = destination.with_suffix(".json.tmp")
            destination.write_text("old", encoding="utf-8")
            unrelated.write_text("other-owner", encoding="utf-8")
            with (
                patch.object(
                    Path, "replace", side_effect=PermissionError("文件被锁定")
                ),
                self.assertRaises(PermissionError),
            ):
                write_json_atomic({"value": 1}, destination)
            self.assertEqual(destination.read_text(encoding="utf-8"), "old")
            self.assertEqual(unrelated.read_text(encoding="utf-8"), "other-owner")
            self.assertEqual(len(list(Path(directory).iterdir())), 2)

    def test_cli_io_failure_and_input_preservation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            request = root / "请求.json"
            write_json_atomic(request_record(), request)
            before = request.read_bytes()
            stream = StringIO()
            self.assertEqual(
                main([str(request), "--result", str(request)], stdout=stream), 2
            )
            self.assertEqual(request.read_bytes(), before)
            stream = StringIO()
            self.assertEqual(
                main([str(request), "--result", str(root)], stdout=stream), 2
            )
            self.assertEqual(
                json.loads(stream.getvalue())["error"]["code"], "io.result_write_failed"
            )

    def test_diagnostics_missing_native_and_wsl_not_windows(self):
        missing = {
            "available": False,
            "abi_version": None,
            "required_abi_version": "1.5",
        }
        with patch(
            "agentfem_native.diagnostics.native_kernel_identity", return_value=missing
        ):
            self.assertEqual(runtime_diagnostics()["status"], "success")
            self.assertEqual(
                runtime_diagnostics(require_native=True)["status"], "failed"
            )
        with (
            patch("platform.system", return_value="Linux"),
            patch("platform.release", return_value="microsoft-standard-WSL2"),
        ):
            report = runtime_diagnostics()
        self.assertTrue(report["runtime"]["wsl"])
        self.assertFalse(report["runtime"]["native_windows"])

    def test_installed_diagnostics_without_optional_imports(self):
        program = """
import importlib.abc, json, sys
class RejectOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'scipy', 'petsc4py', 'mpi4py', 'dolfinx', 'basix', 'ufl', 'ffcx'}:
            raise AssertionError('不允许导入可选/第三方有限元组件: ' + fullname)
sys.meta_path.insert(0, RejectOptional())
from agentfem_native.diagnostics import runtime_diagnostics
print(json.dumps(runtime_diagnostics()))
"""
        with TemporaryDirectory() as directory:
            completed = subprocess.run(
                [sys.executable, "-c", program],
                cwd=directory,
                capture_output=True,
                check=False,
                timeout=30,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["status"], "success")
