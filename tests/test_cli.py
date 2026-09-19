# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import json
import unittest
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from agentfem_native.cli import main
from tests.test_contract import request_record


class CommandLineTests(unittest.TestCase):
    def test_success_is_json_on_stdout_with_zero_exit(self) -> None:
        with TemporaryDirectory() as directory:
            request_path = Path(directory) / "request.json"
            request_path.write_text(json.dumps(request_record()), encoding="utf-8")
            output = StringIO()
            status = main([str(request_path)], stdout=output)
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "success")

    def test_contract_failure_has_nonzero_exit(self) -> None:
        request = request_record()
        request["contract_version"] = "future"
        with TemporaryDirectory() as directory:
            request_path = Path(directory) / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            output = StringIO()
            status = main([str(request_path)], stdout=output)
        self.assertEqual(status, 2)
        self.assertEqual(
            json.loads(output.getvalue())["error"]["code"],
            "unsupported_contract_version",
        )

    def test_plan_only_returns_resource_plan_without_executing(self) -> None:
        with TemporaryDirectory() as directory:
            request_path = Path(directory) / "request.json"
            request_path.write_text(json.dumps(request_record()), encoding="utf-8")
            output = StringIO()
            status = main([str(request_path), "--plan-only"], stdout=output)
        result = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(result["status"], "planned")
        self.assertIn("peak_bytes_upper_bound", result["plan"])
        self.assertNotIn("fields", result)

    def test_invalid_json_is_structured_and_result_write_is_atomic(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            request_path = root / "bad.json"
            result_path = root / "nested" / "result.json"
            request_path.write_text("{", encoding="utf-8")
            status = main(
                [str(request_path), "--result", str(result_path)],
                stdout=StringIO(),
            )
            result = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(status, 2)
        self.assertEqual(result["error"]["code"], "invalid_input_file")


if __name__ == "__main__":
    unittest.main()
