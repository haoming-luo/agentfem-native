# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from tools.bootstrap_env import environment_python


class BootstrapEnvironmentTests(unittest.TestCase):
    def test_interpreter_path_is_native_on_posix_and_windows(self) -> None:
        environment = Path("project") / ".venv"
        with patch("tools.bootstrap_env.sys.platform", "win32"):
            self.assertEqual(
                environment_python(environment),
                environment / "Scripts" / "python.exe",
            )
        with patch("tools.bootstrap_env.sys.platform", "linux"):
            self.assertEqual(
                environment_python(environment), environment / "bin" / "python"
            )


if __name__ == "__main__":
    unittest.main()
