# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Create the repository-local development environment on every Tier-1 OS."""

from __future__ import annotations

import argparse
import subprocess
import sys
import venv
from pathlib import Path


def environment_python(environment: Path) -> Path:
    if sys.platform == "win32":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-scipy", action="store_true", help="install sparse provider"
    )
    parser.add_argument(
        "--agentfem",
        metavar="SOURCE",
        help="also install AgentFEM from a local checkout or package requirement",
    )
    arguments = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    environment = repository / ".venv"
    if not environment_python(environment).exists():
        venv.EnvBuilder(with_pip=True).create(environment)
    package = ".[scipy]" if arguments.with_scipy else "."
    subprocess.run(
        [
            str(environment_python(environment)),
            "-m",
            "pip",
            "install",
            "--editable",
            package,
        ],
        cwd=repository,
        check=True,
    )
    if arguments.agentfem:
        source = Path(arguments.agentfem).expanduser()
        target = str(source.resolve()) if source.exists() else arguments.agentfem
        command = [str(environment_python(environment)), "-m", "pip", "install"]
        if source.exists():
            command.append("--editable")
        command.append(target)
        subprocess.run(command, cwd=repository, check=True)
    print(f"AgentFEM Native environment is ready: {environment}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
