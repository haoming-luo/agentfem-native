# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Run a portable full-output C++20/Rust/NumPy ABI comparison."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import agentfem_native._p1_native as cpp_kernel


def _rust_library(repository: Path) -> Path:
    suffix = {"darwin": ".dylib", "win32": ".dll"}.get(sys.platform, ".so")
    candidates = sorted(
        (repository / "native_spikes" / "rust" / "target" / "release").glob(
            f"*agentfem_native_p1_rust*{suffix}"
        )
    )
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one Rust cdylib, found {candidates!r}.")
    return candidates[0]


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(repository / "benchmarks" / "language_kernel_comparison.py"),
        "--cpp-library",
        str(Path(cpp_kernel.__file__).resolve()),
        "--rust-library",
        str(_rust_library(repository)),
        "--resolution",
        "32",
        "--repeats",
        "5",
    ]
    return subprocess.run(command, cwd=repository, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
