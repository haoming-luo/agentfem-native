# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Create a deterministic digest manifest for built AgentFEM Native wheels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(directory: Path) -> dict[str, object]:
    wheels = sorted(directory.glob("*.whl"), key=lambda path: path.name)
    if not wheels:
        raise ValueError(f"No wheels found beneath {directory}.")
    if len({path.name for path in wheels}) != len(wheels):
        raise ValueError("Wheel filenames must be unique.")
    return {
        "schema": "agentfem.native-wheel-manifest",
        "schema_version": "0.1.0",
        "source_revision": os.environ.get("GITHUB_SHA", "local-uncommitted"),
        "wheels": [
            {
                "filename": path.name,
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in wheels
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    manifest = build_manifest(arguments.directory)
    arguments.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
