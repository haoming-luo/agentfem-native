# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import hashlib
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tools.build_artifact_manifest import build_manifest


class ArtifactManifestTests(unittest.TestCase):
    def test_manifest_is_sorted_hashed_and_revision_bound(self) -> None:
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            first = directory / "agentfem_native-0.4.0a1-cp311-abi3-win_amd64.whl"
            second = directory / "agentfem_native-0.4.0a1-cp311-abi3-manylinux.whl"
            first.write_bytes(b"windows")
            second.write_bytes(b"linux")
            with patch.dict(os.environ, {"GITHUB_SHA": "abc123"}):
                manifest = build_manifest(directory)
        self.assertEqual(manifest["source_revision"], "abc123")
        wheels = manifest["wheels"]
        self.assertEqual([record["filename"] for record in wheels], sorted((first.name, second.name)))
        records = {record["filename"]: record for record in wheels}
        self.assertEqual(records[first.name]["sha256"], hashlib.sha256(b"windows").hexdigest())
        self.assertEqual(records[first.name]["size_bytes"], 7)

    def test_empty_directory_is_rejected(self) -> None:
        with TemporaryDirectory() as directory, self.assertRaisesRegex(
            ValueError, "No wheels"
        ):
            build_manifest(Path(directory))


if __name__ == "__main__":
    unittest.main()
