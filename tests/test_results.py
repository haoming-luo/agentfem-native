# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from agentfem_native.mesh import unit_square_two_triangles
from agentfem_native.results import write_legacy_vtk


class ResultTests(unittest.TestCase):
    def test_legacy_vtk_contains_mesh_and_nodal_field(self) -> None:
        mesh = unit_square_two_triangles()
        with TemporaryDirectory() as directory:
            path = write_legacy_vtk(
                Path(directory) / "nested" / "solution.vtk",
                mesh,
                mesh.points[:, 0],
                field_name="temperature field",
            )
            content = path.read_text(encoding="utf-8")
        self.assertIn("POINTS 4 double", content)
        self.assertIn("CELLS 2 8", content)
        self.assertIn("POINT_DATA 4", content)
        self.assertIn("SCALARS temperature_field double 1", content)
        self.assertTrue(content.endswith("\n"))

    def test_result_size_must_match_mesh(self) -> None:
        with TemporaryDirectory() as directory, self.assertRaises(ValueError):
            write_legacy_vtk(
                Path(directory) / "bad.vtk",
                unit_square_two_triangles(),
                np.zeros(3),
            )


if __name__ == "__main__":
    unittest.main()
