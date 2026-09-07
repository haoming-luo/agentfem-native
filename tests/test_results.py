# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from agentfem_native.mesh import unit_square_two_triangles
from agentfem_native.results import write_legacy_vtk, write_mechanics_vtk
from agentfem_native.volume_mesh import unit_cube_tetrahedra


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

    def test_mechanics_vtk_contains_vector_and_tensor_fields(self) -> None:
        mesh = unit_cube_tetrahedra(1)
        with TemporaryDirectory() as directory:
            path = write_mechanics_vtk(
                Path(directory) / "solid.vtk",
                mesh,
                np.zeros((mesh.node_count, 3)),
                np.tile(
                    (1.0, 2.0, 3.0, 0.1, 0.2, 0.3),
                    (mesh.cell_count, 1),
                ),
            )
            content = path.read_text(encoding="utf-8")
        self.assertIn("VECTORS displacement double", content)
        self.assertIn("TENSORS cauchy_stress double", content)
        self.assertIn("CELL_TYPES 6", content)
        self.assertEqual(content.splitlines().count("10"), mesh.cell_count)


if __name__ == "__main__":
    unittest.main()
