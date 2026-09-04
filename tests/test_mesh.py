# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import unittest

import numpy as np

from agentfem_native.mesh import (
    TriangularMesh,
    unit_square_triangles,
    unit_square_two_triangles,
)


class MeshTests(unittest.TestCase):
    def test_unit_square_has_owned_zero_based_topology_and_named_sets(self) -> None:
        mesh = unit_square_two_triangles()
        self.assertEqual(mesh.node_count, 4)
        self.assertEqual(mesh.cell_count, 2)
        np.testing.assert_array_equal(mesh.nodes("left"), (0, 3))
        np.testing.assert_array_equal(mesh.boundary_edges("right"), ((1, 2),))
        with self.assertRaises(ValueError):
            mesh.points[0, 0] = 4.0

    def test_interior_edge_cannot_be_declared_as_boundary(self) -> None:
        with self.assertRaises(ValueError):
            TriangularMesh(
                points=np.array(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
                cells=np.array(((0, 1, 2), (0, 2, 3))),
                boundary_sets={"not_boundary": np.array(((0, 2),))},
            )

    def test_unknown_set_fails_explicitly(self) -> None:
        mesh = unit_square_two_triangles()
        with self.assertRaisesRegex(KeyError, "Unknown node set"):
            mesh.nodes("missing")
        with self.assertRaisesRegex(KeyError, "Unknown boundary set"):
            mesh.boundary_edges("missing")

    def test_structured_unit_square_counts_and_boundaries(self) -> None:
        mesh = unit_square_triangles(3, 2)
        self.assertEqual(mesh.node_count, 12)
        self.assertEqual(mesh.cell_count, 12)
        self.assertEqual(mesh.nodes("boundary").size, 10)
        self.assertEqual(mesh.boundary_edges("bottom").shape, (3, 2))
        self.assertEqual(mesh.boundary_edges("right").shape, (2, 2))

    def test_structured_mesh_resolution_must_be_positive_integer(self) -> None:
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                unit_square_triangles(value)  # type: ignore[arg-type]

    def test_fractional_connectivity_is_rejected_instead_of_truncated(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite integers"):
            TriangularMesh(
                points=np.array(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
                cells=np.array(((0.0, 1.0, 1.5),)),
            )

    def test_duplicate_boundary_edge_is_rejected_regardless_of_orientation(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate edges"):
            TriangularMesh(
                points=np.array(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
                cells=np.array(((0, 1, 2),)),
                boundary_sets={"edge": np.array(((0, 1), (1, 0)))},
            )


if __name__ == "__main__":
    unittest.main()
