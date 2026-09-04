# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import unittest

import numpy as np

from agentfem_native.geometry import AffineTriangleMap
from agentfem_native.quadrature import triangle_rule
from agentfem_native.reference import REFERENCE_TRIANGLE_VERTICES, p1_basis


class GeometryTests(unittest.TestCase):
    def test_reference_vertices_map_to_physical_vertices(self) -> None:
        vertices = np.array(((2.0, -1.0), (5.0, 0.0), (1.0, 3.0)))
        mapping = AffineTriangleMap(vertices)
        np.testing.assert_allclose(
            mapping.map_points(REFERENCE_TRIANGLE_VERTICES), vertices
        )

    def test_area_and_orientation_are_separate(self) -> None:
        positive = AffineTriangleMap(np.array(((0.0, 0.0), (2.0, 0.0), (0.0, 3.0))))
        negative = AffineTriangleMap(np.array(((0.0, 0.0), (0.0, 3.0), (2.0, 0.0))))
        self.assertAlmostEqual(positive.signed_determinant, 6.0)
        self.assertAlmostEqual(negative.signed_determinant, -6.0)
        self.assertAlmostEqual(positive.area, 3.0)
        self.assertAlmostEqual(negative.area, 3.0)

    def test_p1_physical_gradients_reproduce_affine_gradient(self) -> None:
        vertices = np.array(((2.0, -1.0), (5.0, 0.0), (1.0, 3.0)))
        mapping = AffineTriangleMap(vertices)
        nodal_values = 7.0 + 2.0 * vertices[:, 0] - 5.0 * vertices[:, 1]
        recovered_gradient = nodal_values @ mapping.p1_gradients()
        np.testing.assert_allclose(recovered_gradient, (2.0, -5.0), atol=3.0e-15)

    def test_orientation_does_not_change_integral_of_constant(self) -> None:
        rule = triangle_rule(1)
        vertices = np.array(((1.0, 1.0), (4.0, 1.0), (1.0, 5.0)))
        integrals = []
        for ordering in (vertices, vertices[[0, 2, 1]]):
            mapping = AffineTriangleMap(ordering)
            integral = mapping.integration_scale * rule.integrate_values(
                np.ones(rule.points.shape[0])
            )
            integrals.append(integral)
        np.testing.assert_allclose(integrals, (6.0, 6.0))

    def test_physical_p1_interpolation_reproduces_coordinates(self) -> None:
        vertices = np.array(((-1.0, 2.0), (2.0, 4.0), (0.0, 7.0)))
        mapping = AffineTriangleMap(vertices)
        reference_points = np.array(((0.2, 0.3), (0.6, 0.1)))
        by_geometry = mapping.map_points(reference_points)
        by_basis = p1_basis(reference_points) @ vertices
        np.testing.assert_allclose(by_geometry, by_basis)

    def test_invalid_or_degenerate_triangles_are_rejected(self) -> None:
        values = (
            np.zeros((3, 2)),
            np.array(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))),
            np.array(((0.0, 0.0), (1.0, 0.0), (np.nan, 1.0))),
        )
        for vertices in values:
            with self.subTest(vertices=vertices), self.assertRaises(ValueError):
                AffineTriangleMap(vertices)

    def test_vertex_and_jacobian_views_cannot_mutate_geometry(self) -> None:
        mapping = AffineTriangleMap(np.array(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))))
        with self.assertRaises(ValueError):
            mapping.vertices[0, 0] = 4.0
        with self.assertRaises(ValueError):
            mapping.jacobian[0, 0] = 4.0


if __name__ == "__main__":
    unittest.main()
