# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import unittest

import numpy as np

from agentfem_native.reference import (
    REFERENCE_TRIANGLE_VERTICES,
    inside_reference_triangle,
    p1_basis,
    p1_basis_gradients,
)


class ReferenceTriangleTests(unittest.TestCase):
    def test_p1_kronecker_property_at_vertices(self) -> None:
        np.testing.assert_array_equal(p1_basis(REFERENCE_TRIANGLE_VERTICES), np.eye(3))

    def test_p1_partition_of_unity_and_zero_gradient_sum(self) -> None:
        points = np.array(((0.1, 0.2), (0.25, 0.25), (0.0, 1.0)))
        np.testing.assert_allclose(p1_basis(points).sum(axis=1), 1.0)
        np.testing.assert_array_equal(p1_basis_gradients().sum(axis=0), np.zeros(2))

    def test_p1_reproduces_every_affine_scalar_function(self) -> None:
        points = np.array(((0.1, 0.2), (0.25, 0.5), (0.7, 0.1)))
        vertex_values = (
            2.5
            + 3.0 * REFERENCE_TRIANGLE_VERTICES[:, 0]
            - 4.0 * (REFERENCE_TRIANGLE_VERTICES[:, 1])
        )
        interpolated = p1_basis(points) @ vertex_values
        exact = 2.5 + 3.0 * points[:, 0] - 4.0 * points[:, 1]
        np.testing.assert_allclose(interpolated, exact, rtol=0.0, atol=2.0e-15)

    def test_basis_gradients_broadcast_to_point_shape(self) -> None:
        points = np.zeros((2, 4, 2))
        gradients = p1_basis_gradients(points)
        self.assertEqual(gradients.shape, (2, 4, 3, 2))
        np.testing.assert_array_equal(gradients[1, 3], p1_basis_gradients())

    def test_inside_reference_triangle_includes_closed_boundary(self) -> None:
        points = np.array(
            (
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
                (0.2, 0.3),
                (-0.1, 0.2),
                (0.6, 0.5),
            )
        )
        np.testing.assert_array_equal(
            inside_reference_triangle(points),
            np.array((True, True, True, True, False, False)),
        )

    def test_reference_point_shape_and_finiteness_are_checked(self) -> None:
        for bad in (0.0, [1.0], [[1.0, 2.0, 3.0]], [np.nan, 0.0]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                p1_basis(bad)

    def test_negative_tolerance_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            inside_reference_triangle((0.0, 0.0), tolerance=-1.0)


if __name__ == "__main__":
    unittest.main()
