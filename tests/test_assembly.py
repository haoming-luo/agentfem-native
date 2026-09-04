# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import unittest

import numpy as np

from agentfem_native.assembly import (
    assemble_diffusion,
    p1_boundary_flux_load,
    p1_diffusion_stiffness,
    p1_source_load,
)
from agentfem_native.mesh import unit_square_two_triangles
from agentfem_native.reference import REFERENCE_TRIANGLE_VERTICES


class AssemblyTests(unittest.TestCase):
    def test_reference_triangle_stiffness_matches_independent_formula(self) -> None:
        matrix = p1_diffusion_stiffness(REFERENCE_TRIANGLE_VERTICES, 1.0)
        expected = np.array(((1.0, -0.5, -0.5), (-0.5, 0.5, 0.0), (-0.5, 0.0, 0.5)))
        np.testing.assert_allclose(matrix, expected, atol=2.0e-16)
        np.testing.assert_allclose(matrix.sum(axis=1), 0.0, atol=2.0e-16)
        np.testing.assert_allclose(matrix, matrix.T, atol=0.0)

    def test_anisotropic_tensor_stiffness_matches_direct_gradient_formula(self) -> None:
        tensor = np.array(((4.0, 1.0), (1.0, 2.0)))
        matrix = p1_diffusion_stiffness(REFERENCE_TRIANGLE_VERTICES, tensor)
        gradients = np.array(((-1.0, -1.0), (1.0, 0.0), (0.0, 1.0)))
        expected = 0.5 * gradients @ tensor @ gradients.T
        np.testing.assert_allclose(matrix, expected, atol=8.0e-16)

    def test_invalid_conductivity_tensors_are_rejected(self) -> None:
        invalid = (
            0.0,
            -1.0,
            np.array(((1.0, 2.0), (0.0, 1.0))),
            np.array(((1.0, 0.0), (0.0, -1.0))),
            np.full((2, 2), np.nan),
        )
        for conductivity in invalid:
            with self.subTest(conductivity=conductivity), self.assertRaises(ValueError):
                p1_diffusion_stiffness(REFERENCE_TRIANGLE_VERTICES, conductivity)

    def test_constant_source_is_shared_equally_by_element_nodes(self) -> None:
        load = p1_source_load(REFERENCE_TRIANGLE_VERTICES, 2.0)
        np.testing.assert_allclose(load, np.full(3, 1.0 / 3.0), atol=2.0e-16)

    def test_constant_flux_is_shared_equally_by_edge_nodes(self) -> None:
        load = p1_boundary_flux_load(np.array(((0.0, 0.0), (2.0, 0.0))), 3.0)
        np.testing.assert_allclose(load, (3.0, 3.0), atol=5.0e-16)

    def test_coo_assembly_is_symmetric_and_has_constant_null_mode(self) -> None:
        matrix, load = assemble_diffusion(
            unit_square_two_triangles(), conductivity=2.0, source=0.0
        )
        dense = matrix.to_dense()
        np.testing.assert_allclose(dense, dense.T, atol=0.0)
        np.testing.assert_allclose(matrix.matvec(np.ones(4)), 0.0, atol=5.0e-16)
        np.testing.assert_array_equal(load, np.zeros(4))


if __name__ == "__main__":
    unittest.main()
