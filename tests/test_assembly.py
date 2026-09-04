# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import unittest

import numpy as np

from agentfem_native.assembly import (
    assemble_diffusion,
    assemble_diffusion_reference,
    assemble_diffusion_vectorized,
    p1_boundary_flux_load,
    p1_diffusion_stiffness,
    p1_source_load,
    select_assembly_mode,
)
from agentfem_native.mesh import unit_square_triangles, unit_square_two_triangles
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

    def test_vectorized_static_tensor_matches_reference_triplet_by_triplet(
        self,
    ) -> None:
        mesh = unit_square_triangles(5, 3)
        tensor = np.array(((2.0, 0.3), (0.3, 1.0)))
        reference_matrix, reference_load = assemble_diffusion_reference(
            mesh,
            conductivity=tensor,
            source=1.2,
            boundary_fluxes=(("right", 0.7),),
        )
        for chunk_size in (1, 7, 10_000):
            with self.subTest(chunk_size=chunk_size):
                matrix, load = assemble_diffusion_vectorized(
                    mesh,
                    conductivity=tensor,
                    source=1.2,
                    boundary_fluxes=(("right", 0.7),),
                    chunk_size=chunk_size,
                )
                np.testing.assert_array_equal(matrix.rows, reference_matrix.rows)
                np.testing.assert_array_equal(matrix.columns, reference_matrix.columns)
                np.testing.assert_allclose(
                    matrix.data, reference_matrix.data, rtol=0.0, atol=3.0e-15
                )
                np.testing.assert_allclose(load, reference_load, rtol=0.0, atol=3.0e-16)

    def test_vectorized_material_overrides_match_reference(self) -> None:
        mesh = unit_square_triangles(4)
        overrides = {
            index: (2.0 if index % 2 else np.array(((3.0, 0.2), (0.2, 1.0))))
            for index in range(mesh.cell_count)
        }
        reference_matrix, reference_load = assemble_diffusion_reference(
            mesh,
            conductivity=1.0,
            source=-0.25,
            cell_conductivities=overrides,
        )
        matrix, load = assemble_diffusion_vectorized(
            mesh,
            conductivity=1.0,
            source=-0.25,
            cell_conductivities=overrides,
            chunk_size=5,
        )
        np.testing.assert_array_equal(matrix.rows, reference_matrix.rows)
        np.testing.assert_array_equal(matrix.columns, reference_matrix.columns)
        np.testing.assert_allclose(matrix.data, reference_matrix.data, atol=3.0e-15)
        np.testing.assert_allclose(load, reference_load, atol=3.0e-17)

    def test_auto_dispatch_is_conservative_for_callable_fields(self) -> None:
        self.assertEqual(
            select_assembly_mode("auto", conductivity=2.0, source=1.0),
            "vectorized",
        )
        conductivity = lambda points: 1.0 + points[:, 0]
        self.assertEqual(
            select_assembly_mode("auto", conductivity=conductivity, source=1.0),
            "reference",
        )
        with self.assertRaisesRegex(ValueError, "requires static"):
            select_assembly_mode("vectorized", conductivity=conductivity, source=1.0)

    def test_vectorized_chunk_size_and_mode_fail_explicitly(self) -> None:
        mesh = unit_square_two_triangles()
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            assemble_diffusion_vectorized(mesh, conductivity=1.0, chunk_size=0)
        with self.assertRaisesRegex(ValueError, "Unknown assembly mode"):
            assemble_diffusion(mesh, conductivity=1.0, mode="unknown")


if __name__ == "__main__":
    unittest.main()
