# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import unittest

import numpy as np

from agentfem_native.mesh import unit_square_two_triangles
from agentfem_native.native import (
    NATIVE_P1_ABI_VERSION,
    assemble_p1_volume,
    assemble_t3_volume,
    assemble_t4_volume,
    csr_spmv,
    native_kernel_available,
    native_kernel_identity,
)


@unittest.skipUnless(native_kernel_available(), "compiled kernel unavailable")
class NativeKernelTests(unittest.TestCase):
    def test_compiled_identity_is_versioned(self) -> None:
        identity = native_kernel_identity()
        self.assertEqual(identity["name"], "cpp20")
        self.assertTrue(identity["available"])
        self.assertEqual(identity["abi_version"], "1.2")
        self.assertEqual(identity["required_abi_version"], "1.2")
        self.assertEqual(NATIVE_P1_ABI_VERSION, 0x0001_0002)

    def test_global_and_per_cell_conductivity_emit_owned_arrays(self) -> None:
        mesh = unit_square_two_triangles()
        global_result = assemble_p1_volume(mesh.points, mesh.cells, np.eye(2), 2.0)
        per_cell_result = assemble_p1_volume(
            mesh.points,
            mesh.cells,
            np.broadcast_to(np.eye(2), (mesh.cell_count, 2, 2)),
            2.0,
        )
        for global_array, per_cell_array in zip(
            global_result, per_cell_result, strict=True
        ):
            np.testing.assert_array_equal(global_array, per_cell_array)
            self.assertTrue(global_array.flags.owndata)

    def test_invalid_native_input_fails_without_partial_result(self) -> None:
        mesh = unit_square_two_triangles()
        with self.assertRaisesRegex(ValueError, "positive definite"):
            assemble_p1_volume(mesh.points, mesh.cells, np.zeros((2, 2)), 0.0)
        with self.assertRaisesRegex(ValueError, "conductivity"):
            assemble_p1_volume(
                mesh.points,
                mesh.cells,
                np.ones((mesh.cell_count, 2)),
                0.0,
            )

    def test_t3_volume_matches_readable_element_data(self) -> None:
        mesh = unit_square_two_triangles()
        modulus, ratio = 100.0, 0.25
        scale = modulus / (1.0 - ratio * ratio)
        constitutive = scale * np.array(
            ((1.0, ratio, 0.0), (ratio, 1.0, 0.0), (0.0, 0.0, (1.0 - ratio) / 2.0))
        )
        rows, columns, data, load = assemble_t3_volume(
            mesh.points,
            mesh.cells,
            np.broadcast_to(constitutive, (mesh.cell_count, 3, 3)),
            np.array((2.0, -1.0)),
            0.5,
        )
        from agentfem_native.elasticity import (
            LinearElasticMaterial,
            p1_elastic_stiffness,
        )

        expected = p1_elastic_stiffness(
            mesh.points[mesh.cells[0]],
            LinearElasticMaterial(modulus, ratio),
            thickness=0.5,
        )
        np.testing.assert_allclose(data[:36].reshape(6, 6), expected, atol=2.0e-14)
        self.assertEqual(rows.shape, (72,))
        self.assertEqual(columns.shape, (72,))
        np.testing.assert_allclose(load.reshape(4, 2).sum(axis=0), (1.0, -0.5))

    def test_t4_volume_matches_readable_element_data(self) -> None:
        points = np.array(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        )
        cells = np.array(((0, 1, 2, 3),), dtype=np.int64)
        from agentfem_native.solid import SolidElasticMaterial, t4_elastic_stiffness

        material = SolidElasticMaterial(120.0, 0.25)
        rows, columns, data, load = assemble_t4_volume(
            points,
            cells,
            material.constitutive_matrix[None, :, :],
            np.array((6.0, -3.0, 1.5)),
        )
        np.testing.assert_allclose(
            data.reshape(12, 12), t4_elastic_stiffness(points, material), atol=2.0e-14
        )
        np.testing.assert_array_equal(rows, np.repeat(np.arange(12), 12))
        np.testing.assert_array_equal(columns, np.tile(np.arange(12), 12))
        np.testing.assert_allclose(load.reshape(4, 3).sum(axis=0), (1.0, -0.5, 0.25))

    def test_native_csr_spmv_matches_readable_reduction(self) -> None:
        from agentfem_native.sparse import CSRMatrix

        size = 1024
        rows = np.repeat(np.arange(size, dtype=np.int64), 3)
        columns = np.column_stack(
            (
                np.maximum(np.arange(size) - 1, 0),
                np.arange(size),
                np.minimum(np.arange(size) + 1, size - 1),
            )
        ).ravel()
        matrix = CSRMatrix.from_coo(
            (size, size), rows, columns, np.linspace(0.5, 1.5, rows.size)
        )
        vector = np.linspace(-1.0, 2.0, size)
        native = csr_spmv(
            matrix.shape, matrix.indptr, matrix.indices, matrix.data, vector
        )
        np.testing.assert_allclose(
            native, matrix.matvec_reference(vector), atol=1.0e-15
        )
        np.testing.assert_allclose(matrix.matvec(vector), native, atol=0.0)


if __name__ == "__main__":
    unittest.main()
