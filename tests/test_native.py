# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import unittest

import numpy as np

from agentfem_native.mesh import unit_square_two_triangles
from agentfem_native.native import (
    NATIVE_P1_ABI_VERSION,
    assemble_p1_volume,
    assemble_t3_volume,
    native_kernel_available,
    native_kernel_identity,
)


@unittest.skipUnless(native_kernel_available(), "compiled kernel unavailable")
class NativeKernelTests(unittest.TestCase):
    def test_compiled_identity_is_versioned(self) -> None:
        identity = native_kernel_identity()
        self.assertEqual(identity["name"], "cpp20")
        self.assertTrue(identity["available"])
        self.assertEqual(identity["abi_version"], "1.1")
        self.assertEqual(identity["required_abi_version"], "1.1")
        self.assertEqual(NATIVE_P1_ABI_VERSION, 0x0001_0001)

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


if __name__ == "__main__":
    unittest.main()
