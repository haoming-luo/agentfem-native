# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import unittest

import numpy as np

from agentfem_native.mesh import unit_square_two_triangles
from agentfem_native.native import (
    NATIVE_P1_ABI_VERSION,
    assemble_p1_volume,
    native_kernel_available,
    native_kernel_identity,
)


@unittest.skipUnless(native_kernel_available(), "compiled kernel unavailable")
class NativeKernelTests(unittest.TestCase):
    def test_compiled_identity_is_versioned(self) -> None:
        identity = native_kernel_identity()
        self.assertEqual(identity["name"], "cpp20")
        self.assertTrue(identity["available"])
        self.assertEqual(identity["abi_version"], "1.0")
        self.assertEqual(identity["required_abi_version"], "1.0")
        self.assertEqual(NATIVE_P1_ABI_VERSION, 0x0001_0000)

    def test_global_and_per_cell_conductivity_emit_owned_arrays(self) -> None:
        mesh = unit_square_two_triangles()
        global_result = assemble_p1_volume(
            mesh.points, mesh.cells, np.eye(2), 2.0
        )
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


if __name__ == "__main__":
    unittest.main()
