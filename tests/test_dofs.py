# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import unittest

import numpy as np

from agentfem_native.dofs import VectorDofMap


class VectorDofMapTests(unittest.TestCase):
    def test_node_major_component_minor_numbering(self) -> None:
        dofs = VectorDofMap(4, 2)
        self.assertEqual(dofs.size, 8)
        self.assertEqual(dofs.dof(2, 1), 5)
        np.testing.assert_array_equal(dofs.node_dofs([1, 3]), ((2, 3), (6, 7)))
        np.testing.assert_array_equal(
            dofs.cell_dofs([[0, 2, 3]]), ((0, 1, 4, 5, 6, 7),)
        )

    def test_invalid_counts_and_indices_fail(self) -> None:
        for arguments in ((0, 2), (2, 0), (True, 2)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                VectorDofMap(*arguments)
        dofs = VectorDofMap(3, 2)
        with self.assertRaises(ValueError):
            dofs.dof(3, 0)
        with self.assertRaises(ValueError):
            dofs.cell_dofs([[0, 1, 3]])


if __name__ == "__main__":
    unittest.main()
