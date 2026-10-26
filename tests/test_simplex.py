# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""候选网格的分量准入、刚体尺度不变量与无载荷奇异拒绝。"""

import unittest

import numpy as np

from agentfem_native.mesh import TriangularMesh, unit_square_triangles
from agentfem_native.simplex import EssentialBC, SimplexGeometry, connected_components
from agentfem_native.thermal import ThermalOperator
from agentfem_native.thermoelastic import ThermoelasticOperator
from agentfem_native.volume_mesh import TetrahedralMesh, unit_cube_tetrahedra


class SimplexTests(unittest.TestCase):
    def test_rigid_constraint_rank_translation_and_unit_invariant(self):
        for base in (unit_square_triangles(1), unit_cube_tetrahedra(1)):
            d = base.points.shape[1]
            nodes = base.nodes("left")
            dofs = (d * nodes[:, None] + np.arange(d)).ravel()
            for scale, shift in ((1, 0), (1, 1e12), (1e-6, 0), (1e6, 1e12)):
                mesh = type(base)(base.points * scale + shift, base.cells)
                SimplexGeometry(mesh).check_rigid_constraints(
                    EssentialBC(dofs, np.zeros(dofs.size))
                )

    def test_disconnected_zero_load_is_not_unique_equilibrium(self):
        for base in (unit_square_triangles(1), unit_cube_tetrahedra(1)):
            d, n = base.points.shape[1], base.node_count
            mesh = type(base)(
                np.vstack((base.points, base.points + 3)),
                np.vstack((base.cells, base.cells + n)),
            )
            labels = connected_components(mesh)
            self.assertEqual(np.unique(labels).size, 2)
            # 第一块固定全部节点，第二块悬浮，即使零载荷也不能冒充唯一解。
            bc = EssentialBC(np.arange(n * d), np.zeros(n * d))
            with self.assertRaisesRegex(ValueError, "连通分量"):
                ThermoelasticOperator(mesh, 100, 0.25).solve(bc)
            thermal = ThermalOperator(mesh)
            with self.assertRaisesRegex(ValueError, "连通分量"):
                thermal.steady(EssentialBC([0], [0]))
            result = thermal.steady(EssentialBC([0, n], [2, 5]))
            np.testing.assert_allclose(result.temperatures[0, :n], 2, atol=1e-10)
            np.testing.assert_allclose(result.temperatures[0, n:], 5, atol=1e-10)

    def test_isolated_node_and_numbering(self):
        for cls, base in (
            (TriangularMesh, unit_square_triangles(3)),
            (TetrahedralMesh, unit_cube_tetrahedra(2)),
        ):
            mesh = cls(
                np.vstack((base.points, np.full((1, base.points.shape[1]), 4))),
                base.cells,
            )
            with self.assertRaisesRegex(ValueError, "孤立"):
                SimplexGeometry(mesh)
            permutation = np.random.default_rng(19).permutation(base.node_count)
            inverse = np.argsort(permutation)
            renumbered = cls(base.points[permutation], inverse[base.cells])
            np.testing.assert_array_equal(
                connected_components(renumbered), np.zeros(base.node_count)
            )
