# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""自由热膨胀、完全约束热应力与热力闭环。"""

import unittest

import numpy as np

from agentfem_native.mesh import unit_square_triangles
from agentfem_native.simplex import EssentialBC
from agentfem_native.thermal import ThermalBoundary, ThermalOperator
from agentfem_native.thermoelastic import ThermoelasticOperator, solve_thermal_elastic
from agentfem_native.volume_mesh import unit_cube_tetrahedra


class ThermoelasticTests(unittest.TestCase):
    def test_free_expansion_and_blocked_stress(self):
        for mesh in (unit_square_triangles(2), unit_cube_tetrahedra(2)):
            d = mesh.points.shape[1]
            operator = ThermoelasticOperator(
                mesh,
                100,
                0.25,
                mode="plane_stress" if d == 2 else "plane_strain",
                temperature=np.full(mesh.node_count, 30),
                reference_temperature=20,
                expansion=0.001,
            )
            # 各坐标零平面仅固定法向分量，允许解析自由热膨胀。
            dofs = np.concatenate(
                [d * np.flatnonzero(mesh.points[:, j] == 0) + j for j in range(d)]
            )
            result = operator.solve(EssentialBC(dofs, np.zeros(dofs.size)))
            np.testing.assert_allclose(
                result.displacements, 0.01 * mesh.points, atol=1e-11
            )
            np.testing.assert_allclose(result.cell_stress, 0, atol=1e-9)
            self.assertLess(abs(result.strain_energy), 1e-20)
            blocked = operator.solve(
                EssentialBC(
                    np.arange(d * mesh.node_count), np.zeros(d * mesh.node_count)
                )
            )
            expected = -100 * 0.01 / (1 - 0.25 if d == 2 else 1 - 2 * 0.25)
            np.testing.assert_allclose(
                blocked.cell_stress[:, 0, 0], expected, atol=1e-12
            )

    def test_plane_strain_includes_out_of_plane_thermal_stress(self):
        mesh = unit_square_triangles(1)
        operator = ThermoelasticOperator(
            mesh, 100, 0.25, temperature=np.full(mesh.node_count, 10), expansion=0.001
        )
        result = operator.solve(
            EssentialBC(np.arange(2 * mesh.node_count), np.zeros(2 * mesh.node_count))
        )
        for j in range(3):
            np.testing.assert_allclose(result.cell_stress[:, j, j], -2, atol=1e-12)
        self.assertAlmostEqual(result.strain_energy, 0.03)

    def test_gradient_energy_and_thermal_chain(self):
        mesh = unit_square_triangles(3)
        thermal = ThermalOperator(
            mesh,
            boundaries=(
                ThermalBoundary("right", convection=2, ambient_temperature=30),
            ),
        )
        left = mesh.nodes("left")
        dofs = (2 * left[:, None] + np.arange(2)).ravel()
        heat, result = solve_thermal_elastic(
            thermal,
            EssentialBC(left, np.full(left.size, 20)),
            EssentialBC(dofs, np.zeros(dofs.size)),
            young_modulus=100,
            poisson_ratio=0.25,
            expansion=0.001,
            reference_temperature=20,
            mode="plane_stress",
        )
        np.testing.assert_allclose(
            heat.temperatures[0], 20 + (20 / 3) * mesh.points[:, 0], atol=1e-10
        )
        self.assertLess(result.free_residual_norm, 1e-10)
        self.assertGreater(result.displacements[:, 0].max(), 0)
        operator = ThermoelasticOperator(
            mesh, 100, 0.25, temperature=heat.temperatures[0], expansion=0.001
        )
        rng = np.random.default_rng(12)
        u = rng.normal(size=mesh.points.shape) * 0.01
        direction = rng.normal(size=u.shape)
        h = 1e-6
        ep = operator.geometry.volumes @ operator.fields(u + h * direction)[2]
        em = operator.geometry.volumes @ operator.fields(u - h * direction)[2]
        force = operator.stiffness.matvec(u.ravel()) - operator.thermal_load
        self.assertAlmostEqual((ep - em) / (2 * h), force @ direction.ravel(), places=7)

    def test_bad_temperature_and_rigid_motion(self):
        mesh = unit_square_triangles(1)
        with self.assertRaises(ValueError):
            ThermoelasticOperator(mesh, 100, 0.25, temperature=[0])
        operator = ThermoelasticOperator(mesh, 100, 0.25)
        with self.assertRaises(ValueError):
            operator.solve(EssentialBC([0], [0]))
