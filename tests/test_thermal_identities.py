# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""导热的材料跳变、旋转协变与隐式耗散恒等式，真值不依赖另一 FEM。"""

import unittest
from dataclasses import replace

import numpy as np

from agentfem_native.mesh import unit_square_triangles
from agentfem_native.simplex import EssentialBC
from agentfem_native.thermal import ThermalOperator
from agentfem_native.volume_mesh import unit_cube_tetrahedra


class ThermalIdentityTests(unittest.TestCase):
    def test_two_layer_resistance_flux_and_boundary_balance(self):
        for mesh in (unit_square_triangles(4), unit_cube_tetrahedra(4)):
            centers = mesh.points[mesh.cells].mean(axis=1)
            x = mesh.points[:, 0]
            boundary = np.flatnonzero((x == 0) | (x == 1))
            for scale in (1e-3, 1, 1e3):
                left, right = scale, 1000 * scale
                conductivity = np.where(centers[:, 0] < 0.5, left, right)
                resistance = 0.5 / left + 0.5 / right
                exact = np.minimum(x, 0.5) / (resistance * left) + np.maximum(
                    x - 0.5, 0
                ) / (resistance * right)
                operator = ThermalOperator(mesh, conductivity=conductivity)
                result = operator.steady(EssentialBC(boundary, x[boundary]))
                np.testing.assert_allclose(
                    result.temperatures[0], exact, rtol=1e-9, atol=1e-10
                )
                expected_flux = np.zeros_like(result.cell_flux[0])
                expected_flux[:, 0] = -1 / resistance
                np.testing.assert_allclose(
                    result.cell_flux[0],
                    expected_flux,
                    rtol=1e-8,
                    atol=1e-8 / resistance,
                )
                self.assertLess(abs(result.balance_errors[0]), 1e-8 / resistance)

    def test_anisotropic_affine_patch_is_rotation_covariant(self):
        for base in (unit_square_triangles(3), unit_cube_tetrahedra(2)):
            dimension = base.points.shape[1]
            angle = 0.37
            rotation = np.eye(dimension)
            rotation[:2, :2] = [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
            mesh = replace(base, points=base.points @ rotation.T + 3)
            tensor = rotation @ np.diag(np.arange(1, dimension + 1)) @ rotation.T
            gradient = np.arange(1, dimension + 1) * 0.2
            exact = 7 + base.points @ gradient
            boundary = np.flatnonzero(
                np.any((base.points == 0) | (base.points == 1), axis=1)
            )
            result = ThermalOperator(mesh, conductivity=tensor).steady(
                EssentialBC(boundary, exact[boundary])
            )
            expected_flux = -tensor @ rotation @ gradient
            np.testing.assert_allclose(
                result.temperatures[0], exact, rtol=1e-11, atol=1e-11
            )
            np.testing.assert_allclose(
                result.cell_flux[0],
                np.broadcast_to(expected_flux, result.cell_flux[0].shape),
                rtol=1e-10,
                atol=1e-11,
            )

    def test_backward_euler_discrete_dissipation_identity(self):
        for mesh in (unit_square_triangles(4), unit_cube_tetrahedra(2)):
            boundary = np.flatnonzero(
                np.any((mesh.points == 0) | (mesh.points == 1), axis=1)
            )
            initial = np.random.default_rng(34).uniform(size=mesh.node_count)
            initial[boundary] = 0
            operator = ThermalOperator(mesh, conductivity=2, capacity=3)
            dt = 0.03
            result = operator.transient(
                initial,
                EssentialBC(boundary, np.zeros(boundary.size)),
                time_step=dt,
                steps=4,
            )
            for old, new in zip(
                result.temperatures[:-1], result.temperatures[1:], strict=True
            ):
                delta = new - old
                before, after = (
                    0.5 * old @ operator.capacity.matvec(old),
                    0.5 * new @ operator.capacity.matvec(new),
                )
                conduction = dt * new @ operator.stiffness.matvec(new)
                implicit_loss = 0.5 * delta @ operator.capacity.matvec(delta)
                scale = before + after + conduction + implicit_loss
                self.assertLessEqual(
                    abs(after - before + conduction + implicit_loss), 1e-11 * scale
                )
                self.assertLess(after, before)
                self.assertGreater(implicit_loss, 0)


if __name__ == "__main__":
    unittest.main()
