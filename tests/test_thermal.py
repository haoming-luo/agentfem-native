# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""热场的解析、守恒、阶数和非法输入证据。"""

import unittest
from unittest.mock import patch

import numpy as np

from agentfem_native.mesh import unit_square_triangles
from agentfem_native.simplex import EssentialBC
from agentfem_native.sparse import CSRMatrix
from agentfem_native.thermal import ThermalBoundary, ThermalOperator
from agentfem_native.volume_mesh import unit_cube_tetrahedra


class ThermalTests(unittest.TestCase):
    def meshes(self):
        return (unit_square_triangles(3), unit_cube_tetrahedra(2))

    def test_linear_patch_and_total_capacity(self):
        for mesh in self.meshes():
            with self.subTest(dimension=mesh.points.shape[1]):
                operator = ThermalOperator(mesh, conductivity=2, capacity=3)
                boundary = np.flatnonzero(
                    np.any((mesh.points == 0) | (mesh.points == 1), axis=1)
                )
                exact = 10 + mesh.points.sum(axis=1)
                bc = EssentialBC(boundary, exact[boundary])
                with patch.object(
                    CSRMatrix, "to_dense", side_effect=AssertionError("禁止稠密化")
                ):
                    result = operator.steady(bc)
                np.testing.assert_allclose(result.temperatures[0], exact, atol=1e-10)
                np.testing.assert_allclose(result.cell_flux, -2, atol=1e-10)
                self.assertAlmostEqual(operator.capacity.data.sum(), 3)
                self.assertLess(abs(result.balance_errors[0]), 1e-10)

    def test_insulated_heating_and_preconditioner_reuse(self):
        from agentfem_native.providers import NativeSparseProvider

        for mesh in self.meshes():
            operator = ThermalOperator(mesh, capacity=4, source=8)
            original = NativeSparseProvider.prepare_constrained
            calls = []

            def tracked(provider, *args, calls=calls, original=original):
                calls.append(1)
                return original(provider, *args)

            with patch.object(NativeSparseProvider, "prepare_constrained", tracked):
                result = operator.transient(
                    np.full(mesh.node_count, 5.0),
                    EssentialBC.empty(),
                    time_step=0.1,
                    steps=5,
                )
            self.assertEqual(len(calls), 1)
            np.testing.assert_allclose(result.temperatures[-1], 6, atol=1e-11)
            np.testing.assert_allclose(
                result.stored_heat, 20 + 8 * result.times, atol=1e-10
            )
            np.testing.assert_allclose(result.balance_errors, 0, atol=1e-9)
            self.assertFalse(result.temperatures.flags.writeable)

    def test_convection_and_inward_flux(self):
        for mesh in self.meshes():
            left = mesh.nodes("left")
            operator = ThermalOperator(
                mesh,
                conductivity=2,
                boundaries=(
                    ThermalBoundary(
                        "right", inward_flux=3, convection=4, ambient_temperature=7
                    ),
                ),
            )
            result = operator.steady(EssentialBC(left, np.full(left.size, 5.0)))
            # 2*slope = 3 + 4*(7-(5+slope))。
            exact = 5 + 11 / 6 * mesh.points[:, 0]
            np.testing.assert_allclose(result.temperatures[0], exact, atol=1e-10)
            np.testing.assert_allclose(result.cell_flux[0, :, 0], -11 / 3, atol=1e-10)

    def test_backward_euler_first_order_discrete_mode(self):
        mesh = unit_square_triangles(3)
        operator = ThermalOperator(mesh)
        boundary = np.flatnonzero(
            np.any((mesh.points == 0) | (mesh.points == 1), axis=1)
        )
        free = np.setdiff1d(np.arange(mesh.node_count), boundary)
        mass = operator.capacity.to_dense()[np.ix_(free, free)]
        stiffness = operator.stiffness.to_dense()[np.ix_(free, free)]
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.solve(mass, stiffness))
        index = np.argmin(eigenvalues)
        rate = eigenvalues[index].real
        initial = np.zeros(mesh.node_count)
        initial[free] = eigenvectors[:, index].real
        final_time = 0.5 / rate
        errors = []
        for steps in (20, 40):
            result = operator.transient(
                initial,
                EssentialBC(boundary, np.zeros(boundary.size)),
                time_step=final_time / steps,
                steps=steps,
            )
            errors.append(
                np.linalg.norm(result.temperatures[-1] - initial * np.exp(-0.5))
            )
        self.assertGreater(errors[0] / errors[1], 1.9)

    def test_failure_boundaries(self):
        mesh = unit_square_triangles(1)
        for conductivity in (-1, [[1, 2], [0, 1]], float("nan")):
            with self.assertRaises(ValueError):
                ThermalOperator(mesh, conductivity=conductivity)
        operator = ThermalOperator(mesh)
        with self.assertRaises(ValueError):
            operator.steady(EssentialBC.empty())
        with self.assertRaises(ValueError):
            operator.transient(
                np.zeros(mesh.node_count), EssentialBC([0], [1]), time_step=0.1, steps=1
            )
        with self.assertRaises(ValueError):
            operator.transient(
                np.zeros(mesh.node_count), EssentialBC.empty(), time_step=0, steps=1
            )
