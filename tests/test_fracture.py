# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""AT2 均匀解析、不可逆卸载、裂纹轮廓与变分界限证据。"""

import itertools
import unittest
from unittest.mock import patch

import numpy as np

from agentfem_native.finite_strain import NonlinearConvergenceError
from agentfem_native.fracture import (
    AT2FractureOperator,
    box_kkt_residual,
    solve_box_quadratic,
)
from agentfem_native.mesh import unit_square_triangles
from agentfem_native.simplex import EssentialBC
from agentfem_native.sparse import CSRMatrix
from agentfem_native.volume_mesh import unit_cube_tetrahedra


class FractureTests(unittest.TestCase):
    def test_damage_is_invariant_under_energy_unit_scaling(self):
        a = np.array([[4.0, -1], [-1, 3]])
        rows, cols = np.indices(a.shape)
        solutions = []
        for scale in (1e-16, 1, 1e16):
            matrix = CSRMatrix.from_coo(
                a.shape, rows.ravel(), cols.ravel(), (scale * a).ravel()
            )
            solution, _, _ = solve_box_quadratic(
                matrix, scale * np.array([1.0, 2]), [0.2, 0.1], [1, 1], [0.2, 0.1]
            )
            solutions.append(solution)
        np.testing.assert_allclose(
            solutions, np.broadcast_to(solutions[1], (3, 2)), atol=1e-12
        )

    def test_box_solution_matches_enumerated_active_sets(self):
        a = np.array([[4.0, -1, 0], [-1, 3, -1], [0, -1, 2]])
        b = np.array([5.0, 0, -2])
        lower, upper = np.array([0.1, 0.2, 0.3]), np.ones(3)
        rows, cols = np.indices((3, 3))
        matrix = CSRMatrix.from_coo((3, 3), rows.ravel(), cols.ravel(), a.ravel())
        x, kkt, _ = solve_box_quadratic(matrix, b, lower, upper, lower)
        candidates = []
        for flags in itertools.product((-1, 0, 1), repeat=3):
            flags = np.array(flags)
            candidate = np.where(flags < 0, lower, upper)
            free, fixed = np.flatnonzero(flags == 0), np.flatnonzero(flags != 0)
            if free.size:
                candidate[free] = np.linalg.solve(
                    a[np.ix_(free, free)],
                    b[free] - a[np.ix_(free, fixed)] @ candidate[fixed],
                )
            if np.all(candidate >= lower) and np.all(candidate <= upper):
                candidates.append(
                    (0.5 * candidate @ a @ candidate - b @ candidate, candidate)
                )
        expected = min(candidates, key=lambda item: item[0])[1]
        np.testing.assert_allclose(x, expected, atol=1e-12)
        self.assertLess(kkt, 1e-10)
        clipped = np.clip(np.linalg.solve(a, b), lower, upper)
        self.assertGreater(
            np.linalg.norm(box_kkt_residual(clipped, a @ clipped - b, lower, upper)),
            0.1,
        )

    def test_homogeneous_damage_analytic_and_unloading(self):
        for mesh in (unit_square_triangles(2), unit_cube_tetrahedra(1)):
            d = mesh.points.shape[1]
            op = AT2FractureOperator(mesh, 100, 0.25, toughness=0.1, length_scale=0.2)
            displacement = mesh.points * 0.02
            bc = EssentialBC(np.arange(displacement.size), displacement.ravel())
            with patch.object(
                CSRMatrix, "to_dense", side_effect=AssertionError("禁止稠密化")
            ):
                result = op.solve(bc, load_factors=[0.5, 1.0, 0.25])
            psi = op.elastic.fields(displacement)[2][0]
            exact = 2 * psi / (0.1 / 0.2 + 2 * psi)
            np.testing.assert_allclose(result.states[1].damage, exact, atol=1e-10)
            np.testing.assert_allclose(result.states[2].damage, exact, atol=1e-10)
            for earlier, later in zip(result.states, result.states[1:]):
                self.assertTrue(np.all(later.damage >= earlier.damage - 1e-14))
            for state in result.states:
                self.assertLess(state.damage_kkt_residual, 1e-9)
                self.assertTrue(np.all(np.diff(state.alternating_energies) <= 1e-10))
            self.assertEqual(result.final.displacements.shape[1], d)

    def test_seeded_crack_profile_converges_and_never_heals(self):
        errors = []
        for resolution in (16, 32):
            # 二维形状规则网格同时细化两方向，避免仅 x 细化导致长宽比发散。
            mesh = unit_square_triangles(resolution, resolution)
            op = AT2FractureOperator(mesh, 100, 0.25, toughness=1, length_scale=0.1)
            initial = np.zeros(mesh.node_count)
            initial[np.isclose(mesh.points[:, 0], 0.5)] = 1
            bc = EssentialBC(
                np.arange(mesh.node_count * 2), np.zeros(mesh.node_count * 2)
            )
            result = op.solve(bc, load_factors=[0], initial_damage=initial)
            exact = np.cosh((0.5 - np.abs(mesh.points[:, 0] - 0.5)) / 0.1) / np.cosh(5)
            errors.append(np.max(np.abs(result.final.damage - exact)))
            np.testing.assert_array_equal(result.final.damage[initial == 1], 1)
            self.assertGreater(result.final.fracture_energy, 0)
        self.assertGreater(errors[0] / errors[1], 3)

    def test_nonuniform_damage_and_free_mechanical_equilibrium(self):
        mesh = unit_square_triangles(4)
        op = AT2FractureOperator(mesh, 100, 0.25, toughness=1, length_scale=0.2)
        left, right = mesh.nodes("left"), mesh.nodes("right")
        dofs = np.concatenate(((2 * left[:, None] + np.arange(2)).ravel(), 2 * right))
        values = np.concatenate((np.zeros(left.size * 2), np.full(right.size, 0.01)))
        initial = np.zeros(mesh.node_count)
        initial[(mesh.points[:, 0] <= 0.25) & (mesh.points[:, 1] == 0.5)] = 1
        result = op.solve(
            EssentialBC(dofs, values), initial_damage=initial, load_factors=[0.5, 1]
        )
        self.assertLess(result.final.free_residual_norm, 1e-8)
        self.assertTrue(np.all(result.final.damage >= initial))
        self.assertGreater(np.ptp(result.final.damage), 0.5)

    def test_failure_does_not_mutate_initial_damage(self):
        mesh = unit_square_triangles(2)
        op = AT2FractureOperator(mesh, 100, 0.25, toughness=1, length_scale=0.2)
        initial = np.zeros(mesh.node_count)
        bc = EssentialBC(np.arange(mesh.node_count * 2), (mesh.points * 0.1).ravel())
        with self.assertRaises(NonlinearConvergenceError) as caught:
            op.solve(bc, initial_damage=initial, maximum_alternations=1)
        self.assertEqual(caught.exception.accepted_states, ())
        np.testing.assert_array_equal(initial, 0)
        with self.assertRaises(ValueError):
            op.degradation(np.full(mesh.node_count, 1.1))
