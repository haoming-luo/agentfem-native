# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""有限变形的客观性、材料/全局切线、解析 patch 与失败事务证据。"""

import unittest
from unittest.mock import patch

import numpy as np

from agentfem_native.finite_strain import (
    FiniteStrainOperator,
    NonlinearConvergenceError,
    neo_hookean_tangent,
)
from agentfem_native.krylov import gmres
from agentfem_native.materials import NeoHookeanMaterial
from agentfem_native.mesh import unit_square_triangles
from agentfem_native.providers import LinearSolveError
from agentfem_native.simplex import EssentialBC
from agentfem_native.sparse import CSRMatrix
from agentfem_native.volume_mesh import unit_cube_tetrahedra


class FiniteStrainTests(unittest.TestCase):
    def test_force_controlled_large_extension(self):
        for mesh in (unit_square_triangles(3), unit_cube_tetrahedra(2)):
            d = mesh.points.shape[1]
            op = FiniteStrainOperator(mesh, NeoHookeanMaterial(100, 0.25))
            stretch = 1.5
            # 横向位移为零的解析一轴应变，加载为参考面 Piola 力。
            traction = 40 * (stretch - 1 / stretch) + 40 * np.log(stretch) / stretch
            load = op.geometry.mechanical_load(
                tractions={"right": [traction] + [0] * (d - 1)}
            )
            dofs = np.concatenate(
                (
                    d * mesh.nodes("left"),
                    (d * np.arange(mesh.node_count)[:, None] + np.arange(1, d)).ravel(),
                )
            )
            result = op.solve(
                EssentialBC(dofs, np.zeros(dofs.size)), load=load, steps=4
            )
            exact = np.zeros(mesh.points.shape)
            exact[:, 0] = (stretch - 1) * mesh.points[:, 0]
            np.testing.assert_allclose(result.final.displacements, exact, atol=2e-8)
            self.assertGreater(sum(state.iterations for state in result.states), 4)
            reaction = result.final.residual.reshape(mesh.points.shape).sum(axis=0)
            np.testing.assert_allclose(reaction + load.sum(axis=0), 0, atol=2e-8)

    def test_material_tangent_and_rigid_rotation(self):
        material = NeoHookeanMaterial(100, 0.25)
        f = np.array([[1.3, 0.2, 0], [0.1, 0.9, 0.1], [0, 0, 1.1]])
        direction = np.random.default_rng(4).normal(size=(3, 3))
        h = 1e-6
        fd = (
            material.response(f + h * direction).first_piola
            - material.response(f - h * direction).first_piola
        ) / (2 * h)
        a = neo_hookean_tangent(material, f)
        np.testing.assert_allclose(
            np.einsum("ijkl,kl->ij", a, direction), fd, rtol=1e-8, atol=1e-7
        )
        np.testing.assert_allclose(a, a.transpose(2, 3, 0, 1), atol=1e-12)
        theta = 1.2
        rotation = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        self.assertLess(abs(material.response(rotation).energy_density), 1e-12)
        np.testing.assert_allclose(
            material.response(rotation @ f).first_piola,
            rotation @ material.response(f).first_piola,
            atol=1e-12,
        )

    def test_global_tangent_energy_and_rotation(self):
        for mesh in (unit_square_triangles(2), unit_cube_tetrahedra(1)):
            op = FiniteStrainOperator(mesh, NeoHookeanMaterial(100, 0.25))
            rng = np.random.default_rng(18)
            u = rng.normal(size=mesh.points.shape) * 0.01
            direction = rng.normal(size=u.shape)
            k, force, *_ = op.assemble(u)
            h = 1e-7
            plus, minus = op.assemble(u + h * direction), op.assemble(u - h * direction)
            np.testing.assert_allclose(
                k.matvec(direction.ravel()),
                (plus[1] - minus[1]) / (2 * h),
                rtol=2e-8,
                atol=2e-6,
            )
            self.assertAlmostEqual(
                (plus[2] - minus[2]) / (2 * h), force @ direction.ravel(), places=6
            )
            rotation = np.eye(op.geometry.dimension)
            rotation[:2, :2] = [[0, -1], [1, 0]]
            rigid = mesh.points @ rotation.T - mesh.points
            np.testing.assert_allclose(op.assemble(rigid)[1], 0, atol=1e-12)

    def test_large_affine_patch_with_free_interior(self):
        for mesh in (unit_square_triangles(2), unit_cube_tetrahedra(2)):
            d = mesh.points.shape[1]
            material = NeoHookeanMaterial(100, 0.25)
            op = FiniteStrainOperator(mesh, material)
            deformation = np.eye(d)
            deformation[0, 0], deformation[0, 1] = 1.4, 0.2
            exact = mesh.points @ (deformation - np.eye(d)).T
            boundary = np.flatnonzero(
                np.any((mesh.points == 0) | (mesh.points == 1), axis=1)
            )
            dofs = (d * boundary[:, None] + np.arange(d)).ravel()
            bc = EssentialBC(dofs, exact.ravel()[dofs])
            with patch.object(
                CSRMatrix, "to_dense", side_effect=AssertionError("禁止稠密化")
            ):
                result = op.solve(bc, steps=4)
            np.testing.assert_allclose(result.final.displacements, exact, atol=2e-9)
            self.assertAlmostEqual(result.final.load_factor, 1)
            self.assertTrue(np.all(result.final.cell_jacobian > 0))
            self.assertFalse(result.final.displacements.flags.writeable)
            f = np.eye(3)
            f[:d, :d] = deformation
            np.testing.assert_allclose(
                result.final.cell_cauchy,
                np.broadcast_to(
                    material.response(f).cauchy_stress, (mesh.cell_count, 3, 3)
                ),
                atol=2e-7,
            )

    def test_gmres_indefinite_and_singular_failure(self):
        a = np.array([[0.0, 1, 0], [1, -2, 0.5], [0, 0.5, 3]])
        rows, cols = np.indices(a.shape)
        matrix = CSRMatrix.from_coo(a.shape, rows.ravel(), cols.ravel(), a.ravel())
        expected = np.array([2.0, -3, 1])
        result = gmres(matrix, a @ expected)
        np.testing.assert_allclose(result.solution, expected, atol=1e-11)
        zero = CSRMatrix.from_coo((2, 2), [0, 1], [0, 1], [0, 0])
        with self.assertRaises(LinearSolveError):
            gmres(zero, [1, 1])

    def test_cutback_and_committed_state_survives_failure(self):
        mesh = unit_square_triangles(2)
        op = FiniteStrainOperator(mesh, NeoHookeanMaterial(100, 0.25))
        boundary = np.flatnonzero(
            np.any((mesh.points == 0) | (mesh.points == 1), axis=1)
        )
        dofs = (2 * boundary[:, None] + np.arange(2)).ravel()
        exact = mesh.points * [0.4, 0]
        bc = EssentialBC(dofs, exact.ravel()[dofs])
        real = op.assemble
        calls = []

        def once_reject(u, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise ValueError("测试强制首次试探失败")
            return real(u, **kwargs)

        with patch.object(op, "assemble", side_effect=once_reject):
            result = op.solve(bc, steps=2)
        self.assertEqual(result.rejected_steps, 1)
        np.testing.assert_allclose(result.final.displacements, exact, atol=2e-9)
        # 所有节点被强制翻转；若最终步失败，错误只暴露此前合法状态。
        bad = mesh.points * [-2.0, 0]
        with self.assertRaises(NonlinearConvergenceError) as caught:
            op.solve(
                EssentialBC(np.arange(bad.size), bad.ravel()),
                steps=4,
                maximum_cutbacks=3,
            )
        self.assertGreater(len(caught.exception.accepted_states), 0)
        for state in caught.exception.accepted_states:
            self.assertTrue(np.all(state.cell_jacobian > 0))

    def test_finite_thermal_free_expansion(self):
        mesh = unit_cube_tetrahedra(1)
        op = FiniteStrainOperator(
            mesh, NeoHookeanMaterial(100, 0.25), thermal_stretch=1.1
        )
        u = 0.1 * mesh.points
        _, force, energy, _, stress = op.assemble(u)
        np.testing.assert_allclose(force, 0, atol=1e-12)
        np.testing.assert_allclose(stress, 0, atol=1e-12)
        self.assertLess(abs(energy), 1e-12)
