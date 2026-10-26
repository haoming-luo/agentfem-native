# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""T4 质量、离散模态时间细化与重启证据。"""

import unittest
from dataclasses import replace

import numpy as np

from agentfem_native.dynamics import integrate_constrained_linear_dynamics
from agentfem_native.solid import (
    LinearElastic3DProblem,
    SolidDisplacementCondition,
    SolidElasticMaterial,
)
from agentfem_native.solid_dynamics import assemble_t4_mass, build_t4_linear_dynamics
from agentfem_native.volume_mesh import unit_cube_tetrahedra


class SolidDynamicsTests(unittest.TestCase):
    def problem(self):
        return LinearElastic3DProblem(
            unit_cube_tetrahedra(1),
            SolidElasticMaterial(100, 0.25),
            dirichlet=(SolidDisplacementCondition("left", None, (0, 0, 0)),),
        )

    def test_mass_sum_symmetry_positive_and_lumping(self):
        mesh = self.problem().mesh
        mass = assemble_t4_mass(mesh, 2)
        lumped = assemble_t4_mass(mesh, 2, lumped=True)
        self.assertAlmostEqual(mass.data.sum(), 6)
        np.testing.assert_allclose(
            mass.matvec(np.ones(3 * mesh.node_count)), lumped.diagonal(), atol=1e-15
        )
        np.testing.assert_allclose(mass.to_dense(), mass.to_dense().T, atol=1e-15)
        self.assertGreater(np.linalg.eigvalsh(mass.to_dense()).min(), 0)

    def test_modal_newmark_second_order_and_restart(self):
        system, constrained = build_t4_linear_dynamics(
            self.problem(), 2, time_step=0.001, steps=1, lumped_mass=False
        )
        free = np.setdiff1d(np.arange(system.mass.shape[0]), constrained)
        mass = system.mass.to_dense()[np.ix_(free, free)]
        stiffness = system.stiffness.to_dense()[np.ix_(free, free)]
        cholesky = np.linalg.cholesky(mass)
        inverse = np.linalg.inv(cholesky)
        eigenvalues, vectors = np.linalg.eigh(inverse @ stiffness @ inverse.T)
        mode = np.linalg.solve(cholesky.T, vectors[:, 0])
        initial = np.zeros(system.mass.shape[0])
        initial[free] = mode / np.max(np.abs(mode)) * 0.01
        omega = np.sqrt(eigenvalues[0])
        final_time = np.pi / (2 * omega)
        errors = []
        for steps in (20, 40):
            trial = replace(
                system,
                initial_displacement=initial,
                steps=steps,
                time_step=final_time / steps,
            )
            result = integrate_constrained_linear_dynamics(
                trial, constrained, method="newmark_average_acceleration"
            )
            errors.append(np.linalg.norm(result.displacement[-1]))
        self.assertGreater(errors[0] / errors[1], 3.9)
        full = replace(system, initial_displacement=initial, steps=20)
        first = integrate_constrained_linear_dynamics(
            replace(full, steps=8), constrained, method="newmark_average_acceleration"
        )
        restart = integrate_constrained_linear_dynamics(
            replace(full, steps=12),
            constrained,
            method="newmark_average_acceleration",
            restart=first.checkpoint(),
        )
        expected = integrate_constrained_linear_dynamics(
            full, constrained, method="newmark_average_acceleration"
        )
        np.testing.assert_array_equal(
            restart.displacement[-1], expected.displacement[-1]
        )

    def test_mass_and_constraint_failures(self):
        with self.assertRaises(ValueError):
            assemble_t4_mass(self.problem().mesh, -1)
        problem = replace(
            self.problem(),
            dirichlet=(SolidDisplacementCondition("left", None, (1, 0, 0)),),
        )
        with self.assertRaises(ValueError):
            build_t4_linear_dynamics(problem, 1, time_step=0.1, steps=2)
