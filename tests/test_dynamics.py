# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import unittest

import numpy as np

from agentfem_native.dynamics import (
    LinearSecondOrderSystem,
    assemble_t3_mass,
    build_t3_linear_dynamics,
    central_difference_safe_time_step,
    integrate_constrained_linear_dynamics,
    integrate_linear_dynamics,
)
from agentfem_native.elasticity import (
    DisplacementCondition,
    LinearElasticMaterial,
    LinearElasticProblem,
    prepare_linear_elasticity_assembly,
)
from agentfem_native.mesh import unit_square_triangles
from agentfem_native.sparse import CSRMatrix


def _diagonal(values) -> CSRMatrix:
    values = np.asarray(values, dtype=np.float64)
    indices = np.arange(values.size, dtype=np.int64)
    return CSRMatrix.from_coo((values.size, values.size), indices, indices, values)


def _oscillator(time_step: float, steps: int) -> LinearSecondOrderSystem:
    return LinearSecondOrderSystem(
        stiffness=_diagonal((4.0,)),
        mass=_diagonal((1.0,)),
        load=np.zeros(1),
        time_step=time_step,
        steps=steps,
        initial_displacement=np.ones(1),
        initial_velocity=np.zeros(1),
    )


def _t3_eigenmode_system(
    time_step: float, steps: int, *, lumped_mass: bool
) -> tuple[LinearSecondOrderSystem, np.ndarray, float]:
    """构造受约束 T3 的最低离散模态，作为时间积分的独立小型对照。"""

    mesh = unit_square_triangles(2)
    problem = LinearElasticProblem(
        mesh,
        LinearElasticMaterial(100.0, 0.25),
        dirichlet=(DisplacementCondition("left", None, (0.0, 0.0)),),
    )
    prototype, constrained = build_t3_linear_dynamics(
        problem,
        2.0,
        time_step=time_step,
        steps=steps,
        lumped_mass=lumped_mass,
    )
    active = np.setdiff1d(np.arange(prototype.stiffness.shape[0]), constrained)
    stiffness = prototype.stiffness.to_dense()[np.ix_(active, active)]
    mass = prototype.mass.to_dense()[np.ix_(active, active)]
    cholesky = np.linalg.cholesky(mass)
    left_scaled = np.linalg.solve(cholesky, stiffness)
    transformed = np.linalg.solve(cholesky, left_scaled.T).T
    eigenvalues, eigenvectors = np.linalg.eigh(transformed)
    mode = np.linalg.solve(cholesky.T, eigenvectors[:, 0])
    mode /= np.max(np.abs(mode))
    initial = np.zeros(prototype.stiffness.shape[0], dtype=np.float64)
    initial[active] = mode
    system, rebuilt_constrained = build_t3_linear_dynamics(
        problem,
        2.0,
        time_step=time_step,
        steps=steps,
        initial_displacement=initial,
        lumped_mass=lumped_mass,
    )
    np.testing.assert_array_equal(rebuilt_constrained, constrained)
    return system, constrained, float(np.sqrt(eigenvalues[0]))


class T3MassTests(unittest.TestCase):
    def test_consistent_and_lumped_mass_preserve_total_mass(self) -> None:
        mesh = unit_square_triangles(2)
        consistent = assemble_t3_mass(mesh, 2.5, thickness=0.4)
        lumped = assemble_t3_mass(mesh, 2.5, thickness=0.4, lumped=True)
        ones = np.ones(2 * mesh.node_count)
        for matrix in (consistent, lumped):
            nodal = matrix.matvec(ones).reshape(mesh.node_count, 2)
            np.testing.assert_allclose(nodal.sum(axis=0), (1.0, 1.0), atol=2.0e-16)
        np.testing.assert_allclose(lumped.data, consistent.matvec(ones))

    def test_t3_mesh_to_transient_result_preserves_fixed_boundary(self) -> None:
        mesh = unit_square_triangles(2)
        initial = np.zeros((mesh.node_count, 2))
        initial[mesh.nodes("right"), 0] = 0.01
        problem = LinearElasticProblem(
            mesh,
            LinearElasticMaterial(100.0, 0.25),
            dirichlet=(DisplacementCondition("left", None, (0.0, 0.0)),),
        )
        system, constrained = build_t3_linear_dynamics(
            problem,
            1.0,
            time_step=0.001,
            steps=20,
            initial_displacement=initial,
        )
        plan = prepare_linear_elasticity_assembly(problem)
        prepared_system, prepared_constrained = build_t3_linear_dynamics(
            problem,
            1.0,
            time_step=0.001,
            steps=20,
            initial_displacement=initial,
            assembly_plan=plan,
        )
        np.testing.assert_array_equal(prepared_constrained, constrained)
        np.testing.assert_array_equal(
            prepared_system.stiffness.data, system.stiffness.data
        )
        self.assertIs(prepared_system.stiffness.indptr, plan.pattern.indptr)
        result = integrate_constrained_linear_dynamics(system, constrained)
        prepared_result = integrate_constrained_linear_dynamics(
            prepared_system, prepared_constrained
        )
        np.testing.assert_array_equal(prepared_result.displacement, result.displacement)
        np.testing.assert_array_equal(result.displacement[:, constrained], 0.0)
        self.assertEqual(result.reaction.shape, (21, constrained.size))
        self.assertTrue(np.all(np.isfinite(result.energy_balance_error)))


class LinearDynamicsTests(unittest.TestCase):
    def test_explicit_safe_step_is_reported_and_unsafe_step_fails(self) -> None:
        self.assertEqual(central_difference_safe_time_step(_oscillator(0.1, 1)), 1.0)
        load_calls: list[float] = []

        def tracked_load(time: float) -> np.ndarray:
            load_calls.append(time)
            return np.zeros(1)

        unsafe = LinearSecondOrderSystem(
            stiffness=_diagonal((4.0,)),
            mass=_diagonal((1.0,)),
            load=tracked_load,
            time_step=1.01,
            steps=1,
            initial_displacement=np.ones(1),
            initial_velocity=np.zeros(1),
        )
        with self.assertRaisesRegex(ValueError, "超过保守稳定上限"):
            integrate_linear_dynamics(unsafe)
        self.assertEqual(load_calls, [])

    def test_central_difference_has_second_order_error_trend(self) -> None:
        coarse = integrate_linear_dynamics(_oscillator(0.1, 10))
        fine = integrate_linear_dynamics(_oscillator(0.05, 20))
        exact = np.cos(2.0)
        coarse_error = abs(coarse.displacement[-1, 0] - exact)
        fine_error = abs(fine.displacement[-1, 0] - exact)
        self.assertLess(fine_error, coarse_error / 3.5)
        self.assertLess(np.max(np.abs(coarse.total_energy - 2.0)), 0.021)

    def test_newmark_average_acceleration_conserves_undamped_energy(self) -> None:
        result = integrate_linear_dynamics(
            _oscillator(0.1, 100), method="newmark_average_acceleration"
        )
        np.testing.assert_allclose(result.total_energy, 2.0, atol=2.0e-13)
        np.testing.assert_allclose(result.external_work, 0.0, atol=0.0)
        np.testing.assert_allclose(result.energy_balance_error, 0.0, atol=2.0e-13)

    def test_newmark_forced_motion_closes_external_work_ledger(self) -> None:
        system = LinearSecondOrderSystem(
            stiffness=_diagonal((4.0,)),
            mass=_diagonal((1.0,)),
            load=np.ones(1),
            time_step=0.01,
            steps=100,
            initial_displacement=np.zeros(1),
            initial_velocity=np.zeros(1),
        )
        result = integrate_linear_dynamics(
            system, method="newmark_average_acceleration"
        )
        np.testing.assert_allclose(
            result.total_energy - result.total_energy[0],
            result.external_work,
            atol=2.0e-13,
        )

    def test_checkpoint_restart_reproduces_uninterrupted_trajectory(self) -> None:
        complete = integrate_linear_dynamics(_oscillator(0.02, 100))
        first = integrate_linear_dynamics(_oscillator(0.02, 40))
        checkpoint = first.checkpoint()
        resumed = integrate_linear_dynamics(_oscillator(0.02, 60), restart=checkpoint)
        self.assertEqual(checkpoint.step, 40)
        self.assertEqual(len(checkpoint.digest), 64)
        np.testing.assert_array_equal(resumed.displacement, complete.displacement[40:])
        np.testing.assert_array_equal(resumed.velocity, complete.velocity[40:])
        np.testing.assert_array_equal(resumed.acceleration, complete.acceleration[40:])

    def test_newmark_checkpoint_reproduces_uninterrupted_trajectory(self) -> None:
        method = "newmark_average_acceleration"
        complete = integrate_linear_dynamics(_oscillator(0.02, 100), method=method)
        first = integrate_linear_dynamics(_oscillator(0.02, 40), method=method)
        resumed = integrate_linear_dynamics(
            _oscillator(0.02, 60), method=method, restart=first.checkpoint()
        )
        np.testing.assert_array_equal(resumed.displacement, complete.displacement[40:])
        np.testing.assert_array_equal(resumed.velocity, complete.velocity[40:])
        np.testing.assert_array_equal(resumed.acceleration, complete.acceleration[40:])

    def test_t3_discrete_mode_has_second_order_time_refinement(self) -> None:
        for method, lumped_mass in (
            ("central_difference", True),
            ("newmark_average_acceleration", False),
        ):
            with self.subTest(method=method):
                _probe, _, frequency = _t3_eigenmode_system(
                    0.001, 1, lumped_mass=lumped_mass
                )
                duration = 0.5 * np.pi / frequency
                errors = []
                for steps in (20, 40):
                    system, constrained, rebuilt_frequency = _t3_eigenmode_system(
                        duration / steps, steps, lumped_mass=lumped_mass
                    )
                    self.assertAlmostEqual(rebuilt_frequency, frequency)
                    result = integrate_constrained_linear_dynamics(
                        system, constrained, method=method
                    )
                    exact = np.zeros_like(system.initial_displacement)
                    errors.append(
                        float(np.linalg.norm(result.displacement[-1] - exact))
                    )
                self.assertLess(errors[1], errors[0] / 3.5)

    def test_explicit_method_rejects_consistent_mass(self) -> None:
        mass = CSRMatrix.from_coo(
            (2, 2), [0, 0, 1, 1], [0, 1, 0, 1], [2.0, 1.0, 1.0, 2.0]
        )
        system = LinearSecondOrderSystem(
            _diagonal((1.0, 1.0)),
            mass,
            np.zeros(2),
            0.1,
            1,
            np.zeros(2),
            np.zeros(2),
        )
        with self.assertRaisesRegex(ValueError, "diagonal mass"):
            integrate_linear_dynamics(system)

    def test_zero_constraint_reduction_recovers_dynamic_reaction(self) -> None:
        stiffness = CSRMatrix.from_coo(
            (2, 2), [0, 0, 1, 1], [0, 1, 0, 1], [1.0, -1.0, -1.0, 1.0]
        )
        system = LinearSecondOrderSystem(
            stiffness=stiffness,
            mass=_diagonal((1.0, 1.0)),
            load=np.zeros(2),
            time_step=0.02,
            steps=50,
            initial_displacement=np.array((0.0, 1.0)),
            initial_velocity=np.zeros(2),
        )
        constrained = integrate_constrained_linear_dynamics(system, [0])
        reduced = integrate_linear_dynamics(
            LinearSecondOrderSystem(
                _diagonal((1.0,)),
                _diagonal((1.0,)),
                np.zeros(1),
                0.02,
                50,
                np.ones(1),
                np.zeros(1),
            )
        )
        np.testing.assert_array_equal(constrained.displacement[:, 0], 0.0)
        np.testing.assert_array_equal(
            constrained.displacement[:, 1], reduced.displacement[:, 0]
        )
        np.testing.assert_allclose(
            constrained.reaction[:, 0], -constrained.displacement[:, 1], atol=1.0e-15
        )
        self.assertEqual(constrained.checkpoint().displacement.shape, (2,))


if __name__ == "__main__":
    unittest.main()
