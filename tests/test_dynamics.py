# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import unittest

import numpy as np

from agentfem_native.dynamics import (
    LinearSecondOrderSystem,
    assemble_t3_mass,
    integrate_linear_dynamics,
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


class LinearDynamicsTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
