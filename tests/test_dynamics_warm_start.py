# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Newmark 显式初猜实验：全状态、突变载荷、重启和非法策略边界。"""

import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from agentfem_native.dynamics import (
    integrate_constrained_linear_dynamics,
    integrate_linear_dynamics,
)
from agentfem_native.solid import (
    LinearElastic3DProblem,
    SolidDisplacementCondition,
    SolidElasticMaterial,
)
from agentfem_native.solid_dynamics import build_t4_linear_dynamics
from agentfem_native.sparse import conjugate_gradient
from agentfem_native.volume_mesh import unit_cube_tetrahedra
from tests.test_dynamics import _oscillator, _t3_eigenmode_system


class DynamicsWarmStartTests(unittest.TestCase):
    def test_sdof_analytic_and_invalid_policy(self):
        system = _oscillator(0.01, 100)
        actual = integrate_linear_dynamics(
            system, method="newmark_average_acceleration", warm_start=True
        )
        np.testing.assert_allclose(
            actual.displacement[:, 0], np.cos(2 * actual.times), rtol=0, atol=7e-5
        )
        self.assertLess(np.max(np.abs(actual.energy_balance_error)), 1e-11)
        with self.assertRaises(TypeError):
            integrate_linear_dynamics(system, warm_start=1)
        with self.assertRaises(ValueError):
            integrate_linear_dynamics(system, warm_start=True)

    def test_full_histories_for_t3_t4_and_discontinuous_load(self):
        t3, fixed3, _ = _t3_eigenmode_system(0.001, 40, lumped_mass=False)
        problem = LinearElastic3DProblem(
            unit_cube_tetrahedra(1),
            SolidElasticMaterial(100, 0.25),
            dirichlet=(SolidDisplacementCondition("left", None, (0, 0, 0)),),
        )
        initial = np.zeros_like(problem.mesh.points)
        initial[:, 0] = 0.01 * problem.mesh.points[:, 0]
        t4, fixed4 = build_t4_linear_dynamics(
            problem,
            2,
            time_step=0.001,
            steps=40,
            initial_displacement=initial,
            lumped_mass=False,
        )
        for system, fixed in ((t3, fixed3), (t4, fixed4)):
            for jump_load in (False, True):
                with self.subTest(
                    dofs=len(system.initial_displacement), jump=jump_load
                ):
                    if jump_load:
                        system = replace(
                            system,
                            load=lambda t, size=system.mass.shape[0]: np.full(
                                size, 1 if t >= 0.02 else 0
                            ),
                        )
                    cold = integrate_constrained_linear_dynamics(
                        system, fixed, method="newmark_average_acceleration"
                    )
                    reports = []

                    def checked_solve(*args, reports=reports, **kwargs):
                        outcome = conjugate_gradient(*args, **kwargs)
                        reports.append(outcome.report)
                        return outcome

                    with patch(
                        "agentfem_native.dynamics.conjugate_gradient",
                        side_effect=checked_solve,
                    ):
                        warm = integrate_constrained_linear_dynamics(
                            system,
                            fixed,
                            method="newmark_average_acceleration",
                            warm_start=True,
                        )
                    self.assertTrue(
                        all(
                            r.converged and r.residual_norm <= r.threshold
                            for r in reports
                        )
                    )
                    for name in (
                        "displacement",
                        "velocity",
                        "acceleration",
                        "reaction",
                        "kinetic_energy",
                        "strain_energy",
                        "external_work",
                        "energy_balance_error",
                    ):
                        np.testing.assert_allclose(
                            getattr(warm, name),
                            getattr(cold, name),
                            rtol=1e-10,
                            atol=1e-10,
                            err_msg=name,
                        )

    def test_accepted_acceleration_restarts_without_hidden_cache(self):
        system, fixed, _ = _t3_eigenmode_system(0.001, 40, lumped_mass=False)
        kwargs = {"method": "newmark_average_acceleration", "warm_start": True}
        full = integrate_constrained_linear_dynamics(system, fixed, **kwargs)
        first = integrate_constrained_linear_dynamics(
            replace(system, steps=17), fixed, **kwargs
        )
        resumed = integrate_constrained_linear_dynamics(
            replace(system, steps=23), fixed, restart=first.checkpoint(), **kwargs
        )
        for name in (
            "displacement",
            "velocity",
            "acceleration",
            "reaction",
            "kinetic_energy",
            "strain_energy",
        ):
            np.testing.assert_array_equal(
                getattr(resumed, name), getattr(full, name)[17:]
            )
