# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""自动请求不得把时间溢出、已取消请求或非法检查点带入求解。"""

import unittest
from unittest.mock import patch

import numpy as np

from agentfem_native.contract import plan_kernel_request
from agentfem_native.dynamics import (
    DynamicCheckpoint,
    _checkpoint_digest,
    integrate_linear_dynamics,
)
from agentfem_native.multiphysics import plan_multiphysics_request
from agentfem_native.runtime import (
    CancellationToken,
    ExecutionContext,
    NativeExecutionError,
    ResourceBudget,
)
from agentfem_native.simplex import EssentialBC
from agentfem_native.thermal import ThermalOperator
from agentfem_native.time_grid import time_grid, validate_time_interval
from tests.test_contract import dynamics_request as stable_dynamics_request
from tests.test_dynamics import _oscillator
from tests.test_multiphysics import request_for
from tests.test_t4_dynamics_contract import dynamics_request


class TimeAdmissionTests(unittest.TestCase):
    def test_finite_grid_and_unrepresentable_interval(self):
        np.testing.assert_array_equal(
            time_grid(0.125, 4, start_time=1), [1, 1.125, 1.25, 1.375, 1.5]
        )
        np.testing.assert_array_equal(time_grid(0.125, 0), [0])
        for dt, steps, start in (
            (1e308, 2, 0),
            (0.1, 2, 1e20),
            (0.1, 2**54, 0),
            (1, 1, float("nan")),
            (1, 1, -1),
            (0.1, -1, 0),
        ):
            with self.assertRaises(ValueError):
                validate_time_interval(dt, steps, start_time=start)
        with self.assertRaises(TypeError):
            validate_time_interval(True, 1)
        # 两端相邻时间能区分，但中间仍可能因舍入产生重复；执行态再查完整网格。
        validate_time_interval(0.75, 4, start_time=2**52)
        with self.assertRaisesRegex(ValueError, "严格递增"):
            time_grid(0.75, 4, start_time=2**52)

    def test_overflow_rejected_in_system_and_candidate_plan(self):
        with self.assertRaisesRegex(ValueError, "终点"):
            _oscillator(1e308, 2)
        for request in (request_for("thermal_transient"), dynamics_request()):
            request["procedure"].update(time_step=1e308, steps=2)
            with patch(
                "agentfem_native.multiphysics.ThermalOperator",
                side_effect=AssertionError("不应装配"),
            ):
                self.assertEqual(plan_multiphysics_request(request)["status"], "failed")
        stable = stable_dynamics_request()
        stable["procedure"].update(time_step=1e308, steps=2)
        self.assertEqual(plan_kernel_request(stable)["status"], "failed")

    def test_restart_time_and_step_overflow_precede_solver(self):
        vector = np.zeros(1)
        for step, time in ((np.int64(np.iinfo(np.int64).max), 0.0), (1, 1e20)):
            checkpoint = DynamicCheckpoint(
                "central_difference",
                step,
                time,
                vector,
                vector,
                vector,
                _checkpoint_digest(
                    "central_difference", int(step), time, vector, vector, vector
                ),
            )
            with (
                patch(
                    "agentfem_native.dynamics._solve_spd",
                    side_effect=AssertionError("不应求解"),
                ),
                self.assertRaises(ValueError),
            ):
                integrate_linear_dynamics(_oscillator(0.1, 1), restart=checkpoint)

    def test_thermal_overflow_before_preparing_linear_system(self):
        from agentfem_native.mesh import unit_square_triangles

        operator = ThermalOperator(unit_square_triangles(1))
        with (
            patch(
                "agentfem_native.thermal.NativeSparseProvider",
                side_effect=AssertionError("不应准备"),
            ),
            self.assertRaisesRegex(ValueError, "终点"),
        ):
            operator.transient(
                np.zeros(4), EssentialBC.empty(), time_step=1e308, steps=2
            )

    def test_cancellation_and_budget_precede_initial_solve(self):
        token = CancellationToken()
        token.cancel()
        for context in (
            ExecutionContext(cancellation=token),
            ExecutionContext(budget=ResourceBudget(maximum_steps=0)),
        ):
            with (
                patch(
                    "agentfem_native.dynamics._solve_spd",
                    side_effect=AssertionError("不应求解初始质量"),
                ),
                self.assertRaises(NativeExecutionError),
            ):
                integrate_linear_dynamics(_oscillator(0.1, 1), context=context)

    def test_checkpoint_metadata_must_be_valid_even_with_matching_digest(self):
        for method, step, time in (
            ("invalid", 0, 0),
            ("central_difference", -1, 0),
            ("central_difference", 0, float("nan")),
            ("central_difference", 0, -1),
        ):
            vector = np.zeros(1)
            digest = _checkpoint_digest(method, step, time, vector, vector, vector)
            with self.assertRaises(ValueError):
                DynamicCheckpoint(method, step, time, vector, vector, vector, digest)
        first, second = np.zeros(1), np.zeros(2)
        digest = _checkpoint_digest("central_difference", 0, 0, first, second, first)
        with self.assertRaises(ValueError):
            DynamicCheckpoint("central_difference", 0, 0, first, second, first, digest)


if __name__ == "__main__":
    unittest.main()
