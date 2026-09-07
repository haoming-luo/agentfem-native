# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import unittest

from agentfem_native.dynamics import LinearSecondOrderSystem, integrate_linear_dynamics
from agentfem_native.planning import plan_linear_dynamics
from agentfem_native.runtime import (
    CancellationToken,
    ExecutionContext,
    NativeExecutionError,
    ResourceBudget,
    enforce_plan_budget,
)
from agentfem_native.sparse import CSRMatrix


def _system(steps: int) -> LinearSecondOrderSystem:
    matrix = CSRMatrix.from_coo((1, 1), [0], [0], [1.0])
    return LinearSecondOrderSystem(matrix, matrix, [0.0], 0.1, steps, [1.0], [0.0])


class RuntimeControlTests(unittest.TestCase):
    def test_plan_budget_failure_is_structured(self) -> None:
        plan = plan_linear_dynamics(_system(10))
        with self.assertRaises(NativeExecutionError) as captured:
            enforce_plan_budget(plan, ResourceBudget(maximum_dofs=0))
        self.assertEqual(captured.exception.code, "budget.dofs_exceeded")
        self.assertTrue(captured.exception.retryable)
        self.assertEqual(captured.exception.as_dict()["path"], "plan.dof_count")

    def test_progress_can_cancel_at_accepted_boundary(self) -> None:
        token = CancellationToken()
        events = []

        def progress(event):
            events.append(event)
            if event.completed == 2:
                token.cancel()

        context = ExecutionContext(
            cancellation=token,
            progress=progress,
        )
        with self.assertRaises(NativeExecutionError) as captured:
            integrate_linear_dynamics(_system(10), context=context)
        self.assertEqual(captured.exception.code, "execution.cancelled")
        self.assertEqual([event.completed for event in events], [0, 1, 2])

    def test_step_budget_rejects_before_first_transition(self) -> None:
        with self.assertRaises(NativeExecutionError) as captured:
            integrate_linear_dynamics(
                _system(3),
                context=ExecutionContext(budget=ResourceBudget(maximum_steps=2)),
            )
        self.assertEqual(captured.exception.code, "budget.steps_exceeded")


if __name__ == "__main__":
    unittest.main()
