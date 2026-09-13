# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import json
import unittest
from dataclasses import replace

import numpy as np

from agentfem_native import (
    DirichletCondition,
    DisplacementCondition,
    ExecutionContext,
    LinearElastic3DProblem,
    LinearElasticMaterial,
    LinearElasticProblem,
    LinearSecondOrderSystem,
    NativeExecutionError,
    ResourceBudget,
    SolidDisplacementCondition,
    SolidElasticMaterial,
    SteadyDiffusionProblem,
    TriangularMesh,
    execute_plan,
    explain_execution,
    plan_linear_dynamics,
    plan_linear_elasticity,
    plan_linear_elasticity_3d,
    plan_steady_diffusion,
    unit_cube_tetrahedra,
    unit_square_triangles,
)
from agentfem_native.sparse import CSRMatrix


class PlannedExecutionTests(unittest.TestCase):
    def _problem(self, resolution: int = 2) -> SteadyDiffusionProblem:
        return SteadyDiffusionProblem(
            unit_square_triangles(resolution),
            1.0,
            source=2.0,
            dirichlet=(DirichletCondition("boundary", 0.0),),
        )

    def test_plan_execute_explain_keeps_result_and_json_evidence_separate(self) -> None:
        problem = self._problem()
        receipt = execute_plan(plan_steady_diffusion(problem), problem)
        self.assertEqual(receipt.status, "succeeded")
        self.assertEqual(receipt.plan.maturity, "verified")
        self.assertEqual(receipt.result.nodal_values.shape, (9,))
        self.assertEqual(len(receipt.evidence_digest), 64)
        mutated = receipt.evidence
        mutated["claim_maturity"] = "validated"
        self.assertEqual(receipt.evidence["claim_maturity"], "verified")
        with self.assertRaisesRegex(ValueError, "digest"):
            replace(receipt, evidence_digest="0" * 64)
        explanation = explain_execution(receipt)
        self.assertTrue(
            explanation["interpretation"]["maturity_is_not_upgraded_by_execution"]
        )
        self.assertNotIn("result", receipt.as_dict())
        json.dumps(explanation, allow_nan=False, sort_keys=True)

    def test_stale_shape_plan_is_rejected_before_execution(self) -> None:
        plan = plan_steady_diffusion(self._problem(1))
        with self.assertRaises(NativeExecutionError) as captured:
            execute_plan(plan, self._problem(2))
        self.assertEqual(captured.exception.code, "execution.plan_mismatch")

    def test_same_size_but_different_mesh_is_rejected(self) -> None:
        problem = self._problem(1)
        changed_mesh = TriangularMesh(
            problem.mesh.points,
            problem.mesh.cells[::-1],
            node_sets=problem.mesh.node_sets,
            boundary_sets=problem.mesh.boundary_sets,
            cell_sets=problem.mesh.cell_sets,
        )
        changed = SteadyDiffusionProblem(
            changed_mesh,
            1.0,
            source=2.0,
            dirichlet=(DirichletCondition("boundary", 0.0),),
        )
        with self.assertRaises(NativeExecutionError):
            execute_plan(plan_steady_diffusion(problem), changed)

    def test_budget_and_progress_apply_to_static_execution(self) -> None:
        problem = self._problem()
        events = []
        context = ExecutionContext(
            budget=ResourceBudget(maximum_dofs=9), progress=events.append
        )
        execute_plan(plan_steady_diffusion(problem), problem, context=context)
        self.assertEqual([event.completed for event in events], [0, 1])
        execute_plan(
            plan_steady_diffusion(problem),
            problem,
            context=ExecutionContext(budget=ResourceBudget(maximum_steps=0)),
        )
        with self.assertRaises(NativeExecutionError) as captured:
            execute_plan(
                plan_steady_diffusion(problem),
                problem,
                context=ExecutionContext(budget=ResourceBudget(maximum_peak_bytes=0)),
            )
        self.assertEqual(captured.exception.code, "budget.memory_exceeded")

    def test_receipt_evidence_is_deterministic_for_same_runtime_and_result(
        self,
    ) -> None:
        problem = self._problem()
        plan = plan_steady_diffusion(problem)
        first = execute_plan(plan, problem)
        second = execute_plan(plan, problem)
        self.assertEqual(first.evidence_digest, second.evidence_digest)
        np.testing.assert_array_equal(
            first.result.nodal_values, second.result.nodal_values
        )

    def test_elasticity_and_dynamics_dispatches_return_typed_evidence(self) -> None:
        mesh_2d = unit_square_triangles(1)
        elastic_2d = LinearElasticProblem(
            mesh_2d,
            LinearElasticMaterial(10.0, 0.2),
            dirichlet=(DisplacementCondition("boundary", None, (0.0, 0.0)),),
        )
        receipt_2d = execute_plan(plan_linear_elasticity(elastic_2d), elastic_2d)
        self.assertEqual(
            receipt_2d.evidence["result"]["result_kind"],
            "linear_elasticity_2d",
        )
        self.assertEqual(
            receipt_2d.evidence["analysis"],
            {
                "problem_kind": "linear_elasticity_2d",
                "spatial_dimension": 2,
                "element_family": "triangle_p1_vector2",
                "dof_count": 8,
                "assembly_mode": "native",
                "thread_count": 1,
                "thread_workspace_bytes": 0,
                "provider": "native_sparse",
                "claim_maturity": "implemented",
            },
        )

        mesh_3d = unit_cube_tetrahedra(1)
        elastic_3d = LinearElastic3DProblem(
            mesh_3d,
            SolidElasticMaterial(10.0, 0.2),
            dirichlet=(
                SolidDisplacementCondition("left", "x", 0.0),
                SolidDisplacementCondition("front", "y", 0.0),
                SolidDisplacementCondition("bottom", "z", 0.0),
            ),
        )
        receipt_3d = execute_plan(plan_linear_elasticity_3d(elastic_3d), elastic_3d)
        self.assertEqual(
            receipt_3d.evidence["result"]["result_kind"],
            "linear_elasticity_3d",
        )
        self.assertEqual(receipt_3d.evidence["analysis"]["spatial_dimension"], 3)
        self.assertEqual(
            receipt_3d.evidence["analysis"]["element_family"],
            "tetrahedron_p1_vector3",
        )

        diagonal = CSRMatrix.from_coo((2, 2), [0, 1], [0, 1], [1.0, 1.0])
        dynamic = LinearSecondOrderSystem(
            diagonal, diagonal, [0.0, 0.0], 0.1, 2, [0.0, 1.0], [0.0, 0.0]
        )
        receipt_dynamic = execute_plan(
            plan_linear_dynamics(dynamic), dynamic, constrained_dofs=[0]
        )
        self.assertEqual(receipt_dynamic.evidence["result"]["constrained_dof_count"], 1)
        self.assertIsNone(receipt_dynamic.evidence["analysis"]["spatial_dimension"])


if __name__ == "__main__":
    unittest.main()
