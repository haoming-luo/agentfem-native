# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import json
import unittest

from agentfem_native import (
    DirichletCondition,
    DisplacementCondition,
    LinearElasticMaterial,
    LinearElasticProblem,
    SteadyDiffusionProblem,
    native_capabilities,
    plan_linear_elasticity,
    plan_steady_diffusion,
    unit_square_triangles,
)


class PlanningTests(unittest.TestCase):
    def test_diffusion_plan_is_deterministic_and_conservative(self) -> None:
        problem = SteadyDiffusionProblem(
            unit_square_triangles(4),
            1.0,
            dirichlet=(DirichletCondition("left", 0.0),),
        )
        first = plan_steady_diffusion(problem)
        second = plan_steady_diffusion(problem)
        self.assertEqual(first, second)
        self.assertEqual(first.provider, "native_sparse")
        self.assertEqual(first.dof_count, 25)
        self.assertEqual(first.coo_entry_count, 32 * 9)
        self.assertEqual(len(first.digest), 64)
        self.assertGreaterEqual(
            first.peak_bytes_upper_bound, first.csr_bytes_upper_bound
        )
        json.dumps(first.as_dict(), allow_nan=False, sort_keys=True)

    def test_elasticity_plan_reports_vector_dofs_and_maturity(self) -> None:
        mesh = unit_square_triangles(3)
        problem = LinearElasticProblem(
            mesh,
            LinearElasticMaterial(10.0, 0.2),
            dirichlet=(DisplacementCondition("boundary", None, (0.0, 0.0)),),
        )
        plan = plan_linear_elasticity(problem)
        self.assertEqual(plan.dof_count, 2 * mesh.node_count)
        self.assertEqual(plan.coo_entry_count, 36 * mesh.cell_count)
        self.assertEqual(plan.maturity, "implemented")
        self.assertTrue(plan.warnings)

    def test_dense_oracle_plan_is_visibly_warned(self) -> None:
        problem = SteadyDiffusionProblem(
            unit_square_triangles(1),
            1.0,
            dirichlet=(DirichletCondition("left", 0.0),),
        )
        self.assertIn(
            "Dense provider",
            plan_steady_diffusion(problem, provider="numpy").warnings[0],
        )

    def test_capability_manifest_is_json_safe_and_honest(self) -> None:
        capabilities = native_capabilities()
        scientific = {
            item["name"]: item["maturity"] for item in capabilities["scientific"]
        }
        self.assertEqual(scientific["steady_diffusion_p1_triangle"], "verified")
        self.assertEqual(
            scientific["linear_elasticity_t3_plane_stress_strain"], "implemented"
        )
        providers = {item["name"]: item for item in capabilities["linear_algebra"]}
        self.assertTrue(providers["native_sparse"]["available"])
        json.dumps(capabilities, allow_nan=False, sort_keys=True)


if __name__ == "__main__":
    unittest.main()
