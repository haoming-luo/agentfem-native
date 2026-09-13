# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import json
import unittest

from agentfem_native import (
    DirichletCondition,
    DisplacementCondition,
    LinearElastic3DProblem,
    LinearElasticMaterial,
    LinearElasticProblem,
    LinearSecondOrderSystem,
    SolidElasticMaterial,
    SteadyDiffusionProblem,
    native_capabilities,
    plan_linear_dynamics,
    plan_linear_elasticity,
    plan_linear_elasticity_3d,
    plan_steady_diffusion,
    unit_cube_tetrahedra,
    unit_square_triangles,
)
from agentfem_native.sparse import CSRMatrix


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
        self.assertEqual(plan.thread_count, 1)
        self.assertEqual(plan.thread_workspace_bytes, 0)

        parallel = plan_linear_elasticity(
            LinearElasticProblem(
                mesh,
                LinearElasticMaterial(10.0, 0.2),
                assembly_mode="native",
                thread_count=4,
            )
        )
        self.assertEqual(parallel.thread_count, 4)
        self.assertEqual(parallel.thread_workspace_bytes, 4 * parallel.dof_count * 8)
        self.assertEqual(
            parallel.peak_bytes_upper_bound - plan.peak_bytes_upper_bound,
            parallel.thread_workspace_bytes,
        )
        self.assertNotEqual(parallel.digest, plan.digest)
        with self.assertRaisesRegex(ValueError, "大于零"):
            LinearElasticProblem(mesh, LinearElasticMaterial(10.0, 0.2), thread_count=0)
        with self.assertRaisesRegex(TypeError, "正整数"):
            LinearElastic3DProblem(
                unit_cube_tetrahedra(),
                SolidElasticMaterial(10.0, 0.2),
                thread_count=True,
            )
        with self.assertRaisesRegex(ValueError, "native"):
            plan_linear_elasticity(
                LinearElasticProblem(
                    mesh,
                    LinearElasticMaterial(10.0, 0.2),
                    assembly_mode="reference",
                    thread_count=2,
                )
            )

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
        self.assertEqual(scientific["linear_elasticity_t4_3d"], "implemented")
        self.assertEqual(
            scientific["linear_dynamics_central_difference_newmark"], "implemented"
        )
        providers = {item["name"]: item for item in capabilities["linear_algebra"]}
        self.assertTrue(providers["native_sparse"]["available"])
        self.assertIn("block_jacobi", providers["native_sparse"]["preconditioners"])
        self.assertEqual(
            providers["native_sparse"]["graph_lifecycle"],
            ["build_pattern", "fill_values"],
        )
        execution = {item["name"]: item for item in capabilities["execution"]}
        parallel = execution["deterministic_cpu_parallel_assembly"]
        self.assertTrue(parallel["available"])
        self.assertEqual(parallel["elements"], ["T3", "T4"])
        self.assertEqual(parallel["default_thread_count"], 1)
        json.dumps(capabilities, allow_nan=False, sort_keys=True)

    def test_t4_and_dynamics_plans_are_deterministic(self) -> None:
        solid = plan_linear_elasticity_3d(
            LinearElastic3DProblem(
                unit_cube_tetrahedra(), SolidElasticMaterial(10.0, 0.2)
            )
        )
        self.assertEqual(solid.dof_count, 24)
        self.assertEqual(solid.coo_entry_count, 6 * 144)
        self.assertEqual(solid.assembly_mode, "native")
        self.assertEqual(solid.thread_count, 1)
        diagonal = CSRMatrix.from_coo((1, 1), [0], [0], [1.0])
        dynamics = plan_linear_dynamics(
            LinearSecondOrderSystem(diagonal, diagonal, [0.0], 0.1, 10, [0.0], [0.0])
        )
        self.assertEqual(dynamics.dof_count, 1)
        self.assertGreater(
            dynamics.peak_bytes_upper_bound, dynamics.csr_bytes_upper_bound
        )
        self.assertEqual(len(dynamics.digest), 64)


if __name__ == "__main__":
    unittest.main()
