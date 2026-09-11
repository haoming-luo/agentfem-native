# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import unittest

import agentfem_native


class PackageTests(unittest.TestCase):
    def test_public_version_and_gate_one_api(self) -> None:
        self.assertEqual(agentfem_native.__version__, "0.7.0a1")
        self.assertGreaterEqual(
            set(agentfem_native.__all__),
            {
                "AffineTriangleMap",
                "CellMaterial",
                "CGReport",
                "CGResult",
                "CONTRACT_NAME",
                "CONTRACT_VERSION",
                "CSRMatrix",
                "DiffusionResult",
                "DirichletCondition",
                "DisplacementCondition",
                "ExecutionPlan",
                "ExecutionReceipt",
                "KernelRequestError",
                "LinearElasticMaterial",
                "LinearElasticProblem",
                "LinearElasticResult",
                "LinearSolveError",
                "NeumannCondition",
                "NativeSparseProvider",
                "NativeInterchangeBundle",
                "NumpyDenseProvider",
                "ProviderIdentity",
                "ProviderUnavailableError",
                "REFERENCE_TRIANGLE_VERTICES",
                "SteadyDiffusionProblem",
                "TractionCondition",
                "ScipySparseProvider",
                "TriangleQuadrature",
                "TriangularMesh",
                "VectorDofMap",
                "assemble_linear_elasticity",
                "conjugate_gradient",
                "execute_plan",
                "explain_execution",
                "inside_reference_triangle",
                "integrate_reference",
                "kernel_request_schema",
                "kernel_result_schema",
                "lower_agentfem_ir",
                "lower_agentfem_model",
                "native_kernel_available",
                "native_kernel_identity",
                "native_capabilities",
                "p1_basis",
                "p1_basis_gradients",
                "p1_body_force_load",
                "p1_boundary_traction_load",
                "p1_elastic_stiffness",
                "p1_strain_displacement",
                "plan_linear_elasticity",
                "plan_linear_elasticity_3d",
                "plan_steady_diffusion",
                "run_kernel_request",
                "solve_steady_diffusion",
                "solve_linear_elasticity",
                "solve_linear_elasticity_3d",
                "triangle_rule",
                "unit_square_triangles",
                "unit_square_two_triangles",
                "write_legacy_vtk",
                "write_native_bundle",
            },
        )


if __name__ == "__main__":
    unittest.main()
