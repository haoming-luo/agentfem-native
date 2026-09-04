# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import unittest

import agentfem_native


class PackageTests(unittest.TestCase):
    def test_public_version_and_gate_one_api(self) -> None:
        self.assertEqual(agentfem_native.__version__, "0.3.0a1")
        self.assertEqual(
            set(agentfem_native.__all__),
            {
                "AffineTriangleMap",
                "CellMaterial",
                "CONTRACT_NAME",
                "CONTRACT_VERSION",
                "DiffusionResult",
                "DirichletCondition",
                "KernelRequestError",
                "NeumannCondition",
                "NumpyDenseProvider",
                "ProviderIdentity",
                "ProviderUnavailableError",
                "REFERENCE_TRIANGLE_VERTICES",
                "SteadyDiffusionProblem",
                "ScipySparseProvider",
                "TriangleQuadrature",
                "TriangularMesh",
                "inside_reference_triangle",
                "integrate_reference",
                "kernel_request_schema",
                "kernel_result_schema",
                "lower_agentfem_ir",
                "lower_agentfem_model",
                "p1_basis",
                "p1_basis_gradients",
                "run_kernel_request",
                "solve_steady_diffusion",
                "triangle_rule",
                "unit_square_triangles",
                "unit_square_two_triangles",
                "write_legacy_vtk",
            },
        )


if __name__ == "__main__":
    unittest.main()
