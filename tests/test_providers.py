# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from agentfem_native import (
    DirichletCondition,
    DisplacementCondition,
    LinearElasticMaterial,
    LinearElasticProblem,
    NeumannCondition,
    ProviderUnavailableError,
    ScipySparseProvider,
    SteadyDiffusionProblem,
    prepare_linear_elasticity_assembly,
    solve_linear_elasticity,
    solve_steady_diffusion,
    unit_square_triangles,
)
from agentfem_native.providers import (
    LinearSolveOutcome,
    NativeSparseProvider,
    ProviderIdentity,
)


def _problem() -> SteadyDiffusionProblem:
    return SteadyDiffusionProblem(
        mesh=unit_square_triangles(4),
        conductivity=1.0,
        dirichlet=(DirichletCondition("left", 0.0),),
        neumann=(NeumannCondition("right", 1.0),),
    )


class ProviderTests(unittest.TestCase):
    def test_default_provider_is_owned_sparse_baseline(self) -> None:
        result = solve_steady_diffusion(_problem())
        self.assertEqual(result.provider_name, "native_sparse")
        self.assertEqual(result.provider_version, "0.3")

    def test_unknown_provider_fails_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown linear algebra provider"):
            solve_steady_diffusion(_problem(), provider="missing")

    def test_explicit_missing_scipy_provider_fails_with_install_hint(self) -> None:
        with (
            patch.object(ScipySparseProvider, "available", return_value=False),
            self.assertRaisesRegex(ProviderUnavailableError, r"\[scipy\]"),
        ):
            solve_steady_diffusion(_problem(), provider="scipy")

    @unittest.skipUnless(
        importlib.util.find_spec("scipy") is not None, "SciPy unavailable"
    )
    def test_scipy_sparse_matches_numpy_dense(self) -> None:
        dense = solve_steady_diffusion(_problem(), provider="numpy")
        sparse = solve_steady_diffusion(_problem(), provider="scipy")
        self.assertEqual(sparse.provider_name, "scipy_sparse")
        np.testing.assert_allclose(
            sparse.nodal_values, dense.nodal_values, atol=2.0e-15
        )
        self.assertLess(sparse.free_residual_norm, 2.0e-14)

    @unittest.skipUnless(
        importlib.util.find_spec("scipy") is not None, "SciPy unavailable"
    )
    def test_scipy_accepts_prepared_mechanics_csr(self) -> None:
        problem = LinearElasticProblem(
            unit_square_triangles(3),
            LinearElasticMaterial(50.0, 0.2),
            body_force=(0.2, -0.1),
            dirichlet=(DisplacementCondition("left", None, (0.0, 0.0)),),
        )
        plan = prepare_linear_elasticity_assembly(problem)
        prepared = solve_linear_elasticity(
            problem, provider="scipy", assembly_plan=plan
        )
        cold = solve_linear_elasticity(problem, provider="scipy")
        np.testing.assert_allclose(
            prepared.displacements, cold.displacements, rtol=2.0e-13, atol=2.0e-16
        )

    @unittest.skipUnless(
        importlib.util.find_spec("scipy") is not None, "SciPy unavailable"
    )
    def test_auto_provider_keeps_dependency_free_sparse_baseline(self) -> None:
        automatic = solve_steady_diffusion(_problem(), provider="auto")
        self.assertEqual(automatic.provider_name, "native_sparse")

    def test_native_sparse_matches_numpy_without_densification(self) -> None:
        dense = solve_steady_diffusion(_problem(), provider="numpy")
        with patch(
            "agentfem_native.assembly.COOMatrix.to_dense",
            side_effect=AssertionError("production path densified"),
        ):
            sparse = solve_steady_diffusion(_problem(), provider="native")
        np.testing.assert_allclose(sparse.nodal_values, dense.nodal_values, atol=2e-13)
        self.assertLess(sparse.free_residual_norm, 2e-12)

    def test_native_sparse_iteration_limit_fails_with_report(self) -> None:
        provider = NativeSparseProvider(maximum_iterations=0)
        with self.assertRaisesRegex(ValueError, "iteration_limit") as captured:
            solve_steady_diffusion(_problem(), provider=provider)
        self.assertIsNotNone(captured.exception.report)
        self.assertFalse(captured.exception.report.converged)

    def test_availability_probe_does_not_import_scipy(self) -> None:
        self.assertEqual(
            ScipySparseProvider.available(),
            importlib.util.find_spec("scipy") is not None,
        )

    def test_custom_provider_cannot_return_malformed_solution(self) -> None:
        class MalformedProvider:
            def solve_constrained(self, matrix, load, constrained, values):
                return LinearSolveOutcome(
                    solution=np.array((np.nan,)),
                    provider=ProviderIdentity("malformed", "0", "none"),
                )

        with self.assertRaisesRegex(ValueError, "invalid solution vector"):
            solve_steady_diffusion(_problem(), provider=MalformedProvider())


if __name__ == "__main__":
    unittest.main()
