# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import importlib.util
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from agentfem_native import (
    CSRMatrix,
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
    def test_prescribed_values_stay_exact_with_scalar_and_block_jacobi(self):
        dense = (
            np.array(
                [
                    [5, -1, -0.4, 0.2],
                    [-1, 4, 0.3, -0.1],
                    [-0.4, 0.3, 6, -1],
                    [0.2, -0.1, -1, 3],
                ],
                dtype=float,
            )
            * 1e6
        )
        rows, columns = np.nonzero(dense)
        matrix = CSRMatrix.from_coo(dense.shape, rows, columns, dense[rows, columns])
        fixed, free = np.array([1, 3]), np.array([0, 2])
        prescribed = np.array([1.25, -0.75])
        load = np.array([1, 2, 3, 4], dtype=float)
        exact_free = np.linalg.solve(
            dense[np.ix_(free, free)],
            load[free] - dense[np.ix_(free, fixed)] @ prescribed,
        )
        for block_size in (None, 2):
            answer = NativeSparseProvider(block_size=block_size).solve_constrained(
                matrix, load, fixed, prescribed
            )
            np.testing.assert_array_equal(answer.solution[fixed], prescribed)
            np.testing.assert_allclose(
                answer.solution[free], exact_free, rtol=1e-12, atol=1e-13
            )
            self.assertLessEqual(
                answer.convergence.residual_norm, answer.convergence.threshold
            )

    def test_fully_prescribed_solution_requires_no_krylov_iteration(self):
        matrix = CSRMatrix.from_coo((2, 2), [0, 0, 1, 1], [0, 1, 0, 1], [4, -1, -1, 3])
        result = NativeSparseProvider(maximum_iterations=0).solve_constrained(
            matrix, [100, -200], [1, 0], [1.25, -0.75]
        )
        np.testing.assert_array_equal(result.solution, [-0.75, 1.25])
        self.assertEqual(result.convergence.iterations, 0)
        self.assertEqual(result.convergence.residual_norm, 0)

    def test_homogeneous_constraints_keep_implicit_zero_initial_guess(self):
        from agentfem_native.sparse import conjugate_gradient

        matrix = CSRMatrix.from_coo((2, 2), [0, 1], [0, 1], [1, 2])
        with patch(
            "agentfem_native.providers.conjugate_gradient", wraps=conjugate_gradient
        ) as solve:
            NativeSparseProvider().solve_constrained(matrix, [0, 2], [0], [0])
        self.assertIsNone(solve.call_args.kwargs["initial_guess"])

    def test_default_provider_is_owned_sparse_baseline(self) -> None:
        result = solve_steady_diffusion(_problem())
        self.assertEqual(result.provider_name, "native_sparse")
        self.assertEqual(result.provider_version, "0.4")

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
        with self.assertRaisesRegex(ValueError, "块尺寸"):
            NativeSparseProvider(block_size=0)

    def test_native_sparse_plan_reuses_matrix_and_preconditioner(self) -> None:
        matrix = np.array(((4.0, -1.0), (-1.0, 3.0)))
        rows, columns = np.nonzero(matrix)
        sparse = CSRMatrix.from_coo(matrix.shape, rows, columns, matrix[rows, columns])
        provider = NativeSparseProvider()
        plan = provider.prepare_constrained(sparse, [0])
        first = plan.solve([0.0, 2.0], [0.0])
        second = plan.solve([0.0, 4.0], [0.0])
        np.testing.assert_allclose(first.solution, (0.0, 2.0 / 3.0))
        np.testing.assert_allclose(second.solution, 2.0 * first.solution)
        self.assertIs(plan.constrained_matrix.data, plan.preconditioner.matrix_data)
        self.assertEqual(first.provider.version, "0.4")
        with self.assertRaisesRegex(ValueError, "约束后矩阵"):
            replace(plan, constrained_matrix=sparse)

    def test_block_solve_plan_matches_cold_for_multiple_loads(self) -> None:
        dense = np.array(
            (
                (5.0, 1.0, -1.0, 0.0),
                (1.0, 4.0, 0.0, -0.5),
                (-1.0, 0.0, 5.0, 1.0),
                (0.0, -0.5, 1.0, 4.0),
            )
        )
        rows, columns = np.nonzero(dense)
        matrix = CSRMatrix.from_coo(dense.shape, rows, columns, dense[rows, columns])
        provider = NativeSparseProvider(block_size=2)
        constrained = np.array((0, 1), dtype=np.int64)
        values = np.array((0.1, -0.2))
        plan = provider.prepare_constrained(matrix, constrained)
        for scale in (0.5, 1.0, 1.7):
            load = scale * np.array((1.0, -2.0, 3.0, 4.0))
            cold = provider.solve_constrained(matrix, load, constrained, values)
            prepared = plan.solve(load, values)
            np.testing.assert_allclose(prepared.solution, cold.solution, atol=1.0e-14)
            self.assertTrue(prepared.convergence.converged)

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
