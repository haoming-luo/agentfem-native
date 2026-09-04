# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from agentfem_native import (
    DirichletCondition,
    NeumannCondition,
    ProviderUnavailableError,
    ScipySparseProvider,
    SteadyDiffusionProblem,
    solve_steady_diffusion,
    unit_square_triangles,
)
from agentfem_native.providers import LinearSolveOutcome, ProviderIdentity


def _problem() -> SteadyDiffusionProblem:
    return SteadyDiffusionProblem(
        mesh=unit_square_triangles(4),
        conductivity=1.0,
        dirichlet=(DirichletCondition("left", 0.0),),
        neumann=(NeumannCondition("right", 1.0),),
    )


class ProviderTests(unittest.TestCase):
    def test_default_provider_is_explicit_numpy_baseline(self) -> None:
        result = solve_steady_diffusion(_problem())
        self.assertEqual(result.provider_name, "numpy_dense")
        self.assertEqual(result.provider_version, np.__version__)

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
    def test_auto_provider_selects_installed_sparse_backend(self) -> None:
        automatic = solve_steady_diffusion(_problem(), provider="auto")
        self.assertEqual(automatic.provider_name, "scipy_sparse")

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
