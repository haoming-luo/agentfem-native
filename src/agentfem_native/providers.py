# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Replaceable linear-algebra providers for the Native reference kernel."""

from __future__ import annotations

import importlib.util
import warnings
from dataclasses import dataclass
from typing import Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .assembly import COOMatrix
from .sparse import CGReport, CSRMatrix, conjugate_gradient

FloatArray: TypeAlias = NDArray[np.float64]
IndexArray: TypeAlias = NDArray[np.int64]


class ProviderUnavailableError(RuntimeError):
    """Raised when an explicitly requested optional provider is unavailable."""


class LinearSolveError(ValueError):
    """Raised when a provider reaches an explicit nonconverged solve outcome."""

    def __init__(self, message: str, report: CGReport | None = None) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True, slots=True)
class ProviderIdentity:
    name: str
    version: str
    matrix_format: str


@dataclass(frozen=True, slots=True)
class LinearSolveOutcome:
    solution: FloatArray
    provider: ProviderIdentity
    convergence: CGReport | None = None


class LinearAlgebraProvider(Protocol):
    def solve_constrained(
        self,
        matrix: COOMatrix,
        load: FloatArray,
        constrained: IndexArray,
        values: FloatArray,
    ) -> LinearSolveOutcome: ...


def _free_indices(size: int, constrained: IndexArray) -> IndexArray:
    if constrained.ndim != 1 or values_out_of_range(constrained, size):
        raise ValueError("Constrained node indices are invalid for the linear system.")
    return np.setdiff1d(np.arange(size, dtype=np.int64), constrained)


def values_out_of_range(indices: IndexArray, size: int) -> bool:
    return bool(np.any(indices < 0) or np.any(indices >= size))


@dataclass(frozen=True, slots=True)
class NumpyDenseProvider:
    """Auditable small-problem provider backed by ``numpy.linalg.solve``."""

    def solve_constrained(
        self,
        matrix: COOMatrix,
        load: FloatArray,
        constrained: IndexArray,
        values: FloatArray,
    ) -> LinearSolveOutcome:
        dense = matrix.to_dense()
        solution = np.zeros(matrix.shape[0], dtype=np.float64)
        solution[constrained] = values
        free = _free_indices(matrix.shape[0], constrained)
        if free.size:
            reduced = dense[np.ix_(free, free)]
            right_hand_side = load[free] - dense[np.ix_(free, constrained)] @ values
            try:
                solution[free] = np.linalg.solve(reduced, right_hand_side)
            except np.linalg.LinAlgError as error:
                raise LinearSolveError(
                    "Constrained linear system is singular."
                ) from error
        return LinearSolveOutcome(
            solution=solution,
            provider=ProviderIdentity("numpy_dense", np.__version__, "dense"),
        )


@dataclass(frozen=True, slots=True)
class NativeSparseProvider:
    """Dependency-free CSR, CG, and Jacobi production baseline."""

    relative_tolerance: float = 1.0e-12
    absolute_tolerance: float = 1.0e-14
    maximum_iterations: int | None = None
    block_size: int | None = None

    def solve_constrained(
        self,
        matrix: COOMatrix,
        load: FloatArray,
        constrained: IndexArray,
        values: FloatArray,
    ) -> LinearSolveOutcome:
        sparse = CSRMatrix.from_coo(
            matrix.shape, matrix.rows, matrix.columns, matrix.data
        )
        constrained_matrix, constrained_load = sparse.with_dirichlet(
            load, constrained, values
        )
        result = conjugate_gradient(
            constrained_matrix,
            constrained_load,
            relative_tolerance=self.relative_tolerance,
            absolute_tolerance=self.absolute_tolerance,
            maximum_iterations=self.maximum_iterations,
            preconditioner="block_jacobi" if self.block_size else "jacobi",
            block_size=self.block_size,
        )
        if not result.report.converged:
            raise LinearSolveError(
                "Native sparse solve did not converge: "
                f"{result.report.reason} after {result.report.iterations} iterations "
                f"with residual {result.report.residual_norm:.6e}.",
                result.report,
            )
        return LinearSolveOutcome(
            solution=result.solution,
            provider=ProviderIdentity(
                "native_sparse",
                "0.2",
                "bsr-block-jacobi" if self.block_size else "csr",
            ),
            convergence=result.report,
        )


@dataclass(frozen=True, slots=True)
class ScipySparseProvider:
    """Optional CSR direct-solve provider imported only when requested."""

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("scipy") is not None

    def solve_constrained(
        self,
        matrix: COOMatrix,
        load: FloatArray,
        constrained: IndexArray,
        values: FloatArray,
    ) -> LinearSolveOutcome:
        if not self.available():
            raise ProviderUnavailableError(
                "SciPy provider is unavailable; install agentfem-native[scipy]."
            )
        import scipy
        from scipy.sparse import coo_array
        from scipy.sparse.linalg import MatrixRankWarning, spsolve

        sparse = coo_array(
            (matrix.data, (matrix.rows, matrix.columns)),
            shape=matrix.shape,
        ).tocsr()
        solution = np.zeros(matrix.shape[0], dtype=np.float64)
        solution[constrained] = values
        free = _free_indices(matrix.shape[0], constrained)
        if free.size:
            reduced = sparse[free][:, free]
            right_hand_side = load[free] - sparse[free][:, constrained] @ values
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", MatrixRankWarning)
                    solved = spsolve(reduced, right_hand_side)
            except MatrixRankWarning as error:
                raise LinearSolveError(
                    "Constrained linear system is singular."
                ) from error
            solution[free] = np.asarray(solved, dtype=np.float64)
            if not np.all(np.isfinite(solution[free])):
                raise ValueError("Sparse provider returned a non-finite solution.")
        return LinearSolveOutcome(
            solution=solution,
            provider=ProviderIdentity("scipy_sparse", scipy.__version__, "csr"),
        )


def resolve_provider(
    provider: str | LinearAlgebraProvider | None,
) -> LinearAlgebraProvider:
    """Resolve a provider without making optional libraries import requirements."""

    if provider is None:
        return NativeSparseProvider()
    if isinstance(provider, str):
        if provider == "numpy":
            return NumpyDenseProvider()
        if provider in {"native", "native_sparse"}:
            return NativeSparseProvider()
        if provider == "scipy":
            return ScipySparseProvider()
        if provider == "auto":
            return NativeSparseProvider()
        raise ValueError(f"Unknown linear algebra provider {provider!r}.")
    solve = getattr(provider, "solve_constrained", None)
    if not callable(solve):
        raise TypeError("Linear algebra provider must implement solve_constrained().")
    return provider
