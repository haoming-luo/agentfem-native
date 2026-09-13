# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""面向 Native 规范 COO/CSR 的可替换线性代数提供者。"""

from __future__ import annotations

import importlib.util
import warnings
from dataclasses import dataclass
from typing import Protocol, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .assembly import COOMatrix
from .sparse import (
    CGReport,
    CSRConstraintPlan,
    CSRMatrix,
    PreparedPreconditioner,
    conjugate_gradient,
)

FloatArray: TypeAlias = NDArray[np.float64]
IndexArray: TypeAlias = NDArray[np.int64]
SparseMatrix: TypeAlias = COOMatrix | CSRMatrix


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


@dataclass(frozen=True, slots=True)
class NativeSparseSolvePlan:
    """固定 CSR 数值与约束集合的可复用 Native CG 求解计划。"""

    matrix: CSRMatrix
    constraint_plan: CSRConstraintPlan
    constrained_matrix: CSRMatrix
    preconditioner: PreparedPreconditioner
    relative_tolerance: float
    absolute_tolerance: float
    maximum_iterations: int | None
    block_size: int | None

    def __post_init__(self) -> None:
        self.constraint_plan.validate_matrix(self.matrix)
        expected = self.constraint_plan.transform_matrix(self.matrix)
        if (
            self.constrained_matrix.shape != expected.shape
            or not np.array_equal(self.constrained_matrix.indptr, expected.indptr)
            or not np.array_equal(self.constrained_matrix.indices, expected.indices)
            or not np.array_equal(self.constrained_matrix.data, expected.data)
        ):
            raise ValueError("约束后矩阵与约束计划不一致。")
        self.preconditioner.validate_matrix(self.constrained_matrix)
        expected_kind = "block_jacobi" if self.block_size else "jacobi"
        if self.preconditioner.kind != expected_kind:
            raise ValueError("预条件器类型与求解计划的块尺寸不一致。")
        for value, name in (
            (self.relative_tolerance, "相对容差"),
            (self.absolute_tolerance, "绝对容差"),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name}必须为有限非负数。")
        if self.maximum_iterations is not None and (
            isinstance(self.maximum_iterations, (bool, np.bool_))
            or not isinstance(self.maximum_iterations, (int, np.integer))
            or int(self.maximum_iterations) < 0
        ):
            raise ValueError("最大迭代次数必须为非负整数或 None。")

    @property
    def storage_nbytes(self) -> int:
        """报告计划独占和持有数组的可审计字节量，不推断共享去重。"""

        return int(
            self.matrix.storage_nbytes
            + self.constraint_plan.storage_nbytes
            + self.constrained_matrix.storage_nbytes
            + self.preconditioner.storage_nbytes
        )

    @property
    def structure_digest(self) -> str:
        return self.constraint_plan.structure_digest

    def solve(self, load: ArrayLike, values: ArrayLike) -> LinearSolveOutcome:
        """复用约束后矩阵和数值预条件器，只变换当前右端项。"""

        constrained_load = self.constraint_plan.transform_rhs(
            self.matrix, load, values
        )
        result = conjugate_gradient(
            self.constrained_matrix,
            constrained_load,
            relative_tolerance=self.relative_tolerance,
            absolute_tolerance=self.absolute_tolerance,
            maximum_iterations=self.maximum_iterations,
            prepared_preconditioner=self.preconditioner,
        )
        if not result.report.converged:
            raise LinearSolveError(
                "Native 稀疏求解未收敛："
                f"{result.report.reason}，迭代 {result.report.iterations} 次，"
                f"残差 {result.report.residual_norm:.6e}。",
                result.report,
            )
        return LinearSolveOutcome(
            solution=result.solution,
            provider=ProviderIdentity(
                "native_sparse",
                "0.4",
                "bsr-block-jacobi" if self.block_size else "csr",
            ),
            convergence=result.report,
        )


class LinearAlgebraProvider(Protocol):
    def solve_constrained(
        self,
        matrix: SparseMatrix,
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
        matrix: SparseMatrix,
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
    """不依赖外部求解库的 CSR、CG 与 Jacobi 生产基线。"""

    relative_tolerance: float = 1.0e-12
    absolute_tolerance: float = 1.0e-14
    maximum_iterations: int | None = None
    block_size: int | None = None

    def __post_init__(self) -> None:
        if self.block_size is not None and (
            isinstance(self.block_size, (bool, np.bool_))
            or not isinstance(self.block_size, (int, np.integer))
            or int(self.block_size) < 1
        ):
            raise ValueError("块尺寸必须为正整数或 None。")

    def prepare_constrained(
        self,
        matrix: SparseMatrix,
        constrained: ArrayLike,
    ) -> NativeSparseSolvePlan:
        """把固定矩阵和约束准备成可由 AgentFEM 显式拥有的求解资源。"""

        sparse = (
            matrix
            if isinstance(matrix, CSRMatrix)
            else CSRMatrix.from_coo(
                matrix.shape, matrix.rows, matrix.columns, matrix.data
            )
        )
        constraint_plan = CSRConstraintPlan.from_matrix(sparse, constrained)
        constrained_matrix = constraint_plan.transform_matrix(sparse)
        kind = "block_jacobi" if self.block_size else "jacobi"
        preconditioner = PreparedPreconditioner.from_matrix(
            constrained_matrix,
            kind,
            block_size=self.block_size,
        )
        return NativeSparseSolvePlan(
            sparse,
            constraint_plan,
            constrained_matrix,
            preconditioner,
            self.relative_tolerance,
            self.absolute_tolerance,
            self.maximum_iterations,
            self.block_size,
        )

    def solve_constrained(
        self,
        matrix: SparseMatrix,
        load: FloatArray,
        constrained: IndexArray,
        values: FloatArray,
    ) -> LinearSolveOutcome:
        return self.prepare_constrained(matrix, constrained).solve(load, values)


@dataclass(frozen=True, slots=True)
class ScipySparseProvider:
    """Optional CSR direct-solve provider imported only when requested."""

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("scipy") is not None

    def solve_constrained(
        self,
        matrix: SparseMatrix,
        load: FloatArray,
        constrained: IndexArray,
        values: FloatArray,
    ) -> LinearSolveOutcome:
        if not self.available():
            raise ProviderUnavailableError(
                "SciPy provider is unavailable; install agentfem-native[scipy]."
            )
        import scipy
        from scipy.sparse import coo_array, csr_array
        from scipy.sparse.linalg import MatrixRankWarning, spsolve

        if isinstance(matrix, CSRMatrix):
            sparse = csr_array(
                (matrix.data, matrix.indices, matrix.indptr), shape=matrix.shape
            )
        else:
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
