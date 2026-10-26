# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""隔离 IC(0) 实验：自有分解语义，可替换的三角应用；非默认提供者。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .providers import ProviderUnavailableError
from .sparse import (
    CSRMatrix,
    _matrix_infinity_norm,
    _readonly_indices,
    _readonly_values,
)


class IncompleteFactorizationError(ValueError):
    """SPD 输入也可能不完全分解失败；保留明确行号，不自动添加平移。"""

    def __init__(self, row: int, pivot: float):
        self.row = row
        self.pivot = pivot
        super().__init__(
            f"IC0 在第 {row} 行遇到非正或非有限枢轴 {pivot:.6e}；请检查尺度或显式选择其他预条件策略。"
        )


def _reference_factor(lower: CSRMatrix, shift: float):
    factors = lower.data.copy()
    pivots = np.empty(lower.shape[0])
    for row in range(lower.shape[0]):
        start, stop = lower.indptr[row : row + 2]
        columns = lower.indices[start:stop]
        for local, column in enumerate(columns):
            other_start, other_stop = lower.indptr[column : column + 2]
            common, here, there = np.intersect1d(
                columns[:local],
                lower.indices[other_start:other_stop],
                assume_unique=True,
                return_indices=True,
            )
            correction = np.dot(
                factors[start + here] * pivots[common], factors[other_start + there]
            )
            factors[start + local] = (factors[start + local] - correction) / pivots[
                column
            ]
        values = factors[start:stop]
        pivot = float(1.0 + shift - np.dot(values * values, pivots[columns]))
        if not np.isfinite(pivot) or pivot <= 0:
            raise IncompleteFactorizationError(row, pivot)
        pivots[row] = pivot
    return factors, pivots


@dataclass(frozen=True, slots=True, init=False)
class IncompleteCholeskyPattern:
    """仅拥有结构分析；不同材料数值须重新分解，不缓存任何旧枢轴。"""

    shape: tuple[int, int]
    indptr: np.ndarray
    indices: np.ndarray
    rows: np.ndarray
    transposed_order: np.ndarray
    diagonal_slots: np.ndarray
    lower_slots: np.ndarray
    lower_indptr: np.ndarray
    lower_indices: np.ndarray

    def __init__(self, matrix: CSRMatrix):
        if (
            not isinstance(matrix, CSRMatrix)
            or matrix.shape[0] != matrix.shape[1]
            or matrix.shape[0] == 0
        ):
            raise ValueError("IC0 要求非空方形规范 CSR 矩阵。")
        rows = matrix.row_indices()
        order = np.lexsort((rows, matrix.indices))
        if not np.array_equal(rows, matrix.indices[order]) or not np.array_equal(
            matrix.indices, rows[order]
        ):
            raise ValueError("IC0 要求结构对称，缺少镜像位置。")
        diagonal = np.flatnonzero(rows == matrix.indices)
        if diagonal.size != matrix.shape[0]:
            raise ValueError("IC0 要求每行拥有显式对角位置。")
        lower_slots = np.flatnonzero(matrix.indices < rows)
        lower_ptr = np.zeros(matrix.shape[0] + 1, dtype=np.int64)
        lower_ptr[1:] = np.cumsum(
            np.bincount(rows[lower_slots], minlength=matrix.shape[0])
        )
        object.__setattr__(self, "shape", matrix.shape)
        for name, value in {
            "indptr": matrix.indptr,
            "indices": matrix.indices,
            "rows": rows,
            "transposed_order": order,
            "diagonal_slots": diagonal,
            "lower_slots": lower_slots,
            "lower_indptr": lower_ptr,
            "lower_indices": matrix.indices[lower_slots],
        }.items():
            object.__setattr__(self, name, _readonly_indices(value, name=name))

    def validate_matrix(self, matrix: CSRMatrix) -> None:
        if (
            not isinstance(matrix, CSRMatrix)
            or matrix.shape != self.shape
            or not np.array_equal(matrix.indptr, self.indptr)
            or not np.array_equal(matrix.indices, self.indices)
        ):
            raise ValueError("IC0 的 CSR 图已改变，必须重新分析结构。")

    @property
    def storage_nbytes(self) -> int:
        """结构计划自身持有的数组，不包含各次数值分解。"""
        return sum(
            getattr(self, name).nbytes
            for name in (
                "indptr",
                "indices",
                "rows",
                "transposed_order",
                "diagonal_slots",
                "lower_slots",
                "lower_indptr",
                "lower_indices",
            )
        )

    def factor(
        self, matrix: CSRMatrix, *, shift: float = 0.0, backend: str = "reference"
    ) -> IncompleteCholesky:
        """成功后才返回新数值快照；任意失败不修改已有分解或结构。"""
        return IncompleteCholesky(matrix, shift=shift, backend=backend, pattern=self)


@dataclass(frozen=True, slots=True, init=False)
class IncompleteCholesky:
    """对角缩放的零填充 LDLT；只接受显式研究调用，不提供自动回退。"""

    matrix: CSRMatrix
    lower: CSRMatrix
    diagonal: np.ndarray
    scale: np.ndarray
    shift: float
    backend: str
    _norm: float
    _triangular_solve: object
    _scipy_lower: object
    _scipy_upper: object

    def __init__(
        self,
        matrix: CSRMatrix,
        *,
        shift: float = 0.0,
        backend: str = "reference",
        pattern: IncompleteCholeskyPattern | None = None,
    ):
        if (
            not isinstance(matrix, CSRMatrix)
            or matrix.shape[0] != matrix.shape[1]
            or matrix.shape[0] == 0
        ):
            raise ValueError("IC0 要求非空方形规范 CSR 矩阵。")
        if isinstance(shift, (bool, np.bool_)) or not np.isfinite(shift) or shift < 0:
            raise ValueError("IC0 显式平移必须是有限非负标量。")
        if backend not in ("reference", "scipy", "native"):
            raise ValueError("IC0 后端仅支持 reference、scipy 或 native 实验。")
        if pattern is not None and not isinstance(pattern, IncompleteCholeskyPattern):
            raise ValueError("IC0 结构计划必须为 IncompleteCholeskyPattern。")
        graph = pattern if pattern is not None else IncompleteCholeskyPattern(matrix)
        graph.validate_matrix(matrix)
        diagonal = matrix.data[graph.diagonal_slots]
        if np.any(diagonal <= 0):
            raise ValueError("IC0 缩放要求严格正的原矩阵对角。")
        norm = _matrix_infinity_norm(matrix)
        rows = graph.rows
        transposed_data = matrix.data[graph.transposed_order]
        magnitude = np.maximum(np.abs(matrix.data), np.abs(transposed_data))
        with np.errstate(over="ignore"):
            difference = np.abs(matrix.data - transposed_data)
        if np.any(difference > 64 * np.finfo(float).eps * magnitude):
            raise ValueError("IC0 要求数值对称，不接受明显非对称系统。")
        scale = 1.0 / np.sqrt(diagonal)
        # 对允许的装配舍入尾差只对称化预条件器；外层 CG 的原矩阵保持不变。
        scaled = (
            (0.5 * matrix.data + 0.5 * transposed_data)
            * scale[rows]
            * scale[matrix.indices]
        )
        lower_ptr = graph.lower_indptr
        lower_columns, lower_values = graph.lower_indices, scaled[graph.lower_slots]
        if backend == "native":
            from .native import NativeIC0Breakdown, ic0_factor

            try:
                factors, pivots = ic0_factor(
                    lower_ptr,
                    lower_columns,
                    lower_values,
                    np.full(matrix.shape[0], 1.0 + shift),
                )
            except NativeIC0Breakdown as error:
                raise IncompleteFactorizationError(error.row, error.pivot) from error
        else:
            lower = CSRMatrix(matrix.shape, lower_ptr, lower_columns, lower_values)
            factors, pivots = _reference_factor(lower, shift)
        lower = CSRMatrix(matrix.shape, lower_ptr, lower_columns, factors)
        solver = scipy_lower = scipy_upper = None
        if backend == "scipy":
            try:
                from scipy.sparse import csr_array
                from scipy.sparse.linalg import spsolve_triangular
            except ImportError as error:
                raise ProviderUnavailableError(
                    "IC0 的 scipy 三角后端不可用；可显式选择 reference。"
                ) from error
            scipy_lower = csr_array(
                (lower.data.copy(), lower.indices.copy(), lower.indptr.copy()),
                shape=lower.shape,
            )
            scipy_lower.setdiag(1.0)
            scipy_upper = scipy_lower.T.tocsr()
            solver = spsolve_triangular
        for name, value in {
            "matrix": matrix,
            "lower": lower,
            "diagonal": _readonly_values(pivots, name="IC0 枢轴"),
            "scale": _readonly_values(scale, name="IC0 对角缩放"),
            "shift": float(shift),
            "backend": backend,
            "_norm": norm,
            "_triangular_solve": solver,
            "_scipy_lower": scipy_lower,
            "_scipy_upper": scipy_upper,
        }.items():
            object.__setattr__(self, name, value)

    def validate_matrix(self, matrix: CSRMatrix) -> None:
        """只允许同一数值系统；图/数值变化不能继承旧分解。"""
        source = self.matrix
        if (
            matrix.shape != source.shape
            or not np.array_equal(matrix.indptr, source.indptr)
            or not np.array_equal(matrix.indices, source.indices)
        ):
            raise ValueError("IC0 的 CSR 图已改变，必须重新分解。")
        if matrix.data is not source.data and not np.array_equal(
            matrix.data, source.data
        ):
            raise ValueError("IC0 的 CSR 数值已改变，必须重新分解。")

    def infinity_norm(self) -> float:
        return self._norm

    @property
    def storage_nbytes(self) -> int:
        """自有/持有数组字节；可选三角后端的内部副本另计。"""
        return (
            self.matrix.storage_nbytes
            + self.lower.storage_nbytes
            + self.diagonal.nbytes
            + self.scale.nbytes
        )

    def apply(self, values: ArrayLike) -> np.ndarray:
        rhs = np.asarray(values, dtype=float)
        if rhs.shape != (self.matrix.shape[0],) or not np.all(np.isfinite(rhs)):
            raise ValueError("IC0 右端项必须为每行一个有限数。")
        result = self.scale * rhs
        if self.backend == "native":
            from .native import ldlt_solve

            result = ldlt_solve(
                self.lower.indptr,
                self.lower.indices,
                self.lower.data,
                self.diagonal,
                result,
            )
        elif self.backend == "scipy":
            result = self._triangular_solve(
                self._scipy_lower, result, lower=True, unit_diagonal=True
            )
            result = self._triangular_solve(
                self._scipy_upper,
                result / self.diagonal,
                lower=False,
                unit_diagonal=True,
            )
        else:
            lower = self.lower
            for row in range(result.size):
                start, stop = lower.indptr[row : row + 2]
                result[row] -= np.dot(
                    lower.data[start:stop], result[lower.indices[start:stop]]
                )
            result /= self.diagonal
            # 反向逐行散射等价于 L^T 回代；每行列索引唯一，不发生重复覆盖。
            for row in range(result.size - 1, -1, -1):
                start, stop = lower.indptr[row : row + 2]
                result[lower.indices[start:stop]] -= (
                    lower.data[start:stop] * result[row]
                )
        result *= self.scale
        if not np.all(np.isfinite(result)):
            raise ValueError("IC0 三角应用产生非有限结果，请检查缩放。")
        return result
