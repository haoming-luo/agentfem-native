# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Owned deterministic sparse algebra for dependency-free Native execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IndexArray: TypeAlias = NDArray[np.int64]
CGReason = Literal[
    "initial_residual",
    "converged",
    "iteration_limit",
    "breakdown",
]


def _readonly_indices(value: ArrayLike, *, name: str) -> IndexArray:
    raw = np.asarray(value)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer array.")
    if (
        np.issubdtype(raw.dtype, np.unsignedinteger)
        and raw.size
        and int(raw.max()) > np.iinfo(np.int64).max
    ):
        raise ValueError(f"{name} cannot be represented by signed 64-bit indices.")
    result = np.array(raw, dtype=np.int64, copy=True)
    result.setflags(write=False)
    return result


def _readonly_values(value: ArrayLike, *, name: str) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional float array.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    result.setflags(write=False)
    return result


def _validate_shape(shape: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise ValueError("Sparse matrix shape must contain two dimensions.")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
        or int(value) > np.iinfo(np.int64).max
        for value in shape
    ):
        raise ValueError("Sparse matrix dimensions must be nonnegative integers.")
    return int(shape[0]), int(shape[1])


@dataclass(frozen=True, slots=True)
class CSRMatrix:
    """Immutable scalar CSR matrix with a canonical ordered graph."""

    shape: tuple[int, int]
    indptr: IndexArray
    indices: IndexArray
    data: FloatArray

    def __post_init__(self) -> None:
        shape = _validate_shape(self.shape)
        indptr = _readonly_indices(self.indptr, name="CSR indptr")
        indices = _readonly_indices(self.indices, name="CSR indices")
        data = _readonly_values(self.data, name="CSR data")
        if indptr.shape != (shape[0] + 1,):
            raise ValueError("CSR indptr length must equal the row count plus one.")
        if indices.shape != data.shape:
            raise ValueError("CSR indices and data must have equal length.")
        if indptr[0] != 0 or indptr[-1] != data.size:
            raise ValueError("CSR indptr endpoints do not match stored entries.")
        if np.any(np.diff(indptr) < 0):
            raise ValueError("CSR indptr must be nondecreasing.")
        if np.any(indices < 0) or np.any(indices >= shape[1]):
            raise ValueError("CSR column index is out of range.")
        for row in range(shape[0]):
            row_indices = indices[indptr[row] : indptr[row + 1]]
            if np.any(np.diff(row_indices) <= 0):
                raise ValueError(
                    "CSR column indices must be strictly increasing within each row."
                )
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "indptr", indptr)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "data", data)

    @classmethod
    def from_coo(
        cls,
        shape: tuple[int, int],
        rows: ArrayLike,
        columns: ArrayLike,
        data: ArrayLike,
    ) -> CSRMatrix:
        """Canonicalize finite COO entries by row, column, and input order."""

        checked_shape = _validate_shape(shape)
        row_array = _readonly_indices(rows, name="COO rows")
        column_array = _readonly_indices(columns, name="COO columns")
        data_array = _readonly_values(data, name="COO data")
        if row_array.shape != column_array.shape or row_array.shape != data_array.shape:
            raise ValueError("COO rows, columns, and data must have equal length.")
        if np.any(row_array < 0) or np.any(row_array >= checked_shape[0]):
            raise ValueError("COO row index is out of range.")
        if np.any(column_array < 0) or np.any(column_array >= checked_shape[1]):
            raise ValueError("COO column index is out of range.")
        if data_array.size == 0:
            return cls(
                checked_shape,
                np.zeros(checked_shape[0] + 1, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
            )

        input_order = np.arange(data_array.size, dtype=np.int64)
        order = np.lexsort((input_order, column_array, row_array))
        sorted_rows = row_array[order]
        sorted_columns = column_array[order]
        sorted_data = data_array[order]
        group_start = np.empty(sorted_data.size, dtype=bool)
        group_start[0] = True
        group_start[1:] = (sorted_rows[1:] != sorted_rows[:-1]) | (
            sorted_columns[1:] != sorted_columns[:-1]
        )
        starts = np.flatnonzero(group_start)
        canonical_rows = sorted_rows[starts]
        canonical_columns = sorted_columns[starts]
        canonical_data = np.add.reduceat(sorted_data, starts)
        indptr = np.zeros(checked_shape[0] + 1, dtype=np.int64)
        indptr[1:] = np.cumsum(
            np.bincount(canonical_rows, minlength=checked_shape[0]), dtype=np.int64
        )
        return cls(checked_shape, indptr, canonical_columns, canonical_data)

    @property
    def nnz(self) -> int:
        return int(self.data.size)

    @property
    def storage_nbytes(self) -> int:
        return int(self.indptr.nbytes + self.indices.nbytes + self.data.nbytes)

    def row_indices(self) -> IndexArray:
        result = np.repeat(
            np.arange(self.shape[0], dtype=np.int64), np.diff(self.indptr)
        )
        result.setflags(write=False)
        return result

    def matvec(self, vector: ArrayLike) -> FloatArray:
        values = np.asarray(vector, dtype=np.float64)
        if values.shape != (self.shape[1],):
            raise ValueError("Vector size does not match sparse matrix columns.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Sparse matrix-vector input must be finite.")
        result = np.zeros(self.shape[0], dtype=np.float64)
        if not self.nnz:
            return result
        products = self.data * values[self.indices]
        counts = np.diff(self.indptr)
        nonempty = counts > 0
        result[nonempty] = np.add.reduceat(products, self.indptr[:-1][nonempty])
        if not np.all(np.isfinite(result)):
            raise ValueError("Sparse matrix-vector result is non-finite.")
        return result

    def diagonal(self) -> FloatArray:
        length = min(self.shape)
        result = np.empty(length, dtype=np.float64)
        for row in range(length):
            start, stop = int(self.indptr[row]), int(self.indptr[row + 1])
            position = int(np.searchsorted(self.indices[start:stop], row)) + start
            if position >= stop or self.indices[position] != row:
                raise ValueError(f"CSR matrix has no explicit diagonal at row {row}.")
            result[row] = self.data[position]
        return result

    def residual(self, solution: ArrayLike, right_hand_side: ArrayLike) -> FloatArray:
        rhs = np.asarray(right_hand_side, dtype=np.float64)
        if rhs.shape != (self.shape[0],) or not np.all(np.isfinite(rhs)):
            raise ValueError("Right-hand side must be one finite scalar per row.")
        return rhs - self.matvec(solution)

    def to_dense(self) -> FloatArray:
        """Return a diagnostic dense copy; production providers do not call this."""

        result = np.zeros(self.shape, dtype=np.float64)
        rows = self.row_indices()
        result[rows, self.indices] = self.data
        return result

    def with_dirichlet(
        self,
        right_hand_side: ArrayLike,
        constrained: ArrayLike,
        values: ArrayLike,
    ) -> tuple[CSRMatrix, FloatArray]:
        """Return the symmetric strong-Dirichlet system without densification."""

        if self.shape[0] != self.shape[1]:
            raise ValueError("Dirichlet transformation requires a square matrix.")
        rhs = np.array(right_hand_side, dtype=np.float64, copy=True)
        if rhs.shape != (self.shape[0],) or not np.all(np.isfinite(rhs)):
            raise ValueError("Right-hand side must be one finite scalar per row.")
        constrained_array = _readonly_indices(constrained, name="Constrained indices")
        prescribed = _readonly_values(values, name="Constrained values")
        if constrained_array.shape != prescribed.shape:
            raise ValueError("Constrained indices and values must have equal length.")
        if np.any(constrained_array < 0) or np.any(constrained_array >= self.shape[0]):
            raise ValueError("Constrained index is out of range.")
        if np.unique(constrained_array).size != constrained_array.size:
            raise ValueError("Constrained indices must be unique.")
        order = np.argsort(constrained_array, kind="stable")
        constrained_array = constrained_array[order]
        prescribed = prescribed[order]

        rows = self.row_indices()
        missing = []
        for index in constrained_array:
            row = int(index)
            start, stop = int(self.indptr[row]), int(self.indptr[row + 1])
            position = int(np.searchsorted(self.indices[start:stop], row)) + start
            if position >= stop or self.indices[position] != row:
                missing.append(row)
        if missing:
            rows = np.concatenate((rows, np.asarray(missing, dtype=np.int64)))
            columns = np.concatenate(
                (self.indices, np.asarray(missing, dtype=np.int64))
            )
            data = np.concatenate((self.data, np.zeros(len(missing), dtype=np.float64)))
            working = CSRMatrix.from_coo(self.shape, rows, columns, data)
        else:
            working = self

        data = np.array(working.data, copy=True)
        is_constrained = np.zeros(self.shape[0], dtype=bool)
        prescribed_by_index = np.zeros(self.shape[0], dtype=np.float64)
        is_constrained[constrained_array] = True
        prescribed_by_index[constrained_array] = prescribed
        for row in range(self.shape[0]):
            start, stop = int(working.indptr[row]), int(working.indptr[row + 1])
            columns = working.indices[start:stop]
            if is_constrained[row]:
                data[start:stop] = 0.0
                position = int(np.searchsorted(columns, row)) + start
                data[position] = 1.0
                rhs[row] = prescribed_by_index[row]
            else:
                selected = is_constrained[columns]
                if np.any(selected):
                    rhs[row] -= float(
                        data[start:stop][selected]
                        @ prescribed_by_index[columns[selected]]
                    )
                    data[start:stop][selected] = 0.0
        return CSRMatrix(working.shape, working.indptr, working.indices, data), rhs


@dataclass(frozen=True, slots=True)
class BSRMatrix:
    """Immutable square-block CSR matrix for node-major vector fields."""

    block_shape: tuple[int, int]
    block_size: int
    indptr: IndexArray
    indices: IndexArray
    data: FloatArray

    def __post_init__(self) -> None:
        block_shape = _validate_shape(self.block_shape)
        if (
            isinstance(self.block_size, (bool, np.bool_))
            or not isinstance(self.block_size, (int, np.integer))
            or int(self.block_size) < 1
        ):
            raise ValueError("BSR block size must be a positive integer.")
        block_size = int(self.block_size)
        indptr = _readonly_indices(self.indptr, name="BSR indptr")
        indices = _readonly_indices(self.indices, name="BSR indices")
        values = np.array(self.data, dtype=np.float64, copy=True)
        if values.ndim != 3 or values.shape[1:] != (block_size, block_size):
            raise ValueError("BSR data must have shape (nnzb, block_size, block_size).")
        if not np.all(np.isfinite(values)):
            raise ValueError("BSR data must contain only finite values.")
        if indptr.shape != (block_shape[0] + 1,):
            raise ValueError("BSR indptr length must equal block rows plus one.")
        if indices.shape != (values.shape[0],):
            raise ValueError("BSR indices and data must have equal block counts.")
        if indptr[0] != 0 or indptr[-1] != values.shape[0]:
            raise ValueError("BSR indptr endpoints do not match stored blocks.")
        if np.any(np.diff(indptr) < 0):
            raise ValueError("BSR indptr must be nondecreasing.")
        if np.any(indices < 0) or np.any(indices >= block_shape[1]):
            raise ValueError("BSR block-column index is out of range.")
        for row in range(block_shape[0]):
            row_indices = indices[indptr[row] : indptr[row + 1]]
            if np.any(np.diff(row_indices) <= 0):
                raise ValueError(
                    "BSR block-column indices must be strictly increasing per row."
                )
        values.setflags(write=False)
        object.__setattr__(self, "block_shape", block_shape)
        object.__setattr__(self, "block_size", block_size)
        object.__setattr__(self, "indptr", indptr)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "data", values)

    @classmethod
    def from_csr(cls, matrix: CSRMatrix, block_size: int) -> BSRMatrix:
        """Group a canonical scalar CSR graph into deterministic dense blocks."""

        if (
            isinstance(block_size, (bool, np.bool_))
            or not isinstance(block_size, (int, np.integer))
            or int(block_size) < 1
        ):
            raise ValueError("BSR block size must be a positive integer.")
        size = int(block_size)
        if matrix.shape[0] % size or matrix.shape[1] % size:
            raise ValueError("Scalar CSR dimensions must be divisible by block size.")
        rows = matrix.row_indices()
        block_rows = rows // size
        block_columns = matrix.indices // size
        local_rows = rows % size
        local_columns = matrix.indices % size
        if matrix.nnz == 0:
            shape = (matrix.shape[0] // size, matrix.shape[1] // size)
            return cls(
                shape,
                size,
                np.zeros(shape[0] + 1, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty((0, size, size), dtype=np.float64),
            )
        order = np.lexsort((block_columns, block_rows))
        ordered_rows = block_rows[order]
        ordered_columns = block_columns[order]
        starts_mask = np.empty(order.size, dtype=bool)
        starts_mask[0] = True
        starts_mask[1:] = (ordered_rows[1:] != ordered_rows[:-1]) | (
            ordered_columns[1:] != ordered_columns[:-1]
        )
        starts = np.flatnonzero(starts_mask)
        block_count = starts.size
        group = np.cumsum(starts_mask, dtype=np.int64) - 1
        blocks = np.zeros((block_count, size, size), dtype=np.float64)
        np.add.at(
            blocks,
            (group, local_rows[order], local_columns[order]),
            matrix.data[order],
        )
        canonical_rows = ordered_rows[starts]
        canonical_columns = ordered_columns[starts]
        shape = (matrix.shape[0] // size, matrix.shape[1] // size)
        indptr = np.zeros(shape[0] + 1, dtype=np.int64)
        indptr[1:] = np.cumsum(
            np.bincount(canonical_rows, minlength=shape[0]), dtype=np.int64
        )
        return cls(shape, size, indptr, canonical_columns, blocks)

    @property
    def shape(self) -> tuple[int, int]:
        return (
            self.block_shape[0] * self.block_size,
            self.block_shape[1] * self.block_size,
        )

    @property
    def nnzb(self) -> int:
        return int(self.data.shape[0])

    @property
    def storage_nbytes(self) -> int:
        return int(self.indptr.nbytes + self.indices.nbytes + self.data.nbytes)

    def matvec(self, vector: ArrayLike) -> FloatArray:
        values = np.asarray(vector, dtype=np.float64)
        if values.shape != (self.shape[1],) or not np.all(np.isfinite(values)):
            raise ValueError("Vector must contain one finite scalar per BSR column.")
        blocked = values.reshape(self.block_shape[1], self.block_size)
        result = np.zeros((self.block_shape[0], self.block_size), dtype=np.float64)
        for row in range(self.block_shape[0]):
            start, stop = int(self.indptr[row]), int(self.indptr[row + 1])
            if start != stop:
                result[row] = np.einsum(
                    "bij,bj->i",
                    self.data[start:stop],
                    blocked[self.indices[start:stop]],
                    optimize=True,
                )
        if not np.all(np.isfinite(result)):
            raise ValueError("BSR matrix-vector result is non-finite.")
        return result.ravel()

    def diagonal_blocks(self) -> FloatArray:
        if self.block_shape[0] != self.block_shape[1]:
            raise ValueError("BSR diagonal blocks require a square matrix.")
        result = np.empty(
            (self.block_shape[0], self.block_size, self.block_size),
            dtype=np.float64,
        )
        for row in range(self.block_shape[0]):
            start, stop = int(self.indptr[row]), int(self.indptr[row + 1])
            position = int(np.searchsorted(self.indices[start:stop], row)) + start
            if position >= stop or self.indices[position] != row:
                raise ValueError(f"BSR matrix has no diagonal block at row {row}.")
            result[row] = self.data[position]
        return result

    def to_csr(self) -> CSRMatrix:
        block_rows = np.repeat(
            np.arange(self.block_shape[0], dtype=np.int64), np.diff(self.indptr)
        )
        local = np.arange(self.block_size, dtype=np.int64)
        rows = (
            block_rows[:, None, None] * self.block_size
            + local[None, :, None]
            + np.zeros((self.nnzb, 1, self.block_size), dtype=np.int64)
        )
        columns = (
            self.indices[:, None, None] * self.block_size
            + local[None, None, :]
            + np.zeros((self.nnzb, self.block_size, 1), dtype=np.int64)
        )
        return CSRMatrix.from_coo(
            self.shape, rows.ravel(), columns.ravel(), self.data.ravel()
        )

    def to_dense(self) -> FloatArray:
        return self.to_csr().to_dense()


@dataclass(frozen=True, slots=True)
class CGReport:
    converged: bool
    reason: CGReason
    iterations: int
    initial_residual_norm: float
    residual_norm: float
    threshold: float
    relative_tolerance: float
    absolute_tolerance: float


@dataclass(frozen=True, slots=True)
class CGResult:
    solution: FloatArray
    report: CGReport

    def __post_init__(self) -> None:
        solution = np.array(self.solution, dtype=np.float64, copy=True)
        solution.setflags(write=False)
        object.__setattr__(self, "solution", solution)


def _cg_result(
    solution: FloatArray,
    *,
    converged: bool,
    reason: CGReason,
    iterations: int,
    initial_norm: float,
    residual_norm: float,
    threshold: float,
    rtol: float,
    atol: float,
) -> CGResult:
    return CGResult(
        solution,
        CGReport(
            converged,
            reason,
            iterations,
            initial_norm,
            residual_norm,
            threshold,
            rtol,
            atol,
        ),
    )


def conjugate_gradient(
    matrix: CSRMatrix,
    right_hand_side: ArrayLike,
    *,
    initial_guess: ArrayLike | None = None,
    relative_tolerance: float = 1.0e-12,
    absolute_tolerance: float = 1.0e-14,
    maximum_iterations: int | None = None,
    preconditioner: Literal["none", "jacobi", "block_jacobi"] = "jacobi",
    block_size: int | None = None,
) -> CGResult:
    """Solve an SPD system and return explicit convergence or breakdown evidence."""

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Conjugate gradient requires a square matrix.")
    rhs = np.asarray(right_hand_side, dtype=np.float64)
    if rhs.shape != (matrix.shape[0],) or not np.all(np.isfinite(rhs)):
        raise ValueError("Right-hand side must be one finite scalar per row.")
    for value, name in (
        (relative_tolerance, "Relative tolerance"),
        (absolute_tolerance, "Absolute tolerance"),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if maximum_iterations is None:
        maximum_iterations = max(1, 10 * matrix.shape[0])
    if (
        isinstance(maximum_iterations, (bool, np.bool_))
        or not isinstance(maximum_iterations, (int, np.integer))
        or int(maximum_iterations) < 0
    ):
        raise ValueError("Maximum iterations must be a nonnegative integer.")
    maximum_iterations = int(maximum_iterations)
    if preconditioner not in {"none", "jacobi", "block_jacobi"}:
        raise ValueError(f"Unknown CG preconditioner {preconditioner!r}.")

    if initial_guess is None:
        solution = np.zeros(matrix.shape[1], dtype=np.float64)
    else:
        solution = np.array(initial_guess, dtype=np.float64, copy=True)
        if solution.shape != (matrix.shape[1],) or not np.all(np.isfinite(solution)):
            raise ValueError("Initial guess must be one finite scalar per column.")
    residual = rhs - matrix.matvec(solution)
    initial_norm = float(np.linalg.norm(residual))
    threshold = max(
        float(absolute_tolerance),
        float(relative_tolerance) * float(np.linalg.norm(rhs)),
    )
    if initial_norm <= threshold:
        return _cg_result(
            solution,
            converged=True,
            reason="initial_residual",
            iterations=0,
            initial_norm=initial_norm,
            residual_norm=initial_norm,
            threshold=threshold,
            rtol=float(relative_tolerance),
            atol=float(absolute_tolerance),
        )

    inverse_diagonal: FloatArray | None = None
    inverse_blocks: FloatArray | None = None
    if preconditioner == "jacobi":
        diagonal = matrix.diagonal()
        if not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
            raise ValueError("Jacobi preconditioning requires a positive diagonal.")
        inverse_diagonal = 1.0 / diagonal
    elif preconditioner == "block_jacobi":
        if block_size is None:
            raise ValueError("Block-Jacobi requires an explicit block size.")
        blocks = BSRMatrix.from_csr(matrix, block_size).diagonal_blocks()
        try:
            eigenvalues = np.linalg.eigvalsh(blocks)
            if np.any(eigenvalues <= 0.0):
                raise ValueError(
                    "Block-Jacobi requires symmetric positive-definite diagonal blocks."
                )
            inverse_blocks = np.linalg.inv(blocks)
        except np.linalg.LinAlgError as error:
            raise ValueError("Block-Jacobi diagonal block is singular.") from error

    def apply_preconditioner(values: FloatArray) -> FloatArray:
        if inverse_diagonal is not None:
            return inverse_diagonal * values
        if inverse_blocks is not None:
            blocked = values.reshape(inverse_blocks.shape[0], inverse_blocks.shape[1])
            return np.einsum("bij,bj->bi", inverse_blocks, blocked).ravel()
        return values.copy()

    z = apply_preconditioner(residual)
    rho = float(residual @ z)
    if not np.isfinite(rho) or rho <= 0.0:
        return _cg_result(
            solution,
            converged=False,
            reason="breakdown",
            iterations=0,
            initial_norm=initial_norm,
            residual_norm=initial_norm,
            threshold=threshold,
            rtol=float(relative_tolerance),
            atol=float(absolute_tolerance),
        )
    direction = z.copy()
    residual_norm = initial_norm
    for iteration in range(1, maximum_iterations + 1):
        image = matrix.matvec(direction)
        curvature = float(direction @ image)
        if not np.isfinite(curvature) or curvature <= 0.0:
            return _cg_result(
                solution,
                converged=False,
                reason="breakdown",
                iterations=iteration - 1,
                initial_norm=initial_norm,
                residual_norm=residual_norm,
                threshold=threshold,
                rtol=float(relative_tolerance),
                atol=float(absolute_tolerance),
            )
        alpha = rho / curvature
        solution += alpha * direction
        residual -= alpha * image
        residual_norm = float(np.linalg.norm(residual))
        if not np.isfinite(residual_norm) or not np.all(np.isfinite(solution)):
            return _cg_result(
                solution,
                converged=False,
                reason="breakdown",
                iterations=iteration,
                initial_norm=initial_norm,
                residual_norm=residual_norm,
                threshold=threshold,
                rtol=float(relative_tolerance),
                atol=float(absolute_tolerance),
            )
        if residual_norm <= threshold:
            return _cg_result(
                solution,
                converged=True,
                reason="converged",
                iterations=iteration,
                initial_norm=initial_norm,
                residual_norm=residual_norm,
                threshold=threshold,
                rtol=float(relative_tolerance),
                atol=float(absolute_tolerance),
            )
        z = apply_preconditioner(residual)
        next_rho = float(residual @ z)
        if not np.isfinite(next_rho) or next_rho <= 0.0:
            return _cg_result(
                solution,
                converged=False,
                reason="breakdown",
                iterations=iteration,
                initial_norm=initial_norm,
                residual_norm=residual_norm,
                threshold=threshold,
                rtol=float(relative_tolerance),
                atol=float(absolute_tolerance),
            )
        direction = z + (next_rho / rho) * direction
        rho = next_rho

    return _cg_result(
        solution,
        converged=False,
        reason="iteration_limit",
        iterations=maximum_iterations,
        initial_norm=initial_norm,
        residual_norm=residual_norm,
        threshold=threshold,
        rtol=float(relative_tolerance),
        atol=float(absolute_tolerance),
    )
