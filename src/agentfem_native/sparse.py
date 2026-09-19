# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Native 无外部求解依赖的自主确定性稀疏代数。"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
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
    def _from_pattern(cls, pattern: CSRPattern, data: FloatArray) -> CSRMatrix:
        """从已验证图构造矩阵；这是避免重复复制图数组的内部可信入口。"""

        data.setflags(write=False)
        result = object.__new__(cls)
        object.__setattr__(result, "shape", pattern.shape)
        object.__setattr__(result, "indptr", pattern.indptr)
        object.__setattr__(result, "indices", pattern.indices)
        object.__setattr__(result, "data", data)
        return result

    @classmethod
    def from_coo(
        cls,
        shape: tuple[int, int],
        rows: ArrayLike,
        columns: ArrayLike,
        data: ArrayLike,
    ) -> CSRMatrix:
        """按行、列和原输入顺序规范化有限 COO 贡献。"""

        return CSRPattern.from_coo(shape, rows, columns).fill(data)

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
        if self.nnz >= 2048:
            from .native import csr_spmv, native_kernel_available

            if native_kernel_available():
                return csr_spmv(
                    self.shape, self.indptr, self.indices, self.data, values
                )
        return self.matvec_reference(values)

    def matvec_reference(self, vector: ArrayLike) -> FloatArray:
        """Apply CSR with readable NumPy reductions for differential evidence."""

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
class CSRPattern:
    """可复用的不可变规范 CSR 图，以及原始贡献到数值槽位的映射。"""

    shape: tuple[int, int]
    indptr: IndexArray
    indices: IndexArray
    coo_to_csr: IndexArray

    def __post_init__(self) -> None:
        index_values = np.asarray(self.indices)
        graph = CSRMatrix(
            self.shape,
            self.indptr,
            self.indices,
            np.zeros(index_values.size, dtype=np.float64),
        )
        mapping = _readonly_indices(self.coo_to_csr, name="COO-to-CSR mapping")
        if mapping.size and (np.any(mapping < 0) or np.any(mapping >= graph.nnz)):
            raise ValueError("COO 到 CSR 的映射槽位超出规范图范围。")
        if np.unique(mapping).size != graph.nnz:
            raise ValueError("每个规范 CSR 槽位必须至少对应一个 COO 贡献。")
        object.__setattr__(self, "shape", graph.shape)
        object.__setattr__(self, "indptr", graph.indptr)
        object.__setattr__(self, "indices", graph.indices)
        object.__setattr__(self, "coo_to_csr", mapping)

    @classmethod
    def from_coo(
        cls,
        shape: tuple[int, int],
        rows: ArrayLike,
        columns: ArrayLike,
    ) -> CSRPattern:
        """只规范化 COO 结构，并保存原输入贡献顺序。"""

        checked_shape = _validate_shape(shape)
        row_array = _readonly_indices(rows, name="COO rows")
        column_array = _readonly_indices(columns, name="COO columns")
        if row_array.shape != column_array.shape:
            raise ValueError("COO 行和列必须具有相同长度。")
        if np.any(row_array < 0) or np.any(row_array >= checked_shape[0]):
            raise ValueError("COO 行索引超出矩阵范围。")
        if np.any(column_array < 0) or np.any(column_array >= checked_shape[1]):
            raise ValueError("COO 列索引超出矩阵范围。")
        if row_array.size == 0:
            return cls(
                checked_shape,
                np.zeros(checked_shape[0] + 1, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
            )

        input_order = np.arange(row_array.size, dtype=np.int64)
        order = np.lexsort((input_order, column_array, row_array))
        sorted_rows = row_array[order]
        sorted_columns = column_array[order]
        group_start = np.empty(order.size, dtype=bool)
        group_start[0] = True
        group_start[1:] = (sorted_rows[1:] != sorted_rows[:-1]) | (
            sorted_columns[1:] != sorted_columns[:-1]
        )
        starts = np.flatnonzero(group_start)
        canonical_rows = sorted_rows[starts]
        canonical_columns = sorted_columns[starts]
        sorted_groups = np.cumsum(group_start, dtype=np.int64) - 1
        mapping = np.empty(order.size, dtype=np.int64)
        mapping[order] = sorted_groups
        indptr = np.zeros(checked_shape[0] + 1, dtype=np.int64)
        indptr[1:] = np.cumsum(
            np.bincount(canonical_rows, minlength=checked_shape[0]), dtype=np.int64
        )
        return cls(checked_shape, indptr, canonical_columns, mapping)

    @classmethod
    def from_element_dofs(cls, dof_count: int, cell_dofs: ArrayLike) -> CSRPattern:
        """从节点主序的二维单元 DOF 表生成一次方阵装配图。"""

        if (
            isinstance(dof_count, (bool, np.bool_))
            or not isinstance(dof_count, (int, np.integer))
            or int(dof_count) < 0
            or int(dof_count) > np.iinfo(np.int64).max
        ):
            raise ValueError("DOF 总数必须是非负整数。")
        raw = np.asarray(cell_dofs)
        if raw.ndim != 2 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("单元 DOF 必须是二维整数数组。")
        if (
            np.issubdtype(raw.dtype, np.unsignedinteger)
            and raw.size
            and int(raw.max()) > np.iinfo(np.int64).max
        ):
            raise ValueError("单元 DOF 不能由有符号 64 位索引表示。")
        local_size = raw.shape[1]
        if local_size == 0:
            raise ValueError("每个单元必须至少包含一个 DOF。")
        values = np.asarray(raw, dtype=np.int64)
        if np.any(values < 0) or np.any(values >= int(dof_count)):
            raise ValueError("单元 DOF 超出全局范围。")
        rows = np.repeat(values, local_size, axis=1).ravel()
        columns = np.tile(values, (1, local_size)).ravel()
        return cls.from_coo((int(dof_count), int(dof_count)), rows, columns)

    @property
    def nnz(self) -> int:
        return int(self.indices.size)

    @property
    def coo_entry_count(self) -> int:
        return int(self.coo_to_csr.size)

    @property
    def storage_nbytes(self) -> int:
        return int(self.indptr.nbytes + self.indices.nbytes + self.coo_to_csr.nbytes)

    @property
    def structure_digest(self) -> str:
        """返回与平台本机字节序无关的规范结构摘要。"""

        digest = sha256()
        digest.update(np.asarray(self.shape, dtype="<i8").tobytes())
        for array in (self.indptr, self.indices, self.coo_to_csr):
            digest.update(np.asarray(array, dtype="<i8").tobytes())
        return digest.hexdigest()

    def fill(self, values: ArrayLike) -> CSRMatrix:
        """优先通过生产内核按原始贡献顺序回填，不重新排序图。"""

        contributions = self._checked_contributions(values)
        if self.nnz == 0:
            return CSRMatrix._from_pattern(self, np.empty(0, dtype=np.float64))
        from .native import csr_fill_from_contributions, native_kernel_available

        if native_kernel_available():
            data = csr_fill_from_contributions(self.nnz, self.coo_to_csr, contributions)
            return CSRMatrix._from_pattern(self, data)
        return self._fill_reference_checked(contributions)

    def fill_reference(self, values: ArrayLike) -> CSRMatrix:
        """使用独立 NumPy 数学路径回填，供验证与无编译内核环境使用。"""

        return self._fill_reference_checked(self._checked_contributions(values))

    def _checked_contributions(self, values: ArrayLike) -> FloatArray:
        """统一验证调用边界，同时保留传入贡献的顺序。"""

        contributions = _readonly_values(values, name="COO values")
        if contributions.shape != self.coo_to_csr.shape:
            raise ValueError("COO 数值长度必须与可复用图的贡献映射一致。")
        return contributions

    def _fill_reference_checked(self, contributions: FloatArray) -> CSRMatrix:
        """执行可读参考归并；调用方已经完成形状与有限性验证。"""

        data = np.zeros(self.nnz, dtype=np.float64)
        np.add.at(data, self.coo_to_csr, contributions)
        if not np.all(np.isfinite(data)):
            raise ValueError("COO 归并后的 CSR 数值不是有限数。")
        return CSRMatrix._from_pattern(self, data)


@dataclass(frozen=True, slots=True)
class CSRAssemblyPlan:
    """绑定单元 DOF 顺序与规范稀疏图的不可变重复装配计划。"""

    dof_count: int
    cell_dofs: IndexArray
    pattern: CSRPattern = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.dof_count, (bool, np.bool_))
            or not isinstance(self.dof_count, (int, np.integer))
            or int(self.dof_count) < 0
            or int(self.dof_count) > np.iinfo(np.int64).max
        ):
            raise ValueError("DOF 总数必须是有符号 64 位可表示的非负整数。")
        raw = np.asarray(self.cell_dofs)
        if raw.ndim != 2 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("预备装配的单元 DOF 必须是二维整数数组。")
        if raw.shape[1] == 0:
            raise ValueError("预备装配中每个单元必须至少包含一个 DOF。")
        if (
            np.issubdtype(raw.dtype, np.unsignedinteger)
            and raw.size
            and int(raw.max()) > np.iinfo(np.int64).max
        ):
            raise ValueError("单元 DOF 不能由有符号 64 位索引表示。")
        values = np.array(raw, dtype=np.int64, copy=True)
        count = int(self.dof_count)
        if np.any(values < 0) or np.any(values >= count):
            raise ValueError("预备装配的单元 DOF 超出全局范围。")
        values.setflags(write=False)
        object.__setattr__(self, "dof_count", count)
        object.__setattr__(self, "cell_dofs", values)
        object.__setattr__(self, "pattern", CSRPattern.from_element_dofs(count, values))

    @property
    def cell_count(self) -> int:
        return int(self.cell_dofs.shape[0])

    @property
    def local_size(self) -> int:
        return int(self.cell_dofs.shape[1])

    @property
    def contribution_count(self) -> int:
        return self.pattern.coo_entry_count

    @property
    def nnz(self) -> int:
        return self.pattern.nnz

    @property
    def storage_nbytes(self) -> int:
        return int(self.cell_dofs.nbytes + self.pattern.storage_nbytes)

    @property
    def avoided_coo_index_nbytes(self) -> int:
        """返回串行 native 仅数值路径明确省去的 COO 行列索引字节数。"""

        return 2 * self.contribution_count * np.dtype(np.int64).itemsize

    @property
    def structure_digest(self) -> str:
        return self.pattern.structure_digest

    def validate_layout(self, dof_count: int, cell_dofs: ArrayLike) -> None:
        """确认当前工程问题仍使用计划创建时的全局与局部 DOF 顺序。"""

        raw = np.asarray(cell_dofs)
        if (
            isinstance(dof_count, (bool, np.bool_))
            or not isinstance(dof_count, (int, np.integer))
            or int(dof_count) != self.dof_count
            or raw.shape != self.cell_dofs.shape
        ):
            raise ValueError("当前问题的 DOF 布局与预备 CSR 装配计划不一致。")
        if not np.issubdtype(raw.dtype, np.integer) or not np.array_equal(
            np.asarray(raw, dtype=np.int64), self.cell_dofs
        ):
            raise ValueError("当前问题的单元 DOF 顺序与预备 CSR 装配计划不一致。")

    def fill(self, values: ArrayLike) -> CSRMatrix:
        """按固定单元—局部行—局部列顺序回填一次元素矩阵贡献。"""

        raw = np.asarray(values, dtype=np.float64)
        matrix_shape = (self.cell_count, self.local_size, self.local_size)
        if raw.shape == matrix_shape:
            contributions = raw.reshape(-1)
        elif raw.shape == (self.contribution_count,):
            contributions = raw
        else:
            raise ValueError("元素贡献必须是计划对应的三维局部矩阵或等长一维数组。")
        return self.pattern.fill(contributions)


@dataclass(frozen=True, slots=True)
class CSRConstraintPlan:
    """固定规范 CSR 图和约束集合的强制 Dirichlet 变换计划。"""

    shape: tuple[int, int]
    source_indptr: IndexArray
    source_indices: IndexArray
    constrained: IndexArray
    pattern: CSRPattern = field(init=False, repr=False)
    source_to_target: IndexArray = field(init=False, repr=False)
    retained_source: IndexArray = field(init=False, repr=False)
    lift_source: IndexArray = field(init=False, repr=False)
    lift_rows: IndexArray = field(init=False, repr=False)
    lift_value_positions: IndexArray = field(init=False, repr=False)
    diagonal_slots: IndexArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        shape = _validate_shape(self.shape)
        if shape[0] != shape[1]:
            raise ValueError("约束变换计划要求方阵。")
        graph = CSRMatrix(
            shape,
            self.source_indptr,
            self.source_indices,
            np.zeros(np.asarray(self.source_indices).size, dtype=np.float64),
        )
        constrained = _readonly_indices(self.constrained, name="约束自由度")
        if np.any(constrained < 0) or np.any(constrained >= shape[0]):
            raise ValueError("约束自由度超出矩阵范围。")
        if np.unique(constrained).size != constrained.size:
            raise ValueError("约束自由度不得重复。")

        rows = graph.row_indices()
        missing: list[int] = []
        for raw_index in constrained:
            index = int(raw_index)
            start, stop = int(graph.indptr[index]), int(graph.indptr[index + 1])
            position = int(np.searchsorted(graph.indices[start:stop], index)) + start
            if position >= stop or graph.indices[position] != index:
                missing.append(index)
        augmented_rows = np.concatenate((rows, np.asarray(missing, dtype=np.int64)))
        augmented_columns = np.concatenate(
            (graph.indices, np.asarray(missing, dtype=np.int64))
        )
        augmented = CSRPattern.from_coo(shape, augmented_rows, augmented_columns)
        # 目标图只承担后续数值容器职责；源图到目标槽位的关系由独立映射保存。
        pattern = CSRPattern(
            augmented.shape,
            augmented.indptr,
            augmented.indices,
            np.arange(augmented.nnz, dtype=np.int64),
        )
        source_to_target = _readonly_indices(
            augmented.coo_to_csr[: graph.nnz], name="源图到约束图映射"
        )

        is_constrained = np.zeros(shape[0], dtype=bool)
        is_constrained[constrained] = True
        retained = np.flatnonzero(
            ~is_constrained[rows] & ~is_constrained[graph.indices]
        )
        lift = np.flatnonzero(~is_constrained[rows] & is_constrained[graph.indices])
        value_position_by_dof = np.full(shape[0], -1, dtype=np.int64)
        value_position_by_dof[constrained] = np.arange(constrained.size, dtype=np.int64)
        diagonal_slots = np.empty(constrained.size, dtype=np.int64)
        for offset, raw_index in enumerate(constrained):
            index = int(raw_index)
            start, stop = int(pattern.indptr[index]), int(pattern.indptr[index + 1])
            position = int(np.searchsorted(pattern.indices[start:stop], index)) + start
            if position >= stop or pattern.indices[position] != index:
                raise RuntimeError("约束计划未能建立显式单位对角槽位。")
            diagonal_slots[offset] = position

        object.__setattr__(self, "shape", graph.shape)
        object.__setattr__(self, "source_indptr", graph.indptr)
        object.__setattr__(self, "source_indices", graph.indices)
        object.__setattr__(self, "constrained", constrained)
        object.__setattr__(self, "pattern", pattern)
        object.__setattr__(self, "source_to_target", source_to_target)
        object.__setattr__(
            self, "retained_source", _readonly_indices(retained, name="保留项")
        )
        object.__setattr__(self, "lift_source", _readonly_indices(lift, name="提升项"))
        object.__setattr__(
            self, "lift_rows", _readonly_indices(rows[lift], name="提升项行")
        )
        object.__setattr__(
            self,
            "lift_value_positions",
            _readonly_indices(
                value_position_by_dof[graph.indices[lift]], name="约束值位置"
            ),
        )
        object.__setattr__(
            self,
            "diagonal_slots",
            _readonly_indices(diagonal_slots, name="约束对角槽位"),
        )

    @classmethod
    def from_matrix(
        cls, matrix: CSRMatrix, constrained: ArrayLike
    ) -> CSRConstraintPlan:
        """从已规范矩阵图准备一次约束结构分析。"""

        if not isinstance(matrix, CSRMatrix):
            raise TypeError("约束计划要求 CSRMatrix。")
        result = cls(matrix.shape, matrix.indptr, matrix.indices, constrained)
        # CSRMatrix 已保证图只读；保留同一引用可让稳态路径常数时间确认图身份。
        object.__setattr__(result, "source_indptr", matrix.indptr)
        object.__setattr__(result, "source_indices", matrix.indices)
        return result

    @property
    def storage_nbytes(self) -> int:
        arrays = (
            self.source_indptr,
            self.source_indices,
            self.constrained,
            self.pattern.indptr,
            self.pattern.indices,
            self.pattern.coo_to_csr,
            self.source_to_target,
            self.retained_source,
            self.lift_source,
            self.lift_rows,
            self.lift_value_positions,
            self.diagonal_slots,
        )
        return int(sum(array.nbytes for array in arrays))

    @property
    def structure_digest(self) -> str:
        """返回源图、约束顺序和目标图共同决定的跨平台摘要。"""

        digest = sha256()
        digest.update(np.asarray(self.shape, dtype="<i8").tobytes())
        for array in (
            self.source_indptr,
            self.source_indices,
            self.constrained,
            self.pattern.indptr,
            self.pattern.indices,
        ):
            digest.update(np.asarray(array, dtype="<i8").tobytes())
        return digest.hexdigest()

    def validate_matrix(self, matrix: CSRMatrix) -> None:
        """拒绝把计划用于不同形状或不同规范图。"""

        if not isinstance(matrix, CSRMatrix):
            raise TypeError("约束计划只接受 CSRMatrix。")
        if (
            matrix.shape != self.shape
            or (
                matrix.indptr is not self.source_indptr
                and not np.array_equal(matrix.indptr, self.source_indptr)
            )
            or (
                matrix.indices is not self.source_indices
                and not np.array_equal(matrix.indices, self.source_indices)
            )
        ):
            raise ValueError("CSR 图与约束计划不一致，必须重新准备计划。")

    def transform_matrix(self, matrix: CSRMatrix) -> CSRMatrix:
        """只更新约束后矩阵数值；结果与具体约束值无关。"""

        self.validate_matrix(matrix)
        data = np.zeros(self.pattern.nnz, dtype=np.float64)
        retained_targets = self.source_to_target[self.retained_source]
        data[retained_targets] = matrix.data[self.retained_source]
        data[self.diagonal_slots] = 1.0
        return CSRMatrix._from_pattern(self.pattern, data)

    def transform_rhs(
        self,
        matrix: CSRMatrix,
        right_hand_side: ArrayLike,
        values: ArrayLike,
    ) -> FloatArray:
        """按准备时的约束顺序生成强制 Dirichlet 右端项。"""

        self.validate_matrix(matrix)
        rhs = np.array(right_hand_side, dtype=np.float64, copy=True)
        if rhs.shape != (self.shape[0],) or not np.all(np.isfinite(rhs)):
            raise ValueError("右端项必须为每行一个有限标量。")
        prescribed = _readonly_values(values, name="约束值")
        if prescribed.shape != self.constrained.shape:
            raise ValueError("约束值数量必须与计划的约束自由度数量一致。")
        if self.lift_source.size:
            with np.errstate(over="ignore", invalid="ignore"):
                correction = (
                    -matrix.data[self.lift_source]
                    * prescribed[self.lift_value_positions]
                )
            np.add.at(rhs, self.lift_rows, correction)
        rhs[self.constrained] = prescribed
        if not np.all(np.isfinite(rhs)):
            raise ValueError("约束变换后的右端项不是有限数。")
        return rhs

    def apply(
        self,
        matrix: CSRMatrix,
        right_hand_side: ArrayLike,
        values: ArrayLike,
    ) -> tuple[CSRMatrix, FloatArray]:
        """一次返回约束后矩阵和右端项，供冷路径及差分验证使用。"""

        return (
            self.transform_matrix(matrix),
            self.transform_rhs(matrix, right_hand_side, values),
        )


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
class PreparedPreconditioner:
    """绑定一个 CSR 数值快照的 CG 预条件器。"""

    kind: Literal["none", "jacobi", "block_jacobi"]
    shape: tuple[int, int]
    indptr: IndexArray
    indices: IndexArray
    matrix_data: FloatArray
    inverse_diagonal: FloatArray | None = None
    inverse_blocks: FloatArray | None = None
    block_size: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"none", "jacobi", "block_jacobi"}:
            raise ValueError(f"未知 CG 预条件器 {self.kind!r}。")
        graph = CSRMatrix(self.shape, self.indptr, self.indices, self.matrix_data)
        inverse_diagonal = self.inverse_diagonal
        inverse_blocks = self.inverse_blocks
        if inverse_diagonal is not None:
            inverse_diagonal = _readonly_values(inverse_diagonal, name="Jacobi 逆对角")
        if inverse_blocks is not None:
            blocks = np.array(inverse_blocks, dtype=np.float64, copy=True)
            if blocks.ndim != 3 or not np.all(np.isfinite(blocks)):
                raise ValueError("块 Jacobi 逆块必须是有限三维数组。")
            blocks.setflags(write=False)
            inverse_blocks = blocks
        if self.kind == "none":
            if inverse_diagonal is not None or inverse_blocks is not None:
                raise ValueError("无预条件模式不得携带逆对角数据。")
        elif self.kind == "jacobi":
            if inverse_diagonal is None or inverse_diagonal.shape != (graph.shape[0],):
                raise ValueError("Jacobi 逆对角长度必须等于矩阵行数。")
            if inverse_blocks is not None:
                raise ValueError("Jacobi 预条件器不得携带块逆矩阵。")
        else:
            if (
                self.block_size is None
                or inverse_blocks is None
                or inverse_diagonal is not None
            ):
                raise ValueError("块 Jacobi 预条件器缺少合法块尺寸或逆块。")
            size = int(self.block_size)
            if size < 1 or inverse_blocks.shape != (
                graph.shape[0] // size,
                size,
                size,
            ):
                raise ValueError("块 Jacobi 逆块形状与矩阵不一致。")
        object.__setattr__(self, "shape", graph.shape)
        object.__setattr__(self, "indptr", graph.indptr)
        object.__setattr__(self, "indices", graph.indices)
        object.__setattr__(self, "matrix_data", graph.data)
        object.__setattr__(self, "inverse_diagonal", inverse_diagonal)
        object.__setattr__(self, "inverse_blocks", inverse_blocks)

    @classmethod
    def from_matrix(
        cls,
        matrix: CSRMatrix,
        kind: Literal["none", "jacobi", "block_jacobi"] = "jacobi",
        *,
        block_size: int | None = None,
    ) -> PreparedPreconditioner:
        """构造数值预条件器，并保存防止陈旧复用的矩阵快照。"""

        if kind == "none":
            result = cls(kind, matrix.shape, matrix.indptr, matrix.indices, matrix.data)
            object.__setattr__(result, "matrix_data", matrix.data)
            return result
        if kind == "jacobi":
            diagonal = matrix.diagonal()
            if not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
                raise ValueError("Jacobi 预条件要求严格正的有限对角。")
            result = cls(
                kind,
                matrix.shape,
                matrix.indptr,
                matrix.indices,
                matrix.data,
                inverse_diagonal=1.0 / diagonal,
            )
            object.__setattr__(result, "matrix_data", matrix.data)
            return result
        if kind != "block_jacobi":
            raise ValueError(f"未知 CG 预条件器 {kind!r}。")
        if block_size is None:
            raise ValueError("块 Jacobi 要求显式块尺寸。")
        blocks = BSRMatrix.from_csr(matrix, block_size).diagonal_blocks()
        try:
            eigenvalues = np.linalg.eigvalsh(blocks)
            if np.any(eigenvalues <= 0.0):
                raise ValueError("块 Jacobi 要求对称正定的对角块。")
            inverse_blocks = np.linalg.inv(blocks)
        except np.linalg.LinAlgError as error:
            raise ValueError("块 Jacobi 对角块不可逆。") from error
        result = cls(
            kind,
            matrix.shape,
            matrix.indptr,
            matrix.indices,
            matrix.data,
            inverse_blocks=inverse_blocks,
            block_size=int(block_size),
        )
        object.__setattr__(result, "matrix_data", matrix.data)
        return result

    @property
    def storage_nbytes(self) -> int:
        return int(
            self.indptr.nbytes
            + self.indices.nbytes
            + self.matrix_data.nbytes
            + (0 if self.inverse_diagonal is None else self.inverse_diagonal.nbytes)
            + (0 if self.inverse_blocks is None else self.inverse_blocks.nbytes)
        )

    def validate_matrix(self, matrix: CSRMatrix) -> None:
        """图或数值改变都拒绝旧预条件器，避免错误缓存命中。"""

        if (
            matrix.shape != self.shape
            or (
                matrix.indptr is not self.indptr
                and not np.array_equal(matrix.indptr, self.indptr)
            )
            or (
                matrix.indices is not self.indices
                and not np.array_equal(matrix.indices, self.indices)
            )
        ):
            raise ValueError("CSR 图与预备预条件器不一致，必须重新准备。")
        if matrix.data is not self.matrix_data and not np.array_equal(
            matrix.data, self.matrix_data
        ):
            raise ValueError("CSR 数值已改变，必须重建数值预条件器。")

    def apply(self, values: FloatArray) -> FloatArray:
        """应用已准备的数值预条件器。"""

        if self.inverse_diagonal is not None:
            return self.inverse_diagonal * values
        if self.inverse_blocks is not None:
            assert self.block_size is not None
            blocked = values.reshape(self.inverse_blocks.shape[0], self.block_size)
            return np.einsum("bij,bj->bi", self.inverse_blocks, blocked).ravel()
        return values.copy()


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
    prepared_preconditioner: PreparedPreconditioner | None = None,
) -> CGResult:
    """求解对称正定系统，并返回明确的收敛、上限或失效证据。"""

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
    if prepared_preconditioner is not None:
        if not isinstance(prepared_preconditioner, PreparedPreconditioner):
            raise TypeError("预备预条件器必须为 PreparedPreconditioner。")
        prepared_preconditioner.validate_matrix(matrix)

    if initial_guess is None:
        solution = np.zeros(matrix.shape[1], dtype=np.float64)
    else:
        solution = np.array(initial_guess, dtype=np.float64, copy=True)
        if solution.shape != (matrix.shape[1],) or not np.all(np.isfinite(solution)):
            raise ValueError("Initial guess must be one finite scalar per column.")
    residual = rhs - matrix.matvec(solution)
    initial_norm = float(np.linalg.norm(residual))
    requested_threshold = max(
        float(absolute_tolerance),
        float(relative_tolerance) * float(np.linalg.norm(rhs)),
    )
    absolute_row_sums = np.zeros(matrix.shape[0], dtype=np.float64)
    np.add.at(absolute_row_sums, matrix.row_indices(), np.abs(matrix.data))
    matrix_infinity_norm = (
        0.0 if absolute_row_sums.size == 0 else float(np.max(absolute_row_sums))
    )

    def attainable_threshold(current_solution: FloatArray) -> float:
        """把用户阈值与 binary64 可达到的保守后向误差底线合并。"""

        solution_norm = (
            0.0
            if current_solution.size == 0
            else float(np.linalg.norm(current_solution, ord=np.inf))
        )
        rhs_infinity_norm = (
            0.0 if rhs.size == 0 else float(np.linalg.norm(rhs, ord=np.inf))
        )
        scale = matrix_infinity_norm * solution_norm + rhs_infinity_norm
        roundoff_floor = (
            np.finfo(np.float64).eps * np.sqrt(max(matrix.shape[0], 1)) * scale
        )
        return max(requested_threshold, roundoff_floor)

    threshold = attainable_threshold(solution)
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

    if prepared_preconditioner is None:
        active_preconditioner = PreparedPreconditioner.from_matrix(
            matrix, preconditioner, block_size=block_size
        )
    else:
        active_preconditioner = prepared_preconditioner

    z = active_preconditioner.apply(residual)
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
        threshold = attainable_threshold(solution)
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
            # 递推残差会积累舍入漂移；对外报告收敛前必须用 A*x 复核真实残差。
            verified_residual = rhs - matrix.matvec(solution)
            verified_norm = float(np.linalg.norm(verified_residual))
            if np.isfinite(verified_norm) and verified_norm <= threshold:
                return _cg_result(
                    solution,
                    converged=True,
                    reason="converged",
                    iterations=iteration,
                    initial_norm=initial_norm,
                    residual_norm=verified_norm,
                    threshold=threshold,
                    rtol=float(relative_tolerance),
                    atol=float(absolute_tolerance),
                )
            if not np.isfinite(verified_norm):
                return _cg_result(
                    solution,
                    converged=False,
                    reason="breakdown",
                    iterations=iteration,
                    initial_norm=initial_norm,
                    residual_norm=verified_norm,
                    threshold=threshold,
                    rtol=float(relative_tolerance),
                    atol=float(absolute_tolerance),
                )
            # 残差替换后重新开始共轭方向，避免把不一致递推状态带入下一步。
            residual = verified_residual
            residual_norm = verified_norm
            z = active_preconditioner.apply(residual)
            rho = float(residual @ z)
            if not np.isfinite(rho) or rho <= 0.0:
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
            direction = z.copy()
            continue
        z = active_preconditioner.apply(residual)
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

    verified_residual_norm = float(np.linalg.norm(rhs - matrix.matvec(solution)))
    threshold = attainable_threshold(solution)
    return _cg_result(
        solution,
        converged=False,
        reason="iteration_limit",
        iterations=maximum_iterations,
        initial_norm=initial_norm,
        residual_norm=verified_residual_norm,
        threshold=threshold,
        rtol=float(relative_tolerance),
        atol=float(absolute_tolerance),
    )
