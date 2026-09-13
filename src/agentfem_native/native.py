# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""自研 C++20 装配加速器的轻量、受检 Python 边界。"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

NATIVE_P1_ABI_VERSION: Final = 0x0001_0004

try:
    from . import _p1_native
except ImportError as error:
    _p1_native = None
    _IMPORT_ERROR: ImportError | None = error
else:
    _IMPORT_ERROR = None


class NativeKernelUnavailableError(RuntimeError):
    """Raised when the platform wheel does not contain the compiled kernel."""


def native_kernel_available() -> bool:
    """Return whether the compiled P1 accelerator is importable and ABI-compatible."""

    return bool(
        _p1_native is not None
        and int(_p1_native.abi_version()) == NATIVE_P1_ABI_VERSION
    )


def _format_abi_version(encoded: int) -> str:
    return f"{encoded >> 16}.{encoded & 0xFFFF}"


def native_kernel_identity() -> dict[str, object]:
    """Return stable runtime identity without exposing extension internals."""

    actual = int(_p1_native.abi_version()) if _p1_native is not None else None
    return {
        "name": "cpp20",
        "available": native_kernel_available(),
        "abi_version": _format_abi_version(actual) if actual is not None else None,
        "required_abi_version": _format_abi_version(NATIVE_P1_ABI_VERSION),
    }


def assemble_p1_volume(
    points: NDArray[np.float64],
    cells: NDArray[np.int64],
    conductivity: NDArray[np.float64],
    source: float,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Fill deterministic COO/load arrays through the compiled C ABI."""

    if not native_kernel_available():
        detail = f": {_IMPORT_ERROR}" if _IMPORT_ERROR is not None else ""
        raise NativeKernelUnavailableError(
            f"The AgentFEM Native C++20 accelerator is unavailable{detail}."
        )
    point_values = np.ascontiguousarray(points, dtype=np.float64)
    cell_values = np.ascontiguousarray(cells, dtype=np.int64)
    tensor_values = np.ascontiguousarray(conductivity, dtype=np.float64)
    if point_values.ndim != 2 or point_values.shape[1] != 2:
        raise ValueError("Native points must have shape (node_count, 2).")
    if cell_values.ndim != 2 or cell_values.shape[1] != 3:
        raise ValueError("Native cells must have shape (cell_count, 3).")
    if tensor_values.shape not in {
        (2, 2),
        (cell_values.shape[0], 2, 2),
    }:
        raise ValueError(
            "Native conductivity must have shape (2, 2) or (cell_count, 2, 2)."
        )

    entry_count = cell_values.shape[0] * 9
    rows = np.empty(entry_count, dtype=np.int64)
    columns = np.empty(entry_count, dtype=np.int64)
    data = np.empty(entry_count, dtype=np.float64)
    load = np.empty(point_values.shape[0], dtype=np.float64)
    status = int(
        _p1_native.assemble_into(
            point_values.shape[0],
            cell_values.shape[0],
            point_values,
            cell_values,
            tensor_values,
            float(source),
            rows,
            columns,
            data,
            load,
        )
    )
    messages = {
        1: "Native assembly received a null or empty buffer.",
        2: "Native conductivity must be finite, symmetric, and positive definite.",
        3: "Native cell connectivity or geometry is invalid.",
        4: "Native assembly input must be finite.",
    }
    if status != 0:
        message = messages.get(status, f"Native assembly failed with status {status}.")
        raise ValueError(message)
    return rows, columns, data, load


def assemble_t3_volume(
    points: NDArray[np.float64],
    cells: NDArray[np.int64],
    constitutive: NDArray[np.float64],
    body_force: NDArray[np.float64],
    thickness: float,
    *,
    thread_count: int = 1,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """通过有版本的 C++20 ABI 装配静态 T3 体积项。"""

    if not native_kernel_available():
        detail = f": {_IMPORT_ERROR}" if _IMPORT_ERROR is not None else ""
        raise NativeKernelUnavailableError(
            f"The AgentFEM Native C++20 accelerator is unavailable{detail}."
        )
    point_values = np.ascontiguousarray(points, dtype=np.float64)
    cell_values = np.ascontiguousarray(cells, dtype=np.int64)
    constitutive_values = np.ascontiguousarray(constitutive, dtype=np.float64)
    body_values = np.ascontiguousarray(body_force, dtype=np.float64)
    if point_values.ndim != 2 or point_values.shape[1] != 2:
        raise ValueError("Native points must have shape (node_count, 2).")
    if cell_values.ndim != 2 or cell_values.shape[1] != 3:
        raise ValueError("Native cells must have shape (cell_count, 3).")
    if constitutive_values.shape != (cell_values.shape[0], 3, 3):
        raise ValueError("Native constitutive data must have shape (cell_count, 3, 3).")
    if body_values.shape != (2,):
        raise ValueError("Native body force must have shape (2,).")
    if isinstance(thread_count, (bool, np.bool_)) or not isinstance(
        thread_count, (int, np.integer)
    ):
        raise TypeError("并行装配线程数必须是正整数。")
    if thread_count < 1:
        raise ValueError("并行装配线程数必须大于零。")
    entry_count = cell_values.shape[0] * 36
    rows = np.empty(entry_count, dtype=np.int64)
    columns = np.empty(entry_count, dtype=np.int64)
    data = np.empty(entry_count, dtype=np.float64)
    load = np.empty(point_values.shape[0] * 2, dtype=np.float64)
    assemble = (
        _p1_native.assemble_t3_into
        if thread_count == 1
        else _p1_native.assemble_t3_parallel_into
    )
    status = int(
        assemble(
            point_values.shape[0],
            cell_values.shape[0],
            *(() if thread_count == 1 else (int(thread_count),)),
            point_values,
            cell_values,
            constitutive_values,
            body_values,
            float(thickness),
            rows,
            columns,
            data,
            load,
        )
    )
    messages = {
        1: "Native T3 assembly received a null or empty buffer.",
        2: "Native T3 constitutive matrix must be finite, symmetric, and positive definite.",
        3: "Native T3 cell connectivity or geometry is invalid.",
        4: "Native T3 assembly input must be finite with positive thickness.",
        5: "并行装配线程数必须大于零。",
        6: "Native T3 并行装配无法创建所需线程或临时资源。",
    }
    if status != 0:
        raise ValueError(
            messages.get(status, f"Native T3 assembly failed with status {status}.")
        )
    return rows, columns, data, load


def assemble_t4_volume(
    points: NDArray[np.float64],
    cells: NDArray[np.int64],
    constitutive: NDArray[np.float64],
    body_force: NDArray[np.float64],
    *,
    thread_count: int = 1,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """通过有版本的 C++20 ABI 装配静态 T4 体积项。"""

    if not native_kernel_available():
        detail = f": {_IMPORT_ERROR}" if _IMPORT_ERROR is not None else ""
        raise NativeKernelUnavailableError(
            f"The AgentFEM Native C++20 accelerator is unavailable{detail}."
        )
    point_values = np.ascontiguousarray(points, dtype=np.float64)
    cell_values = np.ascontiguousarray(cells, dtype=np.int64)
    constitutive_values = np.ascontiguousarray(constitutive, dtype=np.float64)
    body_values = np.ascontiguousarray(body_force, dtype=np.float64)
    if point_values.ndim != 2 or point_values.shape[1] != 3:
        raise ValueError("Native T4 points must have shape (node_count, 3).")
    if cell_values.ndim != 2 or cell_values.shape[1] != 4:
        raise ValueError("Native T4 cells must have shape (cell_count, 4).")
    if constitutive_values.shape != (cell_values.shape[0], 6, 6):
        raise ValueError(
            "Native T4 constitutive data must have shape (cell_count, 6, 6)."
        )
    if body_values.shape != (3,):
        raise ValueError("Native T4 body force must have shape (3,).")
    if isinstance(thread_count, (bool, np.bool_)) or not isinstance(
        thread_count, (int, np.integer)
    ):
        raise TypeError("并行装配线程数必须是正整数。")
    if thread_count < 1:
        raise ValueError("并行装配线程数必须大于零。")
    entry_count = cell_values.shape[0] * 144
    rows = np.empty(entry_count, dtype=np.int64)
    columns = np.empty(entry_count, dtype=np.int64)
    data = np.empty(entry_count, dtype=np.float64)
    load = np.empty(point_values.shape[0] * 3, dtype=np.float64)
    assemble = (
        _p1_native.assemble_t4_into
        if thread_count == 1
        else _p1_native.assemble_t4_parallel_into
    )
    status = int(
        assemble(
            point_values.shape[0],
            cell_values.shape[0],
            *(() if thread_count == 1 else (int(thread_count),)),
            point_values,
            cell_values,
            constitutive_values,
            body_values,
            rows,
            columns,
            data,
            load,
        )
    )
    messages = {
        1: "Native T4 assembly received a null or empty buffer.",
        2: "Native T4 constitutive matrix must be finite, symmetric, and positive definite.",
        3: "Native T4 cell connectivity or geometry is invalid.",
        4: "Native T4 assembly input must be finite.",
        5: "并行装配线程数必须大于零。",
        6: "Native T4 并行装配无法创建所需线程或临时资源。",
    }
    if status != 0:
        raise ValueError(
            messages.get(status, f"Native T4 assembly failed with status {status}.")
        )
    return rows, columns, data, load


def csr_spmv(
    shape: tuple[int, int],
    indptr: NDArray[np.int64],
    indices: NDArray[np.int64],
    data: NDArray[np.float64],
    vector: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply canonical CSR through the C++20 ABI and return an owned vector."""

    if not native_kernel_available():
        detail = f": {_IMPORT_ERROR}" if _IMPORT_ERROR is not None else ""
        raise NativeKernelUnavailableError(
            f"The AgentFEM Native C++20 accelerator is unavailable{detail}."
        )
    rows, columns = shape
    pointer_values = np.ascontiguousarray(indptr, dtype=np.int64)
    index_values = np.ascontiguousarray(indices, dtype=np.int64)
    matrix_values = np.ascontiguousarray(data, dtype=np.float64)
    vector_values = np.ascontiguousarray(vector, dtype=np.float64)
    if pointer_values.shape != (rows + 1,):
        raise ValueError("Native CSR indptr shape does not match row count.")
    if index_values.ndim != 1 or matrix_values.shape != index_values.shape:
        raise ValueError("Native CSR indices and data must be equal-length vectors.")
    if vector_values.shape != (columns,):
        raise ValueError("Native CSR vector shape does not match column count.")
    result = np.empty(rows, dtype=np.float64)
    status = int(
        _p1_native.csr_spmv_into(
            rows,
            columns,
            index_values.size,
            pointer_values,
            index_values,
            matrix_values,
            vector_values,
            result,
        )
    )
    messages = {
        1: "Native CSR SpMV received a null or empty buffer.",
        3: "Native CSR structure is invalid.",
        4: "Native CSR data, vector, or result is non-finite.",
    }
    if status != 0:
        raise ValueError(
            messages.get(status, f"Native CSR SpMV failed with status {status}.")
        )
    return result


def csr_fill_from_contributions(
    nonzero_count: int,
    coo_to_csr: NDArray[np.int64],
    contributions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """通过 C++20 ABI 按原贡献顺序生成一份自有 CSR 数值数组。"""

    if not native_kernel_available():
        detail = f"：{_IMPORT_ERROR}" if _IMPORT_ERROR is not None else ""
        raise NativeKernelUnavailableError(
            f"AgentFEM Native C++20 加速器不可用{detail}。"
        )
    if (
        isinstance(nonzero_count, (bool, np.bool_))
        or not isinstance(nonzero_count, (int, np.integer))
        or int(nonzero_count) < 0
    ):
        raise ValueError("CSR 非零槽位数必须是非负整数。")
    mapping = np.ascontiguousarray(coo_to_csr, dtype=np.int64)
    values = np.ascontiguousarray(contributions, dtype=np.float64)
    if mapping.ndim != 1 or values.shape != mapping.shape:
        raise ValueError("COO 映射与贡献必须是等长一维数组。")
    data = np.empty(int(nonzero_count), dtype=np.float64)
    status = int(
        _p1_native.csr_fill_into(
            mapping.size, int(nonzero_count), mapping, values, data
        )
    )
    messages = {
        1: "Native CSR 回填需要一致的非空映射和数值槽位。",
        3: "Native CSR 回填映射槽位超出规范图范围。",
        4: "Native CSR 回填贡献或归并结果不是有限数。",
    }
    if status != 0:
        raise ValueError(
            messages.get(status, f"Native CSR 回填失败，状态码 {status}。")
        )
    return data
