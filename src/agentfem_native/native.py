# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Thin checked Python boundary for the owned C++20 assembly accelerator."""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

NATIVE_P1_ABI_VERSION: Final = 0x0001_0000

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
