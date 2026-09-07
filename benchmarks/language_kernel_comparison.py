# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Compare C++20 and Rust implementations of the identical owned C ABI."""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from agentfem_native import unit_square_triangles
from agentfem_native.assembly import assemble_diffusion_vectorized

ABI_MAJOR = 1


def _load(path: Path):
    library = ctypes.CDLL(str(path.resolve()))
    version = library.afn_p1_abi_version
    version.argtypes = []
    version.restype = ctypes.c_uint32
    encoded = int(version())
    if encoded >> 16 != ABI_MAJOR:
        raise RuntimeError(f"{path} does not implement AgentFEM P1 ABI major 1.")
    function = library.afn_p1_diffusion_assemble
    function.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    function.restype = ctypes.c_int
    return function


def _runner(function, mesh, points, cells, tensor):
    def execute():
        rows = np.empty(mesh.cell_count * 9, dtype=np.int64)
        columns = np.empty_like(rows)
        data = np.empty(rows.size, dtype=np.float64)
        load = np.empty(mesh.node_count, dtype=np.float64)
        status = function(
            mesh.node_count,
            mesh.cell_count,
            points.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            cells.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            tensor.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            1.2,
            rows.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            columns.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            load.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if status != 0:
            raise RuntimeError(f"Kernel failed with status {status}.")
        return rows, columns, data, load

    return execute


def _measure(function, repeats: int):
    for _ in range(5):
        function()
    timings = []
    result = None
    for _ in range(repeats):
        started = perf_counter()
        result = function()
        timings.append(perf_counter() - started)
    assert result is not None
    return result, timings


def _measure_interleaved(left, right, repeats: int):
    for _ in range(5):
        left()
        right()
    left_times = []
    right_times = []
    left_result = right_result = None
    for iteration in range(repeats):
        ordered = ((left, left_times), (right, right_times))
        if iteration % 2:
            ordered = tuple(reversed(ordered))
        for function, timings in ordered:
            started = perf_counter()
            result = function()
            timings.append(perf_counter() - started)
            if function is left:
                left_result = result
            else:
                right_result = result
    assert left_result is not None and right_result is not None
    return left_result, left_times, right_result, right_times


def _difference(left, right) -> dict[str, object]:
    return {
        "row_identity": bool(np.array_equal(left[0], right[0])),
        "column_identity": bool(np.array_equal(left[1], right[1])),
        "max_abs_matrix_difference": float(
            np.max(np.abs(left[2] - right[2]), initial=0.0)
        ),
        "max_abs_load_difference": float(
            np.max(np.abs(left[3] - right[3]), initial=0.0)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp-library", required=True, type=Path)
    parser.add_argument("--rust-library", required=True, type=Path)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=11)
    arguments = parser.parse_args()
    if arguments.resolution < 1 or arguments.repeats < 1:
        parser.error("resolution and repeats must be positive")

    mesh = unit_square_triangles(arguments.resolution)
    points = np.ascontiguousarray(mesh.points, dtype=np.float64)
    cells = np.ascontiguousarray(mesh.cells, dtype=np.int64)
    tensor = np.ascontiguousarray(((2.0, 0.3), (0.3, 1.0)), dtype=np.float64)
    cpp = _runner(_load(arguments.cpp_library), mesh, points, cells, tensor)
    rust = _runner(_load(arguments.rust_library), mesh, points, cells, tensor)

    def vectorized():
        matrix, load = assemble_diffusion_vectorized(
            mesh, conductivity=tensor, source=1.2
        )
        return matrix.rows, matrix.columns, matrix.data, load

    cpp_result, cpp_times, rust_result, rust_times = _measure_interleaved(
        cpp, rust, arguments.repeats
    )
    numpy_result, numpy_times = _measure(vectorized, arguments.repeats)
    cpp_rust = _difference(cpp_result, rust_result)
    cpp_numpy = _difference(cpp_result, numpy_result)
    verified = all(
        comparison["row_identity"]
        and comparison["column_identity"]
        and comparison["max_abs_matrix_difference"] <= 5.0e-14
        and comparison["max_abs_load_difference"] <= 5.0e-14
        for comparison in (cpp_rust, cpp_numpy)
    )
    cpp_median = median(cpp_times)
    rust_median = median(rust_times)
    numpy_median = median(numpy_times)
    record = {
        "benchmark": "cpp20_rust_numpy_p1_abi",
        "abi_compatibility": "major 1 diffusion subset",
        "resolution": arguments.resolution,
        "nodes": mesh.node_count,
        "cells": mesh.cell_count,
        "repeats": arguments.repeats,
        "cpp20": {"seconds": cpp_times, "median_seconds": cpp_median},
        "rust": {"seconds": rust_times, "median_seconds": rust_median},
        "numpy_vectorized": {
            "seconds": numpy_times,
            "median_seconds": numpy_median,
        },
        "ratios": {
            "rust_over_cpp20_time": rust_median / cpp_median,
            "cpp20_speedup_over_numpy": numpy_median / cpp_median,
            "rust_speedup_over_numpy": numpy_median / rust_median,
        },
        "equivalence": {
            "verified": verified,
            "cpp20_vs_rust": cpp_rust,
            "cpp20_vs_numpy": cpp_numpy,
        },
        "os": platform.system(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    print(json.dumps(record, allow_nan=False, sort_keys=True))
    return 0 if verified else 2


if __name__ == "__main__":
    raise SystemExit(main())
