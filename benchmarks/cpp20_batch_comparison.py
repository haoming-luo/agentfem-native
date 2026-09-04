# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Compare the standard-library C++20 spike with vectorized NumPy assembly."""

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


def _load_function(path: Path):
    library = ctypes.CDLL(str(path.resolve()))
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("library", type=Path)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=9)
    arguments = parser.parse_args()
    if arguments.resolution < 1 or arguments.repeats < 1:
        parser.error("resolution and repeats must be positive")

    function = _load_function(arguments.library)
    mesh = unit_square_triangles(arguments.resolution)
    points = np.ascontiguousarray(mesh.points, dtype=np.float64)
    cells = np.ascontiguousarray(mesh.cells, dtype=np.int64)
    tensor = np.ascontiguousarray(((2.0, 0.3), (0.3, 1.0)), dtype=np.float64)

    def cpp20():
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
            raise RuntimeError(f"C++20 spike failed with status {status}.")
        return rows, columns, data, load

    def vectorized():
        matrix, load = assemble_diffusion_vectorized(
            mesh, conductivity=tensor, source=1.2
        )
        return matrix.rows, matrix.columns, matrix.data, load

    cpp_times = []
    vectorized_times = []
    cpp_result = vectorized_result = None
    for _ in range(arguments.repeats):
        started = perf_counter()
        cpp_result = cpp20()
        cpp_times.append(perf_counter() - started)
        started = perf_counter()
        vectorized_result = vectorized()
        vectorized_times.append(perf_counter() - started)
    assert cpp_result is not None and vectorized_result is not None
    differences = [
        float(np.max(np.abs(left - right), initial=0.0))
        for left, right in zip(cpp_result[2:], vectorized_result[2:], strict=True)
    ]
    index_identity = bool(
        np.array_equal(cpp_result[0], vectorized_result[0])
        and np.array_equal(cpp_result[1], vectorized_result[1])
    )
    equivalent = index_identity and max(differences) <= 5.0e-14
    cpp_median = median(cpp_times)
    vectorized_median = median(vectorized_times)
    record = {
        "benchmark": "cpp20_p1_assembly_spike",
        "library": str(arguments.library),
        "resolution": arguments.resolution,
        "nodes": mesh.node_count,
        "cells": mesh.cell_count,
        "repeats": arguments.repeats,
        "cpp20": {"seconds": cpp_times, "median_seconds": cpp_median},
        "numpy_vectorized": {
            "seconds": vectorized_times,
            "median_seconds": vectorized_median,
        },
        "cpp20_speedup_over_numpy_vectorized": vectorized_median / cpp_median,
        "equivalence": {
            "verified": equivalent,
            "index_identity": index_identity,
            "max_abs_matrix_difference": differences[0],
            "max_abs_load_difference": differences[1],
        },
        "os": platform.system(),
        "architecture": platform.machine(),
        "compiler_build": "external",
    }
    print(json.dumps(record, allow_nan=False, sort_keys=True))
    return 0 if equivalent else 2


if __name__ == "__main__":
    raise SystemExit(main())
