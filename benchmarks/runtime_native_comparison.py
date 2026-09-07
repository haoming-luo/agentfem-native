# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Compare the packaged C++20 runtime path with bounded vectorized NumPy."""

from __future__ import annotations

import argparse
import json
import platform
from statistics import median
from time import perf_counter

import numpy as np

from agentfem_native import __version__, unit_square_triangles
from agentfem_native.assembly import (
    assemble_diffusion_native,
    assemble_diffusion_vectorized,
)
from agentfem_native.native import native_kernel_identity


def _measure(function, repeats: int):
    function()
    timings = []
    result = None
    for _ in range(repeats):
        started = perf_counter()
        result = function()
        timings.append(perf_counter() - started)
    assert result is not None
    return result, timings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=11)
    arguments = parser.parse_args()
    if arguments.resolution < 1 or arguments.repeats < 1:
        parser.error("resolution and repeats must be positive")

    mesh = unit_square_triangles(arguments.resolution)
    tensor = np.array(((2.0, 0.3), (0.3, 1.0)))

    def native():
        return assemble_diffusion_native(mesh, conductivity=tensor, source=1.2)

    def vectorized():
        return assemble_diffusion_vectorized(mesh, conductivity=tensor, source=1.2)

    native_result, native_times = _measure(native, arguments.repeats)
    vectorized_result, vectorized_times = _measure(vectorized, arguments.repeats)
    native_matrix, native_load = native_result
    vectorized_matrix, vectorized_load = vectorized_result
    row_identity = bool(np.array_equal(native_matrix.rows, vectorized_matrix.rows))
    column_identity = bool(
        np.array_equal(native_matrix.columns, vectorized_matrix.columns)
    )
    matrix_difference = float(
        np.max(np.abs(native_matrix.data - vectorized_matrix.data), initial=0.0)
    )
    load_difference = float(np.max(np.abs(native_load - vectorized_load), initial=0.0))
    equivalent = (
        row_identity
        and column_identity
        and matrix_difference <= 5.0e-14
        and load_difference <= 5.0e-14
    )
    native_median = median(native_times)
    vectorized_median = median(vectorized_times)
    record = {
        "benchmark": "packaged_native_runtime",
        "agentfem_native": __version__,
        "native_kernel": native_kernel_identity(),
        "resolution": arguments.resolution,
        "nodes": mesh.node_count,
        "cells": mesh.cell_count,
        "coo_entries_with_duplicates": int(native_matrix.data.size),
        "repeats": arguments.repeats,
        "native_seconds": native_times,
        "native_median_seconds": native_median,
        "vectorized_seconds": vectorized_times,
        "vectorized_median_seconds": vectorized_median,
        "speedup": vectorized_median / native_median,
        "equivalence": {
            "verified": equivalent,
            "row_identity": row_identity,
            "column_identity": column_identity,
            "max_abs_matrix_difference": matrix_difference,
            "max_abs_load_difference": load_difference,
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "os": platform.system(),
        "architecture": platform.machine(),
    }
    print(json.dumps(record, allow_nan=False, sort_keys=True))
    return 0 if equivalent else 2


if __name__ == "__main__":
    raise SystemExit(main())
