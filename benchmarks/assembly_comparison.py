# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Compare the bounded vectorized assembly path against the NumPy oracle."""

from __future__ import annotations

import argparse
import json
import platform
import tracemalloc
from hashlib import sha256
from statistics import median
from time import perf_counter

import numpy as np

from agentfem_native import __version__, unit_square_triangles
from agentfem_native.assembly import (
    assemble_diffusion_reference,
    assemble_diffusion_vectorized,
)


def _digest(*arrays: np.ndarray) -> str:
    digest = sha256()
    for array in arrays:
        dtype = "<f8" if np.issubdtype(array.dtype, np.floating) else "<i8"
        digest.update(np.ascontiguousarray(array, dtype=dtype).tobytes())
    return digest.hexdigest()


def _measure(function, repeats: int):
    timings = []
    outcome = None
    for _ in range(repeats):
        started = perf_counter()
        outcome = function()
        timings.append(perf_counter() - started)
    assert outcome is not None
    tracemalloc.start()
    function()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return outcome, timings, peak


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=32_768)
    arguments = parser.parse_args()
    if arguments.resolution < 1 or arguments.repeats < 1 or arguments.chunk_size < 1:
        parser.error("resolution, repeats, and chunk-size must be positive")

    mesh = unit_square_triangles(arguments.resolution)
    tensor = np.array(((2.0, 0.3), (0.3, 1.0)))

    def reference():
        return assemble_diffusion_reference(mesh, conductivity=tensor, source=1.2)

    def vectorized():
        return assemble_diffusion_vectorized(
            mesh,
            conductivity=tensor,
            source=1.2,
            chunk_size=arguments.chunk_size,
        )

    reference_result, reference_times, reference_peak = _measure(
        reference, arguments.repeats
    )
    vectorized_result, vectorized_times, vectorized_peak = _measure(
        vectorized, arguments.repeats
    )
    reference_matrix, reference_load = reference_result
    vectorized_matrix, vectorized_load = vectorized_result
    row_identity = bool(np.array_equal(reference_matrix.rows, vectorized_matrix.rows))
    column_identity = bool(
        np.array_equal(reference_matrix.columns, vectorized_matrix.columns)
    )
    matrix_difference = float(
        np.max(np.abs(reference_matrix.data - vectorized_matrix.data), initial=0.0)
    )
    load_difference = float(
        np.max(np.abs(reference_load - vectorized_load), initial=0.0)
    )
    equivalent = (
        row_identity
        and column_identity
        and matrix_difference <= 5.0e-14
        and load_difference <= 5.0e-14
    )
    reference_median = median(reference_times)
    vectorized_median = median(vectorized_times)
    record = {
        "benchmark": "diffusion_assembly_comparison",
        "agentfem_native": __version__,
        "resolution": arguments.resolution,
        "nodes": mesh.node_count,
        "cells": mesh.cell_count,
        "coo_entries_with_duplicates": int(reference_matrix.data.size),
        "chunk_size": arguments.chunk_size,
        "repeats": arguments.repeats,
        "reference": {
            "seconds": reference_times,
            "median_seconds": reference_median,
            "tracemalloc_peak_bytes": reference_peak,
            "digest": _digest(
                reference_matrix.rows,
                reference_matrix.columns,
                reference_matrix.data,
                reference_load,
            ),
        },
        "vectorized": {
            "seconds": vectorized_times,
            "median_seconds": vectorized_median,
            "tracemalloc_peak_bytes": vectorized_peak,
            "digest": _digest(
                vectorized_matrix.rows,
                vectorized_matrix.columns,
                vectorized_matrix.data,
                vectorized_load,
            ),
        },
        "speedup": reference_median / vectorized_median,
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
