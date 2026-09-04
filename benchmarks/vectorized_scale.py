# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Measure a large vectorized mesh-and-assembly path with algebraic guards."""

from __future__ import annotations

import argparse
import json
import platform
import tracemalloc
from time import perf_counter

import numpy as np

from agentfem_native import __version__, unit_square_triangles
from agentfem_native.assembly import assemble_diffusion_vectorized


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=32_768)
    arguments = parser.parse_args()
    if arguments.resolution < 1 or arguments.chunk_size < 1:
        parser.error("resolution and chunk-size must be positive")

    tracemalloc.start()
    started = perf_counter()
    mesh = unit_square_triangles(arguments.resolution)
    mesh_seconds = perf_counter() - started
    started = perf_counter()
    matrix, load = assemble_diffusion_vectorized(
        mesh,
        conductivity=1.0,
        source=1.0,
        chunk_size=arguments.chunk_size,
    )
    assembly_seconds = perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    null_residual = float(np.max(np.abs(matrix.matvec(np.ones(mesh.node_count)))))
    load_sum = float(load.sum())
    verified = abs(load_sum - 1.0) <= 2.0e-13 and null_residual <= 2.0e-13
    record = {
        "benchmark": "vectorized_diffusion_scale",
        "agentfem_native": __version__,
        "resolution": arguments.resolution,
        "nodes": mesh.node_count,
        "cells": mesh.cell_count,
        "coo_entries_with_duplicates": int(matrix.data.size),
        "chunk_size": arguments.chunk_size,
        "mesh_seconds": mesh_seconds,
        "assembly_seconds": assembly_seconds,
        "cells_per_second": mesh.cell_count / assembly_seconds,
        "coo_entries_per_second": matrix.data.size / assembly_seconds,
        "tracemalloc_peak_bytes": peak,
        "verification": {
            "passed": verified,
            "load_sum": load_sum,
            "constant_null_mode_max_abs": null_residual,
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "os": platform.system(),
        "architecture": platform.machine(),
    }
    print(json.dumps(record, allow_nan=False, sort_keys=True))
    return 0 if verified else 2


if __name__ == "__main__":
    raise SystemExit(main())
