# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Machine-readable NumPy baseline for future production-kernel comparisons."""

from __future__ import annotations

import argparse
import json
import platform
from time import perf_counter

import numpy as np

from agentfem_native import unit_square_triangles
from agentfem_native.assembly import assemble_diffusion


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    arguments = parser.parse_args()
    if arguments.resolution < 1 or arguments.repeats < 1:
        parser.error("resolution and repeats must be positive")

    mesh = unit_square_triangles(arguments.resolution)
    timings = []
    nonzeros = 0
    for _ in range(arguments.repeats):
        started = perf_counter()
        matrix, _ = assemble_diffusion(mesh, conductivity=1.0, source=1.0)
        timings.append(perf_counter() - started)
        nonzeros = int(matrix.data.size)

    record = {
        "benchmark": "reference_diffusion_assembly",
        "implementation": "python_numpy_reference",
        "resolution": arguments.resolution,
        "nodes": mesh.node_count,
        "cells": mesh.cell_count,
        "coo_entries_with_duplicates": nonzeros,
        "repeats": arguments.repeats,
        "seconds": timings,
        "median_seconds": float(np.median(timings)),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "os": platform.system(),
        "architecture": platform.machine(),
    }
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
