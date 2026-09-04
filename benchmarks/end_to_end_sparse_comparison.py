# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Compare native and vectorized assembly through the same SciPy sparse solve."""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import replace
from statistics import median
from time import perf_counter

import numpy as np

from agentfem_native import (
    DirichletCondition,
    SteadyDiffusionProblem,
    __version__,
    solve_steady_diffusion,
    unit_square_triangles,
)


def _measure(problem: SteadyDiffusionProblem, repeats: int):
    solve_steady_diffusion(problem, provider="scipy")
    timings = []
    result = None
    for _ in range(repeats):
        started = perf_counter()
        result = solve_steady_diffusion(problem, provider="scipy")
        timings.append(perf_counter() - started)
    assert result is not None
    return result, timings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=7)
    arguments = parser.parse_args()
    if arguments.resolution < 1 or arguments.repeats < 1:
        parser.error("resolution and repeats must be positive")

    mesh = unit_square_triangles(arguments.resolution)
    base = SteadyDiffusionProblem(
        mesh=mesh,
        conductivity=np.array(((2.0, 0.2), (0.2, 1.0))),
        source=0.5,
        dirichlet=(
            DirichletCondition("left", 0.0),
            DirichletCondition("right", 1.0),
        ),
        assembly_mode="native",
    )
    native_result, native_times = _measure(base, arguments.repeats)
    vectorized_result, vectorized_times = _measure(
        replace(base, assembly_mode="vectorized"), arguments.repeats
    )
    solution_difference = float(
        np.max(
            np.abs(native_result.nodal_values - vectorized_result.nodal_values),
            initial=0.0,
        )
    )
    reaction_difference = abs(
        native_result.total_reaction - vectorized_result.total_reaction
    )
    energy_difference = abs(
        native_result.potential_energy - vectorized_result.potential_energy
    )
    verified = (
        solution_difference <= 5.0e-13
        and reaction_difference <= 5.0e-12
        and energy_difference <= 5.0e-12
    )
    native_median = median(native_times)
    vectorized_median = median(vectorized_times)
    record = {
        "benchmark": "end_to_end_scipy_sparse",
        "agentfem_native": __version__,
        "resolution": arguments.resolution,
        "nodes": mesh.node_count,
        "cells": mesh.cell_count,
        "repeats": arguments.repeats,
        "native_seconds": native_times,
        "native_median_seconds": native_median,
        "vectorized_seconds": vectorized_times,
        "vectorized_median_seconds": vectorized_median,
        "speedup": vectorized_median / native_median,
        "provider": {
            "name": native_result.provider_name,
            "version": native_result.provider_version,
        },
        "equivalence": {
            "verified": verified,
            "max_abs_solution_difference": solution_difference,
            "total_reaction_difference": reaction_difference,
            "potential_energy_difference": energy_difference,
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
