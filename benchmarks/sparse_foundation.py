# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Record reproducible end-to-end evidence for the Native sparse foundation."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from agentfem_native import (
    DirichletCondition,
    DisplacementCondition,
    LinearElasticMaterial,
    LinearElasticProblem,
    NeumannCondition,
    SteadyDiffusionProblem,
    TractionCondition,
    plan_steady_diffusion,
    solve_linear_elasticity,
    solve_steady_diffusion,
    unit_square_triangles,
)
from agentfem_native.assembly import assemble_diffusion
from agentfem_native.sparse import CSRMatrix


def _median_time(operation, repeats: int):
    samples = []
    value = None
    for _ in range(repeats):
        start = perf_counter()
        value = operation()
        samples.append(perf_counter() - start)
    return value, samples, statistics.median(samples)


def _diffusion_record(resolution: int, repeats: int) -> dict[str, object]:
    mesh = unit_square_triangles(resolution)
    problem = SteadyDiffusionProblem(
        mesh,
        1.0,
        dirichlet=(DirichletCondition("left", 0.0),),
        neumann=(NeumannCondition("right", 1.0),),
    )
    (coo, _), assembly_samples, assembly_median = _median_time(
        lambda: assemble_diffusion(
            mesh,
            conductivity=1.0,
            boundary_fluxes=(("right", 1.0),),
        ),
        repeats,
    )
    csr, conversion_samples, conversion_median = _median_time(
        lambda: CSRMatrix.from_coo(coo.shape, coo.rows, coo.columns, coo.data),
        repeats,
    )
    result, solve_samples, solve_median = _median_time(
        lambda: solve_steady_diffusion(problem, provider="native"), repeats
    )
    plan = plan_steady_diffusion(problem)
    return {
        "resolution": resolution,
        "nodes_dofs": mesh.node_count,
        "cells": mesh.cell_count,
        "coo_entries": int(coo.data.size),
        "csr_nnz": csr.nnz,
        "csr_storage_bytes": csr.storage_nbytes,
        "dense_storage_bytes": 8 * mesh.node_count * mesh.node_count,
        "dense_to_csr_storage_ratio": (
            8 * mesh.node_count * mesh.node_count / csr.storage_nbytes
        ),
        "plan_peak_bytes_upper_bound": plan.peak_bytes_upper_bound,
        "assembly_seconds": assembly_samples,
        "assembly_median_seconds": assembly_median,
        "coo_to_csr_seconds": conversion_samples,
        "coo_to_csr_median_seconds": conversion_median,
        "end_to_end_solve_seconds": solve_samples,
        "end_to_end_solve_median_seconds": solve_median,
        "cg_iterations": result.convergence.iterations,
        "cg_reported_residual_norm": result.convergence.residual_norm,
        "free_residual_norm": result.free_residual_norm,
        "maximum_analytical_error": float(
            np.max(np.abs(result.nodal_values - mesh.points[:, 0]))
        ),
        "balance_error": result.total_applied_load + result.total_reaction,
        "provider": result.provider_name,
        "matrix_format": result.provider_matrix_format,
    }


def _elasticity_record(resolution: int, repeats: int) -> dict[str, object]:
    mesh = unit_square_triangles(resolution)
    modulus, poisson_ratio, traction = 200.0, 0.25, 10.0
    problem = LinearElasticProblem(
        mesh,
        LinearElasticMaterial(modulus, poisson_ratio),
        dirichlet=(
            DisplacementCondition("left", "x", 0.0),
            DisplacementCondition("bottom", "y", 0.0),
        ),
        traction=(TractionCondition("right", (traction, 0.0)),),
    )
    result, samples, median = _median_time(
        lambda: solve_linear_elasticity(problem), repeats
    )
    exact = np.column_stack(
        (
            traction / modulus * mesh.points[:, 0],
            -poisson_ratio * traction / modulus * mesh.points[:, 1],
        )
    )
    return {
        "resolution": resolution,
        "displacement_dofs": 2 * mesh.node_count,
        "cells": mesh.cell_count,
        "end_to_end_solve_seconds": samples,
        "end_to_end_solve_median_seconds": median,
        "cg_iterations": result.convergence.iterations,
        "free_residual_norm": result.free_residual_norm,
        "maximum_analytical_error": float(np.max(np.abs(result.displacements - exact))),
        "force_balance_norm": float(
            np.linalg.norm(result.total_applied_force + result.total_reaction)
        ),
        "provider": result.provider_name,
        "maturity": "implemented",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolutions", nargs="+", type=int, default=(32, 64, 128))
    parser.add_argument("--elasticity-resolution", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.repeats < 1 or any(value < 1 for value in arguments.resolutions):
        parser.error("resolutions and repeats must be positive")
    record = {
        "schema": "agentfem.native-sparse-foundation-benchmark",
        "schema_version": "0.1.0",
        "environment": {
            "os": platform.system(),
            "release": platform.release(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "method": {
            "clock": "time.perf_counter",
            "repeats": arguments.repeats,
            "timing": "warm-process repeated wall time; medians reported",
            "claim_boundary": "local evidence only; not a cross-platform claim",
        },
        "diffusion": [
            _diffusion_record(resolution, arguments.repeats)
            for resolution in arguments.resolutions
        ],
        "elasticity": _elasticity_record(
            arguments.elasticity_resolution, arguments.repeats
        ),
    }
    encoded = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.output is None:
        sys.stdout.write(encoded)
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, encoding="utf-8")
        print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
