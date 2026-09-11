# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Reproducible local evidence for the compressed Mechanics Alpha slice."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from agentfem_native import __version__
from agentfem_native.dynamics import LinearSecondOrderSystem, integrate_linear_dynamics
from agentfem_native.elasticity import (
    LinearElasticMaterial,
    LinearElasticProblem,
    assemble_linear_elasticity,
)
from agentfem_native.mesh import unit_square_triangles
from agentfem_native.native import csr_spmv, native_kernel_identity
from agentfem_native.solid import (
    LinearElastic3DProblem,
    SolidDisplacementCondition,
    SolidElasticMaterial,
    SolidTractionCondition,
    assemble_linear_elasticity_3d,
    solve_linear_elasticity_3d,
)
from agentfem_native.sparse import BSRMatrix, CSRMatrix
from agentfem_native.volume_mesh import unit_cube_tetrahedra


def _median(callable_, repetitions: int) -> tuple[float, object]:
    timings = []
    result = None
    for _ in range(repetitions):
        start = time.perf_counter()
        result = callable_()
        timings.append(time.perf_counter() - start)
    return statistics.median(timings), result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    mesh = unit_square_triangles(arguments.resolution)
    material = LinearElasticMaterial(210.0e9, 0.3)
    modes = {}
    matrices = {}
    loads = {}
    for mode in ("reference", "vectorized", "native"):
        problem = LinearElasticProblem(
            mesh,
            material,
            thickness=0.01,
            body_force=(100.0, -250.0),
            assembly_mode=mode,
        )
        median, assembled = _median(
            lambda problem=problem: assemble_linear_elasticity(problem),
            arguments.repetitions,
        )
        matrix, load, _ = assembled
        canonical = CSRMatrix.from_coo(
            matrix.shape, matrix.rows, matrix.columns, matrix.data
        )
        modes[mode] = {"median_seconds": median}
        matrices[mode] = canonical
        loads[mode] = load
    reference = matrices["reference"]
    blocked = BSRMatrix.from_csr(matrices["native"], 2)
    vector = np.linspace(-1.0, 1.0, reference.shape[0])
    reference_scale = float(np.max(np.abs(reference.data)))
    reference_image = matrices["native"].matvec(vector)
    csr_reference_median, csr_reference_image = _median(
        lambda: matrices["native"].matvec_reference(vector), arguments.repetitions
    )
    csr_native_median, csr_native_image = _median(
        lambda: csr_spmv(
            matrices["native"].shape,
            matrices["native"].indptr,
            matrices["native"].indices,
            matrices["native"].data,
            vector,
        ),
        arguments.repetitions,
    )
    t4_mesh = unit_cube_tetrahedra(3)
    t4_problem = LinearElastic3DProblem(
        t4_mesh,
        SolidElasticMaterial(100.0, 0.25),
        dirichlet=(
            SolidDisplacementCondition("left", "x", 0.0),
            SolidDisplacementCondition("front", "y", 0.0),
            SolidDisplacementCondition("bottom", "z", 0.0),
        ),
        traction=(SolidTractionCondition("right", (1.0, 0.0, 0.0)),),
    )
    t4_assembly = {}
    t4_matrices = {}
    t4_loads = {}
    for mode in ("reference", "native"):
        median, assembled = _median(
            lambda mode=mode: assemble_linear_elasticity_3d(t4_problem, assembly=mode),
            arguments.repetitions,
        )
        matrix, load, _ = assembled
        t4_assembly[mode] = {"median_seconds": median}
        t4_matrices[mode] = CSRMatrix.from_coo(
            matrix.shape, matrix.rows, matrix.columns, matrix.data
        )
        t4_loads[mode] = load
    t4_solve_median, t4_result = _median(
        lambda: solve_linear_elasticity_3d(t4_problem), arguments.repetitions
    )
    one = CSRMatrix.from_coo((1, 1), [0], [0], [1.0])
    four = CSRMatrix.from_coo((1, 1), [0], [0], [4.0])
    dynamic = LinearSecondOrderSystem(four, one, [0.0], 0.001, 10_000, [1.0], [0.0])
    dynamic_median, dynamic_result = _median(
        lambda: integrate_linear_dynamics(dynamic), arguments.repetitions
    )
    record = {
        "schema": "agentfem-native.mechanics-alpha-benchmark/0.2",
        "environment": {
            "agentfem_native": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "os": platform.system(),
            "architecture": platform.machine(),
            "native_kernel": native_kernel_identity(),
        },
        "t3": {
            "resolution": arguments.resolution,
            "nodes": mesh.node_count,
            "cells": mesh.cell_count,
            "dofs": 2 * mesh.node_count,
            "timings": modes,
            "native_speedup_over_reference": modes["reference"]["median_seconds"]
            / modes["native"]["median_seconds"],
            "native_max_matrix_difference": float(
                np.max(np.abs(matrices["native"].data - reference.data))
            ),
            "native_relative_matrix_difference": float(
                np.max(np.abs(matrices["native"].data - reference.data))
                / reference_scale
            ),
            "vectorized_max_matrix_difference": float(
                np.max(np.abs(matrices["vectorized"].data - reference.data))
            ),
            "vectorized_relative_matrix_difference": float(
                np.max(np.abs(matrices["vectorized"].data - reference.data))
                / reference_scale
            ),
            "native_max_load_difference": float(
                np.max(np.abs(loads["native"] - loads["reference"]))
            ),
            "csr_bytes": matrices["native"].storage_nbytes,
            "bsr_bytes": blocked.storage_nbytes,
            "bsr_matvec_difference": float(
                np.max(np.abs(blocked.matvec(vector) - reference_image))
            ),
            "bsr_relative_matvec_difference": float(
                np.max(np.abs(blocked.matvec(vector) - reference_image))
                / np.max(np.abs(reference_image))
            ),
            "csr_spmv": {
                "reference_median_seconds": csr_reference_median,
                "native_median_seconds": csr_native_median,
                "native_speedup_over_reference": (
                    csr_reference_median / csr_native_median
                ),
                "maximum_difference": float(
                    np.max(np.abs(csr_native_image - csr_reference_image))
                ),
            },
        },
        "t4": {
            "nodes": t4_mesh.node_count,
            "cells": t4_mesh.cell_count,
            "dofs": 3 * t4_mesh.node_count,
            "assembly_timings": t4_assembly,
            "native_assembly_speedup_over_reference": (
                t4_assembly["reference"]["median_seconds"]
                / t4_assembly["native"]["median_seconds"]
            ),
            "native_max_matrix_difference": float(
                np.max(
                    np.abs(t4_matrices["native"].data - t4_matrices["reference"].data)
                )
            ),
            "native_max_load_difference": float(
                np.max(np.abs(t4_loads["native"] - t4_loads["reference"]))
            ),
            "solve_median_seconds": t4_solve_median,
            "free_residual_norm": t4_result.free_residual_norm,
            "balance_norm": float(
                np.linalg.norm(t4_result.total_applied_force + t4_result.total_reaction)
            ),
        },
        "dynamics": {
            "steps": dynamic.steps,
            "median_seconds": dynamic_median,
            "maximum_energy_drift": float(
                np.max(
                    np.abs(dynamic_result.total_energy - dynamic_result.total_energy[0])
                )
            ),
        },
        "repetitions": arguments.repetitions,
        "claim_boundary": "Local warm-process medians; no cross-platform performance claim.",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(arguments.output)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
