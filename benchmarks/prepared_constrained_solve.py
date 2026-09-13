# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""测量 T3/T4 固定刚度多载荷的冷求解与预备求解生命周期。"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np

from agentfem_native import __version__
from agentfem_native.elasticity import (
    LinearElasticMaterial,
    LinearElasticProblem,
    assemble_linear_elasticity_prepared,
    prepare_linear_elasticity_assembly,
)
from agentfem_native.mesh import unit_square_triangles
from agentfem_native.native import native_kernel_identity
from agentfem_native.providers import NativeSparseProvider, NativeSparseSolvePlan
from agentfem_native.solid import (
    LinearElastic3DProblem,
    SolidElasticMaterial,
    assemble_linear_elasticity_3d_prepared,
    prepare_linear_elasticity_3d_assembly,
)
from agentfem_native.sparse import CSRMatrix
from agentfem_native.volume_mesh import unit_cube_tetrahedra


def _median(operation: Callable[[], object], repetitions: int) -> float:
    operation()
    samples: list[float] = []
    for _ in range(repetitions):
        start = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def _fixed_dofs(nodes: np.ndarray, component_count: int) -> np.ndarray:
    dofs = (
        nodes[:, None] * component_count
        + np.arange(component_count, dtype=np.int64)[None, :]
    )
    return np.sort(dofs.ravel())


def _cold_batch(
    provider: NativeSparseProvider,
    matrix: CSRMatrix,
    loads: tuple[np.ndarray, ...],
    constrained: np.ndarray,
    values: np.ndarray,
) -> None:
    for load in loads:
        provider.solve_constrained(matrix, load, constrained, values)


def _prepared_batch(
    solve_plan: NativeSparseSolvePlan,
    loads: tuple[np.ndarray, ...],
    values: np.ndarray,
) -> None:
    for load in loads:
        solve_plan.solve(load, values)


def _equivalence(
    matrix: CSRMatrix,
    loads: tuple[np.ndarray, ...],
    constrained: np.ndarray,
    provider: NativeSparseProvider,
    component_count: int,
) -> tuple[dict[str, float | bool], NativeSparseSolvePlan]:
    values = np.zeros(constrained.size, dtype=np.float64)
    plan = provider.prepare_constrained(matrix, constrained)
    max_solution_difference = 0.0
    max_free_residual = 0.0
    max_free_residual_relative = 0.0
    max_energy_difference = 0.0
    max_balance_norm = 0.0
    max_balance_relative = 0.0
    max_reported_residual_ratio = 0.0
    free = np.setdiff1d(np.arange(matrix.shape[0]), constrained)
    for load in loads:
        cold = provider.solve_constrained(matrix, load, constrained, values)
        prepared = plan.solve(load, values)
        assert prepared.convergence is not None
        max_solution_difference = max(
            max_solution_difference,
            float(np.max(np.abs(cold.solution - prepared.solution))),
        )
        cold_energy = float(0.5 * cold.solution @ matrix.matvec(cold.solution))
        prepared_energy = float(
            0.5 * prepared.solution @ matrix.matvec(prepared.solution)
        )
        max_energy_difference = max(
            max_energy_difference, abs(cold_energy - prepared_energy)
        )
        residual = matrix.matvec(prepared.solution) - load
        max_free_residual = max(
            max_free_residual, float(np.linalg.norm(residual[free]))
        )
        load_norm = max(float(np.linalg.norm(load)), 1.0)
        max_free_residual_relative = max(
            max_free_residual_relative,
            float(np.linalg.norm(residual[free])) / load_norm,
        )
        max_reported_residual_ratio = max(
            max_reported_residual_ratio,
            prepared.convergence.residual_norm / prepared.convergence.threshold,
        )
        total_applied = load.reshape(-1, component_count).sum(axis=0)
        balance = residual.reshape(-1, component_count).sum(axis=0) + total_applied
        max_balance_norm = max(
            max_balance_norm,
            float(np.linalg.norm(balance)),
        )
        max_balance_relative = max(
            max_balance_relative,
            float(np.linalg.norm(balance))
            / max(float(np.linalg.norm(total_applied)), 1.0),
        )
    return (
        {
            "solutions_match": max_solution_difference <= 1.0e-11,
            "max_solution_absolute_difference": max_solution_difference,
            "max_free_residual_norm": max_free_residual,
            "max_free_residual_relative": max_free_residual_relative,
            "max_strain_energy_absolute_difference": max_energy_difference,
            "max_global_balance_norm": max_balance_norm,
            "max_global_balance_relative": max_balance_relative,
            "max_reported_residual_to_threshold": max_reported_residual_ratio,
        },
        plan,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t3-resolutions", nargs="+", type=int, default=(12, 24))
    parser.add_argument("--t4-resolutions", nargs="+", type=int, default=(2, 4))
    parser.add_argument("--load-count", type=int, default=6)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.load_count < 2:
        parser.error("--load-count 必须至少为 2")
    if arguments.repetitions < 2:
        parser.error("--repetitions 必须至少为 2")
    if any(
        value < 1 for value in (*arguments.t3_resolutions, *arguments.t4_resolutions)
    ):
        parser.error("全部网格分辨率必须为正整数")

    factors = np.linspace(0.5, 1.5, arguments.load_count)
    cases: list[dict[str, object]] = []
    for element, resolutions in (
        ("T3", arguments.t3_resolutions),
        ("T4", arguments.t4_resolutions),
    ):
        for resolution in resolutions:
            if element == "T3":
                mesh = unit_square_triangles(resolution)
                problem = LinearElasticProblem(
                    mesh,
                    LinearElasticMaterial(210.0e9, 0.3),
                    body_force=(2.0e5, -3.0e5),
                    assembly_mode="native",
                    thread_count=4,
                )
                assembly_plan = prepare_linear_elasticity_assembly(problem)
                matrix, base_load, _ = assemble_linear_elasticity_prepared(
                    problem, assembly_plan
                )
                component_count = 2
                constrained = _fixed_dofs(mesh.nodes("left"), component_count)
            else:
                mesh = unit_cube_tetrahedra(resolution)
                problem = LinearElastic3DProblem(
                    mesh,
                    SolidElasticMaterial(210.0e9, 0.3),
                    body_force=(2.0e5, -3.0e5, 1.0e5),
                    thread_count=4,
                )
                assembly_plan = prepare_linear_elasticity_3d_assembly(problem)
                matrix, base_load, _ = assemble_linear_elasticity_3d_prepared(
                    problem, assembly_plan
                )
                component_count = 3
                constrained = _fixed_dofs(mesh.nodes("left"), component_count)

            loads = tuple(factor * base_load for factor in factors)
            values = np.zeros(constrained.size, dtype=np.float64)
            provider = NativeSparseProvider(block_size=component_count)
            preparation_seconds = _median(
                partial(provider.prepare_constrained, matrix, constrained),
                arguments.repetitions,
            )
            equivalence, solve_plan = _equivalence(
                matrix, loads, constrained, provider, component_count
            )

            cold_seconds = _median(
                partial(_cold_batch, provider, matrix, loads, constrained, values),
                arguments.repetitions,
            )
            prepared_seconds = _median(
                partial(_prepared_batch, solve_plan, loads, values),
                arguments.repetitions,
            )
            cases.append(
                {
                    "element": element,
                    "resolution": resolution,
                    "cells": mesh.cell_count,
                    "dofs": matrix.shape[0],
                    "csr_nnz": matrix.nnz,
                    "constraint_count": constrained.size,
                    "load_count": arguments.load_count,
                    "plan_storage_bytes_upper_bound": solve_plan.storage_nbytes,
                    "preparation_median_seconds": preparation_seconds,
                    "cold_batch_median_seconds": cold_seconds,
                    "prepared_batch_median_seconds": prepared_seconds,
                    "steady_state_speedup": cold_seconds / prepared_seconds,
                    "amortized_speedup_including_one_preparation": cold_seconds
                    / (prepared_seconds + preparation_seconds),
                    **equivalence,
                    "structure_digest": solve_plan.structure_digest,
                }
            )

    record = {
        "schema": "agentfem-native.prepared-constrained-solve/0.1",
        "environment": {
            "agentfem_native": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "os": platform.system(),
            "architecture": platform.machine(),
            "native_kernel": native_kernel_identity(),
        },
        "load_factors": factors.tolist(),
        "repetitions": arguments.repetitions,
        "cases": cases,
        "claim_boundary": (
            "固定已装配 CSR 刚度、固定约束集合，测量多个右端项的约束变换、"
            "块 Jacobi 和 CG；不包含元素装配、非线性更新、MPI、GPU 或直接法。"
        ),
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
