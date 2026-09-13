# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""测量 T3/T4 从元素装配到规范 CSR 的冷路径与预备图路径。"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from agentfem_native import __version__
from agentfem_native.elasticity import (
    LinearElasticMaterial,
    LinearElasticProblem,
    assemble_linear_elasticity,
    assemble_linear_elasticity_prepared,
    prepare_linear_elasticity_assembly,
)
from agentfem_native.mesh import unit_square_triangles
from agentfem_native.native import native_kernel_identity
from agentfem_native.solid import (
    LinearElastic3DProblem,
    SolidElasticMaterial,
    assemble_linear_elasticity_3d,
    assemble_linear_elasticity_3d_prepared,
    prepare_linear_elasticity_3d_assembly,
)
from agentfem_native.sparse import CSRMatrix
from agentfem_native.volume_mesh import unit_cube_tetrahedra


def _median(operation: Callable[[], object], repetitions: int) -> float:
    operation()
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def _cold_t3(problem: LinearElasticProblem) -> CSRMatrix:
    coo, _, _ = assemble_linear_elasticity(problem)
    return CSRMatrix.from_coo(coo.shape, coo.rows, coo.columns, coo.data)


def _cold_t4(problem: LinearElastic3DProblem) -> CSRMatrix:
    coo, _, _ = assemble_linear_elasticity_3d(problem)
    return CSRMatrix.from_coo(coo.shape, coo.rows, coo.columns, coo.data)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t3-resolutions", nargs="+", type=int, default=(16, 32, 64))
    parser.add_argument("--t4-resolutions", nargs="+", type=int, default=(2, 4, 8))
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.repetitions < 2:
        parser.error("--repetitions 必须至少为 2")
    if any(
        value < 1 for value in (*arguments.t3_resolutions, *arguments.t4_resolutions)
    ):
        parser.error("全部网格分辨率必须为正整数")

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
                    body_force=(0.3, -0.2),
                    assembly_mode="native",
                    thread_count=4,
                )
                plan = prepare_linear_elasticity_assembly(problem)
                cold = lambda problem=problem: _cold_t3(problem)
                prepared = lambda problem=problem, plan=plan: (
                    assemble_linear_elasticity_prepared(problem, plan)[0]
                )
                cell_count = mesh.cell_count
                dof_count = 2 * mesh.node_count
            else:
                mesh = unit_cube_tetrahedra(resolution)
                problem = LinearElastic3DProblem(
                    mesh,
                    SolidElasticMaterial(210.0e9, 0.3),
                    body_force=(0.3, -0.2, 0.1),
                    thread_count=4,
                )
                plan = prepare_linear_elasticity_3d_assembly(problem)
                cold = lambda problem=problem: _cold_t4(problem)
                prepared = lambda problem=problem, plan=plan: (
                    assemble_linear_elasticity_3d_prepared(problem, plan)[0]
                )
                cell_count = mesh.cell_count
                dof_count = 3 * mesh.node_count

            cold_seconds = _median(cold, arguments.repetitions)
            prepared_seconds = _median(prepared, arguments.repetitions)
            cold_matrix = cold()
            prepared_matrix = prepared()
            cases.append(
                {
                    "element": element,
                    "resolution": resolution,
                    "cells": cell_count,
                    "dofs": dof_count,
                    "coo_entry_count": plan.contribution_count,
                    "csr_nnz": plan.nnz,
                    "plan_storage_bytes": plan.storage_nbytes,
                    "cold_assembly_to_csr_median_seconds": cold_seconds,
                    "prepared_assembly_to_csr_median_seconds": prepared_seconds,
                    "prepared_speedup": cold_seconds / prepared_seconds,
                    "graph_exactly_equal": bool(
                        np.array_equal(cold_matrix.indptr, prepared_matrix.indptr)
                        and np.array_equal(cold_matrix.indices, prepared_matrix.indices)
                    ),
                    "data_exactly_equal": bool(
                        np.array_equal(cold_matrix.data, prepared_matrix.data)
                    ),
                    "structure_digest": plan.structure_digest,
                }
            )

    record = {
        "schema": "agentfem-native.prepared-mechanics-assembly/0.1",
        "environment": {
            "agentfem_native": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "os": platform.system(),
            "architecture": platform.machine(),
            "native_kernel": native_kernel_identity(),
        },
        "repetitions": arguments.repetitions,
        "cases": cases,
        "claim_boundary": (
            "测量元素装配到规范 CSR 的冷路径与复用图路径；两者仍生成 COO 索引缓冲，"
            "不包含约束、预条件器或线性求解。"
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
