# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""记录 T3 规范稀疏图一次构建、多次数值回填的成本边界。"""

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
)
from agentfem_native.mesh import unit_square_triangles
from agentfem_native.native import native_kernel_identity
from agentfem_native.sparse import CSRMatrix, CSRPattern


def _median(operation: Callable[[], object], repetitions: int) -> float:
    operation()
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolutions", type=int, nargs="+", default=(32, 64, 128))
    parser.add_argument("--repetitions", type=int, default=9)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if any(value < 1 for value in arguments.resolutions):
        parser.error("--resolutions 必须全部为正整数")
    if arguments.repetitions < 2:
        parser.error("--repetitions 必须至少为 2")

    cases = []
    for resolution in arguments.resolutions:
        mesh = unit_square_triangles(resolution)
        coo, _, _ = assemble_linear_elasticity(
            LinearElasticProblem(
                mesh,
                LinearElasticMaterial(210.0e9, 0.3),
                assembly_mode="native",
                thread_count=4,
            )
        )
        build_seconds = _median(
            lambda coo=coo: CSRPattern.from_coo(coo.shape, coo.rows, coo.columns),
            arguments.repetitions,
        )
        pattern = CSRPattern.from_coo(coo.shape, coo.rows, coo.columns)
        canonical_seconds = _median(
            lambda coo=coo: CSRMatrix.from_coo(
                coo.shape, coo.rows, coo.columns, coo.data
            ),
            arguments.repetitions,
        )
        production_refill_seconds = _median(
            lambda pattern=pattern, coo=coo: pattern.fill(coo.data),
            arguments.repetitions,
        )
        reference_refill_seconds = _median(
            lambda pattern=pattern, coo=coo: pattern.fill_reference(coo.data),
            arguments.repetitions,
        )
        cold = CSRMatrix.from_coo(coo.shape, coo.rows, coo.columns, coo.data)
        refilled = pattern.fill(coo.data)
        reference = pattern.fill_reference(coo.data)
        cases.append(
            {
                "resolution": resolution,
                "cells": mesh.cell_count,
                "dofs": 2 * mesh.node_count,
                "coo_entry_count": coo.data.size,
                "csr_nnz": pattern.nnz,
                "pattern_storage_bytes": pattern.storage_nbytes,
                "pattern_build_median_seconds": build_seconds,
                "cold_canonicalization_median_seconds": canonical_seconds,
                "production_refill_median_seconds": production_refill_seconds,
                "reference_refill_median_seconds": reference_refill_seconds,
                "production_speedup_over_reference_refill": (
                    reference_refill_seconds / production_refill_seconds
                ),
                "refill_speedup_over_cold_canonicalization": (
                    canonical_seconds / production_refill_seconds
                ),
                "graph_exactly_equal": bool(
                    np.array_equal(cold.indptr, refilled.indptr)
                    and np.array_equal(cold.indices, refilled.indices)
                ),
                "data_exactly_equal": bool(np.array_equal(cold.data, refilled.data)),
                "production_reference_data_exactly_equal": bool(
                    np.array_equal(reference.data, refilled.data)
                ),
                "structure_digest": pattern.structure_digest,
            }
        )

    record = {
        "schema": "agentfem-native.sparse-pattern-reuse/0.2",
        "environment": {
            "agentfem_native": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "os": platform.system(),
            "architecture": platform.machine(),
            "native_kernel": native_kernel_identity(),
        },
        "cases": cases,
        "repetitions": arguments.repetitions,
        "claim_boundary": (
            "只测量已有 COO 到规范 CSR 的冷转换与复用图数值回填；"
            "不包含单元装配、约束或求解，不能解释为端到端加速。"
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
