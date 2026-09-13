# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""记录 T3/T4 确定性 CPU 并行装配的吞吐与数值边界。"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from agentfem_native import __version__
from agentfem_native.elasticity import LinearElasticMaterial
from agentfem_native.mesh import unit_square_triangles
from agentfem_native.native import (
    assemble_t3_volume,
    assemble_t4_volume,
    native_kernel_identity,
)
from agentfem_native.solid import SolidElasticMaterial
from agentfem_native.volume_mesh import unit_cube_tetrahedra

Assembly = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _measure(
    operation: Callable[[], Assembly], repetitions: int
) -> tuple[float, Assembly, bool]:
    operation()  # 预热不计时，减少首次页映射对小型内核的干扰。
    samples: list[float] = []
    first: Assembly | None = None
    repeatable = True
    for _ in range(repetitions):
        start = time.perf_counter()
        result = operation()
        samples.append(time.perf_counter() - start)
        if first is None:
            first = result
        else:
            repeatable = repeatable and all(
                np.array_equal(left, right)
                for left, right in zip(first, result, strict=True)
            )
    assert first is not None
    return statistics.median(samples), first, repeatable


def _thread_sweep(
    operation: Callable[[int], Assembly],
    thread_counts: tuple[int, ...],
    repetitions: int,
) -> dict[str, object]:
    timings: dict[int, float] = {}
    results: dict[int, Assembly] = {}
    repeatable: dict[int, bool] = {}
    for thread_count in thread_counts:
        median, result, deterministic = _measure(
            lambda thread_count=thread_count: operation(thread_count), repetitions
        )
        timings[thread_count] = median
        results[thread_count] = result
        repeatable[thread_count] = deterministic

    serial = results[1]
    records: dict[str, object] = {}
    for thread_count in thread_counts:
        result = results[thread_count]
        records[str(thread_count)] = {
            "median_seconds": timings[thread_count],
            "speedup_over_serial": timings[1] / timings[thread_count],
            "coo_exactly_equal": all(
                np.array_equal(result[index], serial[index]) for index in range(3)
            ),
            "maximum_load_difference": float(np.max(np.abs(result[3] - serial[3]))),
            "repeatable_for_fixed_thread_count": repeatable[thread_count],
        }
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t3-resolution", type=int, default=256)
    parser.add_argument("--t4-resolution", type=int, default=12)
    parser.add_argument("--thread-counts", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    thread_counts = tuple(dict.fromkeys(arguments.thread_counts))
    if (
        not thread_counts
        or thread_counts[0] != 1
        or any(count < 1 for count in thread_counts)
    ):
        parser.error("--thread-counts 必须以 1 开头且全部为正整数")
    if arguments.repetitions < 2:
        parser.error("--repetitions 必须至少为 2")

    t3_mesh = unit_square_triangles(arguments.t3_resolution)
    t3_constitutive = np.broadcast_to(
        LinearElasticMaterial(210.0e9, 0.3).constitutive_matrix,
        (t3_mesh.cell_count, 3, 3),
    )
    t3 = _thread_sweep(
        lambda count: assemble_t3_volume(
            t3_mesh.points,
            t3_mesh.cells,
            t3_constitutive,
            np.array((100.0, -250.0)),
            0.01,
            thread_count=count,
        ),
        thread_counts,
        arguments.repetitions,
    )

    t4_mesh = unit_cube_tetrahedra(arguments.t4_resolution)
    t4_constitutive = np.broadcast_to(
        SolidElasticMaterial(100.0, 0.25).constitutive_matrix,
        (t4_mesh.cell_count, 6, 6),
    )
    t4 = _thread_sweep(
        lambda count: assemble_t4_volume(
            t4_mesh.points,
            t4_mesh.cells,
            t4_constitutive,
            np.array((1.0, -2.0, 0.5)),
            thread_count=count,
        ),
        thread_counts,
        arguments.repetitions,
    )

    record = {
        "schema": "agentfem-native.cpu-parallel-assembly/0.1",
        "environment": {
            "agentfem_native": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "os": platform.system(),
            "architecture": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "native_kernel": native_kernel_identity(),
        },
        "t3": {
            "resolution": arguments.t3_resolution,
            "nodes": t3_mesh.node_count,
            "cells": t3_mesh.cell_count,
            "dofs": 2 * t3_mesh.node_count,
            "threads": t3,
        },
        "t4": {
            "resolution": arguments.t4_resolution,
            "nodes": t4_mesh.node_count,
            "cells": t4_mesh.cell_count,
            "dofs": 3 * t4_mesh.node_count,
            "threads": t4,
        },
        "repetitions": arguments.repetitions,
        "claim_boundary": (
            "本机热进程中位数，仅证明当前 macOS arm64 候选的方向；"
            "跨平台正确性由 Tier-1 CI 验收，不从本记录推断跨平台速度。"
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
