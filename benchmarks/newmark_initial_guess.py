# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""给定同一稀疏动力系统，比较冷/热初猜的完整积分执行和接受历史。"""

import argparse
import platform
from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np

import agentfem_native.dynamics as dynamics_module
import agentfem_native.sparse as sparse_module
from agentfem_native.dynamics import (
    build_t3_linear_dynamics,
    integrate_constrained_linear_dynamics,
)
from agentfem_native.elasticity import (
    DisplacementCondition,
    LinearElasticMaterial,
    LinearElasticProblem,
)
from agentfem_native.jsonio import write_json_atomic
from agentfem_native.mesh import unit_square_triangles
from agentfem_native.solid import (
    LinearElastic3DProblem,
    SolidDisplacementCondition,
    SolidElasticMaterial,
)
from agentfem_native.solid_dynamics import build_t4_linear_dynamics
from agentfem_native.sparse import PreparedPreconditioner, conjugate_gradient
from agentfem_native.volume_mesh import unit_cube_tetrahedra


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("initial_guess", "norm_cache"), default="initial_guess"
    )
    args = parser.parse_args()
    labels = (
        ("scan_each_rhs", "reuse_snapshot_norm")
        if args.mode == "norm_cache"
        else ("cold", "warm")
    )
    records = []
    for dimension, resolution, dt in (
        (2, 24, 0.001),
        (3, 5, 0.001),
        (3, 8, 0.001),
        (3, 8, 0.01),
    ):
        mesh = (
            unit_square_triangles(resolution)
            if dimension == 2
            else unit_cube_tetrahedra(resolution)
        )
        initial = np.zeros_like(mesh.points)
        initial[:, 0] = 0.001 * mesh.points[:, 0]
        if dimension == 2:
            problem = LinearElasticProblem(
                mesh,
                LinearElasticMaterial(100, 0.25),
                dirichlet=(DisplacementCondition("left", None, (0, 0)),),
            )
            builder = build_t3_linear_dynamics
        else:
            problem = LinearElastic3DProblem(
                mesh,
                SolidElasticMaterial(100, 0.25),
                dirichlet=(SolidDisplacementCondition("left", None, (0, 0, 0)),),
            )
            builder = build_t4_linear_dynamics
        system, fixed = builder(
            problem,
            2,
            time_step=dt,
            steps=100,
            initial_displacement=initial,
            lumped_mass=False,
        )

        def solve(warm, system=system, fixed=fixed):
            def rescan(prepared):
                sums = np.zeros(prepared.shape[0])
                rows = np.repeat(np.arange(prepared.shape[0]), np.diff(prepared.indptr))
                np.add.at(sums, rows, np.abs(prepared.matrix_data))
                return float(np.max(sums)) if sums.size else 0.0

            intervention = (
                patch.object(PreparedPreconditioner, "infinity_norm", rescan)
                if args.mode == "norm_cache" and not warm
                else nullcontext()
            )
            with intervention:
                return integrate_constrained_linear_dynamics(
                    system,
                    fixed,
                    method="newmark_average_acceleration",
                    warm_start=warm if args.mode == "initial_guess" else False,
                )

        baseline, candidate = solve(False), solve(True)
        differences = {}
        for name in (
            "displacement",
            "velocity",
            "acceleration",
            "reaction",
            "kinetic_energy",
            "strain_energy",
            "external_work",
            "energy_balance_error",
        ):
            a, b = getattr(candidate, name), getattr(baseline, name)
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10, err_msg=name)
            if args.mode == "norm_cache":
                np.testing.assert_array_equal(a, b)
            differences[name] = float(np.max(np.abs(a - b)))
        samples = {label: [] for label in labels}
        for repeat in range(5):
            for warm in (False, True) if repeat % 2 == 0 else (True, False):
                start = perf_counter()
                solve(warm)
                samples[labels[int(warm)]].append(perf_counter() - start)
        # 另行观测迭代数，不把 Python mock 插桩成本混入性能计时。
        iterations = {}
        for warm in (False, True):
            reports = []

            def monitored(*args, reports=reports, **kwargs):
                outcome = conjugate_gradient(*args, **kwargs)
                assert outcome.report.converged
                assert outcome.report.residual_norm <= outcome.report.threshold
                reports.append(outcome.report)
                return outcome

            with patch(
                "agentfem_native.dynamics.conjugate_gradient", side_effect=monitored
            ):
                solve(warm)
            iterations[labels[int(warm)]] = {
                "initial_mass": reports[0].iterations,
                "effective_system_total": sum(r.iterations for r in reports[1:]),
            }
        records.append(
            {
                "dimension": dimension,
                "resolution": resolution,
                "dofs": mesh.node_count * dimension,
                "cells": mesh.cell_count,
                "time_step": dt,
                "steps": system.steps,
                "seconds": samples,
                "median_speedup": float(
                    np.median(samples[labels[0]]) / np.median(samples[labels[1]])
                ),
                "iterations": iterations,
                "maximum_differences": differences,
                "maximum_relative_energy_balance": float(
                    np.max(np.abs(candidate.energy_balance_error))
                    / candidate.total_energy[0]
                ),
            }
        )
    write_json_atomic(
        {
            "scope": "已给定相同完整 K/M/载荷/初值后的约束消元、预条件准备、100 步积分与反力/历史恢复；不含导入、网格和装配。",
            "strategy": "同版本冷初猜；干预组每个 RHS 重新扫描范数，对照组复用快照范数，不关闭身份验证。"
            if args.mode == "norm_cache"
            else "Newmark 初猜仅改为上一已接受加速度，默认仍为冷初猜；不改变方程/容差。",
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "dynamics_source_sha256": sha256(
                Path(dynamics_module.__file__).read_bytes()
            ).hexdigest(),
            "sparse_source_sha256": sha256(
                Path(sparse_module.__file__).read_bytes()
            ).hexdigest(),
            "cases": records,
        },
        args.output,
    )
    for case in records:
        print(
            case["dimension"],
            case["dofs"],
            case["time_step"],
            case["median_speedup"],
            case["iterations"],
        )


if __name__ == "__main__":
    main()
