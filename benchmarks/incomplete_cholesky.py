# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""IC0 有界筛选：迭代减少必须与准备和完整多载荷总成本同时报告。"""

import argparse
import platform
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np

import agentfem_native.experimental_preconditioners as experimental
from agentfem_native.experimental_preconditioners import IncompleteCholesky
from agentfem_native.jsonio import write_json_atomic
from agentfem_native.mesh import unit_square_triangles
from agentfem_native.native import native_kernel_identity
from agentfem_native.sparse import (
    CSRConstraintPlan,
    PreparedPreconditioner,
    conjugate_gradient,
)
from agentfem_native.thermoelastic import ThermoelasticOperator
from agentfem_native.volume_mesh import unit_cube_tetrahedra


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--native-only",
        action="store_true",
        help="只比较 Jacobi 与自有 C++ IC0，不要求可选 SciPy",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("重复次数必须为正。")
    # 可选后端先热身导入；此处不以一次性导入成本评价稳态数值路径。
    scipy_version = None
    if not args.native_only:
        import scipy
        import scipy.sparse.linalg

        scipy_version = scipy.__version__

    records = []
    labels = (
        ("jacobi", "ic0_native")
        if args.native_only
        else ("jacobi", "ic0_reference", "ic0_scipy")
    )
    for dimension, resolution in ((2, 16), (2, 48), (3, 8)):
        mesh = (
            unit_square_triangles(resolution)
            if dimension == 2
            else unit_cube_tetrahedra(resolution)
        )
        stiffness = ThermoelasticOperator(mesh, 100, 0.25).stiffness
        fixed = (dimension * mesh.nodes("left")[:, None] + np.arange(dimension)).ravel()
        matrix = CSRConstraintPlan.from_matrix(stiffness, fixed).transform_matrix(
            stiffness
        )
        loads = []
        for direction in range(dimension):
            rhs = np.zeros(matrix.shape[0])
            rhs[dimension * mesh.nodes("right") + direction] = 1 / len(
                mesh.nodes("right")
            )
            loads.append(rhs)

        def make(label, matrix=matrix):
            if label == "jacobi":
                return PreparedPreconditioner.from_matrix(matrix)
            return IncompleteCholesky(
                matrix,
                shift=0.1,
                backend=label.removeprefix("ic0_"),
            )

        def solve(prepared, matrix=matrix, loads=loads):
            answers = [
                conjugate_gradient(matrix, rhs, prepared_preconditioner=prepared)
                for rhs in loads
            ]
            assert all(
                a.report.converged and a.report.residual_norm <= a.report.threshold
                for a in answers
            )
            return answers

        baseline = solve(make("jacobi"))
        samples = {
            label: {"prepare": [], "solve_batch": [], "total": []} for label in labels
        }
        iterations, differences, bytes_owned = {}, {}, {}
        for repeat in range(args.repetitions):
            for label in labels if repeat % 2 == 0 else labels[::-1]:
                start = perf_counter()
                prepared = make(label)
                middle = perf_counter()
                answers = solve(prepared)
                end = perf_counter()
                samples[label]["prepare"].append(middle - start)
                samples[label]["solve_batch"].append(end - middle)
                samples[label]["total"].append(end - start)
                iterations[label] = [a.report.iterations for a in answers]
                bytes_owned[label] = prepared.storage_nbytes
                maximum = 0.0
                for actual, reference in zip(answers, baseline, strict=True):
                    np.testing.assert_allclose(
                        actual.solution, reference.solution, rtol=1e-8, atol=1e-10
                    )
                    energy = float(actual.solution @ matrix.matvec(actual.solution))
                    expected_energy = float(
                        reference.solution @ matrix.matvec(reference.solution)
                    )
                    np.testing.assert_allclose(
                        energy, expected_energy, rtol=1e-9, atol=1e-11
                    )
                    maximum = max(
                        maximum,
                        float(np.max(np.abs(actual.solution - reference.solution))),
                    )
                differences[label] = maximum
        record = {
            "dimension": dimension,
            "cells": mesh.cell_count,
            "dofs": matrix.shape[0],
            "right_hand_sides": len(loads),
            "ic0_shift": 0.1,
            "seconds": samples,
            "iterations": iterations,
            "maximum_displacement_difference": differences,
            "held_array_bytes_excluding_optional_provider": bytes_owned,
            "median_total_speedup_against_jacobi": {
                label: float(
                    np.median(samples["jacobi"]["total"])
                    / np.median(samples[label]["total"])
                )
                for label in labels
            },
        }
        records.append(record)
        print(
            dimension,
            matrix.shape[0],
            record["median_total_speedup_against_jacobi"],
            iterations,
            flush=True,
        )
    write_json_atomic(
        {
            "scope": "给定规范约束 CSR 与各向节点载荷后，预条件准备加全部右端项 CG；排除网格、装配及导入。交错测量，不作统计置信度宣称。",
            "repetitions": args.repetitions,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy_version,
            "source_sha256": sha256(
                Path(experimental.__file__).read_bytes()
            ).hexdigest(),
            "native_kernel": native_kernel_identity(),
            "native_source_sha256": sha256(
                (
                    Path(__file__).resolve().parents[1]
                    / "native_spikes/cpp20/src/p1_batch.cpp"
                ).read_bytes()
            ).hexdigest(),
            "maturity": "experimental",
            "cases": records,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
