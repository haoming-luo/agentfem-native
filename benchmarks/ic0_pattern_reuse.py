# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""同图材料扫描：包含首轮结构分析、每轮数值分解与完整 CG，不含装配。"""

import argparse
import platform
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np

import agentfem_native.experimental_preconditioners as implementation
from agentfem_native.experimental_preconditioners import (
    IncompleteCholesky,
    IncompleteCholeskyPattern,
)
from agentfem_native.jsonio import write_json_atomic
from agentfem_native.mesh import unit_square_triangles
from agentfem_native.native import native_kernel_identity
from agentfem_native.sparse import CSRConstraintPlan, conjugate_gradient
from agentfem_native.thermoelastic import ThermoelasticOperator
from agentfem_native.volume_mesh import unit_cube_tetrahedra


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("重复次数必须为正。")
    records = []
    materials = [(100, 0.2), (150, 0.25), (70, 0.3), (250, 0.15)]
    for dimension, resolution in ((2, 16), (2, 48), (3, 8)):
        mesh = (
            unit_square_triangles(resolution)
            if dimension == 2
            else unit_cube_tetrahedra(resolution)
        )
        fixed = (dimension * mesh.nodes("left")[:, None] + np.arange(dimension)).ravel()
        matrices = []
        for young, poisson in materials:
            matrix = ThermoelasticOperator(mesh, young, poisson).stiffness
            matrices.append(
                CSRConstraintPlan.from_matrix(matrix, fixed).transform_matrix(matrix)
            )
        rhs = np.zeros(matrices[0].shape[0])
        rhs[dimension * mesh.nodes("right")] = 1 / len(mesh.nodes("right"))

        def run(reuse, matrices=matrices, rhs=rhs):
            start = perf_counter()
            pattern = IncompleteCholeskyPattern(matrices[0]) if reuse else None
            structure_time = perf_counter() - start
            prepare_time = structure_time
            answers, iterations = [], []
            for matrix in matrices:
                before = perf_counter()
                prepared = IncompleteCholesky(
                    matrix, shift=0.1, backend="native", pattern=pattern
                )
                prepare_time += perf_counter() - before
                answer = conjugate_gradient(
                    matrix, rhs, prepared_preconditioner=prepared
                )
                if (
                    not answer.report.converged
                    or answer.report.residual_norm > answer.report.threshold
                ):
                    raise AssertionError("材料扫描未满足真残差门槛。")
                answers.append(answer.solution)
                iterations.append(answer.report.iterations)
            return (
                answers,
                iterations,
                {
                    "total": perf_counter() - start,
                    "prepare": prepare_time,
                    "structure": structure_time,
                },
            )

        baseline, baseline_iterations, _ = run(False)
        run(True)  # 两路线均执行一次预热，不计入后续交错确认。
        samples = {"fresh": [], "reuse": []}
        max_difference = 0.0
        for repeat in range(args.repetitions):
            for reuse in (False, True) if repeat % 2 == 0 else (True, False):
                answers, iterations, timing = run(reuse)
                if iterations != baseline_iterations:
                    raise AssertionError("同操作顺序的结构复用改变了迭代数。")
                for matrix, answer, expected in zip(
                    matrices, answers, baseline, strict=True
                ):
                    np.testing.assert_allclose(answer, expected, rtol=1e-12, atol=1e-14)
                    np.testing.assert_allclose(
                        answer @ matrix.matvec(answer),
                        expected @ matrix.matvec(expected),
                        rtol=1e-12,
                        atol=1e-14,
                    )
                    max_difference = max(
                        max_difference, float(np.max(np.abs(answer - expected)))
                    )
                samples["reuse" if reuse else "fresh"].append(timing)
        medians = {
            label: {
                key: float(np.median([sample[key] for sample in values]))
                for key in ("total", "prepare")
            }
            for label, values in samples.items()
        }
        records.append(
            {
                "dimension": dimension,
                "resolution": resolution,
                "dofs": matrices[0].shape[0],
                "materials": materials,
                "iterations": baseline_iterations,
                "samples": samples,
                "medians": medians,
                "speedup": medians["fresh"]["total"] / medians["reuse"]["total"],
                "max_displacement_difference": max_difference,
                "pattern_nbytes": IncompleteCholeskyPattern(matrices[0]).storage_nbytes,
            }
        )
    write_json_atomic(
        {
            "scope": __doc__,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "native": native_kernel_identity(),
            "shift": 0.1,
            "repetitions": args.repetitions,
            "source_sha256": sha256(
                Path(implementation.__file__).read_bytes()
            ).hexdigest(),
            "benchmark_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "cases": records,
        },
        args.output,
    )
    for record in records:
        print(record["dofs"], record["speedup"], record["max_displacement_difference"])


if __name__ == "__main__":
    main()
