# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""相同非线性物理问题的完整数值求解对照：仅改变约束图复用。"""

import argparse
import json
import platform
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np

from agentfem_native import (
    EssentialBC,
    FiniteStrainOperator,
    NeoHookeanMaterial,
    SimplexGeometry,
    finite_strain,
    krylov,
    native_kernel_identity,
    unit_cube_tetrahedra,
    unit_square_triangles,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = []
    for dimension, resolution in ((2, 16), (3, 4), (3, 8)):
        mesh = (
            unit_square_triangles(resolution)
            if dimension == 2
            else unit_cube_tetrahedra(resolution)
        )
        transverse = (
            dimension * np.arange(mesh.node_count)[:, None] + np.arange(1, dimension)
        ).ravel()
        dofs = np.concatenate((dimension * mesh.nodes("left"), transverse))
        bc = EssentialBC(dofs, np.zeros(dofs.size))
        traction = 40 * (1.5 - 1 / 1.5) + 40 * np.log(1.5) / 1.5
        load = SimplexGeometry(mesh).mechanical_load(
            tractions={"right": [traction] + [0] * (dimension - 1)}
        )

        def solve(reuse, mesh=mesh, bc=bc, load=load):
            # 计时包含算子/稀疏图准备、全部非线性步、求解和结果复制；网格输入已给定。
            return FiniteStrainOperator(mesh, NeoHookeanMaterial(100, 0.25)).solve(
                bc, load=load, steps=4, reuse_constraint_plan=reuse
            )

        reference = solve(False)
        solve(True)
        timings = {"rebuild_each_newton": [], "reuse_per_solve": []}
        for repeat in range(5):
            routes = (False, True) if repeat % 2 == 0 else (True, False)
            for reuse in routes:
                start = perf_counter()
                result = solve(reuse)
                elapsed = perf_counter() - start
                timings["reuse_per_solve" if reuse else "rebuild_each_newton"].append(
                    elapsed
                )
                assert result.rejected_steps == reference.rejected_steps
                for actual, expected in zip(
                    result.states, reference.states, strict=True
                ):
                    assert actual.iterations == expected.iterations
                    for name in (
                        "displacements",
                        "residual",
                        "cell_jacobian",
                        "cell_cauchy",
                    ):
                        np.testing.assert_array_equal(
                            getattr(actual, name), getattr(expected, name)
                        )
                    assert actual.strain_energy == expected.strain_energy
        exact = np.zeros_like(mesh.points)
        exact[:, 0] = 0.5 * mesh.points[:, 0]
        np.testing.assert_allclose(reference.final.displacements, exact, atol=2e-8)
        cases.append(
            {
                "dimension": dimension,
                "cells": mesh.cell_count,
                "dofs": dimension * mesh.node_count,
                "accepted_steps": len(reference.states),
                "rejected_steps": reference.rejected_steps,
                "newton_iterations": sum(
                    state.iterations for state in reference.states
                ),
                "maximum_displacement_error": float(
                    np.max(np.abs(reference.final.displacements - exact))
                ),
                "trajectory_bitwise_equal": True,
                "seconds": timings,
                "median_solve_speedup": float(
                    np.median(timings["rebuild_each_newton"])
                    / np.median(timings["reuse_per_solve"])
                ),
            }
        )
    output = {
        "scope": "给定网格及节点力后的算子构建、全载荷步求解和结果；不含导入、网格生成与 I/O。",
        "comparison": "同一数学实现、物理输入、容差；仅复用约束图，数值切线不复用。",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "native_kernel": native_kernel_identity(),
        "source_sha256": {
            module.__name__: sha256(Path(module.__file__).read_bytes()).hexdigest()
            for module in (finite_strain, krylov)
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            [
                {
                    key: row[key]
                    for key in (
                        "dimension",
                        "cells",
                        "dofs",
                        "newton_iterations",
                        "median_solve_speedup",
                    )
                }
                for row in cases
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
