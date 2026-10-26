# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""多物理候选的小规模差分与交错计时；只声明当前机器上的装配阶段证据。"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np

from agentfem_native.finite_strain import FiniteStrainOperator, neo_hookean_tangent
from agentfem_native.materials import NeoHookeanMaterial
from agentfem_native.mesh import unit_square_triangles
from agentfem_native.native import native_kernel_identity
from agentfem_native.volume_mesh import unit_cube_tetrahedra


def reference_assembly(operator, displacement):
    """逐单元可读参考：独立材料点与显式局部收缩，不执行全局求解。"""
    g, mesh = operator.geometry, operator.geometry.mesh
    d = g.dimension
    local = np.zeros((mesh.cell_count, (d + 1) * d, (d + 1) * d))
    local_force = np.zeros((mesh.cell_count, d + 1, d))
    energy = 0.0
    for e, cell in enumerate(mesh.cells):
        f = np.eye(3)
        f[:d, :d] += displacement[cell].T @ g.gradients[e]
        response = operator.material.response(f)
        a = neo_hookean_tangent(operator.material, f)
        for i in range(d + 1):
            local_force[e, i] = (
                g.volumes[e] * response.first_piola[:d, :d] @ g.gradients[e, i]
            )
            for j in range(d + 1):
                local[e, i * d : (i + 1) * d, j * d : (j + 1) * d] = g.volumes[
                    e
                ] * np.einsum(
                    "j,ijkl,l->ik",
                    g.gradients[e, i],
                    a[:d, :d, :d, :d],
                    g.gradients[e, j],
                )
        energy += g.volumes[e] * response.energy_density
    return operator.plan.fill(local), g.scatter(local_force, d), energy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = []
    for dimension, resolution in ((2, 8), (2, 16), (2, 24), (3, 2), (3, 3), (3, 4)):
        mesh = (
            unit_square_triangles(resolution)
            if dimension == 2
            else unit_cube_tetrahedra(resolution)
        )
        operator = FiniteStrainOperator(mesh, NeoHookeanMaterial(100, 0.25))
        # 非均匀但正 Jacobian 的位移；每次同一物理状态，明确属于装配微基准。
        u = 0.05 * mesh.points + 0.002 * np.sin(2 * np.pi * mesh.points)
        reference = reference_assembly(operator, u)
        candidate = operator.assemble(u)
        np.testing.assert_array_equal(candidate[0].indptr, reference[0].indptr)
        np.testing.assert_array_equal(candidate[0].indices, reference[0].indices)
        np.testing.assert_allclose(
            candidate[0].data, reference[0].data, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(candidate[1], reference[1], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(candidate[2], reference[2], rtol=1e-12, atol=1e-12)
        timings = {"reference": [], "vectorized": []}
        for repeat in range(5):
            order = (
                ("reference", "vectorized")
                if repeat % 2 == 0
                else ("vectorized", "reference")
            )
            for route in order:
                start = perf_counter()
                if route == "reference":
                    reference_assembly(operator, u)
                else:
                    operator.assemble(u)
                timings[route].append(perf_counter() - start)
        cases.append(
            {
                "dimension": dimension,
                "resolution": resolution,
                "cells": mesh.cell_count,
                "dofs": dimension * mesh.node_count,
                "csr_bytes": candidate[0].storage_nbytes,
                "prepared_graph_bytes": operator.plan.storage_nbytes,
                "matrix_max_abs_difference": float(
                    np.max(np.abs(candidate[0].data - reference[0].data))
                ),
                "force_max_abs_difference": float(
                    np.max(np.abs(candidate[1] - reference[1]))
                ),
                "energy_abs_difference": float(abs(candidate[2] - reference[2])),
                "seconds": timings,
                "median_phase_speedup": float(
                    np.median(timings["reference"]) / np.median(timings["vectorized"])
                ),
            }
        )
    output = {
        "scope": "固定参考几何和 CSR 图后的有限变形装配；不含全局 Newton、构图或 I/O。",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "native_kernel": native_kernel_identity(),
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
                    k: row[k]
                    for k in (
                        "dimension",
                        "cells",
                        "dofs",
                        "median_phase_speedup",
                        "matrix_max_abs_difference",
                    )
                }
                for row in cases
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
