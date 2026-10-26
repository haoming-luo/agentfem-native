# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""三维 T4 线性动力候选复用已验收积分器，单独验证空间与质量语义。"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .dynamics import LinearSecondOrderSystem
from .simplex import SimplexGeometry, positive
from .solid import (
    LinearElastic3DProblem,
    _constraints,
    assemble_linear_elasticity_3d_prepared,
    prepare_linear_elasticity_3d_assembly,
)
from .sparse import CSRMatrix
from .volume_mesh import TetrahedralMesh


def assemble_t4_mass(
    mesh: TetrahedralMesh, density: float, *, lumped: bool = False
) -> CSRMatrix:
    """T4 一致质量为 rho*V*(1+delta_ab)/20，集中质量为其行和。"""
    if not isinstance(mesh, TetrahedralMesh):
        raise TypeError("T4 质量需要四面体网格。")
    density = positive(density, "密度")
    if not isinstance(lumped, (bool, np.bool_)):
        raise TypeError("质量集中选项必须是布尔值。")
    g = SimplexGeometry(mesh)
    n = 3 * mesh.node_count
    if lumped:
        diagonal = g.scatter(
            np.broadcast_to(
                (density * g.volumes / 4)[:, None, None], (mesh.cell_count, 4, 3)
            ),
            3,
        )
        indices = np.arange(n)
        return CSRMatrix.from_coo((n, n), indices, indices, diagonal)
    plan = g.assembly_plan(3)
    local = np.einsum("eab,ij->eaibj", density * g.mass, np.eye(3))
    return plan.fill(local.reshape(mesh.cell_count, 12, 12))


def build_t4_linear_dynamics(
    problem: LinearElastic3DProblem,
    density: float,
    *,
    time_step: float,
    steps: int,
    initial_displacement: ArrayLike | None = None,
    initial_velocity: ArrayLike | None = None,
    lumped_mass: bool = True,
):
    """零固定约束、恒定外载；积分与反力恢复继续由通用二阶程序拥有。"""
    if not isinstance(problem, LinearElastic3DProblem):
        raise TypeError("T4 动力需要三维线弹性问题。")
    plan = prepare_linear_elasticity_3d_assembly(problem)
    stiffness, load, dofs = assemble_linear_elasticity_3d_prepared(problem, plan)
    constrained, values = _constraints(problem, dofs)
    if np.any(values != 0):
        raise ValueError("T4 动力候选只接受零位移固定约束。")
    states = []
    for value in (initial_displacement, initial_velocity):
        raw = np.zeros(dofs.size) if value is None else np.asarray(value, dtype=float)
        if raw.shape == (problem.mesh.node_count, 3):
            raw = raw.ravel()
        if raw.shape != (dofs.size,) or not np.all(np.isfinite(raw)):
            raise ValueError("T4 动力初始状态必须是有限节点向量。")
        if np.any(raw[constrained] != 0):
            raise ValueError("T4 初始状态违反零位移固定约束。")
        states.append(raw)
    return LinearSecondOrderSystem(
        stiffness,
        assemble_t4_mass(problem.mesh, density, lumped=lumped_mass),
        load,
        time_step,
        steps,
        states[0],
        states[1],
    ), constrained
