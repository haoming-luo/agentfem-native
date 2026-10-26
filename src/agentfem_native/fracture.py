# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""二维/三维 AT2 各向同性准静态断裂隔离候选；不含张压分裂。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .finite_strain import (
    NonlinearConvergenceError,
    NonlinearInterruptionError,
    check_accepted,
)
from .providers import LinearSolveError, NativeSparseProvider
from .runtime import ExecutionContext, NativeExecutionError
from .simplex import (
    EssentialBC,
    Mesh,
    finite_array,
    positive,
    positive_integer,
    readonly,
)
from .sparse import CSRMatrix
from .thermoelastic import ThermoelasticOperator


def box_kkt_residual(x, gradient, lower, upper) -> np.ndarray:
    """下界处梯度非负、上界处梯度非正，内部梯度为零。"""
    residual = gradient.copy()
    residual[x <= lower + 1e-13] = np.minimum(residual[x <= lower + 1e-13], 0)
    residual[x >= upper - 1e-13] = np.maximum(residual[x >= upper - 1e-13], 0)
    residual[lower == upper] = 0
    return residual


def solve_box_quadratic(
    matrix: CSRMatrix,
    rhs: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    initial: ArrayLike,
    *,
    maximum_iterations: int = 300,
    context: ExecutionContext | None = None,
):
    """可行活动集法求解凸二次子问题；截断只用于舍入保护，不代替 KKT 求解。"""
    n = matrix.shape[0]
    b = finite_array(rhs, (n,), "损伤右端")
    lo, hi = (
        finite_array(lower, (n,), "损伤下界"),
        finite_array(upper, (n,), "损伤上界"),
    )
    if np.any(lo > hi):
        raise ValueError("损伤下界不能超过上界。")
    x = np.clip(finite_array(initial, (n,), "初始损伤"), lo, hi)
    maximum_iterations = positive_integer(maximum_iterations, "活动集最大迭代")
    diagonal = matrix.diagonal()
    if np.any(diagonal <= 0):
        raise ValueError("损伤二次子问题要求正对角。")
    # d 无量纲；整体缩放 E、Gc 不应改变损伤。约束单位对角加入前先归一化。
    scale = float(np.max(diagonal))
    scaled_matrix = CSRMatrix(
        matrix.shape, matrix.indptr, matrix.indices, matrix.data / scale
    )
    threshold = max(
        1e-12 * float(np.linalg.norm(diagonal)), 1e-9 * float(np.linalg.norm(b))
    )
    ctx = context or ExecutionContext()
    for iteration in range(maximum_iterations):
        ctx.check(
            "fracture.active_set",
            iteration,
            maximum_iterations,
            enforce_step_budget=False,
        )
        gradient = matrix.matvec(x) - b
        residual = float(np.linalg.norm(box_kkt_residual(x, gradient, lo, hi)))
        if residual <= threshold:
            return x, residual, iteration
        active = (
            ((x <= lo + 1e-13) & (gradient >= 0))
            | ((x >= hi - 1e-13) & (gradient <= 0))
            | (lo == hi)
        )
        constrained = np.flatnonzero(active)
        # 内层线性精度须严于外层 KKT，尤其零右端预裂纹轮廓。
        stationary = (
            NativeSparseProvider(
                relative_tolerance=1e-14, absolute_tolerance=threshold / scale * 1e-3
            )
            .solve_constrained(scaled_matrix, b / scale, constrained, x[constrained])
            .solution
        )
        direction = stationary - x
        # 线性迭代的约束尾差不能让上下界重合的预裂纹节点把全局步长压成零。
        direction[active] = 0.0
        length = 1.0
        positive_mask, negative_mask = direction > 1e-15, direction < -1e-15
        if positive_mask.any():
            length = min(
                length,
                float(
                    np.min(
                        (hi[positive_mask] - x[positive_mask])
                        / direction[positive_mask]
                    )
                ),
            )
        if negative_mask.any():
            length = min(
                length,
                float(
                    np.min(
                        (lo[negative_mask] - x[negative_mask])
                        / direction[negative_mask]
                    )
                ),
            )
        if length <= 1e-14:
            # 耦合矩阵的 Newton 方向可能从自由节点的界限向外指。
            # 用可行投影梯度及一维精确最小化离开退化活动集，仍检查原始 KKT。
            direction = np.clip(x - gradient / diagonal, lo, hi) - x
            curvature = float(direction @ matrix.matvec(direction))
            if curvature <= 0:
                raise LinearSolveError("损伤二次子问题失去正曲率。")
            length = min(1.0, -float(gradient @ direction) / curvature)
        next_x = np.clip(x + length * direction, lo, hi)
        if np.array_equal(next_x, x):
            raise LinearSolveError("损伤活动集停滞且 KKT 未闭合。")
        x = next_x
    raise LinearSolveError("损伤活动集达到迭代上限，未返回未经验证的截断解。")


@dataclass(frozen=True)
class FractureState:
    load_factor: float
    displacements: np.ndarray
    damage: np.ndarray
    residual: np.ndarray
    elastic_energy: float
    fracture_energy: float
    potential_energy: float
    damage_kkt_residual: float
    free_residual_norm: float
    alternating_iterations: int
    alternating_energies: np.ndarray

    def __post_init__(self) -> None:
        for name in ("displacements", "damage", "residual", "alternating_energies"):
            object.__setattr__(self, name, readonly(getattr(self, name)))


@dataclass(frozen=True)
class FractureResult:
    states: tuple[FractureState, ...]
    maturity: str = "experimental"
    limitation: str = "各向同性无张压分裂 AT2；压缩也会损伤，不含裂面接触或动态裂纹。"

    @property
    def final(self) -> FractureState:
        return self.states[-1]


class AT2FractureOperator:
    """共享弹性与标量图，显式拥有长度尺度和断裂韧度。"""

    def __init__(
        self,
        mesh: Mesh,
        young_modulus: float,
        poisson_ratio: float,
        *,
        toughness: float,
        length_scale: float,
        residual_stiffness: float = 1e-8,
        mode: str = "plane_strain",
        thickness: float = 1.0,
    ) -> None:
        self.elastic = ThermoelasticOperator(
            mesh, young_modulus, poisson_ratio, mode=mode, thickness=thickness
        )
        self.geometry = g = self.elastic.geometry
        self.toughness = positive(toughness, "断裂韧度")
        self.length_scale = positive(length_scale, "正则化长度")
        self.residual_stiffness = positive(residual_stiffness, "残余刚度")
        if self.residual_stiffness >= 1:
            raise ValueError("残余刚度必须小于一。")
        self.scalar_plan = g.assembly_plan()
        self.local_mass = readonly(g.mass)
        self.local_gradient = readonly(
            g.volumes[:, None, None]
            * np.einsum("eai,ebi->eab", g.gradients, g.gradients)
        )
        self.surface_matrix = self.scalar_plan.fill(
            self.toughness / self.length_scale * self.local_mass
            + self.toughness * self.length_scale * self.local_gradient
        )

    def degradation(self, damage: ArrayLike) -> np.ndarray:
        mesh = self.geometry.mesh
        values = finite_array(damage, (mesh.node_count,), "节点损伤")
        if np.any(values < 0) or np.any(values > 1):
            raise ValueError("损伤必须位于 [0,1]。")
        intact = 1 - values[mesh.cells]
        return (
            np.einsum("ea,eab,eb->e", intact, self.local_mass, intact)
            / self.geometry.volumes
            + self.residual_stiffness
        )

    def damage_system(self, cell_energy: ArrayLike):
        g, mesh = self.geometry, self.geometry.mesh
        psi = finite_array(cell_energy, (mesh.cell_count,), "未退化弹性能密度")
        if np.any(psi < 0):
            raise ValueError("AT2 驱动能必须非负。")
        local = (2 * psi + self.toughness / self.length_scale)[
            :, None, None
        ] * self.local_mass
        local += self.toughness * self.length_scale * self.local_gradient
        rhs = g.scatter(
            np.broadcast_to(
                (2 * psi * g.volumes / (g.dimension + 1))[:, None], mesh.cells.shape
            )
        )
        return self.scalar_plan.fill(local), rhs

    def solve(
        self,
        bc: EssentialBC,
        *,
        load_factors: ArrayLike = (0.25, 0.5, 0.75, 1.0),
        load: ArrayLike | None = None,
        initial_damage: ArrayLike | None = None,
        maximum_alternations: int = 200,
        tolerance: float = 1e-8,
        context: ExecutionContext | None = None,
    ) -> FractureResult:
        """允许卸载；只有位移平衡与损伤 KKT 同时满足时才提交整步。"""
        g, mesh = self.geometry, self.geometry.mesh
        g.check_rigid_constraints(bc)
        factors = np.asarray(load_factors, dtype=float)
        if (
            factors.ndim != 1
            or factors.size == 0
            or not np.all(np.isfinite(factors))
            or np.any(factors < 0)
        ):
            raise ValueError("载荷因子必须是一维有限非负非空序列。")
        maximum_alternations = positive_integer(maximum_alternations, "交替迭代上限")
        tolerance = positive(tolerance, "交替迭代容差")
        accepted_damage = (
            np.zeros(mesh.node_count)
            if initial_damage is None
            else finite_array(initial_damage, (mesh.node_count,), "初始损伤").copy()
        )
        self.degradation(accepted_damage)
        force = (
            np.zeros(mesh.node_count * g.dimension)
            if load is None
            else finite_array(load, (mesh.node_count, g.dimension), "节点外力").ravel()
        )
        free = np.setdiff1d(np.arange(force.size), bc.dofs)
        ctx = context or ExecutionContext()
        ctx.check("fracture.prepare", 0, factors.size)
        states = []
        for step, factor in enumerate(factors):
            check_accepted(ctx, "fracture.step", step, factors.size, states)
            damage = accepted_damage.copy()
            energies = []
            previous_u = None
            converged = False
            try:
                for iteration in range(1, maximum_alternations + 1):
                    ctx.check("fracture.alternate", step, factors.size)
                    degradation = self.degradation(damage)
                    stiffness = self.elastic.plan.fill(
                        self.elastic.local_stiffness * degradation[:, None, None]
                    )
                    u = (
                        NativeSparseProvider(block_size=g.dimension)
                        .solve_constrained(
                            stiffness, factor * force, bc.dofs, factor * bc.values
                        )
                        .solution
                    )
                    psi = self.elastic.fields(u.reshape(mesh.node_count, g.dimension))[
                        2
                    ]
                    damage_matrix, damage_rhs = self.damage_system(psi)
                    next_damage, kkt, _ = solve_box_quadratic(
                        damage_matrix,
                        damage_rhs,
                        accepted_damage,
                        np.ones(mesh.node_count),
                        damage,
                        context=ctx,
                    )
                    next_degradation = self.degradation(next_damage)
                    next_stiffness = self.elastic.plan.fill(
                        self.elastic.local_stiffness * next_degradation[:, None, None]
                    )
                    residual = next_stiffness.matvec(u) - factor * force
                    residual_norm = float(np.linalg.norm(residual[free]))
                    elastic_energy = float(g.volumes @ (next_degradation * psi))
                    fracture_energy = float(
                        0.5 * next_damage @ self.surface_matrix.matvec(next_damage)
                    )
                    potential = (
                        elastic_energy + fracture_energy - float(factor * force @ u)
                    )
                    energies.append(potential)
                    if len(energies) > 1 and energies[-1] > energies[-2] + 1e-10 * max(
                        1, abs(energies[-2])
                    ):
                        raise LinearSolveError("AT2 交替最小化能量上升，拒绝该增量。")
                    delta_d = np.linalg.norm(next_damage - damage, ord=np.inf)
                    delta_u = (
                        np.inf
                        if previous_u is None
                        else np.linalg.norm(u - previous_u)
                        / max(1.0, np.linalg.norm(u))
                    )
                    damage = next_damage
                    previous_u = u
                    scale = max(
                        float(np.linalg.norm(factor * force)),
                        float(np.linalg.norm(next_stiffness.matvec(u))),
                        1e-12,
                    )
                    if (
                        delta_d <= tolerance
                        and delta_u <= tolerance
                        and residual_norm <= max(1e-10, tolerance * scale)
                    ):
                        converged = True
                        break
            except NativeExecutionError as error:
                raise NonlinearInterruptionError(error, tuple(states)) from error
            except LinearSolveError as error:
                raise NonlinearConvergenceError(str(error), tuple(states)) from error
            if not converged:
                raise NonlinearConvergenceError(
                    "AT2 交替迭代未收敛；未提交损伤或位移试探。", tuple(states)
                )
            accepted_damage = damage.copy()
            states.append(
                FractureState(
                    float(factor),
                    u.reshape(mesh.node_count, g.dimension),
                    damage,
                    residual,
                    elastic_energy,
                    fracture_energy,
                    potential,
                    kkt,
                    residual_norm,
                    iteration,
                    np.asarray(energies),
                )
            )
        check_accepted(ctx, "fracture.complete", factors.size, factors.size, states)
        return FractureResult(tuple(states))
