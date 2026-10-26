# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""T3 平面应变/T4 可压缩 Neo-Hookean 总拉格朗日隔离候选。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .krylov import solve_tangent
from .materials import NeoHookeanMaterial, _elastic_parameters
from .providers import LinearSolveError
from .runtime import ExecutionContext, NativeExecutionError
from .simplex import (
    EssentialBC,
    Mesh,
    SimplexGeometry,
    cell_values,
    finite_array,
    positive,
    positive_integer,
    readonly,
)
from .sparse import CSRConstraintPlan


def neo_hookean_tangent(
    material: NeoHookeanMaterial, deformation_gradient: ArrayLike
) -> np.ndarray:
    """解析 dP_iJ/dF_kL，材料能量与项目已有材料点公式完全一致。"""
    f = finite_array(deformation_gradient, (3, 3), "变形梯度")
    material.response(f)  # 共用正 Jacobian 与有限性准入规则。
    mu, lame = _elastic_parameters(material.young_modulus, material.poisson_ratio)
    inverse = np.linalg.inv(f).T
    return (
        mu * np.einsum("ik,jl->ijkl", np.eye(3), np.eye(3))
        + lame * np.einsum("ij,kl->ijkl", inverse, inverse)
        + (mu - lame * np.log(np.linalg.det(f)))
        * np.einsum("il,kj->ijkl", inverse, inverse)
    )


@dataclass(frozen=True)
class FiniteStrainState:
    load_factor: float
    displacements: np.ndarray
    residual: np.ndarray
    cell_jacobian: np.ndarray
    cell_cauchy: np.ndarray
    strain_energy: float
    iterations: int
    free_residual_norm: float
    residual_threshold: float

    def __post_init__(self) -> None:
        for name in ("displacements", "residual", "cell_jacobian", "cell_cauchy"):
            object.__setattr__(self, name, readonly(getattr(self, name)))


@dataclass(frozen=True)
class FiniteStrainResult:
    states: tuple[FiniteStrainState, ...]
    rejected_steps: int
    maturity: str = "experimental"

    @property
    def final(self) -> FiniteStrainState:
        return self.states[-1]


class NonlinearConvergenceError(NativeExecutionError):
    """只暴露已接受状态；不是已经具备身份校验的持久化重启协议。"""

    def __init__(self, message: str, accepted_states: tuple, rejected_steps: int = 0):
        super().__init__(
            message,
            code="solver.nonlinear_nonconvergence",
            path="procedure",
            retryable=True,
            remediations=("reduce_increment", "inspect_constraints"),
        )
        self.accepted_states = accepted_states
        self.rejected_steps = rejected_steps


class NonlinearInterruptionError(NativeExecutionError):
    """取消/预算中断保留只读接受历史，原始机器错误码不变。"""

    def __init__(self, error: NativeExecutionError, states: tuple):
        super().__init__(
            "计算已在安全检查点中断，仅保留已接受状态。"
            if error.code == "execution.cancelled"
            else str(error),
            code=error.code,
            path=error.path,
            retryable=error.retryable,
            remediations=("inspect_accepted_states", "rerun_request"),
        )
        self.accepted_states = states


def check_accepted(context, stage, completed, total, states):
    """检查点可位于试探内部，但对外只返回已提交状态。"""
    try:
        context.check(stage, completed, total)
    except NativeExecutionError as error:
        raise NonlinearInterruptionError(error, tuple(states)) from error


class FiniteStrainOperator:
    """固定参考几何与 CSR 图；每次试探只生成数值、内力和材料输出。"""

    def __init__(
        self,
        mesh: Mesh,
        material: NeoHookeanMaterial,
        *,
        thickness: float = 1.0,
        thermal_stretch: ArrayLike = 1.0,
    ) -> None:
        self.geometry = SimplexGeometry(mesh, thickness)
        self.material = material
        self.plan = self.geometry.assembly_plan(self.geometry.dimension)
        stretch = cell_values(thermal_stretch, mesh.cell_count, "各向同性热伸长")
        if np.any(stretch <= 0):
            raise ValueError("热伸长必须为正。")
        self.thermal_stretch = readonly(stretch)

    def assemble(self, displacement: ArrayLike, *, tangent: bool = True):
        """向量化材料计算；局部切线按节点—分量顺序直接回填既有稀疏图。"""
        g, mesh = self.geometry, self.geometry.mesh
        d = g.dimension
        u = finite_array(displacement, (mesh.node_count, d), "有限变形位移")
        f = np.broadcast_to(np.eye(3), (mesh.cell_count, 3, 3)).copy()
        f[:, :d, :d] += np.einsum("eai,eaj->eij", u[mesh.cells], g.gradients)
        jacobian = np.linalg.det(f)
        if not np.all(np.isfinite(jacobian)) or np.any(jacobian <= 0):
            raise ValueError("有限变形试探产生非正 Jacobian，必须拒绝该试探。")
        s = self.thermal_stretch
        fe = f / s[:, None, None]
        inverse = np.linalg.inv(fe).swapaxes(1, 2)
        logj = np.log(jacobian / s**3)
        mu, lame = _elastic_parameters(
            self.material.young_modulus, self.material.poisson_ratio
        )
        pe = mu * (fe - inverse) + lame * logj[:, None, None] * inverse
        p = pe / s[:, None, None]
        energy = (
            0.5 * mu * (np.einsum("eij,eij->e", fe, fe) - 3)
            - mu * logj
            + 0.5 * lame * logj**2
        )
        local_force = g.volumes[:, None, None] * np.einsum(
            "eij,eaj->eai", p[:, :d, :d], g.gradients
        )
        force = g.scatter(local_force, d)
        cauchy = np.einsum("eij,ekj->eik", p, f) / jacobian[:, None, None]
        matrix = None
        if tangent:
            inv = inverse[:, :d, :d]
            a = mu * np.einsum("ik,jl->ijkl", np.eye(d), np.eye(d))[None, ...]
            a = a + lame * np.einsum("eij,ekl->eijkl", inv, inv)
            a += (mu - lame * logj)[:, None, None, None, None] * np.einsum(
                "eil,ekj->eijkl", inv, inv
            )
            a /= s[:, None, None, None, None] ** 2
            local = np.einsum(
                "eaj,eijkl,ebl,e->eaibk", g.gradients, a, g.gradients, g.volumes
            )
            matrix = self.plan.fill(
                local.reshape(mesh.cell_count, (d + 1) * d, (d + 1) * d)
            )
        return matrix, force, float(g.volumes @ energy), jacobian, cauchy

    def solve(
        self,
        bc: EssentialBC,
        *,
        load: ArrayLike | None = None,
        steps: int = 8,
        maximum_iterations: int = 30,
        maximum_cutbacks: int = 8,
        maximum_accepted_steps: int = 512,
        relative_tolerance: float = 1e-8,
        absolute_tolerance: float = 1e-10,
        context: ExecutionContext | None = None,
        reuse_constraint_plan: bool = True,
    ) -> FiniteStrainResult:
        """静态比例加载，残差线搜索和失败增量二分；接受后才追加状态。"""
        g, mesh = self.geometry, self.geometry.mesh
        if not isinstance(reuse_constraint_plan, bool):
            raise TypeError("约束计划复用开关必须是布尔值。")
        g.check_rigid_constraints(bc)
        steps = positive_integer(steps, "载荷步数")
        maximum_iterations = positive_integer(maximum_iterations, "Newton 最大迭代")
        maximum_cutbacks = positive_integer(maximum_cutbacks, "最大减步次数")
        maximum_accepted_steps = positive_integer(
            maximum_accepted_steps, "最大接受步数"
        )
        if steps > maximum_accepted_steps:
            raise ValueError("初始载荷步数不能超过最大接受步数。")
        rtol = positive(relative_tolerance, "Newton 相对容差")
        atol = positive(absolute_tolerance, "Newton 绝对容差")
        force = (
            np.zeros(mesh.node_count * g.dimension)
            if load is None
            else finite_array(load, (mesh.node_count, g.dimension), "节点外力").ravel()
        )
        free = np.setdiff1d(np.arange(force.size), bc.dofs)
        zero_bc = EssentialBC(bc.dofs, np.zeros(bc.dofs.size))
        current = np.zeros((mesh.node_count, g.dimension))
        factor, increment, rejected = 0.0, 1 / steps, 0
        accepted: list[FiniteStrainState] = []
        constraint_plan = None
        ctx = context or ExecutionContext()
        check_accepted(ctx, "finite_strain.prepare", 0, steps, accepted)
        while factor < 1 - 8 * np.finfo(float).eps:
            if len(accepted) >= maximum_accepted_steps:
                raise NonlinearConvergenceError(
                    "已达到最大接受步数；保留已接受状态。", tuple(accepted), rejected
                )
            target = min(1.0, factor + increment)
            check_accepted(
                ctx,
                "finite_strain.trial",
                len(accepted),
                max(steps, len(accepted) + 1),
                accepted,
            )
            trial = current.copy()
            trial.ravel()[bc.dofs] = target * bc.values
            threshold = None
            converged = False
            try:
                for iteration in range(maximum_iterations + 1):
                    ctx.check("finite_strain.newton", 0, steps)
                    matrix, internal, energy, jacobian, cauchy = self.assemble(trial)
                    residual = internal - target * force
                    norm = float(np.linalg.norm(residual[free]))
                    if threshold is None:
                        threshold = max(
                            atol,
                            rtol
                            * max(norm, float(np.linalg.norm(target * force[free]))),
                        )
                    if norm <= threshold:
                        converged = True
                        break
                    if iteration == maximum_iterations:
                        break
                    if reuse_constraint_plan and constraint_plan is None:
                        constraint_plan = CSRConstraintPlan.from_matrix(
                            matrix, zero_bc.dofs
                        )
                    delta = solve_tangent(
                        matrix,
                        -residual,
                        zero_bc,
                        context=ctx,
                        constraint_plan=constraint_plan,
                    ).solution.reshape(trial.shape)
                    length = 1.0
                    for _ in range(20):
                        candidate = trial + length * delta
                        candidate.ravel()[bc.dofs] = target * bc.values
                        try:
                            candidate_residual = (
                                self.assemble(candidate, tangent=False)[1]
                                - target * force
                            )
                            candidate_norm = np.linalg.norm(candidate_residual[free])
                        except ValueError:
                            candidate_norm = np.inf
                        if candidate_norm <= (1 - 1e-4 * length) * norm:
                            trial = candidate
                            break
                        length *= 0.5
                    else:
                        break
            except NativeExecutionError as error:
                raise NonlinearInterruptionError(error, tuple(accepted)) from error
            except (ValueError, LinearSolveError):
                converged = False
            if not converged:
                rejected += 1
                increment *= 0.5
                if rejected > maximum_cutbacks or increment < np.finfo(float).eps:
                    raise NonlinearConvergenceError(
                        "有限变形增量未收敛；已保留最后接受状态。",
                        tuple(accepted),
                        rejected,
                    )
                continue
            current, factor = trial, target
            accepted.append(
                FiniteStrainState(
                    factor,
                    current,
                    residual,
                    jacobian,
                    cauchy,
                    energy,
                    iteration,
                    norm,
                    threshold,
                )
            )
        check_accepted(
            ctx, "finite_strain.complete", len(accepted), len(accepted), accepted
        )
        return FiniteStrainResult(tuple(accepted), rejected)
