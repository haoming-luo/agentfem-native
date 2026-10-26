# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""自主重启 GMRES：为非正定 Newton 切线提供有限内存稀疏路径。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .providers import LinearSolveError
from .runtime import ExecutionContext
from .simplex import EssentialBC, finite_array, positive, positive_integer, readonly
from .sparse import CSRConstraintPlan, CSRMatrix


@dataclass(frozen=True)
class GMRESResult:
    solution: np.ndarray
    iterations: int
    residual_norm: float
    threshold: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "solution", readonly(self.solution))


def gmres(
    matrix: CSRMatrix,
    rhs: ArrayLike,
    *,
    restart: int = 40,
    maximum_iterations: int = 1000,
    relative_tolerance: float = 1e-10,
    absolute_tolerance: float = 1e-13,
    context: ExecutionContext | None = None,
) -> GMRESResult:
    """双次正交化 Arnoldi；收敛必须通过原始、未预条件化真实残差复核。"""
    n = matrix.shape[0]
    if matrix.shape != (n, n) or n < 1:
        raise ValueError("GMRES 需要非空方阵。")
    b = finite_array(rhs, (n,), "线性右端")
    restart = min(positive_integer(restart, "GMRES 重启维数"), n)
    maximum_iterations = positive_integer(maximum_iterations, "GMRES 最大迭代")
    rtol = positive(relative_tolerance, "GMRES 相对容差", allow_zero=True)
    atol = positive(absolute_tolerance, "GMRES 绝对容差", allow_zero=True)
    threshold = max(atol, rtol * float(np.linalg.norm(b)))
    diagonal = np.abs(matrix.diagonal())
    # 绝对对角缩放不要求切线正定；零对角保持单位缩放。
    scale = np.where(diagonal > np.finfo(float).tiny, diagonal, 1.0)
    x = np.zeros(n)
    iterations = 0
    ctx = context or ExecutionContext()
    while iterations < maximum_iterations:
        ctx.check(
            "krylov.gmres", iterations, maximum_iterations, enforce_step_budget=False
        )
        residual = b - matrix.matvec(x)
        norm = float(np.linalg.norm(residual))
        if norm <= threshold:
            return GMRESResult(x, iterations, norm, threshold)
        r = residual / scale
        beta = np.linalg.norm(r)
        if not np.isfinite(beta) or beta == 0:
            raise LinearSolveError("GMRES 预条件残差退化。")
        width = min(restart, maximum_iterations - iterations)
        basis, hessenberg = np.zeros((n, width + 1)), np.zeros((width + 1, width))
        basis[:, 0] = r / beta
        target = np.zeros(width + 1)
        target[0] = beta
        for j in range(width):
            ctx.check(
                "krylov.gmres",
                iterations,
                maximum_iterations,
                enforce_step_budget=False,
            )
            vector = matrix.matvec(basis[:, j]) / scale
            original_norm = np.linalg.norm(vector)
            for _ in range(2):
                coefficients = basis[:, : j + 1].T @ vector
                hessenberg[: j + 1, j] += coefficients
                vector -= basis[:, : j + 1] @ coefficients
            hessenberg[j + 1, j] = np.linalg.norm(vector)
            breakdown = hessenberg[j + 1, j] <= 64 * np.finfo(float).eps * original_norm
            if not breakdown:
                basis[:, j + 1] = vector / hessenberg[j + 1, j]
            weights = np.linalg.lstsq(
                hessenberg[: j + 2, : j + 1], target[: j + 2], rcond=None
            )[0]
            candidate = x + basis[:, : j + 1] @ weights
            iterations += 1
            norm = float(np.linalg.norm(b - matrix.matvec(candidate)))
            if norm <= threshold:
                return GMRESResult(candidate, iterations, norm, threshold)
            if breakdown:
                raise LinearSolveError(
                    f"GMRES Krylov 子空间退化，真实残差 {norm:.6e}。"
                )
        x = candidate
    raise LinearSolveError(
        f"GMRES 达到迭代上限，真实残差 {norm:.6e}，要求 {threshold:.6e}。"
    )


def solve_tangent(
    matrix: CSRMatrix,
    rhs: ArrayLike,
    bc: EssentialBC,
    *,
    context: ExecutionContext | None = None,
) -> GMRESResult:
    """强约束保持对称，但一般切线允许不定。"""
    bc.validate(matrix.shape[0])
    plan = CSRConstraintPlan.from_matrix(matrix, bc.dofs)
    return gmres(
        plan.transform_matrix(matrix),
        plan.transform_rhs(matrix, rhs, bc.values),
        maximum_iterations=max(200, matrix.shape[0] * 5),
        context=context,
    )
