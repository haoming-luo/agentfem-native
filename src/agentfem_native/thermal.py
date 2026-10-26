# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""二维/三维单纯形稳态与后向 Euler 热传导候选。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .providers import NativeSparseProvider
from .runtime import ExecutionContext
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
from .sparse import CSRMatrix
from .time_grid import time_grid, validate_time_interval


def conductivity_tensors(value: ArrayLike, count: int, dimension: int) -> np.ndarray:
    """执行与预检共享导热张量准入；仅接受舍入尺度的非对称误差。"""
    raw = np.asarray(value, dtype=np.float64)
    if raw.ndim == 0 or raw.shape == (count,):
        tensor = cell_values(raw, count, "导热系数")[:, None, None] * np.eye(dimension)
    elif raw.shape == (dimension, dimension):
        tensor = np.broadcast_to(raw, (count, dimension, dimension))
    else:
        tensor = finite_array(raw, (count, dimension, dimension), "逐单元导热张量")
    scale = np.max(np.abs(tensor), axis=(1, 2))
    asymmetry = np.max(np.abs(tensor - tensor.swapaxes(1, 2)), axis=(1, 2))
    if not np.all(np.isfinite(tensor)) or np.any(
        asymmetry > 64 * np.finfo(float).eps * scale
    ):
        raise ValueError("导热张量必须有限且对称（仅允许相对 64 ε 的舍入差）。")
    tensor = 0.5 * tensor + 0.5 * tensor.swapaxes(1, 2)
    if np.any(np.linalg.eigvalsh(tensor) <= 0):
        raise ValueError("导热张量必须正定。")
    return tensor


@dataclass(frozen=True)
class ThermalBoundary:
    """正 inward_flux 为入热；convection 为非负换热系数。"""

    boundary_set: str
    inward_flux: float = 0.0
    convection: float = 0.0
    ambient_temperature: float = 0.0


@dataclass(frozen=True)
class ThermalResult:
    temperatures: np.ndarray
    times: np.ndarray
    residuals: np.ndarray
    cell_flux: np.ndarray
    stored_heat: np.ndarray
    balance_errors: np.ndarray
    constrained_dofs: np.ndarray
    maturity: str = "implemented"

    def __post_init__(self) -> None:
        for name in (
            "temperatures",
            "times",
            "residuals",
            "cell_flux",
            "stored_heat",
            "balance_errors",
            "constrained_dofs",
        ):
            object.__setattr__(
                self,
                name,
                readonly(
                    getattr(self, name),
                    np.int64 if name == "constrained_dofs" else np.float64,
                ),
            )


class ThermalOperator:
    """显式拥有固定热算子，允许时间推进复用图和数值预条件器。"""

    def __init__(
        self,
        mesh: Mesh,
        conductivity: ArrayLike = 1.0,
        capacity: ArrayLike = 1.0,
        *,
        source: ArrayLike = 0.0,
        boundaries: tuple[ThermalBoundary, ...] = (),
        thickness: float = 1.0,
    ) -> None:
        self.geometry = geometry = SimplexGeometry(mesh, thickness)
        d, count = geometry.dimension, mesh.cell_count
        tensor = conductivity_tensors(conductivity, count, d)
        capacity_values = cell_values(capacity, count, "体积热容")
        if np.any(capacity_values <= 0):
            raise ValueError("体积热容必须为正。")
        self.conductivity = readonly(tensor)
        plan = geometry.assembly_plan()
        local = geometry.volumes[:, None, None] * np.einsum(
            "eai,eij,ebj->eab", geometry.gradients, tensor, geometry.gradients
        )
        stiffness = plan.fill(local)
        self.capacity = plan.fill(geometry.mass * capacity_values[:, None, None])
        load = geometry.scatter(
            np.broadcast_to(
                (geometry.volumes * cell_values(source, count, "体热源") / (d + 1))[
                    :, None
                ],
                (count, d + 1),
            )
        )
        rows, cols, data = (
            [stiffness.row_indices()],
            [stiffness.indices],
            [stiffness.data],
        )
        self.has_convection = False
        self.convection_components: set[int] = set()
        for boundary in boundaries:
            h = positive(boundary.convection, "换热系数", allow_zero=True)
            q, ambient = (
                float(boundary.inward_flux),
                float(boundary.ambient_temperature),
            )
            if not np.isfinite(q) or not np.isfinite(ambient):
                raise ValueError("热边界值必须有限。")
            facets, area = geometry.boundary(boundary.boundary_set)
            np.add.at(load, facets.ravel(), np.repeat(area * (q + h * ambient) / d, d))
            if h and facets.size:
                self.has_convection = True
                self.convection_components.update(
                    geometry.components[facets.ravel()].tolist()
                )
                rows.append(np.repeat(facets, d, axis=1).ravel())
                cols.append(np.tile(facets, (1, d)).ravel())
                data.append(
                    (
                        h
                        * area[:, None, None]
                        * (np.ones((d, d)) + np.eye(d))
                        / (d * (d + 1))
                    ).ravel()
                )
        self.stiffness = CSRMatrix.from_coo(
            stiffness.shape,
            np.concatenate(rows),
            np.concatenate(cols),
            np.concatenate(data),
        )
        self.load = readonly(load)

    def _result(self, temperatures, times, residuals, bc, balances) -> ThermalResult:
        temperatures = np.asarray(temperatures)
        gradients = np.einsum(
            "tea,eai->tei",
            temperatures[:, self.geometry.mesh.cells],
            self.geometry.gradients,
        )
        flux = -np.einsum("eij,tej->tei", self.conductivity, gradients)
        heat = temperatures @ self.capacity.matvec(np.ones(self.capacity.shape[0]))
        return ThermalResult(
            temperatures, times, residuals, flux, heat, balances, bc.dofs
        )

    def steady(self, bc: EssentialBC) -> ThermalResult:
        """固定温度或 Robin 必须锚定温度零空间。"""
        bc.validate(self.stiffness.shape[0])
        anchored = set(self.geometry.components[bc.dofs]) | self.convection_components
        if anchored != set(self.geometry.components):
            raise ValueError("稳态热传导的每个连通分量都需要固定温度或正换热边界。")
        temperature = (
            NativeSparseProvider()
            .solve_constrained(self.stiffness, self.load, bc.dofs, bc.values)
            .solution
        )
        residual = self.stiffness.matvec(temperature) - self.load
        balance = residual.sum() - residual[bc.dofs].sum()
        return self._result([temperature], [0.0], [residual], bc, [balance])

    def transient(
        self,
        initial: ArrayLike,
        bc: EssentialBC,
        *,
        time_step: float,
        steps: int,
        context: ExecutionContext | None = None,
    ) -> ThermalResult:
        """常系数、固定温度边界；每步使用接受温度，固定算子仅准备一次。"""
        dt = positive(time_step, "时间步")
        steps = positive_integer(steps, "步数")
        validate_time_interval(dt, steps)
        n = self.stiffness.shape[0]
        bc.validate(n)
        current = finite_array(initial, (n,), "初始温度").copy()
        if not np.allclose(current[bc.dofs], bc.values, rtol=0, atol=1e-12):
            raise ValueError("初始温度必须满足固定温度约束。")
        ctx = context or ExecutionContext()
        ctx.check("thermal.prepare", 0, steps)
        times = time_grid(dt, steps)
        matrix = CSRMatrix.from_coo(
            (n, n),
            np.concatenate((self.capacity.row_indices(), self.stiffness.row_indices())),
            np.concatenate((self.capacity.indices, self.stiffness.indices)),
            np.concatenate((self.capacity.data / dt, self.stiffness.data)),
        )
        prepared = NativeSparseProvider().prepare_constrained(matrix, bc.dofs)
        temperatures, residuals, balances = [current.copy()], [np.zeros(n)], [0.0]
        for step in range(1, steps + 1):
            ctx.check("thermal.step", step - 1, steps)
            rhs = self.capacity.matvec(current) / dt + self.load
            candidate = prepared.solve(rhs, bc.values).solution
            residual = self.capacity.matvec(candidate - current) / dt
            residual += self.stiffness.matvec(candidate) - self.load
            temperatures.append(candidate.copy())
            residuals.append(residual)
            balances.append(residual.sum() - residual[bc.dofs].sum())
            current = candidate
        ctx.check("thermal.complete", steps, steps)
        return self._result(temperatures, times, residuals, bc, balances)
