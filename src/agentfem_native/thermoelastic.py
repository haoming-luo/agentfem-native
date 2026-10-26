# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""二维/三维小应变顺序热弹性与相场共享的弹性算子。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .providers import NativeSparseProvider
from .simplex import (
    EssentialBC,
    Mesh,
    SimplexGeometry,
    cell_values,
    finite_array,
    readonly,
)


def cell_elastic_parameters(
    young_modulus: ArrayLike, poisson_ratio: ArrayLike, count: int
):
    """逐单元各向同性参数准入；原参数有限不代表派生本构系数不会溢出。"""
    if (
        np.asarray(young_modulus).dtype.kind == "b"
        or np.asarray(poisson_ratio).dtype.kind == "b"
    ):
        raise TypeError("弹性参数必须为实数，不能为布尔值。")
    modulus = cell_values(young_modulus, count, "逐单元杨氏模量")
    ratio = cell_values(poisson_ratio, count, "逐单元泊松比")
    if np.any(modulus <= 0) or np.any((ratio <= -1) | (ratio >= 0.5)):
        raise ValueError("每个单元必须满足 E>0 且 -1<nu<0.5。")
    with np.errstate(over="ignore", invalid="ignore"):
        shear = modulus / (2 * (1 + ratio))
        lame = modulus * ratio / ((1 + ratio) * (1 - 2 * ratio))
    if (
        not np.all(np.isfinite(shear))
        or not np.all(np.isfinite(lame))
        or np.any(shear <= 0)
    ):
        raise ValueError("派生弹性系数溢出或剪切模量下溢，请检查材料值或调整单位。")
    return readonly(shear), readonly(lame)


@dataclass(frozen=True)
class ThermoelasticResult:
    """应力按完整三维张量返回，二维平面应变包含 sigma_zz。"""

    displacements: np.ndarray
    residual: np.ndarray
    constrained_dofs: np.ndarray
    cell_strain: np.ndarray
    cell_stress: np.ndarray
    cell_energy: np.ndarray
    strain_energy: float
    free_residual_norm: float
    maturity: str = "implemented"

    def __post_init__(self) -> None:
        for name in (
            "displacements",
            "residual",
            "constrained_dofs",
            "cell_strain",
            "cell_stress",
            "cell_energy",
        ):
            object.__setattr__(
                self,
                name,
                readonly(
                    getattr(self, name),
                    np.int64 if name == "constrained_dofs" else np.float64,
                ),
            )


class ThermoelasticOperator:
    """固定拓扑与材料准备一次；温度驱动的热载荷由有限元层拥有。"""

    def __init__(
        self,
        mesh: Mesh,
        young_modulus: ArrayLike,
        poisson_ratio: ArrayLike,
        *,
        mode: str = "plane_strain",
        temperature: ArrayLike | None = None,
        reference_temperature: float = 0.0,
        expansion: ArrayLike = 0.0,
        thickness: float = 1.0,
    ) -> None:
        self.geometry = g = SimplexGeometry(mesh, thickness)
        self.shear, self.lame = cell_elastic_parameters(
            young_modulus, poisson_ratio, mesh.cell_count
        )
        if mode not in ("plane_stress", "plane_strain"):
            raise ValueError("二维模式必须是 plane_stress 或 plane_strain。")
        if g.dimension == 3 and mode == "plane_stress":
            raise ValueError("三维实体不能使用平面应力。")
        self.mode = mode
        self.plan = g.assembly_plan(g.dimension)
        d = g.dimension
        lame = (
            2 * self.shear * self.lame / (self.lame + 2 * self.shear)
            if d == 2 and mode == "plane_stress"
            else self.lame
        )
        gradients = g.gradients
        shear_local = self.shear[:, None, None, None, None]
        local = shear_local * np.einsum(
            "eaj,ebj,ik->eaibk", gradients, gradients, np.eye(d)
        )
        local += shear_local * np.einsum("eak,ebi->eaibk", gradients, gradients)
        local += lame[:, None, None, None, None] * np.einsum(
            "eai,ebk->eaibk", gradients, gradients
        )
        size = (d + 1) * d
        self.local_stiffness = readonly(
            local.reshape(mesh.cell_count, size, size) * g.volumes[:, None, None]
        )
        self.stiffness = self.plan.fill(self.local_stiffness)
        if not np.isfinite(reference_temperature):
            raise ValueError("参考温度必须有限。")
        if temperature is None:
            nodal = np.full(mesh.node_count, reference_temperature)
        else:
            nodal = finite_array(temperature, (mesh.node_count,), "节点温度")
        # P1 位移的应变在单元内为常量；热载荷精确使用 P1 温度的单元均值。
        alpha = cell_values(expansion, mesh.cell_count, "线膨胀系数")
        thermal_nodes = alpha[:, None] * (nodal[mesh.cells] - reference_temperature)
        self.thermal_strain = readonly(thermal_nodes.mean(axis=1))
        m = d + 1
        centered = thermal_nodes - self.thermal_strain[:, None]
        self.thermal_variance = readonly(
            np.einsum("ea,eab,eb->e", centered, g.mass, centered) / g.volumes
        )
        thermal_stress = (
            (2 * self.shear + d * lame)
            if mode == "plane_stress"
            else (2 * self.shear + 3 * self.lame)
        ) * self.thermal_strain
        self.local_thermal_load = readonly(
            (
                g.volumes[:, None, None] * thermal_stress[:, None, None] * gradients
            ).reshape(mesh.cell_count, m * d)
        )
        self.thermal_load = readonly(g.scatter(self.local_thermal_load, d))

    def fields(
        self, displacement: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        g, mesh = self.geometry, self.geometry.mesh
        d = g.dimension
        values = finite_array(displacement, (mesh.node_count, d), "节点位移")
        gradient = np.einsum("eai,eaj->eij", values[mesh.cells], g.gradients)
        strain = np.zeros((mesh.cell_count, 3, 3))
        strain[:, :d, :d] = (gradient + gradient.swapaxes(1, 2)) / 2
        a = self.thermal_strain
        if d == 2 and self.mode == "plane_stress":
            strain[:, 2, 2] = a - self.lame / (self.lame + 2 * self.shear) * (
                strain[:, 0, 0] + strain[:, 1, 1] - 2 * a
            )
        elastic = strain - a[:, None, None] * np.eye(3)
        trace = np.trace(elastic, axis1=1, axis2=2)
        stress = 2 * self.shear[:, None, None] * elastic + (self.lame * trace)[
            :, None, None
        ] * np.eye(3)
        energy = 0.5 * np.einsum("eij,eij->e", stress, elastic)
        # 恢复单元积分热能中的温度方差项；不能把平均温度平方冒充平方平均。
        coefficient = (
            (
                2 * self.shear
                + 2 * self.lame * 2 * self.shear / (self.lame + 2 * self.shear)
            )
            if d == 2 and self.mode == "plane_stress"
            else (3 * self.shear + 4.5 * self.lame)
        )
        energy += coefficient * self.thermal_variance
        return strain, stress, energy

    def solve(
        self, bc: EssentialBC, *, load: ArrayLike | None = None
    ) -> ThermoelasticResult:
        g, mesh = self.geometry, self.geometry.mesh
        g.check_rigid_constraints(bc)
        force = (
            np.zeros(mesh.node_count * g.dimension)
            if load is None
            else finite_array(load, (mesh.node_count, g.dimension), "节点外力").ravel()
        )
        rhs = force + self.thermal_load
        solution = (
            NativeSparseProvider(block_size=g.dimension)
            .solve_constrained(self.stiffness, rhs, bc.dofs, bc.values)
            .solution
        )
        residual = self.stiffness.matvec(solution) - rhs
        strain, stress, energy = self.fields(
            solution.reshape(mesh.node_count, g.dimension)
        )
        free = np.setdiff1d(np.arange(solution.size), bc.dofs)
        return ThermoelasticResult(
            solution.reshape(mesh.node_count, g.dimension),
            residual,
            bc.dofs,
            strain,
            stress,
            energy,
            float(g.volumes @ energy),
            float(np.linalg.norm(residual[free])),
        )


def solve_thermal_elastic(
    thermal,
    thermal_bc: EssentialBC,
    mechanical_bc: EssentialBC,
    *,
    young_modulus: ArrayLike,
    poisson_ratio: ArrayLike,
    expansion: ArrayLike,
    reference_temperature: float,
    mode: str = "plane_strain",
    load: ArrayLike | None = None,
):
    """同一网格上的稳态单向热—力闭环；热反馈不在当前模型内。"""
    heat = thermal.steady(thermal_bc)
    mechanical = ThermoelasticOperator(
        thermal.geometry.mesh,
        young_modulus,
        poisson_ratio,
        mode=mode,
        temperature=heat.temperatures[-1],
        expansion=expansion,
        reference_temperature=reference_temperature,
        thickness=thermal.geometry.thickness,
    )
    return heat, mechanical.solve(mechanical_bc, load=load)
