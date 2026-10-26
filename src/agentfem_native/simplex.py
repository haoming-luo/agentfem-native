# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""多物理候选共享的仿射几何、显式约束和不可变数组。"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np
from numpy.typing import ArrayLike

from .mesh import TriangularMesh
from .sparse import CSRAssemblyPlan
from .volume_mesh import TetrahedralMesh

Mesh = TriangularMesh | TetrahedralMesh


def readonly(value: ArrayLike, dtype: object = np.float64) -> np.ndarray:
    """结果与接受状态拥有自己的只读数组，不共享可写试探缓冲。"""
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def finite_array(value: ArrayLike, shape: tuple[int, ...], name: str) -> np.ndarray:
    raw = np.asarray(value, dtype=np.float64)
    if raw.shape != shape or not np.all(np.isfinite(raw)):
        raise ValueError(f"{name}必须是形状 {shape} 的有限数组。")
    return raw


def positive(value: float, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name}必须是有限实数。")
    result = float(value)
    if not np.isfinite(result) or (result < 0 if allow_zero else result <= 0):
        raise ValueError(f"{name}必须有限且{'非负' if allow_zero else '为正'}。")
    return result


def positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name}必须是正整数。")
    if value < 1:
        raise ValueError(f"{name}必须是正整数。")
    return int(value)


def cell_values(value: ArrayLike, count: int, name: str) -> np.ndarray:
    raw = np.asarray(value, dtype=np.float64)
    if raw.ndim == 0:
        raw = np.full(count, float(raw))
    return finite_array(raw, (count,), name)


def connected_components(mesh: Mesh) -> np.ndarray:
    """节点邻接的向量化最小根合并；只保存 O(节点+单元) 数据，不构造稠密图。"""
    if np.unique(mesh.cells).size != mesh.node_count:
        raise ValueError("候选网格含未被单元使用的孤立节点，请删除或修复连通性。")
    labels = np.arange(mesh.node_count, dtype=np.int64)
    while True:
        previous = labels.copy()
        roots = labels[mesh.cells]
        minimum = roots.min(axis=1)
        np.minimum.at(labels, roots.ravel(), np.repeat(minimum, mesh.cells.shape[1]))
        while not np.array_equal(labels, labels[labels]):
            labels = labels[labels]
        if np.array_equal(previous, labels):
            break
    return np.unique(labels, return_inverse=True)[1]


@dataclass(frozen=True)
class EssentialBC:
    """显式自由度约束；向量场采用节点优先、分量其次的编号。"""

    dofs: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        raw = np.asarray(self.dofs)
        if raw.ndim != 1 or (raw.size and not np.issubdtype(raw.dtype, np.integer)):
            raise ValueError("约束自由度必须是一维整数数组。")
        if np.any(raw < 0) or np.unique(raw).size != raw.size:
            raise ValueError("约束自由度必须非负且无重复。")
        if raw.size and int(raw.max()) > np.iinfo(np.int64).max:
            raise ValueError("约束自由度超出有符号 64 位索引范围。")
        values = finite_array(self.values, raw.shape, "约束值")
        order = np.argsort(raw)
        object.__setattr__(self, "dofs", readonly(raw[order], np.int64))
        object.__setattr__(self, "values", readonly(values[order]))

    def validate(self, count: int) -> None:
        if np.any(self.dofs >= count):
            raise ValueError("约束自由度超出当前场的范围。")

    @classmethod
    def empty(cls) -> EssentialBC:
        return cls(np.empty(0, dtype=np.int64), np.empty(0))


def check_component_constraints(
    mesh: Mesh, components: np.ndarray, bc: EssentialBC
) -> None:
    """不装配矩阵的必要刚体准入；平移原点及改变长度单位不改变秩。"""
    d = mesh.points.shape[1]
    bc.validate(mesh.node_count * d)
    for group in np.unique(components):
        nodes = np.flatnonzero(components == group)
        selected = bc.dofs[components[bc.dofs // d] == group]
        points = mesh.points[nodes]
        origin = points[0]
        scale = np.max(np.abs(points - origin))
        x = (mesh.points[selected // d] - origin) / scale
        component = selected % d
        modes = np.zeros((selected.size, 3 if d == 2 else 6))
        modes[np.arange(selected.size), component] = 1
        if d == 2:
            rotation = np.column_stack((-x[:, 1], x[:, 0]))
            modes[:, 2] = rotation[np.arange(selected.size), component]
        else:
            for j in range(3):
                rotation = np.cross(np.eye(3)[j], x)
                modes[:, 3 + j] = rotation[np.arange(selected.size), component]
        if not selected.size or np.linalg.matrix_rank(modes) < modes.shape[1]:
            raise ValueError(f"连通分量 {group} 的位移约束未消除刚体平移与转动。")


@dataclass(frozen=True)
class SimplexGeometry:
    """参考构形的体积与梯度只准备一次；二维体积含显式厚度。"""

    mesh: Mesh
    thickness: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.mesh, (TriangularMesh, TetrahedralMesh)):
            raise TypeError("只支持项目自有 T3/T4 网格。")
        thickness = positive(self.thickness, "厚度")
        dimension = self.mesh.points.shape[1]
        if dimension == 3 and thickness != 1.0:
            raise ValueError("三维实体不使用二维厚度参数。")
        vertices = self.mesh.points[self.mesh.cells]
        jacobian = np.swapaxes(vertices[:, 1:] - vertices[:, :1], 1, 2)
        reference = np.vstack((-np.ones(dimension), np.eye(dimension)))
        gradients = np.einsum("aj,eji->eai", reference, np.linalg.inv(jacobian))
        volume = np.abs(np.linalg.det(jacobian)) / factorial(dimension) * thickness
        for name, value in (("gradients", gradients), ("volumes", volume)):
            object.__setattr__(self, name, readonly(value))
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "thickness", thickness)
        object.__setattr__(
            self, "components", readonly(connected_components(self.mesh), np.int64)
        )

    def assembly_plan(self, components: int = 1) -> CSRAssemblyPlan:
        components = positive_integer(components, "场分量数")
        dofs = (
            self.mesh.cells[:, :, None] * components + np.arange(components)
        ).reshape(self.mesh.cell_count, -1)
        return CSRAssemblyPlan(self.mesh.node_count * components, dofs)

    @property
    def mass(self) -> np.ndarray:
        m = self.dimension + 1
        return (
            self.volumes[:, None, None] * (np.ones((m, m)) + np.eye(m)) / (m * (m + 1))
        )

    def scatter(self, local: np.ndarray, components: int = 1) -> np.ndarray:
        result = np.zeros((self.mesh.node_count, components))
        np.add.at(result, self.mesh.cells.ravel(), local.reshape(-1, components))
        return result.ravel()

    def boundary(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        try:
            facets = self.mesh.boundary_sets[name]
        except KeyError as error:
            raise ValueError(f"未知边界集合：{name}。") from error
        points = self.mesh.points[facets]
        if self.dimension == 2:
            area = np.linalg.norm(points[:, 1] - points[:, 0], axis=1) * self.thickness
        else:
            area = (
                np.linalg.norm(
                    np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]),
                    axis=1,
                )
                / 2
            )
        return facets, area

    def check_rigid_constraints(self, bc: EssentialBC) -> None:
        """逐连通分量的刚体迹秩为必要条件；不替代铰接机构/零能模式检测。"""
        check_component_constraints(self.mesh, self.components, bc)

    def mechanical_load(
        self,
        *,
        body_force: ArrayLike | None = None,
        tractions: dict[str, ArrayLike] | None = None,
    ) -> np.ndarray:
        """参考构形上的恒定体力/面力，返回节点力；不包含随动压力。"""
        d, mesh = self.dimension, self.mesh
        result = np.zeros((mesh.node_count, d))
        if body_force is not None:
            body = finite_array(body_force, (d,), "体力")
            local = (
                self.volumes[:, None, None]
                * np.broadcast_to(body, (mesh.cell_count, d + 1, d))
                / (d + 1)
            )
            result += self.scatter(local, d).reshape(result.shape)
        for name, value in (tractions or {}).items():
            facets, area = self.boundary(name)
            traction = finite_array(value, (d,), "参考面力")
            local = (
                area[:, None, None]
                * np.broadcast_to(traction, (facets.shape[0], d, d))
                / d
            )
            np.add.at(result, facets.ravel(), local.reshape(-1, d))
        return result
