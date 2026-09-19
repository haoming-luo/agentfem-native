# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""自主 T4 三维小应变线弹性的可读、优化与预备装配竖向链路。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .assembly import COOMatrix
from .dofs import VectorDofMap
from .native import assemble_t4_values, assemble_t4_volume, native_kernel_available
from .providers import LinearAlgebraProvider, NativeSparseProvider, resolve_provider
from .sparse import CGReport, CSRAssemblyPlan, CSRMatrix
from .volume_mesh import TetrahedralMesh

FloatArray: TypeAlias = NDArray[np.float64]
Vector3Field = ArrayLike | Callable[[FloatArray], ArrayLike]
Scalar3Field = float | Callable[[FloatArray], ArrayLike]
SolidAssemblyMode = Literal["auto", "reference", "native"]


@dataclass(frozen=True, slots=True)
class SolidElasticMaterial:
    young_modulus: float
    poisson_ratio: float

    def __post_init__(self) -> None:
        if isinstance(self.young_modulus, (bool, np.bool_)) or isinstance(
            self.poisson_ratio, (bool, np.bool_)
        ):
            raise TypeError("Solid elastic parameters must be real scalars.")
        modulus = float(self.young_modulus)
        ratio = float(self.poisson_ratio)
        if not np.isfinite(modulus) or modulus <= 0.0:
            raise ValueError("Young's modulus must be finite and positive.")
        if not np.isfinite(ratio) or ratio <= -1.0 or ratio >= 0.5:
            raise ValueError("Poisson's ratio must satisfy -1 < nu < 0.5.")
        object.__setattr__(self, "young_modulus", modulus)
        object.__setattr__(self, "poisson_ratio", ratio)

    @property
    def constitutive_matrix(self) -> FloatArray:
        modulus = self.young_modulus
        ratio = self.poisson_ratio
        shear = modulus / (2.0 * (1.0 + ratio))
        lame = modulus * ratio / ((1.0 + ratio) * (1.0 - 2.0 * ratio))
        result = np.zeros((6, 6), dtype=np.float64)
        result[:3, :3] = lame
        np.fill_diagonal(result[:3, :3], lame + 2.0 * shear)
        result[3:, 3:] = np.eye(3) * shear
        return result


@dataclass(frozen=True, slots=True)
class SolidCellMaterial:
    cell_set: str
    material: SolidElasticMaterial


@dataclass(frozen=True, slots=True)
class SolidDisplacementCondition:
    node_set: str
    component: Literal["x", "y", "z"] | int | None
    value: Scalar3Field | Vector3Field


@dataclass(frozen=True, slots=True)
class SolidTractionCondition:
    boundary_set: str
    value: Vector3Field


@dataclass(frozen=True, slots=True)
class LinearElastic3DProblem:
    mesh: TetrahedralMesh
    material: SolidElasticMaterial
    body_force: Vector3Field = (0.0, 0.0, 0.0)
    dirichlet: tuple[SolidDisplacementCondition, ...] = ()
    traction: tuple[SolidTractionCondition, ...] = ()
    materials: tuple[SolidCellMaterial, ...] = ()
    thread_count: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.thread_count, (bool, np.bool_)) or not isinstance(
            self.thread_count, (int, np.integer)
        ):
            raise TypeError("T4 装配线程数必须是正整数。")
        if self.thread_count < 1:
            raise ValueError("T4 装配线程数必须大于零。")
        object.__setattr__(self, "thread_count", int(self.thread_count))


@dataclass(frozen=True, slots=True)
class LinearElastic3DResult:
    displacements: FloatArray
    residual: FloatArray
    constrained_dofs: NDArray[np.int64]
    cell_strain: FloatArray
    cell_stress: FloatArray
    free_residual_norm: float
    total_applied_force: FloatArray
    total_reaction: FloatArray
    strain_energy: float
    provider_name: str
    provider_version: str
    provider_matrix_format: str
    convergence: CGReport | None

    def __post_init__(self) -> None:
        for name in (
            "displacements",
            "residual",
            "constrained_dofs",
            "cell_strain",
            "cell_stress",
            "total_applied_force",
            "total_reaction",
        ):
            dtype = np.int64 if name == "constrained_dofs" else np.float64
            value = np.array(getattr(self, name), dtype=dtype, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def reactions(self) -> FloatArray:
        return self.residual[self.constrained_dofs]


def _vector_values(field: Vector3Field, points: FloatArray, *, name: str) -> FloatArray:
    raw = field(points) if callable(field) else field
    values = np.asarray(raw, dtype=np.float64)
    if values.shape == (3,):
        values = np.broadcast_to(values, (points.shape[0], 3))
    if values.shape != (points.shape[0], 3) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must evaluate to one finite 3-vector per point.")
    return values


def _scalar_values(field: Scalar3Field, points: FloatArray, *, name: str) -> FloatArray:
    raw = field(points) if callable(field) else field
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim == 0:
        values = np.full(points.shape[0], float(values))
    if values.shape != (points.shape[0],) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must evaluate to one finite scalar per point.")
    return values


def t4_geometry(vertices: ArrayLike) -> tuple[float, FloatArray]:
    """Return positive volume and physical P1 gradients for one affine T4."""

    points = np.asarray(vertices, dtype=np.float64)
    if points.shape != (4, 3) or not np.all(np.isfinite(points)):
        raise ValueError("T4 vertices must have shape (4, 3) and be finite.")
    jacobian = np.column_stack(
        (points[1] - points[0], points[2] - points[0], points[3] - points[0])
    )
    determinant = float(np.linalg.det(jacobian))
    scale = float(np.max(np.linalg.norm(jacobian, axis=0)))
    if abs(determinant) <= 64.0 * np.finfo(np.float64).eps * scale**3:
        raise ValueError("T4 geometry is degenerate or numerically singular.")
    reference = np.array(
        ((-1.0, -1.0, -1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    return abs(determinant) / 6.0, reference @ np.linalg.inv(jacobian)


def t4_strain_displacement(vertices: ArrayLike) -> FloatArray:
    """Return the constant engineering-strain B matrix for one T4."""

    _, gradients = t4_geometry(vertices)
    result = np.zeros((6, 12), dtype=np.float64)
    result[0, 0::3] = gradients[:, 0]
    result[1, 1::3] = gradients[:, 1]
    result[2, 2::3] = gradients[:, 2]
    result[3, 0::3] = gradients[:, 1]
    result[3, 1::3] = gradients[:, 0]
    result[4, 1::3] = gradients[:, 2]
    result[4, 2::3] = gradients[:, 1]
    result[5, 0::3] = gradients[:, 2]
    result[5, 2::3] = gradients[:, 0]
    return result


def t4_elastic_stiffness(
    vertices: ArrayLike, material: SolidElasticMaterial
) -> FloatArray:
    volume, _ = t4_geometry(vertices)
    strain = t4_strain_displacement(vertices)
    return volume * strain.T @ material.constitutive_matrix @ strain


def t4_body_force_load(vertices: ArrayLike, body_force: Vector3Field) -> FloatArray:
    volume, _ = t4_geometry(vertices)
    points = np.asarray(vertices, dtype=np.float64)
    centroid = points.mean(axis=0, keepdims=True)
    values = _vector_values(body_force, centroid, name="Body force")[0]
    return np.tile(volume * values / 4.0, 4)


def t4_boundary_traction_load(
    vertices: ArrayLike, traction: Vector3Field
) -> FloatArray:
    points = np.asarray(vertices, dtype=np.float64)
    if points.shape != (3, 3) or not np.all(np.isfinite(points)):
        raise ValueError("Boundary face vertices must have shape (3, 3) and be finite.")
    area = 0.5 * float(
        np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0]))
    )
    if area <= 64.0 * np.finfo(np.float64).eps:
        raise ValueError("Boundary face is degenerate.")
    barycentric = np.array(
        ((2 / 3, 1 / 6, 1 / 6), (1 / 6, 2 / 3, 1 / 6), (1 / 6, 1 / 6, 2 / 3))
    )
    physical = barycentric @ points
    values = _vector_values(traction, physical, name="Boundary traction")
    return (area / 3.0 * (barycentric.T @ values)).ravel()


def _constitutive_by_cell(problem: LinearElastic3DProblem) -> FloatArray:
    values = np.broadcast_to(
        problem.material.constitutive_matrix, (problem.mesh.cell_count, 6, 6)
    ).copy()
    assigned = np.zeros(problem.mesh.cell_count, dtype=bool)
    for override in problem.materials:
        cells = problem.mesh.cells_in(override.cell_set)
        if np.any(assigned[cells]):
            raise ValueError("A T4 cell cannot receive multiple material overrides.")
        values[cells] = override.material.constitutive_matrix
        assigned[cells] = True
    return values


def select_solid_assembly_mode(
    problem: LinearElastic3DProblem, assembly: SolidAssemblyMode = "auto"
) -> Literal["reference", "native"]:
    """Select a T4 volume path without executing assembly."""

    if assembly not in {"auto", "reference", "native"}:
        raise ValueError(f"Unknown T4 assembly mode {assembly!r}.")
    if problem.thread_count > 1 and (
        assembly == "reference"
        or (assembly == "auto" and callable(problem.body_force))
        or not native_kernel_available()
    ):
        raise ValueError("多线程 T4 装配要求可用的 native 静态体力路径。")
    if assembly == "native":
        if callable(problem.body_force):
            raise ValueError("Native T4 assembly requires a static body-force vector.")
        if not native_kernel_available():
            raise ValueError("Native T4 assembly is unavailable in this installation.")
        return "native"
    if assembly == "reference":
        return "reference"
    return (
        "native"
        if native_kernel_available() and not callable(problem.body_force)
        else "reference"
    )


def assemble_linear_elasticity_3d(
    problem: LinearElastic3DProblem,
    *,
    assembly: SolidAssemblyMode = "auto",
) -> tuple[COOMatrix, FloatArray, VectorDofMap]:
    mesh = problem.mesh
    dofs = VectorDofMap(mesh.node_count, 3)
    constitutive = _constitutive_by_cell(problem)
    selected = select_solid_assembly_mode(problem, assembly)
    if selected == "native":
        body = _vector_values(problem.body_force, np.zeros((1, 3)), name="Body force")[
            0
        ]
        rows, columns, data, load = assemble_t4_volume(
            mesh.points,
            mesh.cells,
            constitutive,
            body,
            thread_count=problem.thread_count,
        )
    else:
        cell_dofs = dofs.cell_dofs(mesh.cells)
        rows = np.empty(mesh.cell_count * 144, dtype=np.int64)
        columns = np.empty_like(rows)
        data = np.empty(rows.size, dtype=np.float64)
        load = np.zeros(dofs.size, dtype=np.float64)
        for cell_index, cell in enumerate(mesh.cells):
            local_dofs = cell_dofs[cell_index]
            volume, _ = t4_geometry(mesh.points[cell])
            strain = t4_strain_displacement(mesh.points[cell])
            local_stiffness = volume * strain.T @ constitutive[cell_index] @ strain
            local_load = t4_body_force_load(mesh.points[cell], problem.body_force)
            start = 144 * cell_index
            rows[start : start + 144] = np.repeat(local_dofs, 12)
            columns[start : start + 144] = np.tile(local_dofs, 12)
            data[start : start + 144] = local_stiffness.ravel()
            load[local_dofs] += local_load
    _add_t4_tractions(problem, load, dofs)
    return COOMatrix((dofs.size, dofs.size), rows, columns, data), load, dofs


def _add_t4_tractions(
    problem: LinearElastic3DProblem, load: FloatArray, dofs: VectorDofMap
) -> None:
    """按既有边界积分语义把 T4 面力累加到调用方载荷。"""

    mesh = problem.mesh
    for condition in problem.traction:
        for face in mesh.boundary_faces(condition.boundary_set):
            local_load = t4_boundary_traction_load(mesh.points[face], condition.value)
            load[dofs.node_dofs(face).ravel()] += local_load


def prepare_linear_elasticity_3d_assembly(
    problem: LinearElastic3DProblem,
) -> CSRAssemblyPlan:
    """为固定 T4 拓扑建立一次可复用的规范 CSR 装配计划。"""

    if not isinstance(problem, LinearElastic3DProblem):
        raise TypeError("T4 预备装配要求 LinearElastic3DProblem。")
    dofs = VectorDofMap(problem.mesh.node_count, 3)
    return CSRAssemblyPlan(dofs.size, dofs.cell_dofs(problem.mesh.cells))


def assemble_linear_elasticity_3d_prepared(
    problem: LinearElastic3DProblem,
    plan: CSRAssemblyPlan,
    *,
    assembly: SolidAssemblyMode = "auto",
) -> tuple[CSRMatrix, FloatArray, VectorDofMap]:
    """复用已验证稀疏图装配一次 T4 矩阵，同时保留既有载荷语义。"""

    if not isinstance(problem, LinearElastic3DProblem):
        raise TypeError("T4 预备装配要求 LinearElastic3DProblem。")
    if not isinstance(plan, CSRAssemblyPlan):
        raise TypeError("T4 预备装配要求 CSRAssemblyPlan。")
    dofs = VectorDofMap(problem.mesh.node_count, 3)
    plan.validate_layout(dofs.size, dofs.cell_dofs(problem.mesh.cells))
    selected = select_solid_assembly_mode(problem, assembly)
    if selected == "native" and problem.thread_count == 1:
        body = _vector_values(problem.body_force, np.zeros((1, 3)), name="Body force")[
            0
        ]
        data, load = assemble_t4_values(
            problem.mesh.points,
            problem.mesh.cells,
            _constitutive_by_cell(problem),
            body,
        )
        _add_t4_tractions(problem, load, dofs)
        return plan.fill(data), load, dofs
    coo, load, assembled_dofs = assemble_linear_elasticity_3d(
        problem, assembly=assembly
    )
    return plan.fill(coo.data), load, assembled_dofs


def _component(value: Literal["x", "y", "z"] | int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("Solid displacement component must be x, y, z, 0, 1, or 2.")
    mapping = {"x": 0, "y": 1, "z": 2, 0: 0, 1: 1, 2: 2}
    try:
        return mapping[value]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Solid displacement component must be x, y, z, 0, 1, or 2."
        ) from error


def _constraints(
    problem: LinearElastic3DProblem, dofs: VectorDofMap
) -> tuple[NDArray[np.int64], FloatArray]:
    assigned: dict[int, float] = {}
    for condition in problem.dirichlet:
        nodes = problem.mesh.nodes(condition.node_set)
        points = problem.mesh.points[nodes]
        if condition.component is None:
            values = _vector_values(condition.value, points, name="Vector displacement")
            components = (0, 1, 2)
        else:
            component = _component(condition.component)
            values = _scalar_values(condition.value, points, name="Displacement")[
                :, None
            ]
            components = (component,)
        for local_node, node in enumerate(nodes):
            for local_component, component in enumerate(components):
                dof = dofs.dof(int(node), component)
                value = float(values[local_node, local_component])
                if dof in assigned and not np.isclose(
                    assigned[dof], value, rtol=0.0, atol=1e-13
                ):
                    raise ValueError(f"Conflicting displacement values at DOF {dof}.")
                assigned[dof] = value
    if not assigned:
        raise ValueError(
            "Three-dimensional elasticity requires displacement constraints."
        )
    constrained = np.array(sorted(assigned), dtype=np.int64)
    values = np.array([assigned[int(index)] for index in constrained])
    coordinates = problem.mesh.points[constrained // 3]
    components = constrained % 3
    modes = np.zeros((constrained.size, 6), dtype=np.float64)
    modes[components == 0, 0] = 1.0
    modes[components == 1, 1] = 1.0
    modes[components == 2, 2] = 1.0
    for row, (point, component) in enumerate(zip(coordinates, components, strict=True)):
        x, y, z = point
        rotations = ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0))
        modes[row, 3:] = [rotation[int(component)] for rotation in rotations]
    if np.linalg.matrix_rank(modes) < 6:
        raise ValueError(
            "Displacement constraints do not remove all six 3D rigid modes."
        )
    return constrained, values


def solve_linear_elasticity_3d(
    problem: LinearElastic3DProblem,
    *,
    provider: str | LinearAlgebraProvider | None = None,
    assembly: SolidAssemblyMode = "auto",
    assembly_plan: CSRAssemblyPlan | None = None,
) -> LinearElastic3DResult:
    if assembly_plan is None:
        matrix, load, dofs = assemble_linear_elasticity_3d(problem, assembly=assembly)
    else:
        matrix, load, dofs = assemble_linear_elasticity_3d_prepared(
            problem, assembly_plan, assembly=assembly
        )
    constrained, values = _constraints(problem, dofs)
    selected = (
        NativeSparseProvider(block_size=3)
        if provider is None
        else resolve_provider(provider)
    )
    outcome = selected.solve_constrained(matrix, load, constrained, values)
    solution = np.asarray(outcome.solution, dtype=np.float64)
    residual = matrix.matvec(solution) - load
    free = np.setdiff1d(np.arange(dofs.size, dtype=np.int64), constrained)
    cell_dofs = dofs.cell_dofs(problem.mesh.cells)
    constitutive = _constitutive_by_cell(problem)
    strain = np.empty((problem.mesh.cell_count, 6), dtype=np.float64)
    stress = np.empty_like(strain)
    for index, cell in enumerate(problem.mesh.cells):
        strain[index] = (
            t4_strain_displacement(problem.mesh.points[cell])
            @ solution[cell_dofs[index]]
        )
        stress[index] = constitutive[index] @ strain[index]
    reaction = np.zeros(3, dtype=np.float64)
    np.add.at(reaction, constrained % 3, residual[constrained])
    return LinearElastic3DResult(
        solution.reshape(problem.mesh.node_count, 3),
        residual,
        constrained,
        strain,
        stress,
        float(np.linalg.norm(residual[free])),
        load.reshape(problem.mesh.node_count, 3).sum(axis=0),
        reaction,
        float(0.5 * solution @ matrix.matvec(solution)),
        outcome.provider.name,
        outcome.provider.version,
        outcome.provider.matrix_format,
        outcome.convergence,
    )
