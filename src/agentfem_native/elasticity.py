# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Readable T3 small-strain elasticity vertical slice."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .assembly import COOMatrix
from .dofs import VectorDofMap
from .geometry import AffineTriangleMap
from .mesh import TriangularMesh
from .providers import LinearAlgebraProvider, resolve_provider
from .quadrature import triangle_rule
from .reference import p1_basis
from .sparse import CGReport

FloatArray: TypeAlias = NDArray[np.float64]
VectorField = ArrayLike | Callable[[FloatArray], ArrayLike]
ScalarBoundaryField = float | Callable[[FloatArray], ArrayLike]
VectorBoundaryField = ArrayLike | Callable[[FloatArray], ArrayLike]
PlaneModel = Literal["plane_stress", "plane_strain"]


@dataclass(frozen=True, slots=True)
class LinearElasticMaterial:
    young_modulus: float
    poisson_ratio: float
    model: PlaneModel = "plane_stress"

    def __post_init__(self) -> None:
        if isinstance(self.young_modulus, (bool, np.bool_)) or isinstance(
            self.poisson_ratio, (bool, np.bool_)
        ):
            raise TypeError("Elastic material parameters must be real scalars.")
        try:
            modulus = float(self.young_modulus)
            ratio = float(self.poisson_ratio)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Elastic material parameters must be finite scalars."
            ) from error
        if not np.isfinite(modulus) or modulus <= 0.0:
            raise ValueError("Young's modulus must be finite and positive.")
        if not np.isfinite(ratio) or ratio <= -1.0 or ratio >= 0.5:
            raise ValueError("Poisson's ratio must satisfy -1 < nu < 0.5.")
        if self.model not in {"plane_stress", "plane_strain"}:
            raise ValueError(f"Unknown two-dimensional material model {self.model!r}.")
        object.__setattr__(self, "young_modulus", modulus)
        object.__setattr__(self, "poisson_ratio", ratio)

    @property
    def constitutive_matrix(self) -> FloatArray:
        modulus = float(self.young_modulus)
        ratio = float(self.poisson_ratio)
        if self.model == "plane_stress":
            scale = modulus / (1.0 - ratio * ratio)
            return scale * np.array(
                (
                    (1.0, ratio, 0.0),
                    (ratio, 1.0, 0.0),
                    (0.0, 0.0, 0.5 * (1.0 - ratio)),
                )
            )
        scale = modulus / ((1.0 + ratio) * (1.0 - 2.0 * ratio))
        return scale * np.array(
            (
                (1.0 - ratio, ratio, 0.0),
                (ratio, 1.0 - ratio, 0.0),
                (0.0, 0.0, 0.5 * (1.0 - 2.0 * ratio)),
            )
        )


@dataclass(frozen=True, slots=True)
class DisplacementCondition:
    node_set: str
    component: Literal["x", "y"] | int | None
    value: ScalarBoundaryField | VectorBoundaryField


@dataclass(frozen=True, slots=True)
class TractionCondition:
    boundary_set: str
    value: VectorBoundaryField


@dataclass(frozen=True, slots=True)
class LinearElasticProblem:
    mesh: TriangularMesh
    material: LinearElasticMaterial
    thickness: float = 1.0
    body_force: VectorField = (0.0, 0.0)
    dirichlet: tuple[DisplacementCondition, ...] = ()
    traction: tuple[TractionCondition, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.thickness, (bool, np.bool_)):
            raise TypeError("Elasticity thickness must be a real scalar.")
        try:
            thickness = float(self.thickness)
        except (TypeError, ValueError) as error:
            raise ValueError("Elasticity thickness must be a finite scalar.") from error
        if not np.isfinite(thickness) or thickness <= 0.0:
            raise ValueError("Elasticity thickness must be finite and positive.")
        object.__setattr__(self, "thickness", thickness)


@dataclass(frozen=True, slots=True)
class LinearElasticResult:
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
            array = np.array(getattr(self, name), dtype=dtype, copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def reactions(self) -> FloatArray:
        return self.residual[self.constrained_dofs]


def _vector_values(field: VectorField, points: FloatArray, *, name: str) -> FloatArray:
    raw = field(points) if callable(field) else field
    values = np.asarray(raw, dtype=np.float64)
    if values.shape == (2,):
        values = np.broadcast_to(values, (points.shape[0], 2))
    if values.shape != (points.shape[0], 2) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must evaluate to one finite 2-vector per point.")
    return values


def _scalar_values(
    field: ScalarBoundaryField, points: FloatArray, *, name: str
) -> FloatArray:
    raw = field(points) if callable(field) else field
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim == 0:
        values = np.full(points.shape[0], float(values))
    if values.shape != (points.shape[0],) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must evaluate to one finite scalar per point.")
    return values


def p1_strain_displacement(vertices: ArrayLike) -> FloatArray:
    """Return the constant engineering-strain B matrix for one T3."""

    gradients = AffineTriangleMap(np.asarray(vertices, dtype=np.float64)).p1_gradients()
    result = np.zeros((3, 6), dtype=np.float64)
    result[0, 0::2] = gradients[:, 0]
    result[1, 1::2] = gradients[:, 1]
    result[2, 0::2] = gradients[:, 1]
    result[2, 1::2] = gradients[:, 0]
    return result


def p1_elastic_stiffness(
    vertices: ArrayLike,
    material: LinearElasticMaterial,
    *,
    thickness: float = 1.0,
) -> FloatArray:
    """Return one six-by-six T3 small-strain stiffness matrix."""

    if not np.isfinite(thickness) or thickness <= 0.0:
        raise ValueError("Elasticity thickness must be finite and positive.")
    geometry = AffineTriangleMap(np.asarray(vertices, dtype=np.float64))
    strain_displacement = p1_strain_displacement(vertices)
    return (
        float(thickness)
        * geometry.area
        * strain_displacement.T
        @ material.constitutive_matrix
        @ strain_displacement
    )


def p1_body_force_load(
    vertices: ArrayLike,
    body_force: VectorField,
    *,
    thickness: float = 1.0,
) -> FloatArray:
    """Integrate one body-force field into node-major vector loads."""

    if not np.isfinite(thickness) or thickness <= 0.0:
        raise ValueError("Elasticity thickness must be finite and positive.")
    geometry = AffineTriangleMap(np.asarray(vertices, dtype=np.float64))
    rule = triangle_rule(2)
    points = geometry.map_points(rule.points)
    values = _vector_values(body_force, points, name="Body force")
    nodal = (
        float(thickness)
        * geometry.integration_scale
        * (p1_basis(rule.points).T @ (rule.weights[:, None] * values))
    )
    return nodal.ravel()


def p1_boundary_traction_load(
    edge_vertices: ArrayLike,
    traction: VectorBoundaryField,
    *,
    thickness: float = 1.0,
) -> FloatArray:
    """Integrate one physical traction field along a straight boundary edge."""

    if not np.isfinite(thickness) or thickness <= 0.0:
        raise ValueError("Elasticity thickness must be finite and positive.")
    vertices = np.asarray(edge_vertices, dtype=np.float64)
    if vertices.shape != (2, 2) or not np.all(np.isfinite(vertices)):
        raise ValueError("Boundary edge vertices must have shape (2, 2) and be finite.")
    length = float(np.linalg.norm(vertices[1] - vertices[0]))
    if length <= 32.0 * np.finfo(np.float64).eps:
        raise ValueError("Boundary edge is degenerate.")
    offset = 1.0 / (2.0 * np.sqrt(3.0))
    coordinates = np.array((0.5 - offset, 0.5 + offset))
    weights = np.array((0.5, 0.5))
    points = (1.0 - coordinates[:, None]) * vertices[0] + coordinates[
        :, None
    ] * vertices[1]
    values = _vector_values(traction, points, name="Boundary traction")
    basis = np.column_stack((1.0 - coordinates, coordinates))
    return (float(thickness) * length * (basis.T @ (weights[:, None] * values))).ravel()


def assemble_linear_elasticity(
    problem: LinearElasticProblem,
) -> tuple[COOMatrix, FloatArray, VectorDofMap]:
    """Assemble the readable deterministic T3 elasticity system."""

    mesh = problem.mesh
    dofs = VectorDofMap(mesh.node_count, 2)
    cell_dofs = dofs.cell_dofs(mesh.cells)
    entry_count = mesh.cell_count * 36
    rows = np.empty(entry_count, dtype=np.int64)
    columns = np.empty(entry_count, dtype=np.int64)
    data = np.empty(entry_count, dtype=np.float64)
    load = np.zeros(dofs.size, dtype=np.float64)
    for cell_index, cell in enumerate(mesh.cells):
        local_dofs = cell_dofs[cell_index]
        local_stiffness = p1_elastic_stiffness(
            mesh.points[cell], problem.material, thickness=problem.thickness
        )
        local_load = p1_body_force_load(
            mesh.points[cell], problem.body_force, thickness=problem.thickness
        )
        start = cell_index * 36
        stop = start + 36
        rows[start:stop] = np.repeat(local_dofs, 6)
        columns[start:stop] = np.tile(local_dofs, 6)
        data[start:stop] = local_stiffness.ravel()
        load[local_dofs] += local_load
    for condition in problem.traction:
        for edge in mesh.boundary_edges(condition.boundary_set):
            local_load = p1_boundary_traction_load(
                mesh.points[edge], condition.value, thickness=problem.thickness
            )
            load[dofs.node_dofs(edge).ravel()] += local_load
    return COOMatrix((dofs.size, dofs.size), rows, columns, data), load, dofs


def _component_index(component: Literal["x", "y"] | int) -> int:
    if isinstance(component, (bool, np.bool_)):
        raise TypeError("Displacement component must be 'x', 'y', 0, 1, or None.")
    if component in {"x", 0}:
        return 0
    if component in {"y", 1}:
        return 1
    raise ValueError("Displacement component must be 'x', 'y', 0, 1, or None.")


def _collect_displacements(
    problem: LinearElasticProblem, dofs: VectorDofMap
) -> tuple[NDArray[np.int64], FloatArray]:
    assigned: dict[int, float] = {}
    for condition in problem.dirichlet:
        nodes = problem.mesh.nodes(condition.node_set)
        points = problem.mesh.points[nodes]
        if condition.component is None:
            values = _vector_values(
                condition.value, points, name="Vector displacement value"
            )
            components = (0, 1)
        else:
            component = _component_index(condition.component)
            values = _scalar_values(condition.value, points, name="Displacement value")[
                :, None
            ]
            components = (component,)
        for local_node, node in enumerate(nodes):
            for local_component, component in enumerate(components):
                dof = dofs.dof(int(node), component)
                value = float(values[local_node, local_component])
                if dof in assigned and not np.isclose(
                    assigned[dof], value, rtol=0.0, atol=1.0e-13
                ):
                    raise ValueError(f"Conflicting displacement values at DOF {dof}.")
                assigned[dof] = value
    if not assigned:
        raise ValueError("Elasticity requires displacement constraints.")
    constrained = np.array(sorted(assigned), dtype=np.int64)
    values = np.array([assigned[int(dof)] for dof in constrained], dtype=np.float64)
    coordinates = problem.mesh.points[constrained // 2]
    components = constrained % 2
    rigid_trace = np.zeros((constrained.size, 3), dtype=np.float64)
    rigid_trace[components == 0, 0] = 1.0
    rigid_trace[components == 1, 1] = 1.0
    rigid_trace[components == 0, 2] = -coordinates[components == 0, 1]
    rigid_trace[components == 1, 2] = coordinates[components == 1, 0]
    if np.linalg.matrix_rank(rigid_trace) < 3:
        raise ValueError(
            "Displacement constraints do not remove all three in-plane rigid modes."
        )
    return constrained, values


def solve_linear_elasticity(
    problem: LinearElasticProblem,
    *,
    provider: str | LinearAlgebraProvider | None = None,
) -> LinearElasticResult:
    """Assemble, solve, and recover one 2D T3 small-strain problem."""

    matrix, load, dofs = assemble_linear_elasticity(problem)
    constrained, values = _collect_displacements(problem, dofs)
    outcome = resolve_provider(provider).solve_constrained(
        matrix, load, constrained, values
    )
    solution = np.asarray(outcome.solution, dtype=np.float64)
    if solution.shape != (dofs.size,) or not np.all(np.isfinite(solution)):
        raise ValueError("Linear algebra provider returned an invalid solution vector.")
    residual = matrix.matvec(solution) - load
    free = np.setdiff1d(np.arange(dofs.size, dtype=np.int64), constrained)
    cell_dofs = dofs.cell_dofs(problem.mesh.cells)
    strain = np.empty((problem.mesh.cell_count, 3), dtype=np.float64)
    stress = np.empty_like(strain)
    constitutive = problem.material.constitutive_matrix
    for cell_index, cell in enumerate(problem.mesh.cells):
        strain[cell_index] = (
            p1_strain_displacement(problem.mesh.points[cell])
            @ solution[cell_dofs[cell_index]]
        )
        stress[cell_index] = constitutive @ strain[cell_index]
    reaction_vector = np.zeros(2, dtype=np.float64)
    np.add.at(reaction_vector, constrained % 2, residual[constrained])
    return LinearElasticResult(
        displacements=solution.reshape(problem.mesh.node_count, 2),
        residual=residual,
        constrained_dofs=constrained,
        cell_strain=strain,
        cell_stress=stress,
        free_residual_norm=float(np.linalg.norm(residual[free])),
        total_applied_force=load.reshape(problem.mesh.node_count, 2).sum(axis=0),
        total_reaction=reaction_vector,
        strain_energy=float(0.5 * solution @ matrix.matvec(solution)),
        provider_name=outcome.provider.name,
        provider_version=outcome.provider.version,
        provider_matrix_format=outcome.provider.matrix_format,
        convergence=outcome.convergence,
    )
