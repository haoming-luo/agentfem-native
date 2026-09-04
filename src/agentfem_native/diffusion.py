# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""First complete serial steady-diffusion vertical slice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .assembly import COOMatrix, ScalarField, assemble_diffusion
from .mesh import TriangularMesh


FloatArray: TypeAlias = NDArray[np.float64]
BoundaryValue = float | Callable[[FloatArray], ArrayLike]


@dataclass(frozen=True, slots=True)
class DirichletCondition:
    node_set: str
    value: BoundaryValue


@dataclass(frozen=True, slots=True)
class NeumannCondition:
    boundary_set: str
    flux: ScalarField


@dataclass(frozen=True, slots=True)
class SteadyDiffusionProblem:
    mesh: TriangularMesh
    conductivity: float
    source: ScalarField = 0.0
    dirichlet: tuple[DirichletCondition, ...] = ()
    neumann: tuple[NeumannCondition, ...] = ()


@dataclass(frozen=True, slots=True)
class DiffusionResult:
    """Nodal solution plus transparent algebraic and balance evidence."""

    nodal_values: FloatArray
    residual: FloatArray
    constrained_nodes: NDArray[np.int64]
    free_residual_norm: float
    total_applied_load: float
    total_reaction: float
    potential_energy: float

    def __post_init__(self) -> None:
        for name in ("nodal_values", "residual", "constrained_nodes"):
            dtype = np.int64 if name == "constrained_nodes" else np.float64
            array = np.array(getattr(self, name), dtype=dtype, copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def reactions(self) -> FloatArray:
        return self.residual[self.constrained_nodes]


def _boundary_values(value: BoundaryValue, points: FloatArray) -> FloatArray:
    raw = value(points) if callable(value) else value
    result = np.asarray(raw, dtype=np.float64)
    if result.ndim == 0:
        result = np.full(points.shape[0], float(result))
    if result.shape != (points.shape[0],) or not np.all(np.isfinite(result)):
        raise ValueError("Dirichlet value must evaluate to one finite scalar per node.")
    return result


def _collect_dirichlet(problem: SteadyDiffusionProblem) -> tuple[NDArray[np.int64], FloatArray]:
    assigned: dict[int, float] = {}
    for condition in problem.dirichlet:
        nodes = problem.mesh.nodes(condition.node_set)
        values = _boundary_values(condition.value, problem.mesh.points[nodes])
        for node, value in zip(nodes, values, strict=True):
            index = int(node)
            scalar = float(value)
            if index in assigned and not np.isclose(assigned[index], scalar, rtol=0.0, atol=1e-13):
                raise ValueError(f"Conflicting Dirichlet values at node {index}.")
            assigned[index] = scalar
    if not assigned:
        raise ValueError("At least one Dirichlet node is required for the reference solver.")
    nodes = np.array(sorted(assigned), dtype=np.int64)
    values = np.array([assigned[int(node)] for node in nodes], dtype=np.float64)
    return nodes, values


def _solve_constrained(
    matrix: COOMatrix,
    load: FloatArray,
    constrained: NDArray[np.int64],
    values: FloatArray,
) -> FloatArray:
    dense = matrix.to_dense()
    solution = np.zeros(matrix.shape[0], dtype=np.float64)
    solution[constrained] = values
    free = np.setdiff1d(np.arange(matrix.shape[0], dtype=np.int64), constrained)
    if free.size:
        reduced = dense[np.ix_(free, free)]
        right_hand_side = load[free] - dense[np.ix_(free, constrained)] @ values
        try:
            solution[free] = np.linalg.solve(reduced, right_hand_side)
        except np.linalg.LinAlgError as error:
            raise ValueError("Constrained diffusion system is singular.") from error
    return solution


def solve_steady_diffusion(problem: SteadyDiffusionProblem) -> DiffusionResult:
    """Assemble and solve a scalar P1 steady-diffusion problem."""

    fluxes = tuple((condition.boundary_set, condition.flux) for condition in problem.neumann)
    matrix, load = assemble_diffusion(
        problem.mesh,
        conductivity=problem.conductivity,
        source=problem.source,
        boundary_fluxes=fluxes,
    )
    constrained, values = _collect_dirichlet(problem)
    solution = _solve_constrained(matrix, load, constrained, values)
    residual = matrix.matvec(solution) - load
    free = np.setdiff1d(np.arange(problem.mesh.node_count, dtype=np.int64), constrained)
    free_norm = float(np.linalg.norm(residual[free]))
    potential = float(0.5 * solution @ matrix.matvec(solution) - solution @ load)
    return DiffusionResult(
        nodal_values=solution,
        residual=residual,
        constrained_nodes=constrained,
        free_residual_norm=free_norm,
        total_applied_load=float(load.sum()),
        total_reaction=float(residual[constrained].sum()),
        potential_energy=potential,
    )
