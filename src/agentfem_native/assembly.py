# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Deterministic serial element integration and COO assembly."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import AffineTriangleMap
from .mesh import TriangularMesh
from .quadrature import triangle_rule
from .reference import p1_basis

FloatArray: TypeAlias = NDArray[np.float64]
ScalarField = float | Callable[[FloatArray], ArrayLike]
Conductivity = float | ArrayLike | Callable[[FloatArray], ArrayLike]


def _field_values(field: ScalarField, points: FloatArray, *, name: str) -> FloatArray:
    raw = field(points) if callable(field) else field
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim == 0:
        values = np.full(points.shape[0], float(values))
    if values.shape != (points.shape[0],) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must evaluate to one finite scalar per point.")
    return values


def _conductivity_tensors(conductivity: Conductivity, points: FloatArray) -> FloatArray:
    raw = conductivity(points) if callable(conductivity) else conductivity
    values = np.asarray(raw, dtype=np.float64)
    count = points.shape[0]
    if values.ndim == 0:
        tensors = np.broadcast_to(float(values) * np.eye(2), (count, 2, 2)).copy()
    elif values.shape == (count,):
        tensors = values[:, None, None] * np.broadcast_to(np.eye(2), (count, 2, 2))
    elif values.shape == (2, 2):
        tensors = np.broadcast_to(values, (count, 2, 2)).copy()
    elif values.shape == (count, 2, 2):
        tensors = values.copy()
    else:
        raise ValueError(
            "Conductivity must be a scalar, a 2x2 tensor, or evaluate to one per point."
        )
    if not np.all(np.isfinite(tensors)):
        raise ValueError("Conductivity must contain only finite values.")
    for tensor in tensors:
        scale = max(1.0, float(np.linalg.norm(tensor, ord=np.inf)))
        tolerance = 64.0 * np.finfo(np.float64).eps * scale
        if not np.allclose(tensor, tensor.T, rtol=0.0, atol=tolerance):
            raise ValueError("Conductivity tensor must be symmetric.")
        if float(np.linalg.eigvalsh(tensor).min()) <= tolerance:
            raise ValueError("Conductivity tensor must be positive definite.")
    return tensors


@dataclass(frozen=True, slots=True)
class COOMatrix:
    """Small provider-neutral coordinate sparse matrix with duplicate entries."""

    shape: tuple[int, int]
    rows: NDArray[np.int64]
    columns: NDArray[np.int64]
    data: FloatArray

    def __post_init__(self) -> None:
        if len(self.shape) != 2 or self.shape[0] < 0 or self.shape[1] < 0:
            raise ValueError(
                "Sparse matrix shape must contain two nonnegative dimensions."
            )
        rows = np.array(self.rows, dtype=np.int64, copy=True)
        columns = np.array(self.columns, dtype=np.int64, copy=True)
        data = np.array(self.data, dtype=np.float64, copy=True)
        if rows.ndim != 1 or columns.shape != rows.shape or data.shape != rows.shape:
            raise ValueError(
                "COO rows, columns, and data must be equal-length vectors."
            )
        if np.any(rows < 0) or np.any(rows >= self.shape[0]):
            raise ValueError("COO row index is out of range.")
        if np.any(columns < 0) or np.any(columns >= self.shape[1]):
            raise ValueError("COO column index is out of range.")
        if not np.all(np.isfinite(data)):
            raise ValueError("COO entries must be finite.")
        for array in (rows, columns, data):
            array.setflags(write=False)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "data", data)

    def to_dense(self) -> FloatArray:
        result = np.zeros(self.shape, dtype=np.float64)
        np.add.at(result, (self.rows, self.columns), self.data)
        return result

    def matvec(self, vector: ArrayLike) -> FloatArray:
        values = np.asarray(vector, dtype=np.float64)
        if values.shape != (self.shape[1],):
            raise ValueError("Vector size does not match sparse matrix columns.")
        result = np.zeros(self.shape[0], dtype=np.float64)
        np.add.at(result, self.rows, self.data * values[self.columns])
        return result


def p1_diffusion_stiffness(
    vertices: ArrayLike, conductivity: Conductivity
) -> FloatArray:
    """Return one scalar P1 diffusion element matrix."""

    geometry = AffineTriangleMap(np.asarray(vertices, dtype=np.float64))
    gradients = geometry.p1_gradients()
    rule = triangle_rule(2)
    physical_points = geometry.map_points(rule.points)
    tensors = _conductivity_tensors(conductivity, physical_points)
    result = np.zeros((3, 3), dtype=np.float64)
    for weight, tensor in zip(rule.weights, tensors, strict=True):
        result += (
            geometry.integration_scale * weight * (gradients @ tensor @ gradients.T)
        )
    return result


def p1_source_load(vertices: ArrayLike, source: ScalarField) -> FloatArray:
    """Integrate one scalar source against the three P1 test functions."""

    geometry = AffineTriangleMap(np.asarray(vertices, dtype=np.float64))
    rule = triangle_rule(2)
    physical_points = geometry.map_points(rule.points)
    values = _field_values(source, physical_points, name="Source")
    basis = p1_basis(rule.points)
    return geometry.integration_scale * (basis.T @ (rule.weights * values))


def p1_boundary_flux_load(edge_vertices: ArrayLike, flux: ScalarField) -> FloatArray:
    """Integrate an outward scalar flux against two linear edge functions."""

    vertices = np.asarray(edge_vertices, dtype=np.float64)
    if vertices.shape != (2, 2) or not np.all(np.isfinite(vertices)):
        raise ValueError("Boundary edge vertices must have shape (2, 2) and be finite.")
    tangent = vertices[1] - vertices[0]
    length = float(np.linalg.norm(tangent))
    if length <= 32.0 * np.finfo(np.float64).eps:
        raise ValueError("Boundary edge is degenerate.")
    offset = 1.0 / (2.0 * np.sqrt(3.0))
    coordinates = np.array((0.5 - offset, 0.5 + offset))
    weights = np.array((0.5, 0.5))
    points = (1.0 - coordinates[:, None]) * vertices[0] + coordinates[
        :, None
    ] * vertices[1]
    values = _field_values(flux, points, name="Boundary flux")
    basis = np.column_stack((1.0 - coordinates, coordinates))
    return length * (basis.T @ (weights * values))


def assemble_diffusion(
    mesh: TriangularMesh,
    *,
    conductivity: Conductivity,
    source: ScalarField = 0.0,
    boundary_fluxes: tuple[tuple[str, ScalarField], ...] = (),
    cell_conductivities: dict[int, Conductivity] | None = None,
) -> tuple[COOMatrix, FloatArray]:
    """Assemble the serial P1 diffusion matrix and load vector."""

    entry_count = mesh.cell_count * 9
    rows = np.empty(entry_count, dtype=np.int64)
    columns = np.empty(entry_count, dtype=np.int64)
    data = np.empty(entry_count, dtype=np.float64)
    load = np.zeros(mesh.node_count, dtype=np.float64)

    cursor = 0
    overrides = {} if cell_conductivities is None else dict(cell_conductivities)
    if any(index < 0 or index >= mesh.cell_count for index in overrides):
        raise ValueError("Cell conductivity override index is out of range.")
    for cell_index, cell in enumerate(mesh.cells):
        local_conductivity = overrides.get(cell_index, conductivity)
        local_stiffness = p1_diffusion_stiffness(mesh.points[cell], local_conductivity)
        local_load = p1_source_load(mesh.points[cell], source)
        for local_row, global_row in enumerate(cell):
            load[global_row] += local_load[local_row]
            for local_column, global_column in enumerate(cell):
                rows[cursor] = global_row
                columns[cursor] = global_column
                data[cursor] = local_stiffness[local_row, local_column]
                cursor += 1

    for set_name, flux in boundary_fluxes:
        for edge in mesh.boundary_edges(set_name):
            local_load = p1_boundary_flux_load(mesh.points[edge], flux)
            load[edge] += local_load

    return COOMatrix((mesh.node_count, mesh.node_count), rows, columns, data), load
