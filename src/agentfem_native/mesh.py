# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Owned serial triangle meshes for the Gate 1 reference kernel."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IndexArray: TypeAlias = NDArray[np.int64]
MESH_VALIDATION_CHUNK_SIZE = 65_536


def _readonly_float_array(value: ArrayLike, shape_tail: tuple[int, ...]) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != len(shape_tail) + 1 or array.shape[1:] != shape_tail:
        raise ValueError(
            f"Expected array with shape (n, {', '.join(map(str, shape_tail))})."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("Mesh coordinates must be finite.")
    array.setflags(write=False)
    return array


def _readonly_indices(value: ArrayLike, width: int | None = None) -> IndexArray:
    numeric = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
        raise ValueError("Mesh indices must be finite integers.")
    if np.any(numeric < np.iinfo(np.int64).min) or np.any(
        numeric > np.iinfo(np.int64).max
    ):
        raise ValueError("Mesh index cannot be represented as a 64-bit integer.")
    array = np.array(numeric, dtype=np.int64, copy=True)
    expected_dimension = 1 if width is None else 2
    if array.ndim != expected_dimension or (
        width is not None and array.shape[1] != width
    ):
        suffix = "(n,)" if width is None else f"(n, {width})"
        raise ValueError(f"Expected index array with shape {suffix}.")
    array.setflags(write=False)
    return array


def _named_indices(
    values: Mapping[str, ArrayLike],
    *,
    width: int | None,
    node_count: int,
) -> Mapping[str, IndexArray]:
    result: dict[str, IndexArray] = {}
    for raw_name, value in values.items():
        name = str(raw_name).strip()
        if not name or name in result:
            raise ValueError("Mesh set names must be nonempty and unique.")
        indices = _readonly_indices(value, width)
        if np.any(indices < 0) or np.any(indices >= node_count):
            raise ValueError(f"Mesh set {name!r} contains an out-of-range node index.")
        if width is None and np.unique(indices).size != indices.size:
            raise ValueError(f"Node set {name!r} contains duplicate indices.")
        if width is not None:
            normalized = np.sort(indices, axis=1)
            if np.unique(normalized, axis=0).shape[0] != indices.shape[0]:
                raise ValueError(f"Boundary set {name!r} contains duplicate edges.")
        result[name] = indices
    return MappingProxyType(result)


def _validate_cells(points: FloatArray, cells: IndexArray) -> None:
    duplicate_nodes = (
        (cells[:, 0] == cells[:, 1])
        | (cells[:, 1] == cells[:, 2])
        | (cells[:, 2] == cells[:, 0])
    )
    if np.any(duplicate_nodes):
        raise ValueError("Each triangle cell must reference three distinct nodes.")
    for start in range(0, cells.shape[0], MESH_VALIDATION_CHUNK_SIZE):
        chunk = cells[start : start + MESH_VALIDATION_CHUNK_SIZE]
        vertices = points[chunk]
        first_columns = vertices[:, 1] - vertices[:, 0]
        second_columns = vertices[:, 2] - vertices[:, 0]
        determinants = (
            first_columns[:, 0] * second_columns[:, 1]
            - second_columns[:, 0] * first_columns[:, 1]
        )
        edge_scale = np.maximum(
            np.linalg.norm(first_columns, axis=1),
            np.linalg.norm(second_columns, axis=1),
        )
        threshold = 32.0 * np.finfo(np.float64).eps * edge_scale**2
        if np.any(np.abs(determinants) <= threshold):
            raise ValueError("Triangle is degenerate or numerically singular.")


def _edge_keys(edges: IndexArray, node_count: int) -> IndexArray:
    lower = np.minimum(edges[..., 0], edges[..., 1])
    upper = np.maximum(edges[..., 0], edges[..., 1])
    return lower * node_count + upper


def _validate_declared_boundaries(
    cells: IndexArray,
    boundary_sets: Mapping[str, IndexArray],
    node_count: int,
) -> None:
    if not boundary_sets:
        return
    declared = np.unique(
        np.concatenate(
            [_edge_keys(edges, node_count) for edges in boundary_sets.values()]
        )
    )
    counts = np.zeros(declared.size, dtype=np.int64)
    local_edges = np.array(((0, 1), (1, 2), (2, 0)), dtype=np.int64)
    for start in range(0, cells.shape[0], MESH_VALIDATION_CHUNK_SIZE):
        chunk = cells[start : start + MESH_VALIDATION_CHUNK_SIZE]
        keys = _edge_keys(chunk[:, local_edges], node_count).ravel()
        positions = np.searchsorted(declared, keys)
        inside = positions < declared.size
        matched = np.zeros(keys.size, dtype=bool)
        matched[inside] = declared[positions[inside]] == keys[inside]
        np.add.at(counts, positions[matched], 1)
    for name, edges in boundary_sets.items():
        keys = _edge_keys(edges, node_count)
        positions = np.searchsorted(declared, keys)
        invalid = counts[positions] != 1
        if np.any(invalid):
            edge = tuple(edges[int(np.flatnonzero(invalid)[0])])
            raise ValueError(f"Edge {edge} in {name!r} is not a mesh boundary edge.")


@dataclass(frozen=True, slots=True)
class TriangularMesh:
    """A serial, zero-based, owned 2D triangle mesh with named boundary sets."""

    points: FloatArray
    cells: IndexArray
    node_sets: Mapping[str, IndexArray] = field(default_factory=dict)
    boundary_sets: Mapping[str, IndexArray] = field(default_factory=dict)
    cell_sets: Mapping[str, IndexArray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = _readonly_float_array(self.points, (2,))
        cells = _readonly_indices(self.cells, 3)
        if points.shape[0] < 3 or cells.shape[0] < 1:
            raise ValueError(
                "A triangle mesh requires at least three points and one cell."
            )
        if np.any(cells < 0) or np.any(cells >= points.shape[0]):
            raise ValueError("Cell connectivity contains an out-of-range node index.")
        _validate_cells(points, cells)

        node_sets = _named_indices(
            self.node_sets, width=None, node_count=points.shape[0]
        )
        boundary_sets = _named_indices(
            self.boundary_sets,
            width=2,
            node_count=points.shape[0],
        )
        cell_sets = _named_indices(
            self.cell_sets,
            width=None,
            node_count=cells.shape[0],
        )
        _validate_declared_boundaries(cells, boundary_sets, points.shape[0])

        object.__setattr__(self, "points", points)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "node_sets", node_sets)
        object.__setattr__(self, "boundary_sets", boundary_sets)
        object.__setattr__(self, "cell_sets", cell_sets)

    @property
    def node_count(self) -> int:
        return int(self.points.shape[0])

    @property
    def cell_count(self) -> int:
        return int(self.cells.shape[0])

    def nodes(self, name: str) -> IndexArray:
        try:
            return self.node_sets[name]
        except KeyError as error:
            raise KeyError(f"Unknown node set {name!r}.") from error

    def boundary_edges(self, name: str) -> IndexArray:
        try:
            return self.boundary_sets[name]
        except KeyError as error:
            raise KeyError(f"Unknown boundary set {name!r}.") from error

    def cells_in(self, name: str) -> IndexArray:
        try:
            return self.cell_sets[name]
        except KeyError as error:
            raise KeyError(f"Unknown cell set {name!r}.") from error


def unit_square_two_triangles() -> TriangularMesh:
    """Return the deterministic two-cell mesh used by the first diffusion slice."""

    return TriangularMesh(
        points=np.array(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        cells=np.array(((0, 1, 2), (0, 2, 3))),
        node_sets={"left": np.array((0, 3)), "right": np.array((1, 2))},
        boundary_sets={
            "bottom": np.array(((0, 1),)),
            "right": np.array(((1, 2),)),
            "top": np.array(((2, 3),)),
            "left": np.array(((3, 0),)),
        },
    )


def unit_square_triangles(nx: int, ny: int | None = None) -> TriangularMesh:
    """Create a structured unit-square triangle mesh with named boundaries."""

    if isinstance(nx, bool) or not isinstance(nx, int) or nx < 1:
        raise ValueError("nx must be a positive integer.")
    if ny is None:
        ny = nx
    if isinstance(ny, bool) or not isinstance(ny, int) or ny < 1:
        raise ValueError("ny must be a positive integer.")

    x_coordinates = np.linspace(0.0, 1.0, nx + 1)
    y_coordinates = np.linspace(0.0, 1.0, ny + 1)
    x_grid, y_grid = np.meshgrid(x_coordinates, y_coordinates)
    points = np.column_stack((x_grid.ravel(), y_grid.ravel()))

    lower_left = (
        np.arange(ny, dtype=np.int64)[:, None] * (nx + 1)
        + np.arange(nx, dtype=np.int64)[None, :]
    ).ravel()
    lower_right = lower_left + 1
    upper_left = lower_left + (nx + 1)
    upper_right = upper_left + 1
    first_triangles = np.column_stack((lower_left, lower_right, upper_right))
    second_triangles = np.column_stack((lower_left, upper_right, upper_left))
    cells = np.stack((first_triangles, second_triangles), axis=1).reshape(-1, 3)

    node_sets = {
        "bottom": np.arange(nx + 1, dtype=np.int64),
        "right": np.arange(ny + 1, dtype=np.int64) * (nx + 1) + nx,
        "top": ny * (nx + 1) + np.arange(nx + 1, dtype=np.int64),
        "left": np.arange(ny + 1, dtype=np.int64) * (nx + 1),
    }
    node_sets["boundary"] = np.unique(np.concatenate(tuple(node_sets.values())))
    boundary_sets = {
        "bottom": np.column_stack((node_sets["bottom"][:-1], node_sets["bottom"][1:])),
        "right": np.column_stack((node_sets["right"][:-1], node_sets["right"][1:])),
        "top": np.column_stack((node_sets["top"][1:], node_sets["top"][:-1])),
        "left": np.column_stack((node_sets["left"][1:], node_sets["left"][:-1])),
    }
    return TriangularMesh(
        points=points,
        cells=cells,
        node_sets=node_sets,
        boundary_sets=boundary_sets,
    )
