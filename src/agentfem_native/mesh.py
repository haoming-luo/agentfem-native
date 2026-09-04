# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Owned serial triangle meshes for the Gate 1 reference kernel."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import AffineTriangleMap


FloatArray: TypeAlias = NDArray[np.float64]
IndexArray: TypeAlias = NDArray[np.int64]


def _readonly_float_array(value: ArrayLike, shape_tail: tuple[int, ...]) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != len(shape_tail) + 1 or array.shape[1:] != shape_tail:
        raise ValueError(f"Expected array with shape (n, {', '.join(map(str, shape_tail))}).")
    if not np.all(np.isfinite(array)):
        raise ValueError("Mesh coordinates must be finite.")
    array.setflags(write=False)
    return array


def _readonly_indices(value: ArrayLike, width: int | None = None) -> IndexArray:
    numeric = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
        raise ValueError("Mesh indices must be finite integers.")
    if np.any(numeric < np.iinfo(np.int64).min) or np.any(numeric > np.iinfo(np.int64).max):
        raise ValueError("Mesh index cannot be represented as a 64-bit integer.")
    array = np.array(numeric, dtype=np.int64, copy=True)
    expected_dimension = 1 if width is None else 2
    if array.ndim != expected_dimension or (width is not None and array.shape[1] != width):
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


@dataclass(frozen=True, slots=True)
class TriangularMesh:
    """A serial, zero-based, owned 2D triangle mesh with named boundary sets."""

    points: FloatArray
    cells: IndexArray
    node_sets: Mapping[str, IndexArray] = field(default_factory=dict)
    boundary_sets: Mapping[str, IndexArray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = _readonly_float_array(self.points, (2,))
        cells = _readonly_indices(self.cells, 3)
        if points.shape[0] < 3 or cells.shape[0] < 1:
            raise ValueError("A triangle mesh requires at least three points and one cell.")
        if np.any(cells < 0) or np.any(cells >= points.shape[0]):
            raise ValueError("Cell connectivity contains an out-of-range node index.")
        for cell in cells:
            if np.unique(cell).size != 3:
                raise ValueError("Each triangle cell must reference three distinct nodes.")
            AffineTriangleMap(points[cell])

        node_sets = _named_indices(self.node_sets, width=None, node_count=points.shape[0])
        boundary_sets = _named_indices(
            self.boundary_sets,
            width=2,
            node_count=points.shape[0],
        )
        boundary_edges = self._boundary_edge_keys(cells)
        for name, edges in boundary_sets.items():
            for edge in edges:
                key = tuple(sorted((int(edge[0]), int(edge[1]))))
                if key not in boundary_edges:
                    raise ValueError(f"Edge {tuple(edge)} in {name!r} is not a mesh boundary edge.")

        object.__setattr__(self, "points", points)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "node_sets", node_sets)
        object.__setattr__(self, "boundary_sets", boundary_sets)

    @staticmethod
    def _boundary_edge_keys(cells: IndexArray) -> set[tuple[int, int]]:
        counts: dict[tuple[int, int], int] = {}
        for cell in cells:
            for first, second in ((cell[0], cell[1]), (cell[1], cell[2]), (cell[2], cell[0])):
                key = tuple(sorted((int(first), int(second))))
                counts[key] = counts.get(key, 0) + 1
        return {edge for edge, count in counts.items() if count == 1}

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

    points = np.array(
        [(i / nx, j / ny) for j in range(ny + 1) for i in range(nx + 1)],
        dtype=np.float64,
    )

    def node(i: int, j: int) -> int:
        return j * (nx + 1) + i

    cells: list[tuple[int, int, int]] = []
    for j in range(ny):
        for i in range(nx):
            lower_left = node(i, j)
            lower_right = node(i + 1, j)
            upper_left = node(i, j + 1)
            upper_right = node(i + 1, j + 1)
            cells.extend(
                (
                    (lower_left, lower_right, upper_right),
                    (lower_left, upper_right, upper_left),
                )
            )

    node_sets = {
        "bottom": np.array([node(i, 0) for i in range(nx + 1)]),
        "right": np.array([node(nx, j) for j in range(ny + 1)]),
        "top": np.array([node(i, ny) for i in range(nx + 1)]),
        "left": np.array([node(0, j) for j in range(ny + 1)]),
    }
    node_sets["boundary"] = np.unique(np.concatenate(tuple(node_sets.values())))
    boundary_sets = {
        "bottom": np.array([(node(i, 0), node(i + 1, 0)) for i in range(nx)]),
        "right": np.array([(node(nx, j), node(nx, j + 1)) for j in range(ny)]),
        "top": np.array([(node(i + 1, ny), node(i, ny)) for i in range(nx)]),
        "left": np.array([(node(0, j + 1), node(0, j)) for j in range(ny)]),
    }
    return TriangularMesh(
        points=points,
        cells=np.array(cells),
        node_sets=node_sets,
        boundary_sets=boundary_sets,
    )
