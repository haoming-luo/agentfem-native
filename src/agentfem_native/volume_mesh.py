# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Owned serial tetrahedral meshes for the Gate 2 three-dimensional spine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IndexArray = NDArray[np.int64]


def _points(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.ndim != 2 or result.shape[1] != 3 or not np.all(np.isfinite(result)):
        raise ValueError("Tetrahedral points must have shape (n, 3) and be finite.")
    result.setflags(write=False)
    return result


def _indices(value: ArrayLike, width: int | None, *, name: str) -> IndexArray:
    raw = np.asarray(value)
    dimension = 1 if width is None else 2
    if raw.ndim != dimension or (width is not None and raw.shape[1] != width):
        suffix = "(n,)" if width is None else f"(n, {width})"
        raise ValueError(f"{name} must have shape {suffix}.")
    if not np.issubdtype(raw.dtype, np.integer):
        numeric = np.asarray(raw, dtype=np.float64)
        if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
            raise ValueError(f"{name} must contain finite integers.")
    result = np.array(raw, dtype=np.int64, copy=True)
    result.setflags(write=False)
    return result


def _sets(
    values: Mapping[str, ArrayLike],
    *,
    width: int | None,
    upper_bound: int,
) -> Mapping[str, IndexArray]:
    result: dict[str, IndexArray] = {}
    for raw_name, value in values.items():
        name = str(raw_name).strip()
        if not name or name in result:
            raise ValueError("Mesh set names must be nonempty and unique.")
        indices = _indices(value, width, name=f"Mesh set {name!r}")
        if np.any(indices < 0) or np.any(indices >= upper_bound):
            raise ValueError(f"Mesh set {name!r} contains an out-of-range index.")
        normalized = np.sort(indices, axis=1) if width is not None else indices
        if np.unique(normalized, axis=0).shape[0] != indices.shape[0]:
            raise ValueError(f"Mesh set {name!r} contains duplicate entities.")
        result[name] = indices
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class TetrahedralMesh:
    """Zero-based affine T4 mesh with named nodes, faces, and cells."""

    points: FloatArray
    cells: IndexArray
    node_sets: Mapping[str, IndexArray] = field(default_factory=dict)
    boundary_sets: Mapping[str, IndexArray] = field(default_factory=dict)
    cell_sets: Mapping[str, IndexArray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = _points(self.points)
        cells = _indices(self.cells, 4, name="Tetrahedral cells")
        if points.shape[0] < 4 or cells.shape[0] < 1:
            raise ValueError(
                "A tetrahedral mesh requires at least four points and one cell."
            )
        if np.any(cells < 0) or np.any(cells >= points.shape[0]):
            raise ValueError("Tetrahedral connectivity contains an out-of-range node.")
        if np.any(np.apply_along_axis(lambda row: np.unique(row).size, 1, cells) != 4):
            raise ValueError("Each tetrahedron must reference four distinct nodes.")
        vertices = points[cells]
        jacobians = np.stack(
            (
                vertices[:, 1] - vertices[:, 0],
                vertices[:, 2] - vertices[:, 0],
                vertices[:, 3] - vertices[:, 0],
            ),
            axis=2,
        )
        determinants = np.linalg.det(jacobians)
        scales = np.max(np.linalg.norm(jacobians, axis=1), axis=1)
        threshold = 64.0 * np.finfo(np.float64).eps * scales**3
        if np.any(np.abs(determinants) <= threshold):
            raise ValueError("Tetrahedron is degenerate or numerically singular.")
        node_sets = _sets(self.node_sets, width=None, upper_bound=points.shape[0])
        boundary_sets = _sets(self.boundary_sets, width=3, upper_bound=points.shape[0])
        cell_sets = _sets(self.cell_sets, width=None, upper_bound=cells.shape[0])
        counts: dict[tuple[int, int, int], int] = {}
        for cell in cells:
            for local in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
                key = tuple(sorted(int(cell[index]) for index in local))
                counts[key] = counts.get(key, 0) + 1
        for name, faces in boundary_sets.items():
            for face in faces:
                if counts.get(tuple(sorted(map(int, face))), 0) != 1:
                    raise ValueError(
                        f"Face {tuple(face)} in {name!r} is not a boundary face."
                    )
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

    def boundary_faces(self, name: str) -> IndexArray:
        try:
            return self.boundary_sets[name]
        except KeyError as error:
            raise KeyError(f"Unknown boundary set {name!r}.") from error

    def cells_in(self, name: str) -> IndexArray:
        try:
            return self.cell_sets[name]
        except KeyError as error:
            raise KeyError(f"Unknown cell set {name!r}.") from error


def unit_cube_tetrahedra(resolution: int = 1) -> TetrahedralMesh:
    """Create a conforming six-T4-per-cube mesh with named Cartesian faces."""

    if (
        isinstance(resolution, bool)
        or not isinstance(resolution, int)
        or resolution < 1
    ):
        raise ValueError("Cube resolution must be a positive integer.")
    axis = np.linspace(0.0, 1.0, resolution + 1)
    points = np.asarray(
        [
            (axis[i], axis[j], axis[k])
            for k in range(resolution + 1)
            for j in range(resolution + 1)
            for i in range(resolution + 1)
        ],
        dtype=np.float64,
    )

    def node(i: int, j: int, k: int) -> int:
        return (k * (resolution + 1) + j) * (resolution + 1) + i

    cells: list[tuple[int, int, int, int]] = []
    for k in range(resolution):
        for j in range(resolution):
            for i in range(resolution):
                a = node(i, j, k)
                b = node(i + 1, j, k)
                c = node(i + 1, j + 1, k)
                d = node(i, j + 1, k)
                e = node(i, j, k + 1)
                f = node(i + 1, j, k + 1)
                g = node(i + 1, j + 1, k + 1)
                h = node(i, j + 1, k + 1)
                cells.extend(
                    (
                        (a, b, c, g),
                        (a, c, d, g),
                        (a, d, h, g),
                        (a, h, e, g),
                        (a, e, f, g),
                        (a, f, b, g),
                    )
                )
    cell_array = np.asarray(cells, dtype=np.int64)
    face_counts: dict[tuple[int, int, int], tuple[int, tuple[int, int, int]]] = {}
    for cell in cell_array:
        for local in ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)):
            face = tuple(int(cell[index]) for index in local)
            key = tuple(sorted(face))
            count, _ = face_counts.get(key, (0, face))
            face_counts[key] = (count + 1, face)
    boundaries: dict[str, list[tuple[int, int, int]]] = {
        "left": [],
        "right": [],
        "front": [],
        "back": [],
        "bottom": [],
        "top": [],
    }
    for count, face in face_counts.values():
        if count != 1:
            continue
        coordinates = points[np.asarray(face)]
        for name, axis_index, value in (
            ("left", 0, 0.0),
            ("right", 0, 1.0),
            ("front", 1, 0.0),
            ("back", 1, 1.0),
            ("bottom", 2, 0.0),
            ("top", 2, 1.0),
        ):
            if np.all(coordinates[:, axis_index] == value):
                boundaries[name].append(face)
                break
    node_sets = {
        name: np.flatnonzero(points[:, axis_index] == value)
        for name, axis_index, value in (
            ("left", 0, 0.0),
            ("right", 0, 1.0),
            ("front", 1, 0.0),
            ("back", 1, 1.0),
            ("bottom", 2, 0.0),
            ("top", 2, 1.0),
        )
    }
    node_sets["boundary"] = np.unique(np.concatenate(tuple(node_sets.values())))
    return TetrahedralMesh(
        points,
        cell_array,
        node_sets=node_sets,
        boundary_sets={
            name: np.asarray(faces, dtype=np.int64)
            for name, faces in boundaries.items()
        },
    )
