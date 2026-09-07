# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Portable result artifacts for the serial reference kernel."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from numpy.typing import ArrayLike

from .mesh import TriangularMesh
from .volume_mesh import TetrahedralMesh


def write_legacy_vtk(
    path: str | Path,
    mesh: TriangularMesh,
    nodal_values: ArrayLike,
    *,
    field_name: str = "solution",
) -> Path:
    """Write a portable ASCII VTK triangle mesh with one nodal scalar field."""

    destination = Path(path)
    values = np.asarray(nodal_values, dtype=np.float64)
    if values.shape != (mesh.node_count,) or not np.all(np.isfinite(values)):
        raise ValueError("Nodal result must contain one finite scalar per mesh node.")
    normalized_name = field_name.strip().replace(" ", "_")
    if not normalized_name or any(character.isspace() for character in normalized_name):
        raise ValueError("VTK field name must be a nonempty token.")

    lines = [
        "# vtk DataFile Version 3.0",
        "AgentFEM Native steady diffusion",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        f"POINTS {mesh.node_count} double",
    ]
    lines.extend(f"{x:.17g} {y:.17g} 0" for x, y in mesh.points)
    lines.append(f"CELLS {mesh.cell_count} {mesh.cell_count * 4}")
    lines.extend(f"3 {int(a)} {int(b)} {int(c)}" for a, b, c in mesh.cells)
    lines.append(f"CELL_TYPES {mesh.cell_count}")
    lines.extend("5" for _ in range(mesh.cell_count))
    lines.extend(
        (
            f"POINT_DATA {mesh.node_count}",
            f"SCALARS {normalized_name} double 1",
            "LOOKUP_TABLE default",
        )
    )
    lines.extend(f"{value:.17g}" for value in values)
    content = "\n".join(lines) + "\n"

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(content)
            temporary_path = Path(handle.name)
        temporary_path.replace(destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return destination


def write_mechanics_vtk(
    path: str | Path,
    mesh: TriangularMesh | TetrahedralMesh,
    displacements: ArrayLike,
    cell_stress: ArrayLike,
) -> Path:
    """Write portable legacy VTK displacement vectors and Cauchy stress tensors."""

    destination = Path(path)
    dimension = int(mesh.points.shape[1])
    values = np.asarray(displacements, dtype=np.float64)
    stress = np.asarray(cell_stress, dtype=np.float64)
    components = 3 if dimension == 2 else 6
    if values.shape != (mesh.node_count, dimension) or not np.all(np.isfinite(values)):
        raise ValueError("Displacements must contain one finite vector per mesh node.")
    if stress.shape != (mesh.cell_count, components) or not np.all(np.isfinite(stress)):
        raise ValueError("Cell stress has an invalid shape or non-finite value.")
    if dimension == 2:
        points = np.column_stack((mesh.points, np.zeros(mesh.node_count)))
        vectors = np.column_stack((values, np.zeros(mesh.node_count)))
        width = 3
        cell_type = 5
        tensors = np.zeros((mesh.cell_count, 3, 3), dtype=np.float64)
        tensors[:, 0, 0] = stress[:, 0]
        tensors[:, 1, 1] = stress[:, 1]
        tensors[:, 0, 1] = tensors[:, 1, 0] = stress[:, 2]
    else:
        points = mesh.points
        vectors = values
        width = 4
        cell_type = 10
        tensors = np.zeros((mesh.cell_count, 3, 3), dtype=np.float64)
        tensors[:, 0, 0] = stress[:, 0]
        tensors[:, 1, 1] = stress[:, 1]
        tensors[:, 2, 2] = stress[:, 2]
        tensors[:, 0, 1] = tensors[:, 1, 0] = stress[:, 3]
        tensors[:, 1, 2] = tensors[:, 2, 1] = stress[:, 4]
        tensors[:, 0, 2] = tensors[:, 2, 0] = stress[:, 5]
    lines = [
        "# vtk DataFile Version 3.0",
        "AgentFEM Native mechanics result",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        f"POINTS {mesh.node_count} double",
    ]
    lines.extend(
        " ".join(f"{coordinate:.17g}" for coordinate in point) for point in points
    )
    lines.append(f"CELLS {mesh.cell_count} {mesh.cell_count * (width + 1)}")
    lines.extend(
        f"{width} " + " ".join(str(int(node)) for node in cell) for cell in mesh.cells
    )
    lines.append(f"CELL_TYPES {mesh.cell_count}")
    lines.extend(str(cell_type) for _ in range(mesh.cell_count))
    lines.extend((f"POINT_DATA {mesh.node_count}", "VECTORS displacement double"))
    lines.extend(
        " ".join(f"{component:.17g}" for component in vector) for vector in vectors
    )
    lines.extend((f"CELL_DATA {mesh.cell_count}", "TENSORS cauchy_stress double"))
    for tensor in tensors:
        lines.extend(
            " ".join(f"{component:.17g}" for component in row) for row in tensor
        )
    content = "\n".join(lines) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(content)
            temporary_path = Path(handle.name)
        temporary_path.replace(destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return destination
