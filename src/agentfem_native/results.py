# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Portable result artifacts for the serial reference kernel."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from numpy.typing import ArrayLike

from .mesh import TriangularMesh


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
