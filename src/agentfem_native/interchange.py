# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Deterministic, reconstructable internal mesh/result interchange."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias
from uuid import uuid4

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .mesh import TriangularMesh
from .volume_mesh import TetrahedralMesh

FloatArray: TypeAlias = NDArray[np.float64]
Mesh: TypeAlias = TriangularMesh | TetrahedralMesh
INTERCHANGE_FORMAT = "agentfem-native.bundle/0.1"
MAX_BUNDLE_BYTES = 64 * 1024 * 1024


def _readonly_field(value: ArrayLike, expected: int, *, name: str) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.ndim not in {1, 2} or result.shape[0] != expected:
        raise ValueError(f"{name} must have one scalar or vector value per entity.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class NativeInterchangeBundle:
    mesh: Mesh
    point_fields: dict[str, FloatArray] = field(default_factory=dict)
    cell_fields: dict[str, FloatArray] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.mesh, (TriangularMesh, TetrahedralMesh)):
            raise TypeError("Interchange mesh must be triangular or tetrahedral.")
        point_fields = {
            _field_name(name): _readonly_field(
                value, self.mesh.node_count, name=f"Point field {name!r}"
            )
            for name, value in self.point_fields.items()
        }
        cell_fields = {
            _field_name(name): _readonly_field(
                value, self.mesh.cell_count, name=f"Cell field {name!r}"
            )
            for name, value in self.cell_fields.items()
        }
        metadata = dict(self.metadata)
        try:
            json.dumps(metadata, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Interchange metadata must be finite JSON data."
            ) from error
        object.__setattr__(self, "point_fields", point_fields)
        object.__setattr__(self, "cell_fields", cell_fields)
        object.__setattr__(self, "metadata", metadata)


def _field_name(name: object) -> str:
    if not isinstance(name, str) or not name or name.strip() != name:
        raise ValueError("Interchange field names must be nonempty trimmed strings.")
    return name


def _sets(values: dict[str, NDArray[np.int64]]) -> dict[str, list[object]]:
    return {
        name: np.asarray(indices).tolist() for name, indices in sorted(values.items())
    }


def _payload(bundle: NativeInterchangeBundle) -> dict[str, object]:
    mesh = bundle.mesh
    if isinstance(mesh, TriangularMesh):
        cell_type = "triangle3"
        boundary_sets = _sets(mesh.boundary_sets)
    else:
        cell_type = "tetrahedron4"
        boundary_sets = _sets(mesh.boundary_sets)
    return {
        "format": INTERCHANGE_FORMAT,
        "mesh": {
            "cell_type": cell_type,
            "points": mesh.points.tolist(),
            "cells": mesh.cells.tolist(),
            "node_sets": _sets(mesh.node_sets),
            "boundary_sets": boundary_sets,
            "cell_sets": _sets(mesh.cell_sets),
        },
        "point_fields": {
            name: values.tolist()
            for name, values in sorted(bundle.point_fields.items())
        },
        "cell_fields": {
            name: values.tolist() for name, values in sorted(bundle.cell_fields.items())
        },
        "metadata": bundle.metadata,
    }


def write_native_bundle(path: str | Path, bundle: NativeInterchangeBundle) -> Path:
    """Atomically write canonical UTF-8 JSON without platform-specific tools."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        _payload(bundle),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(encoded) > MAX_BUNDLE_BYTES:
        raise ValueError("Interchange bundle exceeds the 64 MiB internal limit.")
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(encoded)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object with string keys.")
    return value


def read_native_bundle(path: str | Path) -> NativeInterchangeBundle:
    """Read and fully validate a reconstructable Native bundle."""

    source = Path(path)
    if source.stat().st_size > MAX_BUNDLE_BYTES:
        raise ValueError("Interchange bundle exceeds the 64 MiB internal limit.")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Interchange bundle is not valid UTF-8 JSON.") from error
    root = _mapping(payload, name="Interchange root")
    if root.get("format") != INTERCHANGE_FORMAT:
        raise ValueError("Unsupported AgentFEM Native interchange format.")
    mesh_data = _mapping(root.get("mesh"), name="Interchange mesh")
    node_sets = _mapping(mesh_data.get("node_sets", {}), name="Node sets")
    boundary_sets = _mapping(mesh_data.get("boundary_sets", {}), name="Boundary sets")
    cell_sets = _mapping(mesh_data.get("cell_sets", {}), name="Cell sets")
    common = {
        "points": mesh_data.get("points"),
        "cells": mesh_data.get("cells"),
        "node_sets": node_sets,
        "boundary_sets": boundary_sets,
        "cell_sets": cell_sets,
    }
    cell_type = mesh_data.get("cell_type")
    if cell_type == "triangle3":
        mesh: Mesh = TriangularMesh(**common)
    elif cell_type == "tetrahedron4":
        mesh = TetrahedralMesh(**common)
    else:
        raise ValueError("Interchange cell type must be triangle3 or tetrahedron4.")
    point_fields = _mapping(root.get("point_fields", {}), name="Point fields")
    cell_fields = _mapping(root.get("cell_fields", {}), name="Cell fields")
    metadata = _mapping(root.get("metadata", {}), name="Metadata")
    return NativeInterchangeBundle(mesh, point_fields, cell_fields, metadata)
