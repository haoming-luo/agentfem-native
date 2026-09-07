# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Owned deterministic degree-of-freedom numbering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

IndexArray: TypeAlias = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class VectorDofMap:
    """Node-major, component-minor numbering for a fixed-width vector field."""

    node_count: int
    component_count: int

    def __post_init__(self) -> None:
        for value, name in (
            (self.node_count, "node_count"),
            (self.component_count, "component_count"),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")
        object.__setattr__(self, "node_count", int(self.node_count))
        object.__setattr__(self, "component_count", int(self.component_count))

    @property
    def size(self) -> int:
        return self.node_count * self.component_count

    def dof(self, node: int, component: int) -> int:
        if (
            isinstance(node, (bool, np.bool_))
            or not isinstance(node, (int, np.integer))
            or int(node) < 0
            or int(node) >= self.node_count
        ):
            raise ValueError("Node index is out of range for the DOF map.")
        if (
            isinstance(component, (bool, np.bool_))
            or not isinstance(component, (int, np.integer))
            or int(component) < 0
            or int(component) >= self.component_count
        ):
            raise ValueError("Component index is out of range for the DOF map.")
        return int(node) * self.component_count + int(component)

    def node_dofs(self, nodes: ArrayLike) -> IndexArray:
        raw = np.asarray(nodes)
        if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("Node indices must be a one-dimensional integer array.")
        checked = np.asarray(raw, dtype=np.int64)
        if np.any(checked < 0) or np.any(checked >= self.node_count):
            raise ValueError("Node index is out of range for the DOF map.")
        result = (
            checked[:, None] * self.component_count
            + np.arange(self.component_count, dtype=np.int64)[None, :]
        )
        result.setflags(write=False)
        return result

    def cell_dofs(self, cells: ArrayLike) -> IndexArray:
        raw = np.asarray(cells)
        if raw.ndim != 2 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError(
                "Cell connectivity must be a two-dimensional integer array."
            )
        checked = np.asarray(raw, dtype=np.int64)
        if np.any(checked < 0) or np.any(checked >= self.node_count):
            raise ValueError("Cell node index is out of range for the DOF map.")
        result = (
            checked[:, :, None] * self.component_count
            + np.arange(self.component_count, dtype=np.int64)[None, None, :]
        ).reshape(checked.shape[0], checked.shape[1] * self.component_count)
        result.setflags(write=False)
        return result
