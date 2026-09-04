# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Affine geometry mapping for independently defined P1 triangles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .reference import p1_basis_gradients


FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AffineTriangleMap:
    """Affine map from the reference triangle to three physical 2D vertices."""

    vertices: FloatArray
    _jacobian: FloatArray = field(init=False, repr=False)
    _determinant: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        vertices = np.array(self.vertices, dtype=np.float64, copy=True)
        if vertices.shape != (3, 2):
            raise ValueError("Triangle vertices must have shape (3, 2).")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("Triangle vertices must be finite.")

        jacobian = np.column_stack(
            (vertices[1] - vertices[0], vertices[2] - vertices[0])
        )
        determinant = float(np.linalg.det(jacobian))
        edge_scale = max(
            float(np.linalg.norm(jacobian[:, 0])),
            float(np.linalg.norm(jacobian[:, 1])),
        )
        threshold = 32.0 * np.finfo(np.float64).eps * edge_scale**2
        if abs(determinant) <= threshold:
            raise ValueError("Triangle is degenerate or numerically singular.")

        vertices.setflags(write=False)
        jacobian.setflags(write=False)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "_jacobian", jacobian)
        object.__setattr__(self, "_determinant", determinant)

    @property
    def jacobian(self) -> FloatArray:
        """Return a read-only copy of the affine Jacobian."""

        result = self._jacobian.copy()
        result.setflags(write=False)
        return result

    @property
    def signed_determinant(self) -> float:
        """Return the orientation-sensitive determinant of the map."""

        return self._determinant

    @property
    def integration_scale(self) -> float:
        """Return the positive reference-to-physical area scale."""

        return abs(self._determinant)

    @property
    def area(self) -> float:
        """Return the positive physical triangle area."""

        return 0.5 * self.integration_scale

    def map_points(self, reference_points: ArrayLike) -> FloatArray:
        """Map points with shape ``(..., 2)`` into physical coordinates."""

        points = np.asarray(reference_points, dtype=np.float64)
        if points.ndim == 0 or points.shape[-1] != 2:
            raise ValueError("Reference points must have shape (..., 2).")
        if not np.all(np.isfinite(points)):
            raise ValueError("Reference points must be finite.")
        return points @ self._jacobian.T + self.vertices[0]

    def transform_gradients(self, reference_gradients: ArrayLike) -> FloatArray:
        """Transform row-wise reference gradients into physical gradients."""

        gradients = np.asarray(reference_gradients, dtype=np.float64)
        if gradients.ndim == 0 or gradients.shape[-1] != 2:
            raise ValueError("Reference gradients must have shape (..., 2).")
        if not np.all(np.isfinite(gradients)):
            raise ValueError("Reference gradients must be finite.")
        return gradients @ np.linalg.inv(self._jacobian)

    def p1_gradients(self) -> FloatArray:
        """Return physical gradients of the three P1 basis functions."""

        return self.transform_gradients(p1_basis_gradients())
