# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Reference-cell definitions for the independent mathematical kernel."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray: TypeAlias = NDArray[np.float64]

REFERENCE_TRIANGLE_VERTICES: FloatArray = np.array(
    ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
    dtype=np.float64,
)
REFERENCE_TRIANGLE_VERTICES.setflags(write=False)

_P1_GRADIENTS: FloatArray = np.array(
    ((-1.0, -1.0), (1.0, 0.0), (0.0, 1.0)),
    dtype=np.float64,
)
_P1_GRADIENTS.setflags(write=False)


def _points(value: ArrayLike) -> FloatArray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim == 0 or points.shape[-1] != 2:
        raise ValueError("Reference points must have shape (..., 2).")
    if not np.all(np.isfinite(points)):
        raise ValueError("Reference points must contain only finite values.")
    return points


def p1_basis(points: ArrayLike) -> FloatArray:
    """Evaluate the three nodal P1 basis functions on the reference triangle.

    Points may have shape ``(2,)`` or ``(..., 2)``. The returned array has the
    same leading dimensions and a final basis-function dimension of length 3.
    Evaluation outside the triangle is allowed for algebraic use.
    """

    coordinates = _points(points)
    xi = coordinates[..., 0]
    eta = coordinates[..., 1]
    return np.stack((1.0 - xi - eta, xi, eta), axis=-1)


def p1_basis_gradients(points: ArrayLike | None = None) -> FloatArray:
    """Return constant reference gradients for the three P1 basis functions."""

    if points is None:
        return _P1_GRADIENTS.copy()
    coordinates = _points(points)
    shape = coordinates.shape[:-1] + _P1_GRADIENTS.shape
    return np.broadcast_to(_P1_GRADIENTS, shape).copy()


def inside_reference_triangle(
    points: ArrayLike,
    *,
    tolerance: float = 0.0,
) -> NDArray[np.bool_]:
    """Return whether points lie in the closed reference triangle."""

    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError("Tolerance must be finite and nonnegative.")
    coordinates = _points(points)
    xi = coordinates[..., 0]
    eta = coordinates[..., 1]
    return (
        (xi >= -tolerance)
        & (eta >= -tolerance)
        & (xi + eta <= 1.0 + tolerance)
    )
