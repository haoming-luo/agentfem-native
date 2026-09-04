# SPDX-License-Identifier: LicenseRef-AgentFEM-Native-Draft
"""Auditable quadrature rules for the reference triangle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .reference import inside_reference_triangle


FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TriangleQuadrature:
    """Points and measure-inclusive weights on the reference triangle."""

    points: FloatArray
    weights: FloatArray
    exact_degree: int

    def __post_init__(self) -> None:
        points = np.array(self.points, dtype=np.float64, copy=True)
        weights = np.array(self.weights, dtype=np.float64, copy=True)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Quadrature points must have shape (n, 2).")
        if weights.shape != (points.shape[0],):
            raise ValueError("Quadrature weights must have shape (n,).")
        if self.exact_degree < 0:
            raise ValueError("Exact polynomial degree must be nonnegative.")
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(weights)):
            raise ValueError("Quadrature data must be finite.")
        if not np.all(inside_reference_triangle(points, tolerance=1.0e-15)):
            raise ValueError("Quadrature points must lie in the reference triangle.")
        if np.any(weights <= 0.0):
            raise ValueError("Quadrature weights must be positive.")
        points.setflags(write=False)
        weights.setflags(write=False)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "weights", weights)

    def integrate_values(self, values: ArrayLike) -> NDArray[np.float64] | np.float64:
        """Integrate values whose leading dimension indexes quadrature points."""

        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 0 or array.shape[0] != self.points.shape[0]:
            raise ValueError(
                "Values must have the quadrature-point count as their first dimension."
            )
        result = np.tensordot(self.weights, array, axes=(0, 0))
        return np.float64(result) if np.ndim(result) == 0 else result


def triangle_rule(exact_degree: int) -> TriangleQuadrature:
    """Return the smallest available rule exact through the requested degree."""

    if isinstance(exact_degree, bool) or not isinstance(exact_degree, int):
        raise TypeError("Exact degree must be an integer.")
    if exact_degree < 0:
        raise ValueError("Exact degree must be nonnegative.")
    if exact_degree <= 1:
        return TriangleQuadrature(
            points=np.array(((1.0 / 3.0, 1.0 / 3.0),)),
            weights=np.array((0.5,)),
            exact_degree=1,
        )
    if exact_degree <= 2:
        return TriangleQuadrature(
            points=np.array(
                (
                    (1.0 / 6.0, 1.0 / 6.0),
                    (2.0 / 3.0, 1.0 / 6.0),
                    (1.0 / 6.0, 2.0 / 3.0),
                )
            ),
            weights=np.full(3, 1.0 / 6.0),
            exact_degree=2,
        )
    raise NotImplementedError("Reference triangle quadrature above degree 2 is unavailable.")


def integrate_reference(
    function: Callable[[FloatArray], ArrayLike],
    *,
    exact_degree: int,
) -> NDArray[np.float64] | np.float64:
    """Integrate a vectorized callable over the reference triangle."""

    rule = triangle_rule(exact_degree)
    return rule.integrate_values(function(rule.points))
