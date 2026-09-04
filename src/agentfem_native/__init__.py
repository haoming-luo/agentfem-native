# SPDX-License-Identifier: LicenseRef-AgentFEM-Native-Draft
"""Independent executable mathematics for the AgentFEM Native Engine."""

from .geometry import AffineTriangleMap
from .quadrature import TriangleQuadrature, integrate_reference, triangle_rule
from .reference import (
    REFERENCE_TRIANGLE_VERTICES,
    inside_reference_triangle,
    p1_basis,
    p1_basis_gradients,
)

__version__ = "0.0.0"

__all__ = (
    "AffineTriangleMap",
    "REFERENCE_TRIANGLE_VERTICES",
    "TriangleQuadrature",
    "inside_reference_triangle",
    "integrate_reference",
    "p1_basis",
    "p1_basis_gradients",
    "triangle_rule",
)
