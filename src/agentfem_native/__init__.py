# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Independent executable mathematics for the AgentFEM Native Engine."""

from .geometry import AffineTriangleMap
from .diffusion import (
    DiffusionResult,
    DirichletCondition,
    NeumannCondition,
    SteadyDiffusionProblem,
    solve_steady_diffusion,
)
from .mesh import TriangularMesh, unit_square_triangles, unit_square_two_triangles
from .quadrature import TriangleQuadrature, integrate_reference, triangle_rule
from .reference import (
    REFERENCE_TRIANGLE_VERTICES,
    inside_reference_triangle,
    p1_basis,
    p1_basis_gradients,
)
from .results import write_legacy_vtk

__version__ = "0.1.0a1"

__all__ = (
    "AffineTriangleMap",
    "DiffusionResult",
    "DirichletCondition",
    "NeumannCondition",
    "REFERENCE_TRIANGLE_VERTICES",
    "SteadyDiffusionProblem",
    "TriangleQuadrature",
    "TriangularMesh",
    "inside_reference_triangle",
    "integrate_reference",
    "p1_basis",
    "p1_basis_gradients",
    "solve_steady_diffusion",
    "triangle_rule",
    "unit_square_triangles",
    "unit_square_two_triangles",
    "write_legacy_vtk",
)
