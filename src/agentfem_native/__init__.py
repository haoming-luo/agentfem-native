# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Independent executable mathematics for the AgentFEM Native Engine."""

__version__ = "0.3.0a1"

from .contract import (
    CONTRACT_NAME,
    CONTRACT_VERSION,
    KernelRequestError,
    kernel_request_schema,
    kernel_result_schema,
    lower_agentfem_ir,
    lower_agentfem_model,
    run_kernel_request,
)
from .diffusion import (
    CellMaterial,
    DiffusionResult,
    DirichletCondition,
    NeumannCondition,
    SteadyDiffusionProblem,
    solve_steady_diffusion,
)
from .geometry import AffineTriangleMap
from .mesh import TriangularMesh, unit_square_triangles, unit_square_two_triangles
from .providers import (
    NumpyDenseProvider,
    ProviderIdentity,
    ProviderUnavailableError,
    ScipySparseProvider,
)
from .quadrature import TriangleQuadrature, integrate_reference, triangle_rule
from .reference import (
    REFERENCE_TRIANGLE_VERTICES,
    inside_reference_triangle,
    p1_basis,
    p1_basis_gradients,
)
from .results import write_legacy_vtk

__all__ = (
    "CONTRACT_NAME",
    "CONTRACT_VERSION",
    "REFERENCE_TRIANGLE_VERTICES",
    "AffineTriangleMap",
    "CellMaterial",
    "DiffusionResult",
    "DirichletCondition",
    "KernelRequestError",
    "NeumannCondition",
    "NumpyDenseProvider",
    "ProviderIdentity",
    "ProviderUnavailableError",
    "ScipySparseProvider",
    "SteadyDiffusionProblem",
    "TriangleQuadrature",
    "TriangularMesh",
    "inside_reference_triangle",
    "integrate_reference",
    "kernel_request_schema",
    "kernel_result_schema",
    "lower_agentfem_ir",
    "lower_agentfem_model",
    "p1_basis",
    "p1_basis_gradients",
    "run_kernel_request",
    "solve_steady_diffusion",
    "triangle_rule",
    "unit_square_triangles",
    "unit_square_two_triangles",
    "write_legacy_vtk",
)
