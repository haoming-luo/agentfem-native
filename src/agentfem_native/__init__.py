# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Independent executable mathematics for the AgentFEM Native Engine."""

__version__ = "0.5.0a1"

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
from .dofs import VectorDofMap
from .elasticity import (
    DisplacementCondition,
    LinearElasticMaterial,
    LinearElasticProblem,
    LinearElasticResult,
    TractionCondition,
    assemble_linear_elasticity,
    p1_body_force_load,
    p1_boundary_traction_load,
    p1_elastic_stiffness,
    p1_strain_displacement,
    solve_linear_elasticity,
)
from .geometry import AffineTriangleMap
from .mesh import TriangularMesh, unit_square_triangles, unit_square_two_triangles
from .native import native_kernel_available, native_kernel_identity
from .planning import (
    ExecutionPlan,
    native_capabilities,
    plan_linear_elasticity,
    plan_steady_diffusion,
)
from .providers import (
    LinearSolveError,
    NativeSparseProvider,
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
from .sparse import CGReport, CGResult, CSRMatrix, conjugate_gradient

__all__ = (
    "CONTRACT_NAME",
    "CONTRACT_VERSION",
    "REFERENCE_TRIANGLE_VERTICES",
    "AffineTriangleMap",
    "CGReport",
    "CGResult",
    "CSRMatrix",
    "CellMaterial",
    "DiffusionResult",
    "DirichletCondition",
    "DisplacementCondition",
    "ExecutionPlan",
    "KernelRequestError",
    "LinearElasticMaterial",
    "LinearElasticProblem",
    "LinearElasticResult",
    "LinearSolveError",
    "NativeSparseProvider",
    "NeumannCondition",
    "NumpyDenseProvider",
    "ProviderIdentity",
    "ProviderUnavailableError",
    "ScipySparseProvider",
    "SteadyDiffusionProblem",
    "TractionCondition",
    "TriangleQuadrature",
    "TriangularMesh",
    "VectorDofMap",
    "assemble_linear_elasticity",
    "conjugate_gradient",
    "inside_reference_triangle",
    "integrate_reference",
    "kernel_request_schema",
    "kernel_result_schema",
    "lower_agentfem_ir",
    "lower_agentfem_model",
    "native_capabilities",
    "native_kernel_available",
    "native_kernel_identity",
    "p1_basis",
    "p1_basis_gradients",
    "p1_body_force_load",
    "p1_boundary_traction_load",
    "p1_elastic_stiffness",
    "p1_strain_displacement",
    "plan_linear_elasticity",
    "plan_steady_diffusion",
    "run_kernel_request",
    "solve_linear_elasticity",
    "solve_steady_diffusion",
    "triangle_rule",
    "unit_square_triangles",
    "unit_square_two_triangles",
    "write_legacy_vtk",
)
