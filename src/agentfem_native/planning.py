# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Execution-free resource planning and capability negotiation."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import TYPE_CHECKING

from . import __version__
from .assembly import select_assembly_mode
from .elasticity import select_elasticity_assembly_mode
from .native import native_kernel_identity
from .providers import ScipySparseProvider

if TYPE_CHECKING:
    from .diffusion import SteadyDiffusionProblem
    from .dynamics import LinearSecondOrderSystem
    from .elasticity import LinearElasticProblem
    from .solid import LinearElastic3DProblem


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Conservative JSON-safe work and memory estimate without assembly."""

    problem_kind: str
    maturity: str
    cell_type: str
    node_count: int
    cell_count: int
    dof_count: int
    coo_entry_count: int
    csr_nnz_upper_bound: int
    csr_bytes_upper_bound: int
    peak_bytes_upper_bound: int
    provider: str
    assembly_mode: str
    warnings: tuple[str, ...]
    digest: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _plan(
    *,
    problem_kind: str,
    maturity: str,
    cell_type: str,
    node_count: int,
    cell_count: int,
    dof_count: int,
    entries_per_cell: int,
    provider: str,
    assembly_mode: str,
    warnings: tuple[str, ...] = (),
) -> ExecutionPlan:
    coo_entries = cell_count * entries_per_cell
    csr_upper = coo_entries
    csr_bytes = 8 * (dof_count + 1) + 16 * csr_upper
    coo_bytes = 24 * coo_entries
    solver_work_bytes = 8 * dof_count * 10
    peak_bytes = csr_bytes + coo_bytes + solver_work_bytes
    payload: dict[str, object] = {
        "problem_kind": problem_kind,
        "maturity": maturity,
        "cell_type": cell_type,
        "node_count": node_count,
        "cell_count": cell_count,
        "dof_count": dof_count,
        "coo_entry_count": coo_entries,
        "csr_nnz_upper_bound": csr_upper,
        "csr_bytes_upper_bound": csr_bytes,
        "peak_bytes_upper_bound": peak_bytes,
        "provider": provider,
        "assembly_mode": assembly_mode,
        "warnings": warnings,
    }
    digest = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ExecutionPlan(**payload, digest=digest)  # type: ignore[arg-type]


def _provider_name(provider: str | object | None) -> str:
    if provider is None:
        return "native_sparse"
    if isinstance(provider, str):
        if provider == "auto":
            return "native_sparse"
        if provider == "native":
            return "native_sparse"
        if provider == "numpy":
            return "numpy_dense"
        if provider == "scipy":
            return "scipy_sparse"
        if provider not in {"native_sparse", "numpy_dense", "scipy_sparse"}:
            raise ValueError(f"Unknown linear algebra provider {provider!r}.")
        return provider
    return type(provider).__name__


def plan_steady_diffusion(
    problem: SteadyDiffusionProblem, *, provider: str | object | None = None
) -> ExecutionPlan:
    """Estimate a diffusion request without assembling or solving it."""

    cell_conductivities: dict[int, object] = {}
    for material in problem.materials:
        for cell in problem.mesh.cells_in(material.cell_set):
            index = int(cell)
            if index in cell_conductivities:
                raise ValueError(f"Multiple materials are assigned to cell {index}.")
            cell_conductivities[index] = material.conductivity
    assembly = select_assembly_mode(
        problem.assembly_mode,
        conductivity=problem.conductivity,
        source=problem.source,
        cell_conductivities=cell_conductivities,
    )
    warnings = ()
    if _provider_name(provider) == "numpy_dense":
        warnings = ("Dense provider is intended only for bounded oracle problems.",)
    return _plan(
        problem_kind="steady_diffusion",
        maturity="verified",
        cell_type="triangle_p1",
        node_count=problem.mesh.node_count,
        cell_count=problem.mesh.cell_count,
        dof_count=problem.mesh.node_count,
        entries_per_cell=9,
        provider=_provider_name(provider),
        assembly_mode=assembly,
        warnings=warnings,
    )


def plan_linear_elasticity(
    problem: LinearElasticProblem, *, provider: str | object | None = None
) -> ExecutionPlan:
    """Estimate the implemented T3 reference slice without executing it."""

    warnings = (
        "T3 linear elasticity is implemented but Gate 2 verification is incomplete.",
    )
    if _provider_name(provider) == "numpy_dense":
        warnings += ("Dense provider is intended only for bounded oracle problems.",)
    return _plan(
        problem_kind="linear_elasticity_2d",
        maturity="implemented",
        cell_type="triangle_p1_vector2",
        node_count=problem.mesh.node_count,
        cell_count=problem.mesh.cell_count,
        dof_count=2 * problem.mesh.node_count,
        entries_per_cell=36,
        provider=_provider_name(provider),
        assembly_mode=select_elasticity_assembly_mode(problem),
        warnings=warnings,
    )


def plan_linear_elasticity_3d(
    problem: LinearElastic3DProblem, *, provider: str | object | None = None
) -> ExecutionPlan:
    """Estimate the readable T4 three-dimensional vertical slice."""

    warnings = (
        "T4 linear elasticity is implemented but Gate 2 verification is incomplete.",
    )
    if _provider_name(provider) == "numpy_dense":
        warnings += ("Dense provider is intended only for bounded oracle problems.",)
    return _plan(
        problem_kind="linear_elasticity_3d",
        maturity="implemented",
        cell_type="tetrahedron_p1_vector3",
        node_count=problem.mesh.node_count,
        cell_count=problem.mesh.cell_count,
        dof_count=3 * problem.mesh.node_count,
        entries_per_cell=144,
        provider=_provider_name(provider),
        assembly_mode="reference",
        warnings=warnings,
    )


def plan_linear_dynamics(system: LinearSecondOrderSystem) -> ExecutionPlan:
    """Estimate an owned linear second-order procedure without executing it."""

    dofs = system.stiffness.shape[0]
    entries = system.stiffness.nnz + system.mass.nnz
    csr_upper = entries
    csr_bytes = 16 * entries + 16 * (dofs + 1)
    history_bytes = 8 * (system.steps + 1) * (3 * dofs + 4)
    payload: dict[str, object] = {
        "problem_kind": "linear_dynamics",
        "maturity": "implemented",
        "cell_type": "preassembled_second_order_system",
        "node_count": 0,
        "cell_count": 0,
        "dof_count": dofs,
        "coo_entry_count": entries,
        "csr_nnz_upper_bound": csr_upper,
        "csr_bytes_upper_bound": csr_bytes,
        "peak_bytes_upper_bound": csr_bytes + history_bytes + 12 * 8 * dofs,
        "provider": "native_sparse",
        "assembly_mode": "preassembled",
        "warnings": (
            "Linear dynamics is implemented but Gate 3 verification is incomplete.",
        ),
    }
    digest = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ExecutionPlan(**payload, digest=digest)  # type: ignore[arg-type]


def native_capabilities() -> dict[str, object]:
    """Return installed runtime capabilities without importing optional providers."""

    return {
        "backend": {"name": "native", "version": __version__},
        "platform": {
            "os": platform.system(),
            "architecture": platform.machine(),
            "tier_1_contract": ["Windows-x86_64", "macOS-arm64/x86_64", "Linux-x86_64"],
        },
        "scientific": [
            {
                "name": "steady_diffusion_p1_triangle",
                "maturity": "verified",
            },
            {
                "name": "linear_elasticity_t3_plane_stress_strain",
                "maturity": "implemented",
            },
            {
                "name": "linear_elasticity_t4_3d",
                "maturity": "implemented",
            },
            {
                "name": "linear_dynamics_central_difference_newmark",
                "maturity": "implemented",
            },
        ],
        "linear_algebra": [
            {
                "name": "native_sparse",
                "available": True,
                "format": "csr_bsr",
                "solvers": ["cg"],
                "preconditioners": ["jacobi", "block_jacobi"],
            },
            {
                "name": "numpy_dense",
                "available": True,
                "role": "bounded_oracle",
            },
            {
                "name": "scipy_sparse",
                "available": ScipySparseProvider.available(),
                "role": "optional_provider",
            },
        ],
        "native_kernel": native_kernel_identity(),
    }
