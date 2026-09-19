# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""不执行计算的资源规划与能力协商。"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import TYPE_CHECKING

import numpy as np

from . import __version__
from .assembly import select_assembly_mode
from .elasticity import select_elasticity_assembly_mode
from .native import native_kernel_identity
from .providers import ScipySparseProvider
from .solid import SolidAssemblyMode, select_solid_assembly_mode

if TYPE_CHECKING:
    from .diffusion import SteadyDiffusionProblem
    from .dynamics import LinearSecondOrderSystem
    from .elasticity import LinearElasticProblem
    from .mesh import TriangularMesh
    from .solid import LinearElastic3DProblem
    from .volume_mesh import TetrahedralMesh


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """不执行装配，给出保守且可 JSON 序列化的工作量与内存估计。"""

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
    thread_count: int
    thread_workspace_bytes: int
    structure_digest: str
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
    structure_digest: str,
    thread_count: int = 1,
    warnings: tuple[str, ...] = (),
) -> ExecutionPlan:
    coo_entries = cell_count * entries_per_cell
    csr_upper = coo_entries
    csr_bytes = 8 * (dof_count + 1) + 16 * csr_upper
    coo_bytes = 24 * coo_entries
    solver_work_bytes = 8 * dof_count * 10
    thread_workspace_bytes = 8 * dof_count * thread_count if thread_count > 1 else 0
    peak_bytes = csr_bytes + coo_bytes + solver_work_bytes + thread_workspace_bytes
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
        "thread_count": thread_count,
        "thread_workspace_bytes": thread_workspace_bytes,
        "structure_digest": structure_digest,
        "warnings": warnings,
    }
    digest = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ExecutionPlan(**payload, digest=digest)  # type: ignore[arg-type]


def _digest_array(digest: object, values: object, dtype: str) -> None:
    array = np.ascontiguousarray(values, dtype=dtype)
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())


def _mesh_structure_digest(mesh: TriangularMesh | TetrahedralMesh) -> str:
    """Bind geometry, connectivity, and named-set identity without physics."""

    digest = sha256()
    _digest_array(digest, mesh.points, "<f8")
    _digest_array(digest, mesh.cells, "<i8")
    for collection_name in ("node_sets", "boundary_sets", "cell_sets"):
        digest.update(collection_name.encode("ascii"))
        collection = getattr(mesh, collection_name)
        for name, values in sorted(collection.items()):
            encoded = name.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
            _digest_array(digest, values, "<i8")
    return digest.hexdigest()


def _dynamic_structure_digest(system: LinearSecondOrderSystem) -> str:
    digest = sha256()
    for matrix in (system.stiffness, system.mass):
        _digest_array(digest, matrix.indptr, "<i8")
        _digest_array(digest, matrix.indices, "<i8")
        _digest_array(digest, matrix.data, "<f8")
    _digest_array(digest, system.initial_displacement, "<f8")
    _digest_array(digest, system.initial_velocity, "<f8")
    _digest_array(digest, (system.time_step,), "<f8")
    _digest_array(digest, (system.steps,), "<i8")
    return digest.hexdigest()


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
        warnings = ("稠密提供者仅用于规模受限的对照问题。",)
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
        structure_digest=_mesh_structure_digest(problem.mesh),
        warnings=warnings,
    )


def plan_linear_elasticity(
    problem: LinearElasticProblem, *, provider: str | object | None = None
) -> ExecutionPlan:
    """在不执行装配和求解的前提下规划已验证的 T3 线性静力范围。"""

    warnings: tuple[str, ...] = ()
    if _provider_name(provider) == "numpy_dense":
        warnings += ("稠密提供者仅用于规模受限的对照问题。",)
    assembly = select_elasticity_assembly_mode(problem)
    thread_count = problem.thread_count if assembly == "native" else 1
    return _plan(
        problem_kind="linear_elasticity_2d",
        maturity="verified",
        cell_type="triangle_p1_vector2",
        node_count=problem.mesh.node_count,
        cell_count=problem.mesh.cell_count,
        dof_count=2 * problem.mesh.node_count,
        entries_per_cell=36,
        provider=_provider_name(provider),
        assembly_mode=assembly,
        thread_count=thread_count,
        structure_digest=_mesh_structure_digest(problem.mesh),
        warnings=warnings,
    )


def plan_linear_elasticity_3d(
    problem: LinearElastic3DProblem,
    *,
    provider: str | object | None = None,
    assembly: SolidAssemblyMode = "auto",
) -> ExecutionPlan:
    """在不执行装配和求解的前提下规划已验证的 T4 线性静力范围。"""

    warnings: tuple[str, ...] = ()
    if _provider_name(provider) == "numpy_dense":
        warnings += ("稠密提供者仅用于规模受限的对照问题。",)
    selected = select_solid_assembly_mode(problem, assembly)
    thread_count = problem.thread_count if selected == "native" else 1
    return _plan(
        problem_kind="linear_elasticity_3d",
        maturity="verified",
        cell_type="tetrahedron_p1_vector3",
        node_count=problem.mesh.node_count,
        cell_count=problem.mesh.cell_count,
        dof_count=3 * problem.mesh.node_count,
        entries_per_cell=144,
        provider=_provider_name(provider),
        assembly_mode=selected,
        thread_count=thread_count,
        structure_digest=_mesh_structure_digest(problem.mesh),
        warnings=warnings,
    )


def plan_linear_dynamics(system: LinearSecondOrderSystem) -> ExecutionPlan:
    """不执行时间积分，估计自主线性二阶动力过程。"""

    from .dynamics import central_difference_safe_time_step

    dofs = system.stiffness.shape[0]
    entries = system.stiffness.nnz + system.mass.nnz
    csr_upper = entries
    csr_bytes = 16 * entries + 16 * (dofs + 1)
    history_bytes = 8 * (system.steps + 1) * (3 * dofs + 4)
    warnings = ["线性动力过程已实现，但 Gate 3 验证尚未完成。"]
    try:
        safe_time_step = central_difference_safe_time_step(system)
    except ValueError:
        warnings.append("质量矩阵不是正对角形式；中心差分不可用，可选择 Newmark。")
    else:
        if np.isfinite(safe_time_step):
            ratio = system.time_step / safe_time_step
            warnings.append(
                "中心差分时间步预检："
                f"请求 {system.time_step:.17g}，保守上限 {safe_time_step:.17g}，"
                f"比值 {ratio:.6g}。"
            )
        else:
            warnings.append("中心差分预检未发现有限刚度步长约束。")
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
        "thread_count": 1,
        "thread_workspace_bytes": 0,
        "structure_digest": _dynamic_structure_digest(system),
        "warnings": tuple(warnings),
    }
    digest = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ExecutionPlan(**payload, digest=digest)  # type: ignore[arg-type]


def native_capabilities() -> dict[str, object]:
    """不导入可选提供者，返回当前安装环境的运行能力。"""

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
                "maturity": "verified",
            },
            {
                "name": "linear_elasticity_t4_3d",
                "maturity": "verified",
            },
            {
                "name": "linear_dynamics_central_difference_newmark",
                "maturity": "implemented",
                "explicit_stability_preflight": "conservative_infinity_norm_bound",
                "implicit_effective_matrix_reuse": True,
            },
        ],
        "execution": [
            {
                "name": "deterministic_cpu_parallel_assembly",
                "available": native_kernel_identity()["available"],
                "maturity": "implemented",
                "elements": ["T3", "T4"],
                "thread_control": "explicit",
                "default_thread_count": 1,
            }
        ],
        "linear_algebra": [
            {
                "name": "native_sparse",
                "available": True,
                "format": "csr_bsr",
                "solvers": ["cg"],
                "preconditioners": ["jacobi", "block_jacobi"],
                "operators": ["csr_spmv_cpp20", "csr_numeric_fill_cpp20"],
                "graph_lifecycle": [
                    "build_pattern",
                    "validate_layout",
                    "fill_values",
                ],
                "prepared_assembly": {
                    "elements": ["T3", "T4"],
                    "maturity": "implemented",
                    "serial_native_output": "values_and_load_only",
                    "avoided_index_bytes_per_contribution": 16,
                },
                "prepared_solve": {
                    "scope": "fixed_matrix_multiple_rhs",
                    "constraints": "strong_dirichlet",
                    "preconditioners": ["jacobi", "block_jacobi"],
                    "maturity": "implemented",
                },
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
