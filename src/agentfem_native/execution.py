# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""执行已接受的资源计划，并生成紧凑、机器可读且摘要绑定的证据收据。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Literal, TypeAlias, cast

import numpy as np

from . import __version__
from .diffusion import DiffusionResult, SteadyDiffusionProblem, solve_steady_diffusion
from .dynamics import (
    Integrator,
    LinearDynamicsResult,
    LinearSecondOrderSystem,
    integrate_constrained_linear_dynamics,
    integrate_linear_dynamics,
)
from .elasticity import (
    LinearElasticProblem,
    LinearElasticResult,
    solve_linear_elasticity,
)
from .native import native_kernel_identity
from .planning import (
    ExecutionPlan,
    plan_linear_dynamics,
    plan_linear_elasticity,
    plan_linear_elasticity_3d,
    plan_steady_diffusion,
)
from .runtime import ExecutionContext, NativeExecutionError, enforce_plan_budget
from .solid import (
    LinearElastic3DProblem,
    LinearElastic3DResult,
    SolidAssemblyMode,
    solve_linear_elasticity_3d,
)

ExecutableRequest: TypeAlias = (
    SteadyDiffusionProblem
    | LinearElasticProblem
    | LinearElastic3DProblem
    | LinearSecondOrderSystem
)
ExecutionResult: TypeAlias = (
    DiffusionResult | LinearElasticResult | LinearElastic3DResult | LinearDynamicsResult
)


@dataclass(frozen=True, slots=True)
class ExecutionReceipt:
    """把一次成功结果与可复现、JSON 安全的证据绑定。"""

    plan: ExecutionPlan
    result: ExecutionResult
    evidence_json: str
    evidence_digest: str
    status: Literal["succeeded"] = "succeeded"

    def __post_init__(self) -> None:
        value = json.loads(self.evidence_json)
        canonical = json.dumps(
            value, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
        if not isinstance(value, dict) or canonical != self.evidence_json:
            raise ValueError("Execution receipt evidence must be a canonical object.")
        if (
            sha256(self.evidence_json.encode("utf-8")).hexdigest()
            != self.evidence_digest
        ):
            raise ValueError("Execution receipt evidence digest does not match.")
        if self.status != "succeeded":
            raise ValueError(
                "A result receipt can only represent successful execution."
            )

    @property
    def evidence(self) -> dict[str, object]:
        """返回新的证据对象，避免调用方破坏收据的摘要不变量。"""

        value = json.loads(self.evidence_json)
        assert isinstance(value, dict)
        return value

    def as_dict(self) -> dict[str, object]:
        """返回紧凑收据；大型数值结果数组继续与证据分开保存。"""

        return {
            "status": self.status,
            "plan": self.plan.as_dict(),
            "evidence": self.evidence,
            "evidence_digest": self.evidence_digest,
        }


def _provider_argument(name: str) -> str:
    aliases = {
        "native_sparse": "native",
        "numpy_dense": "numpy",
        "scipy_sparse": "scipy",
    }
    return aliases.get(name, name)


def _expected_plan(plan: ExecutionPlan, request: ExecutableRequest) -> ExecutionPlan:
    provider = _provider_argument(plan.provider)
    if isinstance(request, SteadyDiffusionProblem):
        return plan_steady_diffusion(request, provider=provider)
    if isinstance(request, LinearElasticProblem):
        return plan_linear_elasticity(request, provider=provider)
    if isinstance(request, LinearElastic3DProblem):
        assembly = cast(SolidAssemblyMode, plan.assembly_mode)
        return plan_linear_elasticity_3d(request, provider=provider, assembly=assembly)
    if isinstance(request, LinearSecondOrderSystem):
        return plan_linear_dynamics(request)
    raise TypeError(f"Unsupported execution request {type(request).__name__!r}.")


def _convergence_evidence(result: object) -> dict[str, object] | None:
    report = getattr(result, "convergence", None)
    if report is None:
        return None
    return asdict(report)


def _result_evidence(result: ExecutionResult) -> dict[str, object]:
    if isinstance(result, DiffusionResult):
        return {
            "result_kind": "steady_diffusion",
            "free_residual_norm": result.free_residual_norm,
            "balance_error": result.total_applied_load + result.total_reaction,
            "potential_energy": result.potential_energy,
            "assembly_mode": result.assembly_mode,
            "provider": {
                "name": result.provider_name,
                "version": result.provider_version,
                "matrix_format": result.provider_matrix_format,
            },
            "convergence": _convergence_evidence(result),
        }
    if isinstance(result, (LinearElasticResult, LinearElastic3DResult)):
        return {
            "result_kind": (
                "linear_elasticity_2d"
                if isinstance(result, LinearElasticResult)
                else "linear_elasticity_3d"
            ),
            "free_residual_norm": result.free_residual_norm,
            "balance_norm": float(
                np.linalg.norm(result.total_applied_force + result.total_reaction)
            ),
            "strain_energy": result.strain_energy,
            "constrained_dof_count": int(result.constrained_dofs.size),
            "provider": {
                "name": result.provider_name,
                "version": result.provider_version,
                "matrix_format": result.provider_matrix_format,
            },
            "convergence": _convergence_evidence(result),
        }
    return {
        "result_kind": "linear_dynamics",
        "method": result.method,
        "accepted_state_count": int(result.times.size),
        "final_time": float(result.times[-1]),
        "maximum_energy_balance_error": float(
            np.max(np.abs(result.energy_balance_error))
        ),
        "final_total_energy": float(result.total_energy[-1]),
        "final_external_work": float(result.external_work[-1]),
        "constrained_dof_count": int(result.constrained_dofs.size),
    }


def _analysis_evidence(plan: ExecutionPlan) -> dict[str, object]:
    """生成 AgentFEM 可直接审查、但不提升成熟度的计算摘要。"""

    dimensions: dict[str, int | None] = {
        "steady_diffusion": 2,
        "linear_elasticity_2d": 2,
        "linear_elasticity_3d": 3,
        "linear_dynamics": None,
    }
    return {
        "problem_kind": plan.problem_kind,
        "spatial_dimension": dimensions.get(plan.problem_kind),
        "element_family": plan.cell_type,
        "dof_count": plan.dof_count,
        "assembly_mode": plan.assembly_mode,
        "provider": plan.provider,
        "claim_maturity": plan.maturity,
    }


def _receipt(plan: ExecutionPlan, result: ExecutionResult) -> ExecutionReceipt:
    evidence = {
        "engine": {"name": "agentfem_native", "version": __version__},
        "native_kernel": native_kernel_identity(),
        "plan_digest": plan.digest,
        "claim_maturity": plan.maturity,
        "analysis": _analysis_evidence(plan),
        "result": _result_evidence(result),
    }
    encoded = json.dumps(
        evidence, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return ExecutionReceipt(
        plan, result, encoded.decode("utf-8"), sha256(encoded).hexdigest()
    )


def execute_plan(
    plan: ExecutionPlan,
    request: ExecutableRequest,
    *,
    context: ExecutionContext | None = None,
    method: Integrator = "central_difference",
    constrained_dofs: object | None = None,
) -> ExecutionReceipt:
    """Validate a resource plan, execute it, and return result evidence.

    The plan digest binds resource shape, provider, assembly path, and maturity. It
    is deliberately not a serialization or identity of callable physics inputs.
    """

    if not isinstance(plan, ExecutionPlan):
        raise TypeError("execute_plan requires an ExecutionPlan.")
    expected = _expected_plan(plan, request)
    if expected != plan:
        raise NativeExecutionError(
            "Execution request no longer matches the accepted resource plan.",
            code="execution.plan_mismatch",
            path="plan.digest",
            retryable=True,
            remediations=("replan_request",),
        )
    if context is not None:
        enforce_plan_budget(plan, context.budget)
    provider = _provider_argument(plan.provider)
    if isinstance(request, SteadyDiffusionProblem):
        if context is not None:
            context.check("steady_diffusion", 0, 1, enforce_step_budget=False)
        result: ExecutionResult = solve_steady_diffusion(request, provider=provider)
        if context is not None:
            context.check("steady_diffusion", 1, 1, enforce_step_budget=False)
    elif isinstance(request, LinearElasticProblem):
        if context is not None:
            context.check("linear_elasticity_2d", 0, 1, enforce_step_budget=False)
        result = solve_linear_elasticity(request, provider=provider)
        if context is not None:
            context.check("linear_elasticity_2d", 1, 1, enforce_step_budget=False)
    elif isinstance(request, LinearElastic3DProblem):
        if context is not None:
            context.check("linear_elasticity_3d", 0, 1, enforce_step_budget=False)
        result = solve_linear_elasticity_3d(
            request,
            provider=provider,
            assembly=cast(SolidAssemblyMode, plan.assembly_mode),
        )
        if context is not None:
            context.check("linear_elasticity_3d", 1, 1, enforce_step_budget=False)
    elif isinstance(request, LinearSecondOrderSystem):
        if constrained_dofs is None:
            result = integrate_linear_dynamics(request, method=method, context=context)
        else:
            result = integrate_constrained_linear_dynamics(
                request,
                constrained_dofs,
                method=method,
                context=context,
            )
    else:
        raise TypeError(f"Unsupported execution request {type(request).__name__!r}.")
    return _receipt(plan, result)


def explain_execution(receipt: ExecutionReceipt) -> dict[str, object]:
    """Return evidence and explicit claim boundaries without re-running a solve."""

    if not isinstance(receipt, ExecutionReceipt):
        raise TypeError("explain_execution requires an ExecutionReceipt.")
    return {
        **receipt.as_dict(),
        "interpretation": {
            "claim_maturity": receipt.plan.maturity,
            "maturity_is_not_upgraded_by_execution": True,
            "plan_digest_scope": (
                "resource shape, structural identity, provider, assembly path, "
                "maturity, and warnings; not a full serialization of physics inputs"
            ),
            "numerical_arrays_available_on": "receipt.result",
        },
    }
