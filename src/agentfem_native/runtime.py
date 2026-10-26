# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""面向 Agent 的预算、取消、进度与稳定错误边界。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from threading import Event


class NativeExecutionError(RuntimeError):
    """机器错误码保持稳定，面向人的解释使用中文。"""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        path: str,
        retryable: bool,
        remediations: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.path = path
        self.retryable = retryable
        self.remediations = remediations

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": str(self),
            "path": self.path,
            "retryable": self.retryable,
            "remediations": list(self.remediations),
        }


@dataclass(frozen=True, slots=True)
class ResourceBudget:
    maximum_dofs: int | None = None
    maximum_peak_bytes: int | None = None
    maximum_steps: int | None = None

    def __post_init__(self) -> None:
        for name in ("maximum_dofs", "maximum_peak_bytes", "maximum_steps"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} 必须是非负整数或 None。")


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    stage: str
    completed: int
    total: int
    fraction: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class CancellationToken:
    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


@dataclass(slots=True)
class ExecutionContext:
    budget: ResourceBudget = ResourceBudget()
    cancellation: CancellationToken | None = None
    progress: Callable[[ProgressEvent], None] | None = None

    def check(
        self,
        stage: str,
        completed: int,
        total: int,
        *,
        enforce_step_budget: bool = True,
    ) -> None:
        if total < 0 or completed < 0 or completed > total:
            raise ValueError("进度计数必须满足 0 <= completed <= total。")
        if (
            enforce_step_budget
            and self.budget.maximum_steps is not None
            and total > self.budget.maximum_steps
        ):
            raise NativeExecutionError(
                f"请求 {total} 步，超过预算 {self.budget.maximum_steps}。",
                code="budget.steps_exceeded",
                path="procedure.steps",
                retryable=True,
                remediations=("reduce_steps", "increase_step_budget"),
            )
        if self.cancellation is not None and self.cancellation.cancelled:
            raise NativeExecutionError(
                "计算已在安全检查点取消，未接受的试探不能作为结果。",
                code="execution.cancelled",
                path=stage,
                retryable=True,
                remediations=("restart_from_last_checkpoint",),
            )
        if self.progress is not None:
            self.progress(
                ProgressEvent(
                    stage, completed, total, completed / total if total else 1.0
                )
            )
        if self.cancellation is not None and self.cancellation.cancelled:
            raise NativeExecutionError(
                "计算已在安全检查点取消，未接受的试探不能作为结果。",
                code="execution.cancelled",
                path=stage,
                retryable=True,
                remediations=("restart_from_last_checkpoint",),
            )


def enforce_plan_budget(plan: object, budget: ResourceBudget) -> None:
    """保守估计超过预算时，在执行之前拒绝计划。"""

    for attribute, maximum, code in (
        ("dof_count", budget.maximum_dofs, "budget.dofs_exceeded"),
        ("peak_bytes_upper_bound", budget.maximum_peak_bytes, "budget.memory_exceeded"),
    ):
        value = getattr(plan, attribute, None)
        if maximum is not None and (not isinstance(value, int) or value > maximum):
            raise NativeExecutionError(
                f"计划 {attribute}={value} 超过预算 {maximum}。",
                code=code,
                path=f"plan.{attribute}",
                retryable=True,
                remediations=("reduce_problem_size", "increase_resource_budget"),
            )
