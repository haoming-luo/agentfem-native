# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""面向 AgentFEM 的版本化、JSON 安全 Kernel Contract。"""

from __future__ import annotations

import json
import platform
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import TypeAlias, cast

import numpy as np

from . import __version__
from .diffusion import (
    CellMaterial,
    DiffusionResult,
    DirichletCondition,
    NeumannCondition,
    SteadyDiffusionProblem,
)
from .dynamics import (
    DynamicCheckpoint,
    Integrator,
    LinearDynamicsResult,
    LinearSecondOrderSystem,
    TimeScale,
    UnsafeTimeStepError,
)
from .elasticity import (
    DisplacementCondition,
    ElasticCellMaterial,
    LinearElasticMaterial,
    LinearElasticProblem,
    LinearElasticResult,
    TractionCondition,
)
from .execution import (
    ExecutableRequest,
    ExecutionResult,
    execute_plan,
    execute_t3_linear_dynamics_plan,
)
from .mesh import TriangularMesh
from .native import native_kernel_identity
from .planning import (
    ExecutionPlan,
    plan_linear_dynamics,
    plan_linear_elasticity,
    plan_linear_elasticity_3d,
    plan_steady_diffusion,
    plan_t3_linear_dynamics,
)
from .providers import LinearSolveError, ProviderUnavailableError
from .results import write_legacy_vtk, write_mechanics_vtk
from .runtime import (
    ExecutionContext,
    NativeExecutionError,
    ResourceBudget,
    enforce_plan_budget,
)
from .solid import (
    LinearElastic3DProblem,
    LinearElastic3DResult,
    SolidAssemblyMode,
    SolidCellMaterial,
    SolidDisplacementCondition,
    SolidElasticMaterial,
    SolidTractionCondition,
)
from .volume_mesh import TetrahedralMesh

JsonMapping: TypeAlias = Mapping[str, object]
CONTRACT_NAME = "agentfem.native-kernel-request"
CONTRACT_VERSION = "0.3.0"
SUPPORTED_CONTRACT_VERSIONS = ("0.1.0", "0.2.0", CONTRACT_VERSION)
AFIR_SCHEMA = "agentfem.af-ir"
AFIR_VERSION = "0.1.0"


class KernelRequestError(ValueError):
    """带稳定错误码和 JSON 路径的请求或能力失败。"""

    def __init__(self, code: str, message: str, *, path: str = "$") -> None:
        super().__init__(message)
        self.code = code
        self.path = path

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": str(self), "path": self.path}


@dataclass(frozen=True, slots=True)
class _DynamicContractInput:
    """保持无装配的 T3 动力输入，直到资源计划被接受。"""

    problem: LinearElasticProblem
    density: float
    time_step: float
    steps: int
    initial_displacement: object | None
    initial_velocity: object | None
    load_scale: TimeScale
    lumped_mass: bool


@dataclass(frozen=True, slots=True)
class _ParsedRequest:
    """解析后的拥有对象；不会保存 AgentFEM 或外部后端对象。"""

    version: str
    problem: ExecutableRequest | _DynamicContractInput
    provider: str
    outputs: JsonMapping
    requested_assembly: str
    method: Integrator | None = None
    restart: DynamicCheckpoint | None = None
    budget: ResourceBudget = field(default_factory=ResourceBudget)


def _mapping(value: object, path: str) -> JsonMapping:
    if not isinstance(value, Mapping):
        raise KernelRequestError("invalid_request", "此处应为对象。", path=path)
    return value


def _sequence(value: object, path: str) -> list[object]:
    if not isinstance(value, (list, tuple)):
        raise KernelRequestError("invalid_request", "此处应为数组。", path=path)
    return list(value)


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise KernelRequestError("invalid_request", "此处应为非空字符串。", path=path)
    return value.strip()


def _finite_scalar(value: object, path: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)):
        raise KernelRequestError("invalid_request", "此处应为有限数值。", path=path)
    result = float(value)
    if not np.isfinite(result):
        raise KernelRequestError("invalid_request", "此处应为有限数值。", path=path)
    return result


def _positive_integer(value: object, path: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise KernelRequestError("invalid_request", "此处应为正整数。", path=path)
    result = int(value)
    if result < 1:
        raise KernelRequestError("invalid_request", "此处应为正整数。", path=path)
    return result


def _nonnegative_integer(value: object, path: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise KernelRequestError("invalid_request", "此处应为非负整数。", path=path)
    result = int(value)
    if result < 0:
        raise KernelRequestError("invalid_request", "此处应为非负整数。", path=path)
    return result


def _positive_scalar(value: object, path: str) -> float:
    result = _finite_scalar(value, path)
    if result <= 0.0:
        raise KernelRequestError("invalid_request", "此处应为正有限数值。", path=path)
    return result


def _finite_vector(value: object, dimension: int, path: str) -> tuple[float, ...]:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise KernelRequestError(
            "invalid_request", f"此处应为长度 {dimension} 的有限向量。", path=path
        ) from error
    if vector.shape != (dimension,) or not np.all(np.isfinite(vector)):
        raise KernelRequestError(
            "invalid_request", f"此处应为长度 {dimension} 的有限向量。", path=path
        )
    return tuple(float(item) for item in vector)


def _finite_array(value: object, path: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise KernelRequestError(
            "invalid_request", "此处应为有限数值数组。", path=path
        ) from error
    if not np.all(np.isfinite(array)):
        raise KernelRequestError("invalid_request", "此处应为有限数值数组。", path=path)
    return array


def _conductivity(value: object, path: str) -> float | np.ndarray:
    if isinstance(value, (int, float)) and not isinstance(value, (bool, np.bool_)):
        return _finite_scalar(value, path)
    try:
        tensor = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise KernelRequestError(
            "invalid_request", "导热系数必须是标量或 2x2 张量。", path=path
        ) from error
    if tensor.shape != (2, 2) or not np.all(np.isfinite(tensor)):
        raise KernelRequestError(
            "invalid_request", "导热张量必须是有限的 2x2 数组。", path=path
        )
    return tensor


def _named_arrays(value: object, path: str) -> dict[str, object]:
    return {str(name): data for name, data in _mapping(value, path).items()}


def _version(request: JsonMapping) -> str:
    if request.get("contract") != CONTRACT_NAME:
        raise KernelRequestError(
            "invalid_contract",
            f"contract 必须为 {CONTRACT_NAME!r}。",
            path="$.contract",
        )
    version = request.get("contract_version")
    if version not in SUPPORTED_CONTRACT_VERSIONS:
        raise KernelRequestError(
            "unsupported_contract_version",
            f"支持的 contract_version 为 {SUPPORTED_CONTRACT_VERSIONS!r}。",
            path="$.contract_version",
        )
    return cast(str, version)


def _provider(procedure: JsonMapping) -> str:
    provider = _text(
        procedure.get("linear_algebra", "native"), "$.procedure.linear_algebra"
    )
    if provider not in {"native", "native_sparse", "numpy", "scipy", "auto"}:
        raise KernelRequestError(
            "unsupported_provider",
            f"未知的线性代数提供者 {provider!r}。",
            path="$.procedure.linear_algebra",
        )
    return provider


def _assembly(procedure: JsonMapping, *, dimension: int, heat: bool = False) -> str:
    value = _text(procedure.get("assembly", "auto"), "$.procedure.assembly")
    allowed = {"auto", "reference", "vectorized", "native"}
    if not heat and dimension == 3:
        allowed.remove("vectorized")
    if value not in allowed:
        raise KernelRequestError(
            "unsupported_assembly",
            f"当前物理与维数不支持装配路径 {value!r}。",
            path="$.procedure.assembly",
        )
    return value


def _triangular_mesh(record: JsonMapping) -> TriangularMesh:
    node_sets = _named_arrays(record.get("node_sets", {}), "$.mesh.node_sets")
    boundary_sets = _named_arrays(
        record.get("boundary_sets", {}), "$.mesh.boundary_sets"
    )
    cell_sets = _named_arrays(record.get("cell_sets", {}), "$.mesh.cell_sets")
    try:
        return TriangularMesh(
            points=record.get("points"),
            cells=record.get("cells"),
            node_sets=node_sets,
            boundary_sets=boundary_sets,
            cell_sets=cell_sets,
        )
    except (TypeError, ValueError) as error:
        raise KernelRequestError("invalid_model", str(error), path="$.mesh") from error


def _tetrahedral_mesh(record: JsonMapping) -> TetrahedralMesh:
    node_sets = _named_arrays(record.get("node_sets", {}), "$.mesh.node_sets")
    boundary_sets = _named_arrays(
        record.get("boundary_sets", {}), "$.mesh.boundary_sets"
    )
    cell_sets = _named_arrays(record.get("cell_sets", {}), "$.mesh.cell_sets")
    try:
        return TetrahedralMesh(
            points=record.get("points"),
            cells=record.get("cells"),
            node_sets=node_sets,
            boundary_sets=boundary_sets,
            cell_sets=cell_sets,
        )
    except (TypeError, ValueError) as error:
        raise KernelRequestError("invalid_model", str(error), path="$.mesh") from error


def _heat_problem(
    request: JsonMapping, mesh_record: JsonMapping, procedure: JsonMapping
) -> SteadyDiffusionProblem:
    physics = _mapping(request.get("physics"), "$.physics")
    dirichlet = []
    for index, raw in enumerate(_sequence(request.get("dirichlet", []), "$.dirichlet")):
        record = _mapping(raw, f"$.dirichlet[{index}]")
        dirichlet.append(
            DirichletCondition(
                _text(record.get("node_set"), f"$.dirichlet[{index}].node_set"),
                _finite_scalar(record.get("value"), f"$.dirichlet[{index}].value"),
            )
        )
    neumann = []
    for index, raw in enumerate(_sequence(request.get("neumann", []), "$.neumann")):
        record = _mapping(raw, f"$.neumann[{index}]")
        neumann.append(
            NeumannCondition(
                _text(record.get("boundary_set"), f"$.neumann[{index}].boundary_set"),
                _finite_scalar(record.get("flux"), f"$.neumann[{index}].flux"),
            )
        )
    materials = []
    for index, raw in enumerate(_sequence(request.get("materials", []), "$.materials")):
        record = _mapping(raw, f"$.materials[{index}]")
        materials.append(
            CellMaterial(
                _text(record.get("name"), f"$.materials[{index}].name"),
                _text(record.get("cell_set"), f"$.materials[{index}].cell_set"),
                _conductivity(
                    record.get("conductivity"), f"$.materials[{index}].conductivity"
                ),
            )
        )
    assembly = _assembly(procedure, dimension=2, heat=True)
    return SteadyDiffusionProblem(
        mesh=_triangular_mesh(mesh_record),
        conductivity=_conductivity(
            physics.get("conductivity"), "$.physics.conductivity"
        ),
        source=_finite_scalar(physics.get("source", 0.0), "$.physics.source"),
        dirichlet=tuple(dirichlet),
        neumann=tuple(neumann),
        materials=tuple(materials),
        assembly_mode=assembly,
    )


def _elastic_material_2d(record: object, path: str) -> LinearElasticMaterial:
    material = _mapping(record, path)
    model = _text(material.get("model", "plane_stress"), f"{path}.model")
    if model not in {"plane_stress", "plane_strain"}:
        raise KernelRequestError(
            "invalid_request",
            "二维材料模型必须为 plane_stress 或 plane_strain。",
            path=f"{path}.model",
        )
    modulus = _finite_scalar(material.get("young_modulus"), f"{path}.young_modulus")
    ratio = _finite_scalar(material.get("poisson_ratio"), f"{path}.poisson_ratio")
    try:
        return LinearElasticMaterial(modulus, ratio, model)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise KernelRequestError("invalid_model", str(error), path=path) from error


def _elastic_material_3d(record: object, path: str) -> SolidElasticMaterial:
    material = _mapping(record, path)
    if "model" in material:
        raise KernelRequestError(
            "invalid_request",
            "三维各向同性材料不接受二维 plane_stress/plane_strain 模型字段。",
            path=f"{path}.model",
        )
    modulus = _finite_scalar(material.get("young_modulus"), f"{path}.young_modulus")
    ratio = _finite_scalar(material.get("poisson_ratio"), f"{path}.poisson_ratio")
    try:
        return SolidElasticMaterial(modulus, ratio)
    except (TypeError, ValueError) as error:
        raise KernelRequestError("invalid_model", str(error), path=path) from error


def _condition_value(
    record: JsonMapping, dimension: int, index: int
) -> tuple[str | None, float | tuple[float, ...]]:
    path = f"$.dirichlet[{index}]"
    raw_component = record.get("component")
    if raw_component is None:
        return None, _finite_vector(record.get("value"), dimension, f"{path}.value")
    component = _text(raw_component, f"{path}.component")
    allowed = ("x", "y") if dimension == 2 else ("x", "y", "z")
    if component not in allowed:
        raise KernelRequestError(
            "invalid_request",
            f"{dimension}D 位移分量必须属于 {allowed!r}。",
            path=f"{path}.component",
        )
    return component, _finite_scalar(record.get("value"), f"{path}.value")


def _mechanics_problem(
    request: JsonMapping,
    mesh_record: JsonMapping,
    procedure: JsonMapping,
    dimension: int,
) -> LinearElasticProblem | LinearElastic3DProblem:
    physics = _mapping(request.get("physics"), "$.physics")
    if dimension == 3 and "thickness" in physics:
        raise KernelRequestError(
            "invalid_request",
            "三维实体力学不接受二维厚度字段。",
            path="$.physics.thickness",
        )
    body_force = _finite_vector(
        physics.get("body_force", [0.0] * dimension), dimension, "$.physics.body_force"
    )
    thread_count = _positive_integer(
        procedure.get("thread_count", 1), "$.procedure.thread_count"
    )
    dirichlet_records = _sequence(request.get("dirichlet", []), "$.dirichlet")
    traction_records = _sequence(request.get("traction", []), "$.traction")
    material_records = _sequence(request.get("materials", []), "$.materials")
    if dimension == 2:
        thickness = _finite_scalar(physics.get("thickness", 1.0), "$.physics.thickness")
        if thickness <= 0.0:
            raise KernelRequestError(
                "invalid_request",
                "二维力学厚度必须大于零。",
                path="$.physics.thickness",
            )
        dirichlet = []
        for index, raw in enumerate(dirichlet_records):
            record = _mapping(raw, f"$.dirichlet[{index}]")
            component, value = _condition_value(record, 2, index)
            dirichlet.append(
                DisplacementCondition(
                    _text(record.get("node_set"), f"$.dirichlet[{index}].node_set"),
                    component,  # type: ignore[arg-type]
                    value,
                )
            )
        traction = []
        for index, raw in enumerate(traction_records):
            record = _mapping(raw, f"$.traction[{index}]")
            traction.append(
                TractionCondition(
                    _text(
                        record.get("boundary_set"), f"$.traction[{index}].boundary_set"
                    ),
                    _finite_vector(
                        record.get("value"), 2, f"$.traction[{index}].value"
                    ),
                )
            )
        materials = []
        for index, raw in enumerate(material_records):
            record = _mapping(raw, f"$.materials[{index}]")
            materials.append(
                ElasticCellMaterial(
                    _text(record.get("cell_set"), f"$.materials[{index}].cell_set"),
                    _elastic_material_2d(
                        record.get("material"), f"$.materials[{index}].material"
                    ),
                )
            )
        return LinearElasticProblem(
            mesh=_triangular_mesh(mesh_record),
            material=_elastic_material_2d(
                physics.get("material"), "$.physics.material"
            ),
            thickness=thickness,
            body_force=body_force,
            dirichlet=tuple(dirichlet),
            traction=tuple(traction),
            materials=tuple(materials),
            assembly_mode=_assembly(procedure, dimension=2),  # type: ignore[arg-type]
            thread_count=thread_count,
        )
    dirichlet_3d = []
    for index, raw in enumerate(dirichlet_records):
        record = _mapping(raw, f"$.dirichlet[{index}]")
        component, value = _condition_value(record, 3, index)
        dirichlet_3d.append(
            SolidDisplacementCondition(
                _text(record.get("node_set"), f"$.dirichlet[{index}].node_set"),
                component,  # type: ignore[arg-type]
                value,
            )
        )
    traction_3d = []
    for index, raw in enumerate(traction_records):
        record = _mapping(raw, f"$.traction[{index}]")
        traction_3d.append(
            SolidTractionCondition(
                _text(record.get("boundary_set"), f"$.traction[{index}].boundary_set"),
                _finite_vector(record.get("value"), 3, f"$.traction[{index}].value"),
            )
        )
    materials_3d = []
    for index, raw in enumerate(material_records):
        record = _mapping(raw, f"$.materials[{index}]")
        materials_3d.append(
            SolidCellMaterial(
                _text(record.get("cell_set"), f"$.materials[{index}].cell_set"),
                _elastic_material_3d(
                    record.get("material"), f"$.materials[{index}].material"
                ),
            )
        )
    return LinearElastic3DProblem(
        mesh=_tetrahedral_mesh(mesh_record),
        material=_elastic_material_3d(physics.get("material"), "$.physics.material"),
        body_force=body_force,
        dirichlet=tuple(dirichlet_3d),
        traction=tuple(traction_3d),
        materials=tuple(materials_3d),
        thread_count=thread_count,
    )


def _dynamic_checkpoint(value: object, method: Integrator) -> DynamicCheckpoint | None:
    if value is None:
        return None
    record = _mapping(value, "$.restart")
    checkpoint_method = _text(record.get("method"), "$.restart.method")
    if checkpoint_method not in {
        "central_difference",
        "newmark_average_acceleration",
    }:
        raise KernelRequestError(
            "invalid_request", "重启积分方法不受支持。", path="$.restart.method"
        )
    if checkpoint_method != method:
        raise KernelRequestError(
            "restart_method_mismatch",
            "重启检查点的积分方法与本次请求不一致。",
            path="$.restart.method",
        )
    try:
        return DynamicCheckpoint(
            cast(Integrator, checkpoint_method),
            _nonnegative_integer(record.get("step"), "$.restart.step"),
            _finite_scalar(record.get("time"), "$.restart.time"),
            _finite_array(record.get("displacement"), "$.restart.displacement"),
            _finite_array(record.get("velocity"), "$.restart.velocity"),
            _finite_array(record.get("acceleration"), "$.restart.acceleration"),
            _text(record.get("digest"), "$.restart.digest"),
        )
    except ValueError as error:
        raise KernelRequestError(
            "invalid_checkpoint", str(error), path="$.restart"
        ) from error


def _load_scale(value: object, *, start_time: float, end_time: float) -> TimeScale:
    if not isinstance(value, Mapping):
        return _finite_scalar(value, "$.procedure.load_scale")
    record = _mapping(value, "$.procedure.load_scale")
    times = _finite_array(record.get("times"), "$.procedure.load_scale.times")
    values = _finite_array(record.get("values"), "$.procedure.load_scale.values")
    if (
        times.ndim != 1
        or values.ndim != 1
        or times.size < 2
        or values.size != times.size
    ):
        raise KernelRequestError(
            "invalid_load_curve",
            "载荷曲线需要至少两个、数量相同的一维 times 与 values。",
            path="$.procedure.load_scale",
        )
    if np.any(np.diff(times) <= 0.0):
        raise KernelRequestError(
            "invalid_load_curve",
            "载荷曲线 times 必须严格递增。",
            path="$.procedure.load_scale.times",
        )
    tolerance = (
        16.0 * np.finfo(np.float64).eps * max(1.0, abs(start_time), abs(end_time))
    )
    if times[0] > start_time + tolerance or times[-1] < end_time - tolerance:
        raise KernelRequestError(
            "load_curve_range",
            "载荷曲线必须覆盖本次执行的完整绝对时间区间。",
            path="$.procedure.load_scale.times",
        )
    owned_times = np.array(times, copy=True)
    owned_values = np.array(values, copy=True)

    def interpolate(time: float) -> float:
        return float(np.interp(time, owned_times, owned_values))

    return interpolate


def _resource_budget(request: JsonMapping) -> ResourceBudget:
    execution = _mapping(request.get("execution", {}), "$.execution")
    budget = _mapping(execution.get("budget", {}), "$.execution.budget")

    def optional(name: str) -> int | None:
        value = budget.get(name)
        if value is None:
            return None
        return _nonnegative_integer(value, f"$.execution.budget.{name}")

    return ResourceBudget(
        maximum_dofs=optional("maximum_dofs"),
        maximum_peak_bytes=optional("maximum_peak_bytes"),
        maximum_steps=optional("maximum_steps"),
    )


def _dynamic_problem(
    request: JsonMapping,
    mesh_record: JsonMapping,
    procedure: JsonMapping,
) -> tuple[
    _DynamicContractInput,
    Integrator,
    DynamicCheckpoint | None,
]:
    method_text = _text(
        procedure.get("integrator", "central_difference"),
        "$.procedure.integrator",
    )
    if method_text not in {
        "central_difference",
        "newmark_average_acceleration",
    }:
        raise KernelRequestError(
            "unsupported_integrator",
            f"不支持线性动力积分方法 {method_text!r}。",
            path="$.procedure.integrator",
        )
    method = cast(Integrator, method_text)
    mass = _text(procedure.get("mass", "lumped"), "$.procedure.mass")
    if mass not in {"lumped", "consistent"}:
        raise KernelRequestError(
            "invalid_request",
            "质量形式必须为 lumped 或 consistent。",
            path="$.procedure.mass",
        )
    if method == "central_difference" and mass != "lumped":
        raise KernelRequestError(
            "unsupported_mass",
            "中心差分当前只接受集中质量 mass=lumped。",
            path="$.procedure.mass",
        )
    time_step = _positive_scalar(procedure.get("time_step"), "$.procedure.time_step")
    steps = _nonnegative_integer(procedure.get("steps"), "$.procedure.steps")
    restart = _dynamic_checkpoint(request.get("restart"), method)
    initial_record = _mapping(request.get("initial_state", {}), "$.initial_state")
    if restart is not None and initial_record:
        raise KernelRequestError(
            "ambiguous_initial_state",
            "重启请求不能同时携带 initial_state。",
            path="$.initial_state",
        )
    start_time = 0.0 if restart is None else restart.time
    load_scale = _load_scale(
        procedure.get("load_scale", 1.0),
        start_time=start_time,
        end_time=start_time + steps * time_step,
    )
    physics = _mapping(request.get("physics"), "$.physics")
    density = _positive_scalar(physics.get("density"), "$.physics.density")
    problem = _mechanics_problem(request, mesh_record, procedure, 2)
    if not isinstance(problem, LinearElasticProblem):
        raise KernelRequestError(
            "unsupported_capability", "线性动力契约当前只支持二维 T3。", path="$.study"
        )
    _validate_named_references(problem)
    initial_values: dict[str, np.ndarray | None] = {}
    for name in ("displacement", "velocity"):
        raw = initial_record.get(name)
        if raw is None:
            initial_values[name] = None
            continue
        values = _finite_array(raw, f"$.initial_state.{name}")
        if values.shape != (problem.mesh.node_count, 2):
            raise KernelRequestError(
                "invalid_request",
                "T3 动力初始状态必须是每节点两个分量的数组。",
                path=f"$.initial_state.{name}",
            )
        initial_values[name] = np.array(values, copy=True)
    dof_count = 2 * problem.mesh.node_count
    if restart is not None and restart.displacement.shape != (dof_count,):
        raise KernelRequestError(
            "invalid_checkpoint",
            "重启检查点自由度数量与本次模型不一致。",
            path="$.restart",
        )
    dynamic = _DynamicContractInput(
        problem,
        density,
        time_step,
        steps,
        initial_values["displacement"],
        initial_values["velocity"],
        load_scale,
        mass == "lumped",
    )
    return dynamic, method, restart


def _validate_named_references(problem: ExecutableRequest) -> None:
    """在规划前检查命名引用，避免 Agent 把拼写错误带入装配阶段。"""

    mesh = problem.mesh
    for index, condition in enumerate(problem.dirichlet):
        if condition.node_set not in mesh.node_sets:
            raise KernelRequestError(
                "invalid_model",
                f"未知节点集（Unknown node set）{condition.node_set!r}。",
                path=f"$.dirichlet[{index}].node_set",
            )
    boundary_conditions = (
        problem.neumann
        if isinstance(problem, SteadyDiffusionProblem)
        else problem.traction
    )
    boundary_key = (
        "neumann" if isinstance(problem, SteadyDiffusionProblem) else "traction"
    )
    for index, condition in enumerate(boundary_conditions):
        if condition.boundary_set not in mesh.boundary_sets:
            raise KernelRequestError(
                "invalid_model",
                f"未知边界集 {condition.boundary_set!r}。",
                path=f"$.{boundary_key}[{index}].boundary_set",
            )
    for index, material in enumerate(problem.materials):
        if material.cell_set not in mesh.cell_sets:
            raise KernelRequestError(
                "invalid_model",
                f"未知单元集 {material.cell_set!r}。",
                path=f"$.materials[{index}].cell_set",
            )


def _problem_from_request(request: JsonMapping) -> _ParsedRequest:
    version = _version(request)
    _text(request.get("request_id"), "$.request_id")
    study = _mapping(request.get("study"), "$.study")
    analysis = study.get("analysis")
    dimension = study.get("dimension")
    if dimension not in {2, 3}:
        raise KernelRequestError(
            "unsupported_capability",
            "当前契约只支持 dimension=2 或 3。",
            path="$.study.dimension",
        )
    physics_name = study.get("physics")
    mesh_record = _mapping(request.get("mesh"), "$.mesh")
    procedure = _mapping(request.get("procedure", {}), "$.procedure")
    provider = _provider(procedure)
    outputs = _mapping(request.get("outputs", {}), "$.outputs")
    budget = _resource_budget(request)
    if analysis == "linear_transient":
        if version != CONTRACT_VERSION:
            raise KernelRequestError(
                "unsupported_capability",
                "线性动力学需要 Kernel Contract 0.3.0。",
                path="$.contract_version",
            )
        if physics_name != "solid_mechanics" or dimension != 2:
            raise KernelRequestError(
                "unsupported_capability",
                "线性动力契约当前只支持二维 T3 固体力学。",
                path="$.study",
            )
        if procedure.get("kind") != "linear_dynamics":
            raise KernelRequestError(
                "unsupported_capability",
                "线性瞬态 procedure.kind 必须为 linear_dynamics。",
                path="$.procedure.kind",
            )
        if provider not in {"native", "native_sparse", "auto"}:
            raise KernelRequestError(
                "unsupported_provider",
                "线性动力契约当前只使用自主 native_sparse 提供者。",
                path="$.procedure.linear_algebra",
            )
        if "neumann" in request:
            raise KernelRequestError(
                "invalid_request",
                "固体动力学请求不接受热传导 neumann 字段。",
                path="$.neumann",
            )
        if "vtk" in outputs:
            raise KernelRequestError(
                "unsupported_output",
                "Kernel Contract 0.3 尚不支持时间序列 VTK。",
                path="$.outputs.vtk",
            )
        problem, method, restart = _dynamic_problem(request, mesh_record, procedure)
        return _ParsedRequest(
            version,
            problem,
            "native_sparse",
            outputs,
            "preassembled",
            method,
            restart,
            budget,
        )
    if analysis != "linear_static":
        raise KernelRequestError(
            "unsupported_capability",
            "当前契约支持 linear_static 和 0.3.0 的 linear_transient。",
            path="$.study.analysis",
        )
    if physics_name == "heat_transfer" and dimension == 2:
        if "traction" in request:
            raise KernelRequestError(
                "invalid_request",
                "热传导请求不接受力学 traction 字段。",
                path="$.traction",
            )
        if procedure.get("kind", "steady_diffusion") != "steady_diffusion":
            raise KernelRequestError(
                "unsupported_capability",
                "热传导 procedure.kind 必须为 steady_diffusion。",
                path="$.procedure.kind",
            )
        problem: ExecutableRequest = _heat_problem(request, mesh_record, procedure)
        _validate_named_references(problem)
        return _ParsedRequest(
            version, problem, provider, outputs, problem.assembly_mode, budget=budget
        )
    if physics_name == "solid_mechanics" and dimension in {2, 3}:
        if "neumann" in request:
            raise KernelRequestError(
                "invalid_request",
                "固体力学请求不接受热传导 neumann 字段。",
                path="$.neumann",
            )
        if version == "0.1.0":
            raise KernelRequestError(
                "unsupported_capability",
                "Kernel Contract 0.1 不包含线性力学，请使用 0.2.0。",
                path="$.contract_version",
            )
        if procedure.get("kind", "linear_elasticity") != "linear_elasticity":
            raise KernelRequestError(
                "unsupported_capability",
                "固体力学 procedure.kind 必须为 linear_elasticity。",
                path="$.procedure.kind",
            )
        assembly = _assembly(procedure, dimension=int(dimension))
        problem = _mechanics_problem(request, mesh_record, procedure, int(dimension))
        _validate_named_references(problem)
        return _ParsedRequest(
            version, problem, provider, outputs, assembly, budget=budget
        )
    raise KernelRequestError(
        "unsupported_capability",
        "当前契约支持二维稳态热传导和二维/三维线性固体力学。",
        path="$.study",
    )


def _execution_plan(parsed: _ParsedRequest) -> ExecutionPlan:
    if isinstance(parsed.problem, _DynamicContractInput):
        dynamic = parsed.problem
        return plan_t3_linear_dynamics(
            dynamic.problem,
            dynamic.density,
            time_step=dynamic.time_step,
            steps=dynamic.steps,
            lumped_mass=dynamic.lumped_mass,
        )
    if isinstance(parsed.problem, SteadyDiffusionProblem):
        return plan_steady_diffusion(parsed.problem, provider=parsed.provider)
    if isinstance(parsed.problem, LinearElasticProblem):
        return plan_linear_elasticity(parsed.problem, provider=parsed.provider)
    if isinstance(parsed.problem, LinearElastic3DProblem):
        return plan_linear_elasticity_3d(
            parsed.problem,
            provider=parsed.provider,
            assembly=cast(SolidAssemblyMode, parsed.requested_assembly),
        )
    if isinstance(parsed.problem, LinearSecondOrderSystem):
        return plan_linear_dynamics(parsed.problem)
    raise KernelRequestError("unsupported_capability", "当前契约不支持此问题类型。")


def _artifact_path(directory: Path, portable_path: object) -> tuple[Path, str]:
    text = _text(portable_path, "$.outputs.vtk.path")
    if "\\" in text:
        raise KernelRequestError(
            "invalid_artifact_path",
            "产物路径必须使用可移植的 '/' 分隔符。",
            path="$.outputs.vtk.path",
        )
    pure = PurePosixPath(text)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise KernelRequestError(
            "invalid_artifact_path",
            "产物路径必须是不能逃逸根目录的相对路径。",
            path="$.outputs.vtk.path",
        )
    return directory.joinpath(*pure.parts), pure.as_posix()


def _runtime(result: ExecutionResult, plan: ExecutionPlan) -> dict[str, object]:
    convergence = getattr(result, "convergence", None)
    provider = {
        "name": getattr(result, "provider_name", "native_sparse"),
        "version": getattr(result, "provider_version", __version__),
        "matrix_format": getattr(result, "provider_matrix_format", "csr"),
    }
    if convergence is not None:
        provider["convergence"] = {
            "converged": convergence.converged,
            "reason": convergence.reason,
            "iterations": convergence.iterations,
            "initial_residual_norm": convergence.initial_residual_norm,
            "residual_norm": convergence.residual_norm,
            "threshold": convergence.threshold,
        }
    return {
        "os": platform.system(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "assembly": {"name": plan.assembly_mode, "thread_count": plan.thread_count},
        "native_kernel": native_kernel_identity(),
        "linear_algebra_provider": provider,
    }


def _request_digest(request: JsonMapping) -> str:
    encoded = json.dumps(
        request, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _response_version(request: object) -> str:
    if (
        isinstance(request, Mapping)
        and request.get("contract_version") in SUPPORTED_CONTRACT_VERSIONS
    ):
        return cast(str, request.get("contract_version"))
    return CONTRACT_VERSION


def _failure_result(
    request: object,
    request_id: str,
    failure: KernelRequestError | NativeExecutionError,
) -> dict[str, object]:
    version = _response_version(request)
    error = failure.as_dict()
    if version != CONTRACT_VERSION:
        error = {
            key: value
            for key, value in error.items()
            if key in {"code", "message", "path"}
        }
    return {
        "contract": CONTRACT_NAME,
        "contract_version": version,
        "request_id": request_id,
        "status": "failed",
        "backend": {"name": "native", "version": __version__},
        "error": error,
    }


def plan_kernel_request(request: JsonMapping) -> dict[str, object]:
    """不装配、不求解，返回 AgentFEM 可审查的能力与资源计划。"""

    request_id = (
        str(request.get("request_id", "unknown"))
        if isinstance(request, Mapping)
        else "unknown"
    )
    try:
        mapping = _mapping(request, "$")
        parsed = _problem_from_request(mapping)
        if parsed.version == "0.1.0":
            raise KernelRequestError(
                "unsupported_capability",
                "执行前资源预检需要 Kernel Contract 0.2.0 或 0.3.0。",
                path="$.contract_version",
            )
        plan = _execution_plan(parsed)
        enforce_plan_budget(plan, parsed.budget)
        if isinstance(parsed.problem, _DynamicContractInput):
            ExecutionContext(parsed.budget).check(
                "linear_dynamics", 0, parsed.problem.steps
            )
        return {
            "contract": CONTRACT_NAME,
            "contract_version": parsed.version,
            "request_id": request_id,
            "status": "planned",
            "backend": {"name": "native", "version": __version__},
            "request_digest": _request_digest(mapping),
            "plan": plan.as_dict(),
            "warnings": list(plan.warnings),
        }
    except KernelRequestError as error:
        failure = error
    except NativeExecutionError as error:
        failure = error
    except (KeyError, TypeError, ValueError) as error:
        failure = KernelRequestError("invalid_model", str(error))
    return _failure_result(request, request_id, failure)


def _fields_and_quantities(
    result: ExecutionResult,
) -> tuple[dict[str, object], dict[str, object]]:
    if isinstance(result, DiffusionResult):
        return (
            {
                "solution": {
                    "association": "node",
                    "components": 1,
                    "values": result.nodal_values.tolist(),
                }
            },
            {
                "free_residual_norm": result.free_residual_norm,
                "total_applied_load": result.total_applied_load,
                "total_reaction": result.total_reaction,
                "potential_energy": result.potential_energy,
            },
        )
    if isinstance(result, (LinearElasticResult, LinearElastic3DResult)):
        return (
            {
                "displacement": {
                    "association": "node",
                    "components": int(result.displacements.shape[1]),
                    "values": result.displacements.tolist(),
                },
                "strain": {
                    "association": "cell",
                    "components": int(result.cell_strain.shape[1]),
                    "values": result.cell_strain.tolist(),
                },
                "stress": {
                    "association": "cell",
                    "components": int(result.cell_stress.shape[1]),
                    "values": result.cell_stress.tolist(),
                },
            },
            {
                "free_residual_norm": result.free_residual_norm,
                "total_applied_force": result.total_applied_force.tolist(),
                "total_reaction": result.total_reaction.tolist(),
                "balance_norm": float(
                    np.linalg.norm(result.total_applied_force + result.total_reaction)
                ),
                "strain_energy": result.strain_energy,
            },
        )
    if isinstance(result, LinearDynamicsResult):
        displacement = result.displacement.reshape(result.times.size, -1, 2)
        velocity = result.velocity.reshape(result.times.size, -1, 2)
        acceleration = result.acceleration.reshape(result.times.size, -1, 2)
        return (
            {
                "displacement": {
                    "association": "node",
                    "components": 2,
                    "values": displacement.tolist(),
                },
                "velocity": {
                    "association": "node",
                    "components": 2,
                    "values": velocity.tolist(),
                },
                "acceleration": {
                    "association": "node",
                    "components": 2,
                    "values": acceleration.tolist(),
                },
                "reaction": {
                    "association": "constrained_dof",
                    "components": 1,
                    "dofs": result.constrained_dofs.tolist(),
                    "values": result.reaction.tolist(),
                },
            },
            {
                "times": result.times.tolist(),
                "kinetic_energy": result.kinetic_energy.tolist(),
                "strain_energy": result.strain_energy.tolist(),
                "total_energy": result.total_energy.tolist(),
                "external_work": result.external_work.tolist(),
                "energy_balance_error": result.energy_balance_error.tolist(),
                "maximum_energy_balance_error": float(
                    np.max(np.abs(result.energy_balance_error))
                ),
            },
        )
    raise KernelRequestError("unsupported_capability", "当前契约不支持此结果类型。")


def _checkpoint_record(result: LinearDynamicsResult) -> dict[str, object]:
    checkpoint = result.checkpoint()
    return {
        "method": checkpoint.method,
        "step": checkpoint.step,
        "time": checkpoint.time,
        "displacement": checkpoint.displacement.tolist(),
        "velocity": checkpoint.velocity.tolist(),
        "acceleration": checkpoint.acceleration.tolist(),
        "digest": checkpoint.digest,
    }


def _artifacts(
    parsed: _ParsedRequest,
    result: ExecutionResult,
    artifact_directory: str | Path | None,
) -> list[dict[str, object]]:
    if "vtk" not in parsed.outputs:
        return []
    if artifact_directory is None:
        raise KernelRequestError(
            "artifact_root_required",
            "请求 VTK 输出时必须提供 artifact_directory。",
            path="$.outputs.vtk",
        )
    vtk = _mapping(parsed.outputs["vtk"], "$.outputs.vtk")
    path, portable = _artifact_path(Path(artifact_directory), vtk.get("path"))
    if isinstance(parsed.problem, SteadyDiffusionProblem) and isinstance(
        result, DiffusionResult
    ):
        write_legacy_vtk(
            path,
            parsed.problem.mesh,
            result.nodal_values,
            field_name=str(vtk.get("field_name", "temperature")),
        )
    elif isinstance(
        parsed.problem, (LinearElasticProblem, LinearElastic3DProblem)
    ) and isinstance(result, (LinearElasticResult, LinearElastic3DResult)):
        write_mechanics_vtk(
            path, parsed.problem.mesh, result.displacements, result.cell_stress
        )
    else:
        raise KernelRequestError("unsupported_capability", "当前结果不能写入 VTK。")
    return [
        {
            "name": "vtk",
            "path": portable,
            "sha256": sha256(path.read_bytes()).hexdigest(),
        }
    ]


def _execution_context(
    request_budget: ResourceBudget,
    supplied: ExecutionContext | None,
) -> ExecutionContext:
    """合并请求与调用方上限，始终采用两者中更严格的资源边界。"""

    external = ResourceBudget() if supplied is None else supplied.budget

    def tighter(left: int | None, right: int | None) -> int | None:
        if left is None:
            return right
        if right is None:
            return left
        return min(left, right)

    budget = ResourceBudget(
        maximum_dofs=tighter(request_budget.maximum_dofs, external.maximum_dofs),
        maximum_peak_bytes=tighter(
            request_budget.maximum_peak_bytes, external.maximum_peak_bytes
        ),
        maximum_steps=tighter(request_budget.maximum_steps, external.maximum_steps),
    )
    return ExecutionContext(
        budget=budget,
        cancellation=None if supplied is None else supplied.cancellation,
        progress=None if supplied is None else supplied.progress,
    )


def run_kernel_request(
    request: JsonMapping,
    *,
    artifact_directory: str | Path | None = None,
    context: ExecutionContext | None = None,
) -> dict[str, object]:
    """执行一个契约请求，并返回结构化结果或可定位失败。"""

    request_id = (
        str(request.get("request_id", "unknown"))
        if isinstance(request, Mapping)
        else "unknown"
    )
    try:
        mapping = _mapping(request, "$")
        parsed = _problem_from_request(mapping)
        plan = _execution_plan(parsed)
        execution_context = _execution_context(parsed.budget, context)
        if isinstance(parsed.problem, _DynamicContractInput):
            dynamic = parsed.problem
            receipt = execute_t3_linear_dynamics_plan(
                plan,
                dynamic.problem,
                dynamic.density,
                time_step=dynamic.time_step,
                steps=dynamic.steps,
                initial_displacement=dynamic.initial_displacement,
                initial_velocity=dynamic.initial_velocity,
                load_scale=dynamic.load_scale,
                lumped_mass=dynamic.lumped_mass,
                method=cast(Integrator, parsed.method),
                restart=parsed.restart,
                context=execution_context,
            )
        else:
            receipt = execute_plan(plan, parsed.problem, context=execution_context)
        result = receipt.result
        fields, quantities = _fields_and_quantities(result)
        capabilities = [
            "steady_diffusion_p1_triangle:verified",
            "native_sparse_cg:implemented",
            "linear_elasticity_t3_plane_stress_strain:verified",
        ]
        if parsed.version != "0.1.0":
            capabilities.append("linear_elasticity_t4_3d:verified")
        if isinstance(result, LinearDynamicsResult):
            capabilities.append(
                "linear_dynamics_t3_central_difference_newmark:verified"
            )
        response: dict[str, object] = {
            "contract": CONTRACT_NAME,
            "contract_version": parsed.version,
            "request_id": request_id,
            "status": "success",
            "backend": {"name": "native", "version": __version__},
            "capabilities": capabilities,
            "quantities": quantities,
            "fields": fields,
            "artifacts": _artifacts(parsed, result, artifact_directory),
            "runtime": _runtime(result, plan),
            "warnings": [] if parsed.version == "0.1.0" else list(plan.warnings),
        }
        if isinstance(result, LinearDynamicsResult):
            response["checkpoint"] = _checkpoint_record(result)
        if parsed.version != "0.1.0":
            response["plan"] = plan.as_dict()
            response["evidence"] = {
                "request_digest": _request_digest(mapping),
                "execution_digest": receipt.evidence_digest,
                "execution": receipt.evidence,
            }
        return response
    except KernelRequestError as error:
        failure = error
    except NativeExecutionError as error:
        failure = error
    except UnsafeTimeStepError as error:
        failure = KernelRequestError(
            "unsafe_time_step", str(error), path="$.procedure.time_step"
        )
    except ProviderUnavailableError as error:
        failure = KernelRequestError(
            "provider_unavailable", str(error), path="$.procedure"
        )
    except LinearSolveError as error:
        failure = KernelRequestError(
            "linear_solve_failed", str(error), path="$.procedure.linear_algebra"
        )
    except OSError as error:
        failure = KernelRequestError(
            "artifact_write_failed", str(error), path="$.outputs"
        )
    except (KeyError, TypeError, ValueError) as error:
        failure = KernelRequestError("invalid_model", str(error))
    return _failure_result(request, request_id, failure)


def lower_agentfem_ir(document: JsonMapping) -> dict[str, object]:
    """从公开 AF-IR 0.1 的可执行扩展降级，不导入 AgentFEM。"""

    record = _mapping(document, "$")
    if (
        record.get("schema") != AFIR_SCHEMA
        or record.get("schema_version") != AFIR_VERSION
    ):
        raise KernelRequestError(
            "unsupported_agentfem_ir", f"需要 {AFIR_SCHEMA} {AFIR_VERSION}。"
        )
    if record.get("document_type") != "model":
        raise KernelRequestError(
            "unsupported_agentfem_ir", "AF-IR document_type 必须为 'model'。"
        )
    root = _mapping(record.get("root"), "$.root")
    extension = _mapping(root.get("native_kernel"), "$.root.native_kernel")
    request = dict(extension)
    request.update(
        {
            "contract": CONTRACT_NAME,
            "contract_version": CONTRACT_VERSION,
            "request_id": str(
                extension.get("request_id", root.get("name", "agentfem-model"))
            ),
            "study": _mapping(root.get("study"), "$.root.study"),
        }
    )
    _problem_from_request(request)
    return request


def lower_agentfem_model(model: object) -> dict[str, object]:
    """只调用 AgentFEM 公开的 ``to_ir`` 边界。"""

    method = getattr(model, "to_ir", None)
    if not callable(method):
        raise TypeError("AgentFEM 模型适配器需要公开的 to_ir() 方法。")
    document = method()
    if hasattr(document, "as_dict") and callable(document.as_dict):
        document = document.as_dict()
    return lower_agentfem_ir(_mapping(document, "$"))


def _schema(kind: str, version: str | None) -> dict[str, object]:
    selected = CONTRACT_VERSION if version is None else version
    if selected not in SUPPORTED_CONTRACT_VERSIONS:
        raise ValueError(f"不支持的 Kernel Contract Schema 版本 {selected!r}。")
    short = ".".join(selected.split(".")[:2])
    schema = (
        resources.files("agentfem_native")
        .joinpath("schemas")
        .joinpath(f"kernel-{kind}-{short}.schema.json")
    )
    return json.loads(schema.read_text(encoding="utf-8"))


def kernel_request_schema(version: str | None = None) -> dict[str, object]:
    """读取随包安装的 JSON Schema 2020-12 请求定义。"""

    return _schema("request", version)


def kernel_result_schema(version: str | None = None) -> dict[str, object]:
    """读取随包安装的 JSON Schema 2020-12 结果定义。"""

    return _schema("result", version)
