# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""面向 AgentFEM 的多物理候选 JSON 边界；显式版本与成熟度。"""

from __future__ import annotations

import json
from dataclasses import asdict
from hashlib import sha256
from importlib.resources import files
from types import SimpleNamespace

import numpy as np

from .dynamics import LinearSecondOrderSystem, integrate_constrained_linear_dynamics
from .finite_strain import (
    FiniteStrainOperator,
    NonlinearConvergenceError,
    NonlinearInterruptionError,
    check_accepted,
)
from .fracture import AT2FractureOperator
from .materials import NeoHookeanMaterial, _elastic_parameters
from .mesh import TriangularMesh
from .providers import LinearSolveError
from .runtime import (
    ExecutionContext,
    NativeExecutionError,
    ResourceBudget,
    enforce_plan_budget,
)
from .simplex import (
    EssentialBC,
    cell_values,
    check_component_constraints,
    connected_components,
    finite_array,
    positive,
    positive_integer,
)
from .solid_dynamics import assemble_t4_mass
from .thermal import ThermalBoundary, ThermalOperator, conductivity_tensors
from .thermoelastic import ThermoelasticOperator
from .time_grid import validate_time_interval
from .volume_mesh import TetrahedralMesh

MULTIPHYSICS_VERSION = "0.4.0-experimental.1"
KINDS = (
    "thermal_steady",
    "thermal_transient",
    "thermoelastic",
    "finite_strain",
    "at2_fracture",
    "linear_dynamics_t4",
)


def multiphysics_capabilities() -> dict:
    """候选能力与稳定 Gate 能力分开报告，不以平台可运行替代科学准入。"""
    return {
        "contract_version": MULTIPHYSICS_VERSION,
        "dimensions": [2, 3],
        "elements": ["T3", "T4"],
        "provider": "native_sparse",
        "required_optional_providers": [],
        "mechanical_loads": [
            "nodal_force",
            "constant_body_force",
            "reference_traction",
        ],
        "kinds": [
            {
                "name": kind,
                "dimensions": [3] if kind == "linear_dynamics_t4" else [2, 3],
                "maturity": "experimental"
                if kind in ("finite_strain", "at2_fracture")
                else "implemented",
            }
            for kind in KINDS
        ],
        "limitations": [
            "热力为单向顺序耦合，不含机械耗散热反馈。",
            "有限变形不含接触、弧长、壳梁或近不可压缩混合公式。",
            "AT2 无张压分裂，压缩亦会损伤；不用于压缩断裂或裂面闭合。",
            "T4 动力 JSON 仅 Newmark、零固定约束、常载荷；无阻尼或 JSON 重启。",
            "面力按参考构形固定方向积分，不含随动压力；体力输入不是加速度。",
        ],
    }


def multiphysics_request_schema() -> dict:
    """安装包内的候选 JSON Schema；语义准入仍由同源预检负责。"""
    return json.loads(
        files("agentfem_native")
        .joinpath("schemas/multiphysics-request-0.4.schema.json")
        .read_text(encoding="utf-8")
    )


def _mapping(value, name: str, allowed: set[str]) -> dict:
    if not isinstance(value, dict):
        raise TypeError(f"{name}必须是对象。")
    extra = set(value) - allowed
    if extra:
        raise ValueError(f"{name}含不支持的字段：{', '.join(sorted(extra))}。")
    return value


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    return value


def _decode(request):
    _mapping(
        request,
        "请求",
        {
            "contract_version",
            "request_id",
            "kind",
            "mesh",
            "material",
            "thermal",
            "mechanical",
            "procedure",
            "budget",
            "experimental",
        },
    )
    if request.get("contract_version") != MULTIPHYSICS_VERSION:
        raise ValueError(f"多物理候选要求 contract_version={MULTIPHYSICS_VERSION}。")
    if "request_id" in request and not isinstance(request["request_id"], str):
        raise TypeError("request_id 必须是字符串。")
    if "experimental" in request and not isinstance(request["experimental"], bool):
        raise TypeError("experimental 必须是布尔值。")
    # 此处也拒绝 NaN/Infinity，摘要只绑定可移植 JSON 数值。
    digest = sha256(
        json.dumps(
            request,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    kind = request.get("kind")
    if kind not in KINDS:
        raise ValueError("不支持的多物理分析类型。")
    if (
        kind in ("finite_strain", "at2_fracture")
        and request.get("experimental") is not True
    ):
        raise ValueError("非线性/断裂候选必须显式给出 experimental=true。")
    raw = _mapping(
        request.get("mesh"),
        "网格",
        {"points", "cells", "node_sets", "boundary_sets", "cell_sets"},
    )
    points = np.asarray(raw["points"], dtype=float)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError("网格坐标维数只能为二或三。")
    d = points.shape[1]
    cls = TriangularMesh if d == 2 else TetrahedralMesh
    mesh = cls(**raw)
    material = _mapping(
        request.get("material", {}),
        "材料",
        {
            "young_modulus",
            "poisson_ratio",
            "mode",
            "expansion",
            "reference_temperature",
            "thermal_stretch",
            "toughness",
            "length_scale",
            "residual_stiffness",
            "thickness",
            "density",
        },
    )
    thermal = _mapping(
        request.get("thermal", {}),
        "热场",
        {
            "conductivity",
            "capacity",
            "source",
            "boundaries",
            "constraints",
            "initial",
            "temperature",
        },
    )
    mechanical = _mapping(
        request.get("mechanical", {}),
        "力学",
        {
            "constraints",
            "load",
            "body_force",
            "tractions",
            "initial_damage",
            "initial_displacement",
            "initial_velocity",
        },
    )
    procedure = _mapping(
        request.get("procedure", {}),
        "程序",
        {
            "steps",
            "time_step",
            "maximum_iterations",
            "maximum_cutbacks",
            "maximum_accepted_steps",
            "relative_tolerance",
            "absolute_tolerance",
            "load_factors",
            "maximum_alternations",
            "tolerance",
            "mass",
        },
    )
    budget = ResourceBudget(
        **_mapping(
            request.get("budget", {}),
            "资源预算",
            {"maximum_dofs", "maximum_peak_bytes", "maximum_steps"},
        )
    )
    material_fields = {
        "thermal_steady": {"thickness"},
        "thermal_transient": {"thickness"},
        "thermoelastic": {
            "young_modulus",
            "poisson_ratio",
            "mode",
            "expansion",
            "reference_temperature",
            "thickness",
        },
        "finite_strain": {
            "young_modulus",
            "poisson_ratio",
            "thermal_stretch",
            "thickness",
        },
        "at2_fracture": {
            "young_modulus",
            "poisson_ratio",
            "mode",
            "toughness",
            "length_scale",
            "residual_stiffness",
            "thickness",
        },
        "linear_dynamics_t4": {"young_modulus", "poisson_ratio", "density"},
    }
    procedure_fields = {
        "thermal_steady": set(),
        "thermoelastic": set(),
        "thermal_transient": {"time_step", "steps"},
        "finite_strain": {
            "steps",
            "maximum_iterations",
            "maximum_cutbacks",
            "maximum_accepted_steps",
            "relative_tolerance",
            "absolute_tolerance",
        },
        "at2_fracture": {"load_factors", "maximum_alternations", "tolerance"},
        "linear_dynamics_t4": {"time_step", "steps", "mass"},
    }
    _mapping(material, "当前分析的材料", material_fields[kind])
    _mapping(procedure, "当前分析的程序", procedure_fields[kind])
    if kind in ("finite_strain", "at2_fracture", "linear_dynamics_t4") and thermal:
        raise ValueError(
            "此候选分析不接受热场子问题；有限变形仅支持显式 thermal_stretch。"
        )
    if kind.startswith("thermal_") and mechanical:
        raise ValueError("纯热分析不接受力学字段。")
    if kind != "at2_fracture" and "initial_damage" in mechanical:
        raise ValueError("当前分析不接受初始损伤。")
    if (
        kind != "linear_dynamics_t4"
        and {"initial_displacement", "initial_velocity"} & mechanical.keys()
    ):
        raise ValueError("位移/速度初值仅用于 T4 线性动力候选。")
    if kind == "thermal_steady" and ("initial" in thermal or "temperature" in thermal):
        raise ValueError("稳态热分析不接受初值或预给定温度场。")
    if kind == "thermal_transient" and "temperature" in thermal:
        raise ValueError("瞬态热分析请使用 initial 字段。")
    if kind == "thermoelastic" and "initial" in thermal:
        raise ValueError("当前热弹性链只接受稳态热或显式 temperature。")
    if (
        kind == "thermoelastic"
        and "temperature" in thermal
        and set(thermal) != {"temperature"}
    ):
        raise ValueError("预给定 temperature 不得同时包含另一热传导问题。")
    return kind, mesh, material, thermal, mechanical, procedure, budget, digest


def _bc(raw, count: int) -> EssentialBC:
    if raw is None:
        return EssentialBC.empty()
    raw = _mapping(raw, "约束", {"dofs", "values"})
    bc = EssentialBC(raw["dofs"], raw["values"])
    bc.validate(count)
    return bc


def _validate_physics(decoded):
    """轻量物理准入不创建算子/CSR；共享材料、系数与约束校验规则。"""
    kind, mesh, material, thermal, mechanical, procedure, _budget, _digest = decoded
    d, n, count = mesh.points.shape[1], mesh.node_count, mesh.cell_count
    components = connected_components(mesh)
    thickness = positive(material.get("thickness", 1.0), "厚度")
    if d == 3 and thickness != 1:
        raise ValueError("三维实体不能使用二维厚度。")
    thermal_bc = _bc(thermal.get("constraints"), n)
    mechanical_bc = _bc(mechanical.get("constraints"), n * d)
    if "load" in mechanical:
        finite_array(mechanical["load"], (n, d), "节点外力")
    if "body_force" in mechanical:
        finite_array(mechanical["body_force"], (d,), "单位参考体积的体力")
    tractions = _mapping(
        mechanical.get("tractions", {}), "参考面力映射", set(mesh.boundary_sets)
    )
    for name, value in tractions.items():
        finite_array(value, (d,), "单位参考面积的面力")
        if not mesh.boundary_sets[name].size:
            raise ValueError(f"参考面力边界 {name} 为空，不能静默忽略载荷。")
    if not kind.startswith("thermal_"):
        check_component_constraints(mesh, components, mechanical_bc)
        _elastic_parameters(material["young_modulus"], material["poisson_ratio"])
        mode = material.get("mode", "plane_strain")
        if mode not in ("plane_strain", "plane_stress") or (
            d == 3 and mode == "plane_stress"
        ):
            raise ValueError("实体/平面本构模式与维数不兼容。")
    if kind == "linear_dynamics_t4":
        if d != 3:
            raise ValueError("linear_dynamics_t4 只支持三维 T4 实体。")
        positive(material["density"], "密度")
        validate_time_interval(
            positive(procedure["time_step"], "动力时间步"),
            positive_integer(procedure["steps"], "动力步数"),
        )
        if procedure.get("mass", "lumped") not in ("lumped", "consistent"):
            raise ValueError("动力质量形式必须是 lumped 或 consistent。")
        if np.any(mechanical_bc.values != 0) or mechanical_bc.dofs.size == n * d:
            raise ValueError("T4 动力要求零固定约束且至少存在一个自由 DOF。")
        for name in ("initial_displacement", "initial_velocity"):
            if name in mechanical:
                values = finite_array(mechanical[name], (n, d), name)
                if np.any(values.ravel()[mechanical_bc.dofs] != 0):
                    raise ValueError(f"{name} 违反零固定约束。")
    if kind == "thermoelastic":
        cell_values(material.get("expansion", 0.0), count, "线膨胀系数")
        finite_array(material.get("reference_temperature", 0.0), (), "参考温度")
    if "temperature" in thermal:
        finite_array(thermal["temperature"], (n,), "节点温度")
    elif kind in ("thermal_steady", "thermal_transient", "thermoelastic"):
        conductivity_tensors(thermal.get("conductivity", 1.0), count, d)
        if np.any(cell_values(thermal.get("capacity", 1.0), count, "体积热容") <= 0):
            raise ValueError("体积热容必须为正。")
        cell_values(thermal.get("source", 0.0), count, "体热源")
        boundaries = thermal.get("boundaries", [])
        if not isinstance(boundaries, list):
            raise TypeError("热边界必须是数组。")
        anchored = set(components[thermal_bc.dofs])
        for item in boundaries:
            boundary = ThermalBoundary(
                **_mapping(
                    item,
                    "热边界",
                    {
                        "boundary_set",
                        "inward_flux",
                        "convection",
                        "ambient_temperature",
                    },
                )
            )
            if boundary.boundary_set not in mesh.boundary_sets:
                raise ValueError(f"未知热边界集合：{boundary.boundary_set}。")
            h = positive(boundary.convection, "换热系数", allow_zero=True)
            finite_array(
                [boundary.inward_flux, boundary.ambient_temperature], (2,), "热边界参数"
            )
            if h > 0:
                anchored.update(
                    components[mesh.boundary_sets[boundary.boundary_set].ravel()]
                )
        if kind != "thermal_transient" and anchored != set(components):
            raise ValueError("稳态热场每个连通分量都需要固定温度或正换热边界。")
        if kind == "thermal_transient":
            validate_time_interval(
                positive(procedure["time_step"], "时间步"),
                positive_integer(procedure["steps"], "时间步数"),
            )
            initial = finite_array(thermal["initial"], (n,), "初始温度")
            if not np.allclose(
                initial[thermal_bc.dofs], thermal_bc.values, rtol=0, atol=1e-12
            ):
                raise ValueError("初始温度必须满足固定温度约束。")
    if kind == "finite_strain":
        if np.any(
            cell_values(material.get("thermal_stretch", 1.0), count, "热伸长") <= 0
        ):
            raise ValueError("热伸长必须为正。")
        for key in ("maximum_iterations", "maximum_cutbacks", "maximum_accepted_steps"):
            if key in procedure:
                positive_integer(procedure[key], key)
        for key in ("relative_tolerance", "absolute_tolerance"):
            if key in procedure:
                positive(procedure[key], key)
    if kind == "at2_fracture":
        positive(material["toughness"], "断裂韧度")
        positive(material["length_scale"], "正则化长度")
        if positive(material.get("residual_stiffness", 1e-8), "残余刚度") >= 1:
            raise ValueError("残余刚度必须小于一。")
        positive_integer(procedure.get("maximum_alternations", 200), "交替迭代上限")
        positive(procedure.get("tolerance", 1e-8), "交替迭代容差")
        if "initial_damage" in mechanical:
            damage = finite_array(mechanical["initial_damage"], (n,), "初始损伤")
            if np.any(damage < 0) or np.any(damage > 1):
                raise ValueError("初始损伤必须位于 [0,1]。")


def _mechanical_load(geometry, mechanical):
    """Native 拥有载荷积分；三种来源相加，均在参考构形中定义。"""
    if not {"load", "body_force", "tractions"} & mechanical.keys():
        return None
    values = geometry.mechanical_load(
        body_force=mechanical.get("body_force"), tractions=mechanical.get("tractions")
    )
    if "load" in mechanical:
        values += np.asarray(mechanical["load"], dtype=float)
    return finite_array(values, geometry.mesh.points.shape, "组合节点力")


def _plan(decoded):
    _validate_physics(decoded)
    kind, mesh, _material, _thermal, _mechanical, procedure, _budget, digest = decoded
    d, nodes, cells = mesh.points.shape[1], mesh.node_count, mesh.cell_count
    steps = positive_integer(
        procedure.get("steps", 8 if kind == "finite_strain" else 1), "步数"
    )
    history = steps + 1
    if kind == "at2_fracture":
        factors = np.asarray(
            procedure.get("load_factors", [0.25, 0.5, 0.75, 1.0]), dtype=float
        )
        if (
            factors.ndim != 1
            or factors.size == 0
            or not np.all(np.isfinite(factors))
            or np.any(factors < 0)
        ):
            raise ValueError("载荷因子必须是非空有限非负数组。")
        steps = history = factors.size
    if kind == "finite_strain":
        history = positive_integer(
            procedure.get("maximum_accepted_steps", 512), "最大接受步数"
        )
        if steps > history:
            raise ValueError("载荷步数超过最大接受步数。")
    if kind in ("thermal_steady", "thermoelastic"):
        steps = history = 1
    dofs = nodes * (
        1
        if kind.startswith("thermal_")
        else d + (1 if kind in ("thermoelastic", "at2_fracture") else 0)
    )
    # 同时覆盖图构造、元素贡献、多个 CSR/预条件器、Krylov 和历史到 JSON 的膨胀。
    # 这是保守工作区估计，不是 Python 解释器/第三方分配器的 RSS 硬限制。
    local_size = (d + 1) * (1 if kind.startswith("thermal_") else d)
    peak = int(
        160 * cells * local_size**2
        + 1024 * dofs
        + 256 * history * (dofs + 12 * cells)
        + 2**20
    )
    plan = {
        "kind": kind,
        "dimension": d,
        "node_count": nodes,
        "cell_count": cells,
        "dof_count": int(dofs),
        "peak_bytes_upper_bound": peak,
        "requested_steps": int(steps),
        "history_capacity": int(history),
        "request_digest": digest,
        "maturity": "experimental"
        if kind in ("finite_strain", "at2_fracture")
        else "implemented",
        "warnings": [
            "新多物理候选尚未完成独立 Gate 验收。",
            "资源值为数组与序列化工作区估计，不是进程 RSS 硬上限。",
        ],
    }
    if kind == "at2_fracture":
        plan["warnings"].append("各向同性 AT2 无张压分裂；压缩也会损伤，不含接触。")
    if kind == "finite_strain":
        plan["warnings"].append(
            "可压缩 T3 平面应变/T4；不含近不可压缩混合、接触、壳梁或弧长跟踪。"
        )
    if kind == "linear_dynamics_t4":
        plan["warnings"].append(
            "三维动力当前仅无阻尼 Newmark、零固定约束及常载荷，尚未完成三维波传播 Gate。"
        )
    plan["plan_digest"] = sha256(
        json.dumps(plan, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return plan


def _context(budget, context):
    original = context or ExecutionContext()
    values = {}
    for key in ("maximum_dofs", "maximum_peak_bytes", "maximum_steps"):
        limits = [
            v
            for v in (getattr(original.budget, key), getattr(budget, key))
            if v is not None
        ]
        values[key] = min(limits) if limits else None
    return ExecutionContext(
        ResourceBudget(**values), original.cancellation, original.progress
    )


def _failure(error, request=None):
    if isinstance(error, LinearSolveError):
        result = {
            "code": "solver.linear_nonconvergence",
            "message": str(error),
            "path": "solver",
            "retryable": True,
            "remediations": ["inspect_constraints", "inspect_conditioning"],
        }
        if error.report is not None:
            result["convergence"] = _json(asdict(error.report))
    elif isinstance(error, NativeExecutionError):
        result = error.as_dict()
    else:
        result = {
            "code": "request.invalid",
            "message": str(error),
            "path": "$",
            "retryable": False,
        }
    if isinstance(error, (NonlinearConvergenceError, NonlinearInterruptionError)):
        result["accepted_states"] = [
            _json(asdict(state)) for state in error.accepted_states
        ]
    return {
        "contract_version": MULTIPHYSICS_VERSION,
        "request_id": request.get("request_id", "unknown")
        if isinstance(request, dict)
        and isinstance(request.get("request_id", "unknown"), str)
        else "unknown",
        "status": "failed",
        "error": result,
    }


def plan_multiphysics_request(
    request: dict, *, context: ExecutionContext | None = None
) -> dict:
    """预检不构造单元算子或 CSR 图；执行前由同一请求再次推导计划。"""
    try:
        decoded = _decode(request)
        plan = _plan(decoded)
        ctx = _context(decoded[-2], context)
        enforce_plan_budget(SimpleNamespace(**plan), ctx.budget)
        ctx.check("multiphysics.plan", 0, plan["requested_steps"])
        return {
            "contract_version": MULTIPHYSICS_VERSION,
            "request_id": request.get("request_id", "unknown"),
            "status": "planned",
            "plan": plan,
        }
    except (ValueError, TypeError, KeyError, NativeExecutionError) as error:
        return _failure(error, request)


def run_multiphysics_request(
    request: dict, *, context: ExecutionContext | None = None
) -> dict:
    """计算、成熟度与数学账本一起返回，供 AgentFEM 审查。"""
    try:
        decoded = _decode(request)
        kind, mesh, material, thermal, mechanical, procedure, budget, digest = decoded
        plan = _plan(decoded)
        ctx = _context(budget, context)
        enforce_plan_budget(SimpleNamespace(**plan), ctx.budget)
        ctx.check("multiphysics.prepare", 0, plan["requested_steps"])
        d = mesh.points.shape[1]
        thermal_bc = _bc(thermal.get("constraints"), mesh.node_count)
        mechanical_bc = _bc(mechanical.get("constraints"), mesh.node_count * d)
        thickness = material.get("thickness", 1.0)
        heat_result = None
        if kind in ("thermal_steady", "thermal_transient", "thermoelastic"):
            if kind == "thermoelastic" and "temperature" in thermal:
                temperature = thermal["temperature"]
            else:
                boundaries = tuple(
                    ThermalBoundary(
                        **_mapping(
                            item,
                            "热边界",
                            {
                                "boundary_set",
                                "inward_flux",
                                "convection",
                                "ambient_temperature",
                            },
                        )
                    )
                    for item in thermal.get("boundaries", [])
                )
                operator = ThermalOperator(
                    mesh,
                    conductivity=thermal.get("conductivity", 1.0),
                    capacity=thermal.get("capacity", 1.0),
                    source=thermal.get("source", 0.0),
                    boundaries=boundaries,
                    thickness=thickness,
                )
                if kind == "thermal_transient":
                    heat_result = operator.transient(
                        thermal["initial"],
                        thermal_bc,
                        time_step=procedure["time_step"],
                        steps=procedure["steps"],
                        context=ctx,
                    )
                else:
                    heat_result = operator.steady(thermal_bc)
                temperature = heat_result.temperatures[-1]
            if kind == "thermoelastic":
                elastic = ThermoelasticOperator(
                    mesh,
                    material["young_modulus"],
                    material["poisson_ratio"],
                    mode=material.get("mode", "plane_strain"),
                    temperature=temperature,
                    expansion=material.get("expansion", 0.0),
                    reference_temperature=material.get("reference_temperature", 0.0),
                    thickness=thickness,
                )
                result = elastic.solve(
                    mechanical_bc, load=_mechanical_load(elastic.geometry, mechanical)
                )
            else:
                result = heat_result
        elif kind == "linear_dynamics_t4":
            elastic = ThermoelasticOperator(
                mesh, material["young_modulus"], material["poisson_ratio"]
            )
            mass = assemble_t4_mass(
                mesh,
                material["density"],
                lumped=procedure.get("mass", "lumped") == "lumped",
            )
            shape = (mesh.node_count, 3)
            load = _mechanical_load(elastic.geometry, mechanical)
            system = LinearSecondOrderSystem(
                elastic.stiffness,
                mass,
                (np.zeros(shape) if load is None else load).ravel(),
                procedure["time_step"],
                procedure["steps"],
                np.asarray(
                    mechanical.get("initial_displacement", np.zeros(shape))
                ).ravel(),
                np.asarray(mechanical.get("initial_velocity", np.zeros(shape))).ravel(),
            )
            result = integrate_constrained_linear_dynamics(
                system,
                mechanical_bc.dofs,
                method="newmark_average_acceleration",
                context=ctx,
            )
        elif kind == "finite_strain":
            operator = FiniteStrainOperator(
                mesh,
                NeoHookeanMaterial(
                    material["young_modulus"], material["poisson_ratio"]
                ),
                thickness=thickness,
                thermal_stretch=material.get("thermal_stretch", 1.0),
            )
            result = operator.solve(
                mechanical_bc,
                load=_mechanical_load(operator.geometry, mechanical),
                context=ctx,
                **procedure,
            )
        else:
            operator = AT2FractureOperator(
                mesh,
                material["young_modulus"],
                material["poisson_ratio"],
                toughness=material["toughness"],
                length_scale=material["length_scale"],
                residual_stiffness=material.get("residual_stiffness", 1e-8),
                mode=material.get("mode", "plane_strain"),
                thickness=thickness,
            )
            result = operator.solve(
                mechanical_bc,
                load=_mechanical_load(operator.geometry, mechanical),
                initial_damage=mechanical.get("initial_damage"),
                context=ctx,
                **procedure,
            )
        check_accepted(
            ctx,
            "multiphysics.complete",
            plan["requested_steps"],
            plan["requested_steps"],
            getattr(result, "states", ()),
        )
        fields = _json(asdict(result))
        if kind == "linear_dynamics_t4":
            fields["maturity"] = "implemented"
            fields["dof_layout"] = "node_major_xyz"
        if kind == "thermoelastic" and heat_result is not None:
            fields["thermal"] = _json(asdict(heat_result))
        result_digest = sha256(
            json.dumps(fields, allow_nan=False, sort_keys=True).encode()
        ).hexdigest()
        return {
            "contract_version": MULTIPHYSICS_VERSION,
            "request_id": request.get("request_id", "unknown"),
            "status": "success",
            "plan": plan,
            "result": fields,
            "evidence": {
                "request_digest": digest,
                "plan_digest": plan["plan_digest"],
                "result_digest": result_digest,
                "result_digest_role": "exact_payload_integrity",
                "repeatability": "numerical_tolerance_not_bitwise_guarantee",
                "maturity": plan["maturity"],
                "finite_element_owner": "agentfem_native",
            },
        }
    except (ValueError, TypeError, KeyError, NativeExecutionError) as error:
        return _failure(error, request)
