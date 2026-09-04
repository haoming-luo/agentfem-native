# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Versioned JSON-safe Kernel Contract and external AgentFEM AF-IR adapter."""

from __future__ import annotations

import json
import platform
from collections.abc import Mapping
from hashlib import sha256
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import TypeAlias

import numpy as np

from . import __version__
from .diffusion import (
    CellMaterial,
    DirichletCondition,
    NeumannCondition,
    SteadyDiffusionProblem,
    solve_steady_diffusion,
)
from .mesh import TriangularMesh
from .providers import ProviderUnavailableError
from .results import write_legacy_vtk

JsonMapping: TypeAlias = Mapping[str, object]
CONTRACT_NAME = "agentfem.native-kernel-request"
CONTRACT_VERSION = "0.1.0"
AFIR_SCHEMA = "agentfem.af-ir"
AFIR_VERSION = "0.1.0"


class KernelRequestError(ValueError):
    """Addressable request or capability failure."""

    def __init__(self, code: str, message: str, *, path: str = "$") -> None:
        super().__init__(message)
        self.code = code
        self.path = path

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": str(self), "path": self.path}


def _mapping(value: object, path: str) -> JsonMapping:
    if not isinstance(value, Mapping):
        raise KernelRequestError("invalid_request", "Expected an object.", path=path)
    return value


def _sequence(value: object, path: str) -> list[object]:
    if not isinstance(value, (list, tuple)):
        raise KernelRequestError("invalid_request", "Expected an array.", path=path)
    return list(value)


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise KernelRequestError(
            "invalid_request", "Expected a nonempty string.", path=path
        )
    return value.strip()


def _finite_scalar(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise KernelRequestError(
            "invalid_request", "Expected a finite number.", path=path
        )
    result = float(value)
    if not np.isfinite(result):
        raise KernelRequestError(
            "invalid_request", "Expected a finite number.", path=path
        )
    return result


def _conductivity(value: object, path: str) -> float | np.ndarray:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _finite_scalar(value, path)
    try:
        tensor = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise KernelRequestError(
            "invalid_request", "Conductivity must be a scalar or 2x2 tensor.", path=path
        ) from error
    if tensor.shape != (2, 2):
        raise KernelRequestError(
            "invalid_request", "Conductivity tensor must have shape 2x2.", path=path
        )
    return tensor


def _named_arrays(value: object, path: str) -> dict[str, object]:
    mapping = _mapping(value, path)
    return {str(name): data for name, data in mapping.items()}


def _problem_from_request(
    request: JsonMapping,
) -> tuple[SteadyDiffusionProblem, str, JsonMapping]:
    if request.get("contract") != CONTRACT_NAME:
        raise KernelRequestError(
            "invalid_contract",
            f"contract must be {CONTRACT_NAME!r}.",
            path="$.contract",
        )
    if request.get("contract_version") != CONTRACT_VERSION:
        raise KernelRequestError(
            "unsupported_contract_version",
            f"Supported contract_version is {CONTRACT_VERSION!r}.",
            path="$.contract_version",
        )
    _text(request.get("request_id"), "$.request_id")

    study = _mapping(request.get("study"), "$.study")
    if (
        study.get("analysis") != "linear_static"
        or study.get("physics") != "heat_transfer"
    ):
        raise KernelRequestError(
            "unsupported_capability",
            "Only linear_static heat_transfer is implemented.",
            path="$.study",
        )
    if study.get("dimension") != 2:
        raise KernelRequestError(
            "unsupported_capability",
            "Only dimension=2 is implemented.",
            path="$.study.dimension",
        )

    mesh_record = _mapping(request.get("mesh"), "$.mesh")
    try:
        mesh = TriangularMesh(
            points=mesh_record.get("points"),
            cells=mesh_record.get("cells"),
            node_sets=_named_arrays(
                mesh_record.get("node_sets", {}), "$.mesh.node_sets"
            ),
            boundary_sets=_named_arrays(
                mesh_record.get("boundary_sets", {}), "$.mesh.boundary_sets"
            ),
            cell_sets=_named_arrays(
                mesh_record.get("cell_sets", {}), "$.mesh.cell_sets"
            ),
        )
    except (TypeError, ValueError) as error:
        raise KernelRequestError("invalid_model", str(error), path="$.mesh") from error

    physics = _mapping(request.get("physics"), "$.physics")
    conductivity = _conductivity(physics.get("conductivity"), "$.physics.conductivity")
    source = _finite_scalar(physics.get("source", 0.0), "$.physics.source")

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

    procedure = _mapping(request.get("procedure", {}), "$.procedure")
    if procedure.get("kind", "steady_diffusion") != "steady_diffusion":
        raise KernelRequestError(
            "unsupported_capability",
            "Only procedure kind 'steady_diffusion' is implemented.",
            path="$.procedure.kind",
        )
    provider = _text(
        procedure.get("linear_algebra", "numpy"), "$.procedure.linear_algebra"
    )
    if provider not in {"numpy", "scipy", "auto"}:
        raise KernelRequestError(
            "unsupported_provider",
            f"Unknown linear algebra provider {provider!r}.",
            path="$.procedure.linear_algebra",
        )
    assembly_mode = _text(procedure.get("assembly", "auto"), "$.procedure.assembly")
    if assembly_mode not in {"auto", "reference", "vectorized"}:
        raise KernelRequestError(
            "unsupported_assembly",
            f"Unknown assembly mode {assembly_mode!r}.",
            path="$.procedure.assembly",
        )
    outputs = _mapping(request.get("outputs", {}), "$.outputs")
    return (
        SteadyDiffusionProblem(
            mesh=mesh,
            conductivity=conductivity,
            source=source,
            dirichlet=tuple(dirichlet),
            neumann=tuple(neumann),
            materials=tuple(materials),
            assembly_mode=assembly_mode,
        ),
        provider,
        outputs,
    )


def _artifact_path(directory: Path, portable_path: object) -> tuple[Path, str]:
    text = _text(portable_path, "$.outputs.vtk.path")
    if "\\" in text:
        raise KernelRequestError(
            "invalid_artifact_path",
            "Artifact paths must use portable '/' separators.",
            path="$.outputs.vtk.path",
        )
    pure = PurePosixPath(text)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise KernelRequestError(
            "invalid_artifact_path",
            "Artifact path must be relative and cannot escape its root.",
            path="$.outputs.vtk.path",
        )
    return directory.joinpath(*pure.parts), pure.as_posix()


def _runtime(
    provider_name: str, provider_version: str, assembly_mode: str
) -> dict[str, object]:
    return {
        "os": platform.system(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "assembly": {"name": assembly_mode},
        "linear_algebra_provider": {
            "name": provider_name,
            "version": provider_version,
        },
    }


def run_kernel_request(
    request: JsonMapping,
    *,
    artifact_directory: str | Path | None = None,
) -> dict[str, object]:
    """Execute one JSON-safe request and return a structured result envelope."""

    request_id = (
        str(request.get("request_id", "unknown"))
        if isinstance(request, Mapping)
        else "unknown"
    )
    try:
        mapping = _mapping(request, "$")
        problem, provider, outputs = _problem_from_request(mapping)
        result = solve_steady_diffusion(problem, provider=provider)
        artifacts: list[dict[str, object]] = []
        if "vtk" in outputs:
            if artifact_directory is None:
                raise KernelRequestError(
                    "artifact_root_required",
                    "artifact_directory is required when VTK output is requested.",
                    path="$.outputs.vtk",
                )
            vtk = _mapping(outputs["vtk"], "$.outputs.vtk")
            path, portable = _artifact_path(Path(artifact_directory), vtk.get("path"))
            write_legacy_vtk(
                path,
                problem.mesh,
                result.nodal_values,
                field_name=str(vtk.get("field_name", "temperature")),
            )
            artifacts.append(
                {
                    "name": "vtk",
                    "path": portable,
                    "sha256": sha256(path.read_bytes()).hexdigest(),
                }
            )
        return {
            "contract": CONTRACT_NAME,
            "contract_version": CONTRACT_VERSION,
            "request_id": request_id,
            "status": "success",
            "backend": {"name": "native", "version": __version__},
            "capabilities": ["steady_diffusion_p1_triangle:verified_local"],
            "quantities": {
                "free_residual_norm": result.free_residual_norm,
                "total_applied_load": result.total_applied_load,
                "total_reaction": result.total_reaction,
                "potential_energy": result.potential_energy,
            },
            "fields": {
                "solution": {
                    "association": "node",
                    "components": 1,
                    "values": result.nodal_values.tolist(),
                }
            },
            "artifacts": artifacts,
            "runtime": _runtime(
                result.provider_name, result.provider_version, result.assembly_mode
            ),
            "warnings": [],
        }
    except KernelRequestError as error:
        failure = error
    except ProviderUnavailableError as error:
        failure = KernelRequestError(
            "provider_unavailable", str(error), path="$.procedure"
        )
    except OSError as error:
        failure = KernelRequestError(
            "artifact_write_failed", str(error), path="$.outputs"
        )
    except (KeyError, TypeError, ValueError) as error:
        failure = KernelRequestError("invalid_model", str(error))
    return {
        "contract": CONTRACT_NAME,
        "contract_version": CONTRACT_VERSION,
        "request_id": request_id,
        "status": "failed",
        "backend": {"name": "native", "version": __version__},
        "error": failure.as_dict(),
    }


def lower_agentfem_ir(document: JsonMapping) -> dict[str, object]:
    """Lower the executable portable subset of public AgentFEM AF-IR 0.1."""

    record = _mapping(document, "$")
    if (
        record.get("schema") != AFIR_SCHEMA
        or record.get("schema_version") != AFIR_VERSION
    ):
        raise KernelRequestError(
            "unsupported_agentfem_ir",
            f"Expected {AFIR_SCHEMA} version {AFIR_VERSION}.",
        )
    if record.get("document_type") != "model":
        raise KernelRequestError(
            "unsupported_agentfem_ir", "AF-IR document_type must be 'model'."
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
    """Call AgentFEM's public ``to_ir`` boundary without importing AgentFEM."""

    method = getattr(model, "to_ir", None)
    if not callable(method):
        raise TypeError("AgentFEM model adapter requires a public to_ir() method.")
    document = method()
    if hasattr(document, "as_dict") and callable(document.as_dict):
        document = document.as_dict()
    return lower_agentfem_ir(_mapping(document, "$"))


def kernel_request_schema() -> dict[str, object]:
    """Load the bundled JSON Schema 2020-12 contract description."""

    schema = (
        resources.files("agentfem_native")
        .joinpath("schemas")
        .joinpath("kernel-request-0.1.schema.json")
    )
    return json.loads(schema.read_text(encoding="utf-8"))


def kernel_result_schema() -> dict[str, object]:
    """Load the bundled JSON Schema 2020-12 result description."""

    schema = (
        resources.files("agentfem_native")
        .joinpath("schemas")
        .joinpath("kernel-result-0.1.schema.json")
    )
    return json.loads(schema.read_text(encoding="utf-8"))
