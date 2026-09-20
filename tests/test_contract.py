# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from agentfem_native import (
    CONTRACT_NAME,
    CONTRACT_VERSION,
    CancellationToken,
    ExecutionContext,
    KernelRequestError,
    kernel_request_schema,
    kernel_result_schema,
    lower_agentfem_ir,
    lower_agentfem_model,
    plan_kernel_request,
    run_kernel_request,
    unit_cube_tetrahedra,
    unit_square_two_triangles,
)
from agentfem_native.native import native_kernel_available


def request_record(*, outputs: dict[str, object] | None = None) -> dict[str, object]:
    mesh = unit_square_two_triangles()
    return {
        "contract": CONTRACT_NAME,
        "contract_version": CONTRACT_VERSION,
        "request_id": "diffusion-001",
        "study": {
            "analysis": "linear_static",
            "physics": "heat_transfer",
            "dimension": 2,
        },
        "mesh": {
            "points": mesh.points.tolist(),
            "cells": mesh.cells.tolist(),
            "node_sets": {
                name: values.tolist() for name, values in mesh.node_sets.items()
            },
            "boundary_sets": {
                name: values.tolist() for name, values in mesh.boundary_sets.items()
            },
            "cell_sets": {},
        },
        "physics": {"conductivity": 1.0, "source": 0.0},
        "dirichlet": [{"node_set": "left", "value": 0.0}],
        "neumann": [{"boundary_set": "right", "flux": 1.0}],
        "materials": [],
        "procedure": {"kind": "steady_diffusion", "linear_algebra": "numpy"},
        "outputs": {} if outputs is None else outputs,
    }


def mechanics_request(dimension: int) -> dict[str, object]:
    mesh = unit_square_two_triangles() if dimension == 2 else unit_cube_tetrahedra()
    value = [0.0] * dimension
    traction = [1.0] + [0.0] * (dimension - 1)
    return {
        "contract": CONTRACT_NAME,
        "contract_version": CONTRACT_VERSION,
        "request_id": f"elasticity-{dimension}d-001",
        "study": {
            "analysis": "linear_static",
            "physics": "solid_mechanics",
            "dimension": dimension,
        },
        "mesh": {
            "points": mesh.points.tolist(),
            "cells": mesh.cells.tolist(),
            "node_sets": {
                name: values.tolist() for name, values in mesh.node_sets.items()
            },
            "boundary_sets": {
                name: values.tolist() for name, values in mesh.boundary_sets.items()
            },
            "cell_sets": {},
        },
        "physics": {
            "material": {"young_modulus": 100.0, "poisson_ratio": 0.25},
            "body_force": value,
        },
        "dirichlet": [{"node_set": "left", "component": None, "value": value}],
        "traction": [{"boundary_set": "right", "value": traction}],
        "materials": [],
        "procedure": {
            "kind": "linear_elasticity",
            "linear_algebra": "numpy",
            "assembly": "reference",
        },
        "outputs": {},
    }


def dynamics_request(*, steps: int = 6) -> dict[str, object]:
    mesh = unit_square_two_triangles()
    initial_displacement = [[0.01 * float(point[0]), 0.0] for point in mesh.points]
    return {
        "contract": CONTRACT_NAME,
        "contract_version": CONTRACT_VERSION,
        "request_id": "dynamics-2d-001",
        "study": {
            "analysis": "linear_transient",
            "physics": "solid_mechanics",
            "dimension": 2,
        },
        "mesh": {
            "points": mesh.points.tolist(),
            "cells": mesh.cells.tolist(),
            "node_sets": {
                name: values.tolist() for name, values in mesh.node_sets.items()
            },
            "boundary_sets": {
                name: values.tolist() for name, values in mesh.boundary_sets.items()
            },
            "cell_sets": {},
        },
        "physics": {
            "material": {"young_modulus": 10.0, "poisson_ratio": 0.25},
            "density": 1.0,
            "body_force": [0.0, 0.0],
        },
        "dirichlet": [{"node_set": "left", "value": [0.0, 0.0]}],
        "traction": [],
        "materials": [],
        "procedure": {
            "kind": "linear_dynamics",
            "linear_algebra": "native_sparse",
            "assembly": "reference",
            "integrator": "newmark_average_acceleration",
            "time_step": 0.01,
            "steps": steps,
            "mass": "consistent",
            "load_scale": {"times": [0.0, 1.0], "values": [0.0, 0.0]},
        },
        "initial_state": {
            "displacement": initial_displacement,
            "velocity": [[0.0, 0.0] for _ in mesh.points],
        },
        "outputs": {},
    }


class ContractTests(unittest.TestCase):
    def test_0_3_dynamics_plan_execute_and_checkpoint_are_json_safe(self) -> None:
        request = dynamics_request()
        with patch(
            "agentfem_native.execution.build_t3_linear_dynamics",
            side_effect=AssertionError("动力预检不得装配矩阵"),
        ):
            planned = plan_kernel_request(request)
        self.assertEqual(planned["status"], "planned")
        self.assertEqual(planned["plan"]["problem_kind"], "linear_dynamics_t3")  # type: ignore[index]
        self.assertEqual(planned["plan"]["maturity"], "verified")  # type: ignore[index]

        result = run_kernel_request(request)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["plan"]["digest"], planned["plan"]["digest"])  # type: ignore[index]
        self.assertEqual(len(result["quantities"]["times"]), 7)  # type: ignore[index]
        self.assertEqual(result["checkpoint"]["step"], 6)  # type: ignore[index]
        self.assertEqual(len(result["checkpoint"]["digest"]), 64)  # type: ignore[index]
        self.assertEqual(
            result["evidence"]["execution"]["analysis"]["spatial_dimension"],  # type: ignore[index]
            2,
        )
        self.assertIn(
            "linear_dynamics_t3_central_difference_newmark:verified",
            result["capabilities"],
        )
        json.dumps(result, allow_nan=False, sort_keys=True)

    def test_0_3_dynamics_checkpoint_restarts_exactly(self) -> None:
        uninterrupted = run_kernel_request(dynamics_request(steps=6))
        first = run_kernel_request(dynamics_request(steps=3))
        restarted_request = dynamics_request(steps=3)
        restarted_request.pop("initial_state")
        restarted_request["restart"] = first["checkpoint"]
        restarted_request["procedure"]["load_scale"] = {  # type: ignore[index]
            "times": [0.0, 1.0],
            "values": [0.0, 0.0],
        }
        restarted = run_kernel_request(restarted_request)
        self.assertEqual(restarted["status"], "success")
        self.assertEqual(
            restarted["fields"]["displacement"]["values"][-1],  # type: ignore[index]
            uninterrupted["fields"]["displacement"]["values"][-1],  # type: ignore[index]
        )
        self.assertEqual(restarted["checkpoint"]["step"], 6)  # type: ignore[index]

    def test_0_3_dynamics_budget_and_cancellation_are_structured(self) -> None:
        request = dynamics_request(steps=3)
        request["execution"] = {"budget": {"maximum_steps": 2}}
        planned = plan_kernel_request(request)
        self.assertEqual(planned["status"], "failed")
        self.assertEqual(planned["error"]["code"], "budget.steps_exceeded")  # type: ignore[index]
        self.assertEqual(planned["error"]["path"], "procedure.steps")  # type: ignore[index]

        request.pop("execution")
        cancellation = CancellationToken()
        cancellation.cancel()
        cancelled = run_kernel_request(
            request, context=ExecutionContext(cancellation=cancellation)
        )
        self.assertEqual(cancelled["status"], "failed")
        self.assertEqual(cancelled["error"]["code"], "execution.cancelled")  # type: ignore[index]
        self.assertTrue(cancelled["error"]["retryable"])  # type: ignore[index]

    def test_0_3_dynamics_rejects_unsafe_contract_combinations(self) -> None:
        request = dynamics_request()
        request["procedure"]["integrator"] = "central_difference"  # type: ignore[index]
        result = run_kernel_request(request)
        self.assertEqual(result["error"]["code"], "unsupported_mass")  # type: ignore[index]
        self.assertEqual(result["error"]["path"], "$.procedure.mass")  # type: ignore[index]

        request["procedure"]["mass"] = "lumped"  # type: ignore[index]
        request["procedure"]["time_step"] = 100.0  # type: ignore[index]
        request["procedure"]["load_scale"] = {  # type: ignore[index]
            "times": [0.0, 1000.0],
            "values": [0.0, 0.0],
        }
        result = run_kernel_request(request)
        self.assertEqual(result["error"]["code"], "unsafe_time_step")  # type: ignore[index]
        self.assertEqual(result["error"]["path"], "$.procedure.time_step")  # type: ignore[index]

        request = dynamics_request()
        request["procedure"]["load_scale"] = {  # type: ignore[index]
            "times": [0.02, 1.0],
            "values": [0.0, 0.0],
        }
        result = plan_kernel_request(request)
        self.assertEqual(result["error"]["code"], "load_curve_range")  # type: ignore[index]
        self.assertEqual(
            result["error"]["path"],
            "$.procedure.load_scale.times",  # type: ignore[index]
        )

    def test_success_result_is_deterministic_json_safe_envelope(self) -> None:
        result = run_kernel_request(request_record())
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["request_id"], "diffusion-001")
        expected = "native" if native_kernel_available() else "vectorized"
        self.assertEqual(result["runtime"]["assembly"]["name"], expected)  # type: ignore[index]
        self.assertEqual(
            result["runtime"]["native_kernel"]["available"],  # type: ignore[index]
            native_kernel_available(),
        )
        values = result["fields"]["solution"]["values"]  # type: ignore[index]
        self.assertEqual(values, [0.0, 1.0, 1.0, 0.0])
        json.dumps(result, allow_nan=False, sort_keys=True)

    def test_contract_0_1_heat_request_remains_backward_compatible(self) -> None:
        request = request_record()
        request["contract_version"] = "0.1.0"
        result = run_kernel_request(request)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["contract_version"], "0.1.0")
        self.assertNotIn("plan", result)
        self.assertNotIn("evidence", result)
        planned = plan_kernel_request(request)
        self.assertEqual(planned["status"], "failed")
        self.assertEqual(planned["error"]["path"], "$.contract_version")  # type: ignore[index]

    def test_contract_0_2_static_mechanics_remains_backward_compatible(self) -> None:
        request = mechanics_request(2)
        request["contract_version"] = "0.2.0"
        planned = plan_kernel_request(request)
        result = run_kernel_request(request)
        self.assertEqual(planned["status"], "planned")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["contract_version"], "0.2.0")
        self.assertEqual(result["plan"]["digest"], planned["plan"]["digest"])  # type: ignore[index]
        cancellation = CancellationToken()
        cancellation.cancel()
        failed = run_kernel_request(
            request, context=ExecutionContext(cancellation=cancellation)
        )
        self.assertEqual(
            set(failed["error"]),  # type: ignore[arg-type]
            {"code", "message", "path"},
        )

    def test_t3_preflight_and_execution_share_one_resource_plan(self) -> None:
        request = mechanics_request(2)
        planned = plan_kernel_request(request)
        self.assertEqual(planned["status"], "planned")
        self.assertEqual(planned["plan"]["problem_kind"], "linear_elasticity_2d")  # type: ignore[index]
        self.assertEqual(planned["plan"]["maturity"], "verified")  # type: ignore[index]

        result = run_kernel_request(request)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["plan"]["digest"], planned["plan"]["digest"])  # type: ignore[index]
        self.assertEqual(result["fields"]["displacement"]["components"], 2)  # type: ignore[index]
        self.assertIn(
            "linear_elasticity_t3_plane_stress_strain:verified",
            result["capabilities"],
        )
        self.assertLess(result["quantities"]["balance_norm"], 1.0e-12)  # type: ignore[index]
        self.assertEqual(
            result["evidence"]["request_digest"],
            planned["request_digest"],  # type: ignore[index]
        )
        self.assertEqual(
            result["evidence"]["execution"]["claim_maturity"],
            "verified",  # type: ignore[index]
        )
        json.dumps(result, allow_nan=False, sort_keys=True)

    def test_t4_contract_executes_and_reports_owned_three_dimensional_fields(
        self,
    ) -> None:
        result = run_kernel_request(mechanics_request(3))
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["plan"]["problem_kind"], "linear_elasticity_3d")  # type: ignore[index]
        self.assertEqual(result["fields"]["displacement"]["components"], 3)  # type: ignore[index]
        self.assertEqual(result["fields"]["stress"]["components"], 6)  # type: ignore[index]
        self.assertIn("linear_elasticity_t4_3d:verified", result["capabilities"])
        self.assertLess(result["quantities"]["balance_norm"], 1.0e-11)  # type: ignore[index]

    def test_invalid_mechanics_component_fails_before_assembly_at_exact_path(
        self,
    ) -> None:
        request = mechanics_request(2)
        request["dirichlet"] = [{"node_set": "left", "component": "z", "value": 0.0}]
        result = run_kernel_request(request)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "invalid_request")  # type: ignore[index]
        self.assertEqual(result["error"]["path"], "$.dirichlet[0].component")  # type: ignore[index]

        solid_request = mechanics_request(3)
        solid_request["physics"]["thickness"] = 1.0  # type: ignore[index]
        solid_result = run_kernel_request(solid_request)
        self.assertEqual(solid_result["status"], "failed")
        self.assertEqual(solid_result["error"]["path"], "$.physics.thickness")  # type: ignore[index]

    def test_native_sparse_contract_reports_convergence_evidence(self) -> None:
        request = request_record()
        request["procedure"] = {
            "kind": "steady_diffusion",
            "linear_algebra": "native",
        }
        result = run_kernel_request(request)
        provider = result["runtime"]["linear_algebra_provider"]  # type: ignore[index]
        self.assertEqual(provider["name"], "native_sparse")
        self.assertEqual(provider["matrix_format"], "csr")
        self.assertTrue(provider["convergence"]["converged"])
        self.assertIn(
            provider["convergence"]["reason"], {"converged", "initial_residual"}
        )

    def test_contract_version_failure_is_addressable(self) -> None:
        request = request_record()
        request["contract_version"] = "99.0"
        result = run_kernel_request(request)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "unsupported_contract_version")  # type: ignore[index]
        self.assertEqual(result["error"]["path"], "$.contract_version")  # type: ignore[index]

    def test_unknown_named_set_is_a_structured_model_failure(self) -> None:
        request = request_record()
        request["dirichlet"] = [{"node_set": "missing", "value": 0.0}]
        planned = plan_kernel_request(request)
        self.assertEqual(planned["status"], "failed")
        self.assertEqual(planned["error"]["path"], "$.dirichlet[0].node_set")  # type: ignore[index]
        result = run_kernel_request(request)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "invalid_model")  # type: ignore[index]
        self.assertIn("Unknown node set", result["error"]["message"])  # type: ignore[index]

    def test_unknown_provider_is_addressable_before_assembly(self) -> None:
        request = request_record()
        request["procedure"] = {
            "kind": "steady_diffusion",
            "linear_algebra": "unknown",
        }
        result = run_kernel_request(request)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "unsupported_provider")  # type: ignore[index]
        self.assertEqual(result["error"]["path"], "$.procedure.linear_algebra")  # type: ignore[index]

    def test_unknown_assembly_is_addressable_before_assembly(self) -> None:
        request = request_record()
        request["procedure"] = {
            "kind": "steady_diffusion",
            "linear_algebra": "numpy",
            "assembly": "unknown",
        }
        result = run_kernel_request(request)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "unsupported_assembly")  # type: ignore[index]
        self.assertEqual(result["error"]["path"], "$.procedure.assembly")  # type: ignore[index]

    @unittest.skipUnless(native_kernel_available(), "compiled kernel unavailable")
    def test_explicit_native_assembly_is_reported(self) -> None:
        request = request_record()
        request["procedure"] = {
            "kind": "steady_diffusion",
            "linear_algebra": "numpy",
            "assembly": "native",
        }
        result = run_kernel_request(request)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["runtime"]["assembly"]["name"], "native")  # type: ignore[index]
        self.assertEqual(result["runtime"]["native_kernel"]["abi_version"], "1.5")  # type: ignore[index]

    def test_portable_vtk_artifact_has_relative_path_and_digest(self) -> None:
        request = request_record(
            outputs={"vtk": {"path": "results/temperature.vtk", "field_name": "T"}}
        )
        with TemporaryDirectory() as directory:
            result = run_kernel_request(request, artifact_directory=directory)
            artifact = result["artifacts"][0]  # type: ignore[index]
            self.assertEqual(artifact["path"], "results/temperature.vtk")
            self.assertEqual(len(artifact["sha256"]), 64)
            self.assertTrue((Path(directory) / "results" / "temperature.vtk").is_file())

    def test_artifact_path_cannot_escape_root_or_use_windows_separator(self) -> None:
        for path in ("../escape.vtk", "/absolute.vtk", r"results\escape.vtk"):
            with self.subTest(path=path), TemporaryDirectory() as directory:
                result = run_kernel_request(
                    request_record(outputs={"vtk": {"path": path}}),
                    artifact_directory=directory,
                )
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["error"]["code"], "invalid_artifact_path")  # type: ignore[index]

    def test_bundled_schema_declares_json_schema_2020_12(self) -> None:
        request_schema = kernel_request_schema()
        result_schema = kernel_result_schema()
        for schema in (request_schema, result_schema):
            self.assertEqual(
                schema["$schema"],
                "https://json-schema.org/draft/2020-12/schema",
            )
            self.assertEqual(
                schema["properties"]["contract_version"]["const"],  # type: ignore[index]
                CONTRACT_VERSION,
            )
        self.assertEqual(
            kernel_request_schema("0.1.0")["properties"]["contract_version"]["const"],  # type: ignore[index]
            "0.1.0",
        )
        self.assertEqual(
            kernel_result_schema("0.1.0")["properties"]["contract_version"]["const"],  # type: ignore[index]
            "0.1.0",
        )
        self.assertEqual(
            kernel_request_schema("0.2.0")["properties"]["contract_version"]["const"],  # type: ignore[index]
            "0.2.0",
        )
        self.assertEqual(
            kernel_result_schema("0.2.0")["properties"]["contract_version"]["const"],  # type: ignore[index]
            "0.2.0",
        )

    def test_mechanics_vtk_artifact_is_portable_and_digest_bound(self) -> None:
        request = mechanics_request(3)
        request["outputs"] = {"vtk": {"path": "results/solid.vtk"}}
        with TemporaryDirectory() as directory:
            result = run_kernel_request(request, artifact_directory=directory)
            self.assertEqual(result["status"], "success")
            artifact = result["artifacts"][0]  # type: ignore[index]
            self.assertEqual(artifact["path"], "results/solid.vtk")
            self.assertEqual(len(artifact["sha256"]), 64)
            self.assertTrue((Path(directory) / "results" / "solid.vtk").is_file())

    def test_agentfem_ir_portable_extension_lowers_without_agentfem_import(
        self,
    ) -> None:
        request = request_record()
        extension = {
            key: value
            for key, value in request.items()
            if key not in {"contract", "contract_version", "study"}
        }
        document = {
            "schema": "agentfem.af-ir",
            "schema_version": "0.1.0",
            "document_type": "model",
            "root": {
                "name": "thermal-model",
                "study": request["study"],
                "native_kernel": extension,
            },
        }
        lowered = lower_agentfem_ir(document)
        self.assertEqual(lowered["contract"], CONTRACT_NAME)
        self.assertEqual(run_kernel_request(lowered)["status"], "success")

        class PublicModel:
            def to_ir(self):
                return document

        self.assertEqual(lower_agentfem_model(PublicModel()), lowered)

    def test_current_non_reconstructable_agentfem_ir_fails_clearly(self) -> None:
        document = {
            "schema": "agentfem.af-ir",
            "schema_version": "0.1.0",
            "document_type": "model",
            "root": {"name": "model", "study": {}, "mesh": {"reconstructable": False}},
        }
        with self.assertRaises(KernelRequestError) as context:
            lower_agentfem_ir(document)
        self.assertEqual(context.exception.path, "$.root.native_kernel")


if __name__ == "__main__":
    unittest.main()
