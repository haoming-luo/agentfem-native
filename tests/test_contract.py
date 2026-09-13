# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agentfem_native import (
    CONTRACT_NAME,
    CONTRACT_VERSION,
    KernelRequestError,
    kernel_request_schema,
    kernel_result_schema,
    lower_agentfem_ir,
    lower_agentfem_model,
    run_kernel_request,
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


class ContractTests(unittest.TestCase):
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
        self.assertEqual(result["runtime"]["native_kernel"]["abi_version"], "1.4")  # type: ignore[index]

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
                "0.1.0",
            )

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
