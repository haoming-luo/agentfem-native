# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""候选契约的二维/三维闭环、资源前置拒绝和兼容边界。"""

import copy
import json
import unittest
from hashlib import sha256
from unittest.mock import patch

import numpy as np

from agentfem_native.mesh import unit_square_triangles
from agentfem_native.multiphysics import (
    MULTIPHYSICS_VERSION,
    multiphysics_capabilities,
    multiphysics_request_schema,
    plan_multiphysics_request,
    run_multiphysics_request,
)
from agentfem_native.providers import LinearSolveError
from agentfem_native.runtime import CancellationToken, ExecutionContext, ResourceBudget
from agentfem_native.volume_mesh import unit_cube_tetrahedra


def request_for(kind, dimension=2):
    mesh = unit_square_triangles(2) if dimension == 2 else unit_cube_tetrahedra(1)
    request = {
        "contract_version": MULTIPHYSICS_VERSION,
        "kind": kind,
        "mesh": {"points": mesh.points.tolist(), "cells": mesh.cells.tolist()},
        "request_id": "multiphysics-test",
    }
    if kind in ("thermal_steady", "thermal_transient", "thermoelastic"):
        nodes = np.flatnonzero(mesh.points[:, 0] == 0)
        request["thermal"] = {
            "conductivity": 1,
            "source": 0,
            "constraints": {
                "dofs": nodes.tolist(),
                "values": np.full(nodes.size, 30).tolist(),
            },
        }
    if kind == "thermal_transient":
        request["thermal"]["initial"] = np.full(mesh.node_count, 30).tolist()
        request["procedure"] = {"time_step": 0.1, "steps": 3}
    if kind in ("thermoelastic", "finite_strain", "at2_fracture"):
        request["material"] = {"young_modulus": 100, "poisson_ratio": 0.25}
        request["mechanical"] = {
            "constraints": {
                "dofs": list(range(dimension * mesh.node_count)),
                "values": (0.01 * mesh.points).ravel().tolist(),
            }
        }
    if kind == "thermoelastic":
        request["material"].update(expansion=0.001, reference_temperature=20)
    if kind == "finite_strain":
        request["experimental"] = True
        request["procedure"] = {"steps": 2, "maximum_accepted_steps": 8}
    if kind == "at2_fracture":
        request["experimental"] = True
        request["material"].update(toughness=1, length_scale=0.2)
        request["procedure"] = {"load_factors": [0.5, 1, 0.25]}
    return request


class MultiphysicsTests(unittest.TestCase):
    def test_failed_requests_keep_correlation_and_solver_error_category(self):
        request = request_for("thermal_steady")
        request["request_id"] = "中文 Windows 作业"
        planned = plan_multiphysics_request(request)
        self.assertEqual(planned["request_id"], request["request_id"])
        with patch(
            "agentfem_native.multiphysics.ThermalOperator.steady",
            side_effect=LinearSolveError("线性迭代未收敛"),
        ):
            failed = run_multiphysics_request(request)
        self.assertEqual(failed["request_id"], request["request_id"])
        self.assertEqual(failed["error"]["code"], "solver.linear_nonconvergence")
        self.assertTrue(failed["error"]["retryable"])
        request["budget"] = {"maximum_dofs": 0}
        self.assertEqual(
            plan_multiphysics_request(request)["request_id"], request["request_id"]
        )

    def test_packaged_schema_and_candidate_identity(self):
        schema = multiphysics_request_schema()
        self.assertEqual(
            schema["properties"]["contract_version"]["const"], MULTIPHYSICS_VERSION
        )
        capabilities = multiphysics_capabilities()
        self.assertEqual(
            {item["name"] for item in capabilities["kinds"]},
            set(schema["properties"]["kind"]["enum"]),
        )
        self.assertEqual(capabilities["required_optional_providers"], [])
        self.assertNotIn(
            "verified", [item["maturity"] for item in capabilities["kinds"]]
        )

    def test_five_routes_in_two_and_three_dimensions(self):
        for dimension in (2, 3):
            for kind in (
                "thermal_steady",
                "thermal_transient",
                "thermoelastic",
                "finite_strain",
                "at2_fracture",
            ):
                with self.subTest(kind=kind, dimension=dimension):
                    request = request_for(kind, dimension)
                    result = run_multiphysics_request(request)
                    self.assertEqual(result["status"], "success", result)
                    self.assertNotEqual(result["evidence"]["maturity"], "verified")
                    self.assertEqual(result["plan"]["dimension"], dimension)
                    json.dumps(result, allow_nan=False)
                    repeated = run_multiphysics_request(request)
                    self.assertEqual(repeated["status"], "success", repeated)
                    self.assertEqual(
                        result["plan"]["request_digest"],
                        repeated["plan"]["request_digest"],
                    )
                    self.assertEqual(
                        result["plan"]["plan_digest"], repeated["plan"]["plan_digest"]
                    )
                    for record in (result, repeated):
                        # 摘要证明这份载荷未被改写，不承诺两次浮点求解逐位相同。
                        payload = json.dumps(
                            record["result"], allow_nan=False, sort_keys=True
                        ).encode()
                        self.assertEqual(
                            record["evidence"]["result_digest"],
                            sha256(payload).hexdigest(),
                        )
                        # 复用此前数学测试的解析判据，不按 OS 或观测尾差放大容差。
                        if kind.startswith("thermal_"):
                            np.testing.assert_allclose(
                                record["result"]["temperatures"], 30, rtol=0, atol=1e-10
                            )
                            np.testing.assert_allclose(
                                record["result"]["cell_flux"], 0, rtol=0, atol=1e-10
                            )
                        if kind == "at2_fracture":
                            # E=100, nu=.25 -> mu=lambda=40；u=.01*x，plane strain / 3D。
                            psi = (
                                40 * dimension * 0.01**2 + 20 * (dimension * 0.01) ** 2
                            )
                            exact = 2 * psi / (5 + 2 * psi)
                            for state in record["result"]["states"][1:]:
                                np.testing.assert_allclose(
                                    state["damage"], exact, rtol=0, atol=1e-10
                                )

    def test_planning_and_budget_precede_operator_construction(self):
        request = request_for("finite_strain")
        with patch(
            "agentfem_native.multiphysics.FiniteStrainOperator",
            side_effect=AssertionError("不能装配"),
        ):
            self.assertEqual(plan_multiphysics_request(request)["status"], "planned")
            request["budget"] = {"maximum_dofs": 1}
            result = run_multiphysics_request(request)
        self.assertEqual(result["error"]["code"], "budget.dofs_exceeded")

    def test_cancellation_stricter_budget_and_digest(self):
        request = request_for("thermal_transient")
        token = CancellationToken()
        token.cancel()
        result = run_multiphysics_request(
            request, context=ExecutionContext(cancellation=token)
        )
        self.assertEqual(result["error"]["code"], "execution.cancelled")
        request["budget"] = {"maximum_steps": 100}
        result = run_multiphysics_request(
            request, context=ExecutionContext(budget=ResourceBudget(maximum_steps=1))
        )
        self.assertEqual(result["error"]["code"], "budget.steps_exceeded")
        first = plan_multiphysics_request(request)["plan"]["request_digest"]
        request["thermal"]["source"] = 2
        self.assertNotEqual(
            first, plan_multiphysics_request(request)["plan"]["request_digest"]
        )

    def test_unsupported_physics_never_silently_ignored(self):
        request = request_for("at2_fracture")
        for group, key, value in (
            ("material", "tension_split", True),
            ("procedure", "time_step", 0.1),
            ("thermal", "temperature", [1]),
        ):
            modified = copy.deepcopy(request)
            modified.setdefault(group, {})[key] = value
            self.assertEqual(run_multiphysics_request(modified)["status"], "failed")
        request["experimental"] = False
        self.assertEqual(run_multiphysics_request(request)["status"], "failed")

    def test_nonlinear_failure_contains_only_accepted_states(self):
        request = request_for("at2_fracture")
        request["procedure"]["maximum_alternations"] = 1
        result = run_multiphysics_request(request)
        self.assertEqual(result["error"]["code"], "solver.nonlinear_nonconvergence")
        self.assertEqual(result["error"]["accepted_states"], [])

    def test_preflight_rejects_physics_before_constructing_operators(self):
        for kind, group, key, value in (
            ("thermal_steady", "thermal", "conductivity", [[1, 1e-7], [0, 1]]),
            ("thermal_transient", "procedure", "time_step", 0),
            ("thermal_transient", "thermal", "initial", [30]),
            ("thermoelastic", "material", "young_modulus", -1),
            ("finite_strain", "material", "thermal_stretch", -1),
            ("at2_fracture", "material", "length_scale", 0),
            ("at2_fracture", "mechanical", "initial_damage", [2] * 9),
        ):
            with self.subTest(kind=kind, key=key):
                request = request_for(kind)
                request.setdefault(group, {})[key] = value
                self.assertEqual(plan_multiphysics_request(request)["status"], "failed")
                self.assertEqual(run_multiphysics_request(request)["status"], "failed")

    def test_cancelled_nonlinear_request_keeps_only_accepted_steps(self):
        for kind, stage in (
            ("finite_strain", "finite_strain.trial"),
            ("at2_fracture", "fracture.step"),
        ):
            token = CancellationToken()

            def cancel_after_one(event, stage=stage, token=token):
                if event.stage == stage and event.completed == 1:
                    token.cancel()

            result = run_multiphysics_request(
                request_for(kind),
                context=ExecutionContext(cancellation=token, progress=cancel_after_one),
            )
            self.assertEqual(result["error"]["code"], "execution.cancelled")
            self.assertEqual(len(result["error"]["accepted_states"]), 1)
            self.assertNotIn(
                "restart_from_last_checkpoint", result["error"]["remediations"]
            )

    def test_missing_material_is_not_a_successful_plan(self):
        request = request_for("finite_strain")
        del request["material"]["young_modulus"]
        self.assertEqual(plan_multiphysics_request(request)["status"], "failed")
