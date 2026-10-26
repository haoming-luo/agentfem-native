# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""三维动力契约检查输入边界，并对照既有类型化路径的完整历史。"""

import copy
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from agentfem_native.dynamics import integrate_constrained_linear_dynamics
from agentfem_native.multiphysics import (
    MULTIPHYSICS_VERSION,
    plan_multiphysics_request,
    run_multiphysics_request,
)
from agentfem_native.runtime import CancellationToken, ExecutionContext
from agentfem_native.solid import (
    LinearElastic3DProblem,
    SolidDisplacementCondition,
    SolidElasticMaterial,
)
from agentfem_native.solid_dynamics import build_t4_linear_dynamics
from agentfem_native.sparse import CGReport, CGResult
from agentfem_native.volume_mesh import unit_cube_tetrahedra


def dynamics_request():
    mesh = unit_cube_tetrahedra(1)
    fixed_nodes = np.flatnonzero(mesh.points[:, 0] == 0)
    fixed = (3 * fixed_nodes[:, None] + np.arange(3)).ravel()
    initial = np.zeros_like(mesh.points)
    initial[:, 0] = 0.01 * mesh.points[:, 0]
    velocity = np.zeros_like(mesh.points)
    velocity[:, 1] = 0.02 * mesh.points[:, 0]
    load = np.zeros_like(mesh.points)
    load[:, 2] = 0.1 * mesh.points[:, 0]
    return {
        "contract_version": MULTIPHYSICS_VERSION,
        "request_id": "三维原生动力",
        "kind": "linear_dynamics_t4",
        "mesh": {"points": mesh.points.tolist(), "cells": mesh.cells.tolist()},
        "material": {"young_modulus": 100, "poisson_ratio": 0.25, "density": 2},
        "mechanical": {
            "constraints": {"dofs": fixed.tolist(), "values": [0] * fixed.size},
            "initial_displacement": initial.tolist(),
            "initial_velocity": velocity.tolist(),
            "load": load.tolist(),
        },
        "procedure": {"time_step": 0.001, "steps": 8},
    }


class T4DynamicsContractTests(unittest.TestCase):
    def test_linear_failure_keeps_numerical_diagnostics_not_input_error(self):
        request = dynamics_request()
        report = CGReport(False, "maximum_iterations", 1, 1.0, 0.1, 1e-12, 1e-12, 1e-14)
        with patch(
            "agentfem_native.dynamics.conjugate_gradient",
            return_value=CGResult(np.zeros(12), report),
        ):
            answer = run_multiphysics_request(request)
        self.assertEqual(answer["error"]["code"], "solver.linear_nonconvergence")
        self.assertEqual(answer["error"]["convergence"]["iterations"], 1)
        self.assertEqual(answer["request_id"], request["request_id"])

    def test_full_histories_match_typed_route_for_both_mass_forms(self):
        request = dynamics_request()
        mesh = unit_cube_tetrahedra(1)
        problem = LinearElastic3DProblem(
            mesh,
            SolidElasticMaterial(100, 0.25),
            dirichlet=(SolidDisplacementCondition("left", None, (0, 0, 0)),),
        )
        for mass_form in ("lumped", "consistent"):
            with self.subTest(mass=mass_form):
                request["procedure"]["mass"] = mass_form
                answer = run_multiphysics_request(request)
                self.assertEqual(answer["status"], "success", answer)
                self.assertEqual(answer["result"]["maturity"], "implemented")
                self.assertEqual(answer["result"]["dof_layout"], "node_major_xyz")
                system, fixed = build_t4_linear_dynamics(
                    problem,
                    2,
                    time_step=0.001,
                    steps=8,
                    initial_displacement=request["mechanical"]["initial_displacement"],
                    initial_velocity=request["mechanical"]["initial_velocity"],
                    lumped_mass=mass_form == "lumped",
                )
                expected = integrate_constrained_linear_dynamics(
                    replace(
                        system, load=np.asarray(request["mechanical"]["load"]).ravel()
                    ),
                    fixed,
                    method="newmark_average_acceleration",
                )
                for name in (
                    "times",
                    "displacement",
                    "velocity",
                    "acceleration",
                    "reaction",
                    "kinetic_energy",
                    "strain_energy",
                    "external_work",
                    "energy_balance_error",
                ):
                    np.testing.assert_allclose(
                        answer["result"][name],
                        getattr(expected, name),
                        rtol=1e-10,
                        atol=1e-12,
                        err_msg=name,
                    )
                self.assertLess(
                    np.max(np.abs(answer["result"]["energy_balance_error"])), 1e-10
                )

    def test_planning_budget_and_cancel_precede_sparse_construction(self):
        request = dynamics_request()
        with (
            patch(
                "agentfem_native.multiphysics.ThermoelasticOperator",
                side_effect=AssertionError("禁止提前装配"),
            ),
            patch(
                "agentfem_native.multiphysics.assemble_t4_mass",
                side_effect=AssertionError("禁止提前质量装配"),
            ),
        ):
            plan = plan_multiphysics_request(request)
            self.assertEqual(plan["status"], "planned", plan)
            self.assertEqual(plan["plan"]["dof_count"], 24)
            self.assertEqual(plan["plan"]["history_capacity"], 9)
            token = CancellationToken()
            token.cancel()
            answer = run_multiphysics_request(
                request, context=ExecutionContext(cancellation=token)
            )
            self.assertEqual(answer["error"]["code"], "execution.cancelled")
            request["budget"] = {"maximum_steps": 7}
            answer = run_multiphysics_request(request)
            self.assertEqual(answer["error"]["code"], "budget.steps_exceeded")

    def test_invalid_physics_rejected_by_same_preflight_and_execution(self):
        original = dynamics_request()
        changes = (
            ("material", "density", 0),
            ("material", "thickness", 2),
            ("procedure", "time_step", -0.1),
            ("procedure", "steps", True),
            ("procedure", "mass", "diagonal_scaled"),
            ("procedure", "warm_start", 1),
            ("procedure", "method", "central_difference"),
            ("mechanical", "initial_velocity", [[1, 0, 0]] * 8),
            ("mechanical", "initial_displacement", [0] * 24),
            (
                "mechanical",
                "constraints",
                {"dofs": list(range(24)), "values": [0] * 24},
            ),
        )
        for section, key, value in changes:
            with self.subTest(section=section, key=key):
                request = copy.deepcopy(original)
                request[section][key] = value
                for entry in (plan_multiphysics_request, run_multiphysics_request):
                    answer = entry(request)
                    self.assertEqual(answer["status"], "failed", answer)
                    self.assertEqual(answer["error"]["code"], "request.invalid")

    def test_zero_default_states_and_three_dimensional_boundary(self):
        request = dynamics_request()
        for name in ("initial_displacement", "initial_velocity", "load"):
            del request["mechanical"][name]
        answer = run_multiphysics_request(request)
        self.assertEqual(answer["status"], "success", answer)
        np.testing.assert_array_equal(
            answer["result"]["displacement"], np.zeros((9, 24))
        )
        from tests.test_multiphysics import request_for

        wrong = request_for("thermoelastic", 2)
        wrong["kind"] = "linear_dynamics_t4"
        wrong.pop("thermal")
        wrong["material"] = request["material"]
        wrong["procedure"] = request["procedure"]
        self.assertIn("三维", plan_multiphysics_request(wrong)["error"]["message"])
