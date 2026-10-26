# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""参考体力/面力由 Native 积分：独立合力力矩、解析响应及候选接入。"""

import copy
import unittest
from unittest.mock import patch

import numpy as np

from agentfem_native.mesh import unit_square_triangles
from agentfem_native.multiphysics import (
    multiphysics_capabilities,
    multiphysics_request_schema,
    plan_multiphysics_request,
    run_multiphysics_request,
)
from agentfem_native.simplex import SimplexGeometry
from agentfem_native.volume_mesh import unit_cube_tetrahedra
from tests.test_multiphysics import request_for
from tests.test_t4_dynamics_contract import dynamics_request


def _mesh(dimension):
    return unit_square_triangles(2) if dimension == 2 else unit_cube_tetrahedra(1)


def _add_boundaries(request, mesh):
    request["mesh"]["boundary_sets"] = {
        name: value.tolist() for name, value in mesh.boundary_sets.items()
    }


class CandidateLoadTests(unittest.TestCase):
    def assert_fields_equal(self, first, second):
        if isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_fields_equal(first[key], second[key])
        elif isinstance(first, list) and first and isinstance(first[0], dict):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second, strict=True):
                self.assert_fields_equal(a, b)
        elif isinstance(first, (str, bool, int)) or first is None:
            self.assertEqual(first, second)
        else:
            np.testing.assert_allclose(first, second, rtol=1e-10, atol=1e-12)

    def test_independent_resultant_moment_and_thickness(self):
        for dimension, thickness in ((2, 0.3), (3, 1.0)):
            mesh = _mesh(dimension)
            geometry = SimplexGeometry(mesh, thickness)
            body, traction = np.arange(1, dimension + 1), np.arange(2, dimension + 2)
            load = geometry.mechanical_load(
                body_force=body, tractions={"right": traction}
            )
            # 单位域的体积/右面面积同为 thickness，合力和力矩用连续均匀载荷直接积分。
            expected_force = thickness * (body + traction)
            np.testing.assert_allclose(
                load.sum(axis=0), expected_force, rtol=1e-14, atol=1e-14
            )
            points3 = np.pad(mesh.points, ((0, 0), (0, 3 - dimension)))
            load3 = np.pad(load, ((0, 0), (0, 3 - dimension)))
            center = np.r_[np.full(dimension, 0.5), np.zeros(3 - dimension)]
            face_center = center.copy()
            face_center[0] = 1
            expected_moment = thickness * (
                np.cross(center, np.r_[body, np.zeros(3 - dimension)])
                + np.cross(face_center, np.r_[traction, np.zeros(3 - dimension)])
            )
            np.testing.assert_allclose(
                np.cross(points3, load3).sum(axis=0),
                expected_moment,
                rtol=1e-14,
                atol=1e-14,
            )

    def test_analytic_longitudinal_traction_and_reactions(self):
        for dimension in (2, 3):
            request = request_for("thermoelastic", dimension)
            mesh = _mesh(dimension)
            _add_boundaries(request, mesh)
            request["thermal"] = {"temperature": np.zeros(mesh.node_count).tolist()}
            request["material"]["expansion"] = 0
            fixed = np.concatenate(
                [dimension * mesh.nodes("left")]
                + [
                    dimension * np.arange(mesh.node_count) + j
                    for j in range(1, dimension)
                ]
            )
            request["mechanical"] = {
                "constraints": {"dofs": fixed.tolist(), "values": [0] * fixed.size},
                "tractions": {"right": [1] + [0] * (dimension - 1)},
            }
            answer = run_multiphysics_request(request)
            self.assertEqual(answer["status"], "success", answer)
            result = answer["result"]
            # E=100, nu=.25，横向约束时 lambda+2mu=120。
            expected = np.zeros_like(mesh.points)
            expected[:, 0] = mesh.points[:, 0] / 120
            np.testing.assert_allclose(
                result["displacements"], expected, rtol=1e-10, atol=1e-12
            )
            residual = np.asarray(result["residual"]).reshape(-1, dimension)
            np.testing.assert_allclose(
                residual.sum(axis=0), [-1] + [0] * (dimension - 1), atol=1e-11
            )

    def test_all_mechanical_routes_match_explicit_equivalent_nodal_load(self):
        requests = [
            (request_for(kind, dimension), _mesh(dimension))
            for kind in ("thermoelastic", "finite_strain", "at2_fracture")
            for dimension in (2, 3)
        ]
        requests.append((dynamics_request(), unit_cube_tetrahedra(1)))
        for request, mesh in requests:
            with self.subTest(kind=request["kind"], dimension=mesh.points.shape[1]):
                _add_boundaries(request, mesh)
                dimension = mesh.points.shape[1]
                body = np.linspace(0.01, 0.02, dimension)
                traction = np.linspace(0.02, 0.03, dimension)
                mechanical = request["mechanical"]
                mechanical["body_force"] = body.tolist()
                mechanical["tractions"] = {"right": traction.tolist()}
                direct = copy.deepcopy(request)
                nodal = SimplexGeometry(mesh).mechanical_load(
                    body_force=body, tractions={"right": traction}
                )
                nodal += np.asarray(mechanical.get("load", np.zeros_like(nodal)))
                direct["mechanical"].pop("body_force")
                direct["mechanical"].pop("tractions")
                direct["mechanical"]["load"] = nodal.tolist()
                first, second = (
                    run_multiphysics_request(request),
                    run_multiphysics_request(direct),
                )
                self.assertEqual(first["status"], "success", first)
                self.assertEqual(second["status"], "success", second)
                self.assert_fields_equal(first["result"], second["result"])

    def test_invalid_loads_and_budget_reject_before_integration(self):
        request = request_for("thermoelastic")
        _add_boundaries(request, _mesh(2))
        for extra in (
            {"body_force": [1]},
            {"tractions": {"unknown": [1, 0]}},
            {"tractions": []},
            {"tractions": {"right": [1, 2, 3]}},
            {"pressure": 1},
        ):
            invalid = copy.deepcopy(request)
            invalid["mechanical"].update(extra)
            with patch(
                "agentfem_native.multiphysics.ThermoelasticOperator",
                side_effect=AssertionError("非法载荷不应装配"),
            ):
                self.assertEqual(plan_multiphysics_request(invalid)["status"], "failed")
        request["mechanical"]["body_force"] = [1, 0]
        request["budget"] = {"maximum_dofs": 0}
        with patch.object(
            SimplexGeometry,
            "mechanical_load",
            side_effect=AssertionError("预算前不应积分"),
        ):
            self.assertEqual(
                run_multiphysics_request(request)["error"]["code"],
                "budget.dofs_exceeded",
            )

    def test_capability_and_packaged_schema_explain_load_units(self):
        self.assertIn(
            "reference_traction", multiphysics_capabilities()["mechanical_loads"]
        )
        fields = multiphysics_request_schema()["properties"]["mechanical"]["properties"]
        self.assertIn("参考体积", fields["body_force"]["description"])
        self.assertIn("非随动", fields["tractions"]["description"])


if __name__ == "__main__":
    unittest.main()
