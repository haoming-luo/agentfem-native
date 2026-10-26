# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""逐单元弹性参数的解析两层杆、能量导数、快照与候选预检。"""

import copy
import unittest
from unittest.mock import patch

import numpy as np

from agentfem_native.materials import NeoHookeanMaterial
from agentfem_native.mesh import unit_square_triangles
from agentfem_native.multiphysics import (
    plan_multiphysics_request,
    run_multiphysics_request,
)
from agentfem_native.simplex import EssentialBC
from agentfem_native.thermoelastic import ThermoelasticOperator, cell_elastic_parameters
from agentfem_native.volume_mesh import unit_cube_tetrahedra
from tests.test_multiphysics import request_for


class CellThermoelasticTests(unittest.TestCase):
    def assert_repeatable(self, first, second):
        scale = max(np.max(np.abs(first), initial=0), np.max(np.abs(second), initial=0))
        np.testing.assert_allclose(
            first, second, rtol=0, atol=64 * np.finfo(float).eps * scale
        )

    def test_two_layer_series_solution_stress_and_energy(self):
        for mesh in (unit_square_triangles(4), unit_cube_tetrahedra(4)):
            dimension = mesh.points.shape[1]
            centers = mesh.points[mesh.cells].mean(axis=1)
            young = np.where(centers[:, 0] < 0.5, 100, 1000)
            operator = ThermoelasticOperator(mesh, young, 0)
            fixed = np.concatenate(
                [
                    dimension * np.flatnonzero(mesh.points[:, j] == 0) + j
                    for j in range(dimension)
                ]
            )
            load = operator.geometry.mechanical_load(
                tractions={"right": [1] + [0] * (dimension - 1)}
            )
            result = operator.solve(EssentialBC(fixed, np.zeros(fixed.size)), load=load)
            x = mesh.points[:, 0]
            expected = np.zeros_like(mesh.points)
            expected[:, 0] = np.minimum(x, 0.5) / 100 + np.maximum(x - 0.5, 0) / 1000
            np.testing.assert_allclose(
                result.displacements, expected, rtol=1e-9, atol=1e-11
            )
            stress = np.zeros_like(result.cell_stress)
            stress[:, 0, 0] = 1
            np.testing.assert_allclose(result.cell_stress, stress, rtol=1e-9, atol=1e-9)
            self.assertAlmostEqual(
                result.strain_energy, 0.5 * (0.5 / 100 + 0.5 / 1000), places=11
            )

    def test_scalar_and_uniform_fields_have_same_owned_assembly(self):
        for mesh in (unit_square_triangles(2), unit_cube_tetrahedra(1)):
            first = ThermoelasticOperator(
                mesh, 100, 0.25, temperature=mesh.points.sum(axis=1), expansion=0.001
            )
            young, poisson = (
                np.full(mesh.cell_count, 100.0),
                np.full(mesh.cell_count, 0.25),
            )
            second = ThermoelasticOperator(
                mesh,
                young,
                poisson,
                temperature=mesh.points.sum(axis=1),
                expansion=0.001,
            )
            np.testing.assert_array_equal(
                first.stiffness.indptr, second.stiffness.indptr
            )
            np.testing.assert_array_equal(
                first.stiffness.indices, second.stiffness.indices
            )
            self.assert_repeatable(first.stiffness.data, second.stiffness.data)
            self.assert_repeatable(first.thermal_load, second.thermal_load)
            before = second.fields(mesh.points * 0.01)
            young[:] = 1
            poisson[:] = -0.5
            for original, current in zip(
                before, second.fields(mesh.points * 0.01), strict=True
            ):
                self.assert_repeatable(original, current)
            with self.assertRaises(ValueError):
                second.shear[0] = 1

    def test_cellwise_thermal_energy_gradient_in_plane_and_solid_modes(self):
        for dimension, mode in (
            (2, "plane_stress"),
            (2, "plane_strain"),
            (3, "plane_strain"),
        ):
            mesh = (
                unit_square_triangles(2) if dimension == 2 else unit_cube_tetrahedra(1)
            )
            operator = ThermoelasticOperator(
                mesh,
                np.linspace(100, 200, mesh.cell_count),
                np.linspace(0.1, 0.3, mesh.cell_count),
                mode=mode,
                temperature=20 + mesh.points.sum(axis=1),
                reference_temperature=20,
                expansion=np.linspace(0.001, 0.003, mesh.cell_count),
            )
            rng = np.random.default_rng(48)
            u = rng.normal(size=mesh.points.shape) * 0.01
            direction = rng.normal(size=u.shape)
            h = 1e-6
            plus = operator.geometry.volumes @ operator.fields(u + h * direction)[2]
            minus = operator.geometry.volumes @ operator.fields(u - h * direction)[2]
            force = operator.stiffness.matvec(u.ravel()) - operator.thermal_load
            np.testing.assert_allclose(
                (plus - minus) / (2 * h),
                force @ direction.ravel(),
                rtol=1e-7,
                atol=1e-9,
            )

    def test_candidate_cellwise_response_and_invalid_preflight(self):
        for dimension in (2, 3):
            request = request_for("thermoelastic", dimension)
            count = len(request["mesh"]["cells"])
            request["thermal"] = {"temperature": [0] * len(request["mesh"]["points"])}
            request["material"].update(
                young_modulus=np.linspace(100, 200, count).tolist(),
                poisson_ratio=[0] * count,
                expansion=0,
            )
            answer = run_multiphysics_request(request)
            self.assertEqual(answer["status"], "success", answer)
            stress = np.asarray(answer["result"]["cell_stress"])
            np.testing.assert_allclose(
                stress[:, 0, 0], 0.01 * np.linspace(100, 200, count), atol=1e-12
            )
            for bad in (
                {"young_modulus": [100]},
                {"poisson_ratio": [0.5] * count},
                {"young_modulus": [-1] * count},
                {"young_modulus": True},
            ):
                invalid = copy.deepcopy(request)
                invalid["material"].update(bad)
                with patch(
                    "agentfem_native.multiphysics.ThermoelasticOperator",
                    side_effect=AssertionError("非法参数不应装配"),
                ):
                    self.assertEqual(
                        plan_multiphysics_request(invalid)["status"], "failed"
                    )
        unsupported = request_for("finite_strain")
        unsupported["material"]["young_modulus"] = [100] * len(
            unsupported["mesh"]["cells"]
        )
        self.assertEqual(plan_multiphysics_request(unsupported)["status"], "failed")

    def test_finite_inputs_with_overflowing_derived_coefficients_are_rejected(self):
        for young, poisson in ((1e308, 0.49), (float(np.nextafter(0.0, 1.0)), 0.25)):
            with self.assertRaisesRegex(ValueError, "派生"):
                cell_elastic_parameters(young, poisson, 2)
            with self.assertRaisesRegex(ValueError, "派生"):
                NeoHookeanMaterial(young, poisson)


if __name__ == "__main__":
    unittest.main()
