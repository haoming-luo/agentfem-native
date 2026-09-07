# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import unittest

import numpy as np

from agentfem_native.materials import (
    J2MaterialPoint,
    J2State,
    NeoHookeanMaterial,
    SmallStrainJ2Material,
)


class NeoHookeanTests(unittest.TestCase):
    def test_proper_rotation_has_zero_energy_and_stress(self) -> None:
        angle = 0.37
        rotation = np.array(
            (
                (np.cos(angle), -np.sin(angle), 0.0),
                (np.sin(angle), np.cos(angle), 0.0),
                (0.0, 0.0, 1.0),
            )
        )
        response = NeoHookeanMaterial(120.0, 0.25).response(rotation)
        self.assertAlmostEqual(response.energy_density, 0.0, places=13)
        np.testing.assert_allclose(response.first_piola, 0.0, atol=3.0e-14)
        np.testing.assert_allclose(response.cauchy_stress, 0.0, atol=3.0e-14)

    def test_first_piola_is_energy_derivative(self) -> None:
        material = NeoHookeanMaterial(80.0, 0.2)
        gradient = np.array(((1.1, 0.04, 0.0), (0.01, 0.95, 0.03), (0.0, 0.02, 1.05)))
        direction = np.array(
            ((0.2, -0.1, 0.05), (0.03, 0.12, -0.04), (0.02, 0.01, -0.08))
        )
        step = 1.0e-6
        difference = (
            material.response(gradient + step * direction).energy_density
            - material.response(gradient - step * direction).energy_density
        ) / (2.0 * step)
        expected = float(np.sum(material.response(gradient).first_piola * direction))
        self.assertAlmostEqual(difference, expected, places=8)

    def test_non_positive_jacobian_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            NeoHookeanMaterial(1.0, 0.2).response(np.diag((1.0, 1.0, -1.0)))

    def test_invalid_elastic_parameters_fail_at_construction(self) -> None:
        with self.assertRaisesRegex(ValueError, "Young"):
            NeoHookeanMaterial(0.0, 0.2)


class J2MaterialTests(unittest.TestCase):
    def test_invalid_plastic_parameters_fail_at_construction(self) -> None:
        with self.assertRaisesRegex(ValueError, "yield"):
            SmallStrainJ2Material(100.0, 0.2, 0.0)
        with self.assertRaisesRegex(ValueError, "Hardening"):
            SmallStrainJ2Material(100.0, 0.2, 1.0, -1.0)

    def test_elastic_and_plastic_paths_land_on_yield_surface(self) -> None:
        material = SmallStrainJ2Material(200.0, 0.25, 2.0, 10.0)
        elastic = material.update(np.diag((0.001, 0.0, 0.0)), J2State.zero())
        self.assertFalse(elastic.yielded)
        plastic = material.update(np.diag((0.1, -0.05, -0.05)), J2State.zero())
        self.assertTrue(plastic.yielded)
        deviator = plastic.stress - np.trace(plastic.stress) / 3.0 * np.eye(3)
        equivalent = np.sqrt(1.5 * np.sum(deviator * deviator))
        expected = (
            material.initial_yield_stress
            + material.hardening_modulus * plastic.state.equivalent_plastic_strain
        )
        self.assertAlmostEqual(equivalent, expected, places=12)

    def test_transaction_commit_and_rollback_are_explicit(self) -> None:
        point = J2MaterialPoint(SmallStrainJ2Material(100.0, 0.2, 1.0, 2.0))
        initial = point.committed_state
        trial = point.begin(np.diag((0.1, -0.05, -0.05)))
        self.assertTrue(trial.yielded)
        rolled_back = point.rollback()
        np.testing.assert_array_equal(
            rolled_back.plastic_strain, initial.plastic_strain
        )
        point.begin(np.diag((0.1, -0.05, -0.05)))
        committed = point.commit()
        self.assertGreater(committed.equivalent_plastic_strain, 0.0)
        with self.assertRaisesRegex(RuntimeError, "no trial"):
            point.commit()


if __name__ == "__main__":
    unittest.main()
