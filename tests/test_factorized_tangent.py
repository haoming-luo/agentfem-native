# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""显式切线缩并实验：等价、热伸长链式导数、全轨迹与准入失败。"""

import unittest

import numpy as np

from agentfem_native import (
    EssentialBC,
    FiniteStrainOperator,
    NeoHookeanMaterial,
    unit_cube_tetrahedra,
    unit_square_triangles,
)


class FactorizedTangentTests(unittest.TestCase):
    def test_full_assembly_and_thermal_chain_directional_derivative(self):
        for mesh in (unit_square_triangles(3), unit_cube_tetrahedra(2)):
            for stretch in (1.0, 1.1):
                arguments = {"thermal_stretch": stretch}
                material = NeoHookeanMaterial(100, 0.25)
                reference = FiniteStrainOperator(mesh, material, **arguments)
                candidate = FiniteStrainOperator(
                    mesh, material, tangent_form="factorized", **arguments
                )
                random = np.random.default_rng(17)
                u = 0.1 * mesh.points + 0.003 * random.normal(size=mesh.points.shape)
                first, second = reference.assemble(u), candidate.assemble(u)
                np.testing.assert_array_equal(first[0].indices, second[0].indices)
                np.testing.assert_allclose(
                    first[0].data, second[0].data, rtol=1e-12, atol=1e-12
                )
                for a, b in zip(first[1:], second[1:], strict=True):
                    np.testing.assert_array_equal(a, b)
                direction = random.normal(size=u.shape)
                h = 1e-7
                delta = (
                    candidate.assemble(u + h * direction, tangent=False)[1]
                    - candidate.assemble(u - h * direction, tangent=False)[1]
                ) / (2 * h)
                np.testing.assert_allclose(
                    second[0].matvec(direction.ravel()), delta, rtol=2e-8, atol=2e-6
                )

    def test_same_physical_stations_and_full_result_history(self):
        for mesh in (unit_square_triangles(3), unit_cube_tetrahedra(2)):
            d = mesh.points.shape[1]
            material = NeoHookeanMaterial(100, 0.25)
            reference = FiniteStrainOperator(mesh, material)
            candidate = FiniteStrainOperator(mesh, material, tangent_form="factorized")
            dofs = (d * mesh.nodes("left")[:, None] + np.arange(d)).ravel()
            bc = EssentialBC(dofs, np.zeros(dofs.size))
            load = reference.geometry.mechanical_load(
                tractions={"right": [10] + [0] * (d - 1)}
            )
            first, second = (
                reference.solve(bc, load=load, steps=4),
                candidate.solve(bc, load=load, steps=4),
            )
            self.assertEqual(first.rejected_steps, second.rejected_steps)
            for a, b in zip(first.states, second.states, strict=True):
                self.assertEqual(a.load_factor, b.load_factor)
                np.testing.assert_allclose(
                    a.displacements, b.displacements, rtol=1e-10, atol=2e-9
                )
                np.testing.assert_allclose(
                    a.cell_cauchy, b.cell_cauchy, rtol=1e-10, atol=2e-7
                )
                np.testing.assert_allclose(
                    a.cell_jacobian, b.cell_jacobian, rtol=1e-10, atol=2e-9
                )
                self.assertLessEqual(b.free_residual_norm, b.residual_threshold)
                self.assertLess(abs(a.strain_energy - b.strain_energy), 1e-8)

    def test_invalid_selection_and_default_is_unchanged(self):
        mesh = unit_square_triangles(1)
        material = NeoHookeanMaterial(100, 0.25)
        self.assertEqual(FiniteStrainOperator(mesh, material).tangent_form, "tensor")
        with self.assertRaises(ValueError):
            FiniteStrainOperator(mesh, material, tangent_form="unknown")
