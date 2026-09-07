# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import unittest

import numpy as np

from agentfem_native.elasticity import (
    DisplacementCondition,
    LinearElasticMaterial,
    LinearElasticProblem,
    TractionCondition,
    p1_body_force_load,
    p1_boundary_traction_load,
    p1_elastic_stiffness,
    p1_strain_displacement,
    solve_linear_elasticity,
)
from agentfem_native.mesh import TriangularMesh, unit_square_triangles
from agentfem_native.reference import REFERENCE_TRIANGLE_VERTICES


def _mesh_with_origin(resolution: int) -> TriangularMesh:
    base = unit_square_triangles(resolution)
    return TriangularMesh(
        base.points,
        base.cells,
        node_sets={**base.node_sets, "origin": np.array((0,))},
        boundary_sets=base.boundary_sets,
    )


class T3ElementTests(unittest.TestCase):
    def test_element_is_symmetric_with_three_rigid_modes(self) -> None:
        material = LinearElasticMaterial(210.0, 0.3)
        stiffness = p1_elastic_stiffness(REFERENCE_TRIANGLE_VERTICES, material)
        np.testing.assert_allclose(stiffness, stiffness.T, atol=0.0)
        translations = (
            np.array((1.0, 0.0, 1.0, 0.0, 1.0, 0.0)),
            np.array((0.0, 1.0, 0.0, 1.0, 0.0, 1.0)),
        )
        rotation = REFERENCE_TRIANGLE_VERTICES[:, ::-1].copy()
        rotation[:, 0] *= -1.0
        for mode in (*translations, rotation.ravel()):
            np.testing.assert_allclose(stiffness @ mode, 0.0, atol=3.0e-14)
        eigenvalues = np.linalg.eigvalsh(stiffness)
        self.assertEqual(np.count_nonzero(eigenvalues > 1.0e-10), 3)

    def test_constant_affine_field_recovers_engineering_strain(self) -> None:
        vertices = np.array(((0.2, -0.1), (1.4, 0.2), (-0.3, 1.1)))
        displacement = np.column_stack(
            (
                0.01 + 0.02 * vertices[:, 0] - 0.03 * vertices[:, 1],
                -0.02 + 0.04 * vertices[:, 0] + 0.05 * vertices[:, 1],
            )
        )
        strain = p1_strain_displacement(vertices) @ displacement.ravel()
        np.testing.assert_allclose(strain, (0.02, 0.05, 0.01), atol=2.0e-17)

    def test_orientation_changes_only_local_numbering(self) -> None:
        material = LinearElasticMaterial(100.0, 0.2, "plane_strain")
        vertices = np.array(((0.0, 0.0), (2.0, 0.0), (0.2, 1.0)))
        baseline = p1_elastic_stiffness(vertices, material)
        permutation = np.array((0, 1, 4, 5, 2, 3))
        reversed_matrix = p1_elastic_stiffness(vertices[[0, 2, 1]], material)
        np.testing.assert_allclose(
            reversed_matrix, baseline[np.ix_(permutation, permutation)], atol=2.0e-14
        )

    def test_constant_load_integrals_have_exact_resultants(self) -> None:
        body = p1_body_force_load(
            REFERENCE_TRIANGLE_VERTICES, (6.0, -3.0), thickness=2.0
        )
        np.testing.assert_allclose(body.reshape(3, 2).sum(axis=0), (6.0, -3.0))
        traction = p1_boundary_traction_load(
            np.array(((0.0, 0.0), (2.0, 0.0))), (4.0, -1.0), thickness=0.5
        )
        np.testing.assert_allclose(traction.reshape(2, 2).sum(axis=0), (4.0, -1.0))

    def test_plane_models_match_closed_form(self) -> None:
        modulus, ratio = 120.0, 0.25
        stress = LinearElasticMaterial(
            modulus, ratio, "plane_stress"
        ).constitutive_matrix
        strain = LinearElasticMaterial(
            modulus, ratio, "plane_strain"
        ).constitutive_matrix
        np.testing.assert_allclose(
            stress,
            modulus
            / (1 - ratio**2)
            * np.array(((1, ratio, 0), (ratio, 1, 0), (0, 0, (1 - ratio) / 2))),
        )
        np.testing.assert_allclose(
            strain,
            modulus
            / ((1 + ratio) * (1 - 2 * ratio))
            * np.array(
                (
                    (1 - ratio, ratio, 0),
                    (ratio, 1 - ratio, 0),
                    (0, 0, (1 - 2 * ratio) / 2),
                )
            ),
        )


class LinearElasticitySolveTests(unittest.TestCase):
    def test_constant_strain_patch_with_interior_node_is_exact(self) -> None:
        points = np.array(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)))
        mesh = TriangularMesh(
            points,
            np.array(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4))),
            node_sets={"boundary": np.array((0, 1, 2, 3))},
        )

        def exact(coordinates):
            return np.column_stack(
                (
                    0.01 + 0.02 * coordinates[:, 0] - 0.03 * coordinates[:, 1],
                    -0.02 + 0.04 * coordinates[:, 0] + 0.05 * coordinates[:, 1],
                )
            )

        material = LinearElasticMaterial(200.0, 0.25)
        result = solve_linear_elasticity(
            LinearElasticProblem(
                mesh,
                material,
                dirichlet=(DisplacementCondition("boundary", None, exact),),
            )
        )
        np.testing.assert_allclose(result.displacements, exact(points), atol=3.0e-16)
        np.testing.assert_allclose(
            result.cell_strain,
            np.tile((0.02, 0.05, 0.01), (mesh.cell_count, 1)),
            atol=3.0e-16,
        )
        expected_stress = material.constitutive_matrix @ np.array((0.02, 0.05, 0.01))
        np.testing.assert_allclose(
            result.cell_stress,
            np.tile(expected_stress, (mesh.cell_count, 1)),
            atol=8.0e-14,
        )
        self.assertLess(result.free_residual_norm, 2.0e-14)

    def test_plane_stress_uniaxial_traction_and_balance_are_exact(self) -> None:
        mesh = _mesh_with_origin(4)
        modulus, ratio, applied = 200.0, 0.25, 10.0
        result = solve_linear_elasticity(
            LinearElasticProblem(
                mesh,
                LinearElasticMaterial(modulus, ratio, "plane_stress"),
                dirichlet=(
                    DisplacementCondition("left", "x", 0.0),
                    DisplacementCondition("bottom", "y", 0.0),
                ),
                traction=(TractionCondition("right", (applied, 0.0)),),
            )
        )
        expected = np.column_stack(
            (
                applied / modulus * mesh.points[:, 0],
                -ratio * applied / modulus * mesh.points[:, 1],
            )
        )
        np.testing.assert_allclose(result.displacements, expected, atol=2.0e-13)
        np.testing.assert_allclose(
            result.cell_stress,
            np.tile((applied, 0.0, 0.0), (mesh.cell_count, 1)),
            atol=3.0e-11,
        )
        np.testing.assert_allclose(
            result.total_applied_force, (applied, 0.0), atol=2.0e-15
        )
        np.testing.assert_allclose(result.total_reaction, (-applied, 0.0), atol=3.0e-12)
        self.assertAlmostEqual(result.strain_energy, 0.25, places=12)
        self.assertEqual(result.provider_name, "native_sparse")
        self.assertTrue(result.convergence and result.convergence.converged)

    def test_native_sparse_and_dense_oracle_match(self) -> None:
        mesh = _mesh_with_origin(3)
        problem = LinearElasticProblem(
            mesh,
            LinearElasticMaterial(123.0, 0.21),
            body_force=(0.2, -0.4),
            dirichlet=(
                DisplacementCondition("left", "x", 0.0),
                DisplacementCondition("bottom", "y", 0.0),
            ),
            traction=(TractionCondition("right", (1.3, 0.1)),),
        )
        native = solve_linear_elasticity(problem, provider="native")
        dense = solve_linear_elasticity(problem, provider="numpy")
        np.testing.assert_allclose(
            native.displacements, dense.displacements, atol=3.0e-13
        )
        np.testing.assert_allclose(native.cell_stress, dense.cell_stress, atol=4.0e-11)

    def test_node_renumbering_preserves_physical_solution(self) -> None:
        baseline = _mesh_with_origin(2)
        permutation = np.array((8, 0, 4, 2, 6, 1, 7, 3, 5))  # new -> old
        inverse = np.empty_like(permutation)
        inverse[permutation] = np.arange(permutation.size)
        renumbered = TriangularMesh(
            baseline.points[permutation],
            inverse[baseline.cells],
            node_sets={
                name: inverse[nodes] for name, nodes in baseline.node_sets.items()
            },
            boundary_sets={
                name: inverse[edges] for name, edges in baseline.boundary_sets.items()
            },
        )

        def solve(mesh):
            return solve_linear_elasticity(
                LinearElasticProblem(
                    mesh,
                    LinearElasticMaterial(50.0, 0.2),
                    dirichlet=(
                        DisplacementCondition("left", "x", 0.0),
                        DisplacementCondition("bottom", "y", 0.0),
                    ),
                    traction=(TractionCondition("right", (2.0, 0.0)),),
                )
            )

        original = solve(baseline)
        reordered = solve(renumbered)
        restored = np.empty_like(reordered.displacements)
        restored[permutation] = reordered.displacements
        np.testing.assert_allclose(restored, original.displacements, atol=3.0e-14)
        np.testing.assert_allclose(
            reordered.cell_stress, original.cell_stress, atol=2.0e-12
        )

    def test_invalid_material_and_underconstraint_fail_explicitly(self) -> None:
        for modulus, ratio in (
            (0.0, 0.3),
            (1.0, 0.5),
            (1.0, -1.0),
            (True, 0.2),
            (1.0, False),
        ):
            with (
                self.subTest(modulus=modulus, ratio=ratio),
                self.assertRaises((TypeError, ValueError)),
            ):
                LinearElasticMaterial(modulus, ratio)
        mesh = unit_square_triangles(1)
        with self.assertRaisesRegex(ValueError, "rigid modes"):
            solve_linear_elasticity(
                LinearElasticProblem(
                    mesh,
                    LinearElasticMaterial(1.0, 0.2),
                    dirichlet=(DisplacementCondition("left", "x", 0.0),),
                )
            )

    def test_conflicting_displacement_values_fail(self) -> None:
        mesh = unit_square_triangles(1)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            solve_linear_elasticity(
                LinearElasticProblem(
                    mesh,
                    LinearElasticMaterial(1.0, 0.2),
                    dirichlet=(
                        DisplacementCondition("left", None, (0.0, 0.0)),
                        DisplacementCondition("left", "x", 1.0),
                    ),
                )
            )


if __name__ == "__main__":
    unittest.main()
