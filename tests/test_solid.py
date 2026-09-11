# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import unittest

import numpy as np

from agentfem_native.solid import (
    LinearElastic3DProblem,
    SolidDisplacementCondition,
    SolidElasticMaterial,
    SolidTractionCondition,
    assemble_linear_elasticity_3d,
    solve_linear_elasticity_3d,
    t4_body_force_load,
    t4_boundary_traction_load,
    t4_elastic_stiffness,
    t4_strain_displacement,
)
from agentfem_native.volume_mesh import TetrahedralMesh, unit_cube_tetrahedra


class TetrahedralMeshTests(unittest.TestCase):
    def test_unit_cube_has_expected_volume_and_boundary_faces(self) -> None:
        mesh = unit_cube_tetrahedra(2)
        self.assertEqual(mesh.node_count, 27)
        self.assertEqual(mesh.cell_count, 48)
        total = 0.0
        for cell in mesh.cells:
            vertices = mesh.points[cell]
            total += (
                abs(
                    np.linalg.det(
                        np.column_stack(
                            (
                                vertices[1] - vertices[0],
                                vertices[2] - vertices[0],
                                vertices[3] - vertices[0],
                            )
                        )
                    )
                )
                / 6.0
            )
        self.assertAlmostEqual(total, 1.0, places=14)
        self.assertEqual(sum(len(faces) for faces in mesh.boundary_sets.values()), 48)

    def test_invalid_or_interior_declared_face_fails(self) -> None:
        base = unit_cube_tetrahedra(1)
        with self.assertRaisesRegex(ValueError, "not a boundary"):
            TetrahedralMesh(
                base.points,
                base.cells,
                boundary_sets={"bad": np.array(((0, 1, 6),))},
            )


class T4ElementTests(unittest.TestCase):
    def test_stiffness_has_six_rigid_modes(self) -> None:
        vertices = np.array(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        )
        stiffness = t4_elastic_stiffness(vertices, SolidElasticMaterial(100.0, 0.25))
        np.testing.assert_allclose(stiffness, stiffness.T, atol=0.0)
        eigenvalues = np.linalg.eigvalsh(stiffness)
        self.assertEqual(np.count_nonzero(eigenvalues > 1.0e-10), 6)

    def test_orientation_changes_only_local_numbering(self) -> None:
        vertices = np.array(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        )
        material = SolidElasticMaterial(70.0, 0.22)
        baseline = t4_elastic_stiffness(vertices, material)
        reordered = t4_elastic_stiffness(vertices[[0, 2, 1, 3]], material)
        permutation = np.array((0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11))
        np.testing.assert_allclose(
            reordered, baseline[np.ix_(permutation, permutation)], atol=2.0e-14
        )

    def test_affine_field_recovers_engineering_strain(self) -> None:
        vertices = np.array(
            ((0.2, -0.1, 0.3), (1.1, 0.0, 0.4), (0.1, 1.2, 0.2), (0.3, 0.1, 1.4))
        )
        gradient = np.array(
            ((0.02, -0.03, 0.04), (0.05, 0.06, -0.01), (-0.02, 0.03, 0.07))
        )
        displacement = np.array((0.1, -0.2, 0.3)) + vertices @ gradient.T
        strain = t4_strain_displacement(vertices) @ displacement.ravel()
        expected = (
            gradient[0, 0],
            gradient[1, 1],
            gradient[2, 2],
            gradient[0, 1] + gradient[1, 0],
            gradient[1, 2] + gradient[2, 1],
            gradient[0, 2] + gradient[2, 0],
        )
        np.testing.assert_allclose(strain, expected, atol=2.0e-16)

    def test_constant_volume_and_face_load_resultants(self) -> None:
        vertices = np.array(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        )
        body = t4_body_force_load(vertices, (6.0, -3.0, 1.5)).reshape(4, 3).sum(axis=0)
        np.testing.assert_allclose(body, (1.0, -0.5, 0.25))
        face = (
            t4_boundary_traction_load(vertices[:3], (2.0, -4.0, 1.0))
            .reshape(3, 3)
            .sum(axis=0)
        )
        np.testing.assert_allclose(face, (1.0, -2.0, 0.5))


class LinearElastic3DTests(unittest.TestCase):
    def test_native_and_reference_t4_assembly_are_equivalent(self) -> None:
        mesh = unit_cube_tetrahedra(2)
        problem = LinearElastic3DProblem(
            mesh,
            SolidElasticMaterial(137.0, 0.21),
            body_force=(0.3, -0.2, 0.1),
        )
        reference_matrix, reference_load, _ = assemble_linear_elasticity_3d(
            problem, assembly="reference"
        )
        native_matrix, native_load, _ = assemble_linear_elasticity_3d(
            problem, assembly="native"
        )
        np.testing.assert_array_equal(native_matrix.rows, reference_matrix.rows)
        np.testing.assert_array_equal(native_matrix.columns, reference_matrix.columns)
        np.testing.assert_allclose(
            native_matrix.data, reference_matrix.data, atol=8.0e-14
        )
        np.testing.assert_allclose(native_load, reference_load, atol=2.0e-16)

    def test_manufactured_t4_interpolation_error_converges_second_order(self) -> None:
        modulus, ratio = 100.0, 0.2
        shear = modulus / (2.0 * (1.0 + ratio))
        lame = modulus * ratio / ((1.0 + ratio) * (1.0 - 2.0 * ratio))

        def exact(points):
            return np.column_stack(
                (
                    points[:, 0] ** 2 * points[:, 1],
                    points[:, 1] ** 2 * points[:, 2],
                    points[:, 2] ** 2 * points[:, 0],
                )
            )

        def body(points):
            x, y, z = points.T
            return np.column_stack(
                (
                    -2.0 * shear * y - 2.0 * (lame + shear) * (y + z),
                    -2.0 * shear * z - 2.0 * (lame + shear) * (x + z),
                    -2.0 * shear * x - 2.0 * (lame + shear) * (y + x),
                )
            )

        errors = []
        for resolution in (1, 2, 4):
            mesh = unit_cube_tetrahedra(resolution)
            result = solve_linear_elasticity_3d(
                LinearElastic3DProblem(
                    mesh,
                    SolidElasticMaterial(modulus, ratio),
                    body_force=body,
                    dirichlet=(SolidDisplacementCondition("boundary", None, exact),),
                )
            )
            centroids = mesh.points[mesh.cells].mean(axis=1)
            interpolated = result.displacements[mesh.cells].mean(axis=1)
            errors.append(
                float(np.sqrt(np.mean((interpolated - exact(centroids)) ** 2)))
            )
        self.assertGreater(errors[0] / errors[1], 3.7)
        self.assertGreater(errors[1] / errors[2], 3.7)

    def test_native_sparse_and_dense_oracle_match(self) -> None:
        mesh = unit_cube_tetrahedra(1)
        problem = LinearElastic3DProblem(
            mesh,
            SolidElasticMaterial(90.0, 0.23),
            body_force=(0.1, -0.2, 0.05),
            dirichlet=(SolidDisplacementCondition("boundary", None, (0.0, 0.0, 0.0)),),
        )
        native = solve_linear_elasticity_3d(problem)
        dense = solve_linear_elasticity_3d(problem, provider="numpy")
        np.testing.assert_allclose(
            native.displacements, dense.displacements, atol=1.0e-14
        )
        np.testing.assert_allclose(native.cell_stress, dense.cell_stress, atol=1.0e-12)

    def test_constant_strain_cube_patch_is_exact(self) -> None:
        mesh = unit_cube_tetrahedra(2)
        gradient = np.array(
            ((0.01, -0.02, 0.03), (0.04, 0.05, -0.01), (-0.02, 0.01, 0.06))
        )

        def exact(points):
            return np.array((0.1, -0.2, 0.05)) + points @ gradient.T

        result = solve_linear_elasticity_3d(
            LinearElastic3DProblem(
                mesh,
                SolidElasticMaterial(200.0, 0.25),
                dirichlet=(SolidDisplacementCondition("boundary", None, exact),),
            )
        )
        np.testing.assert_allclose(
            result.displacements, exact(mesh.points), atol=2.0e-13
        )
        expected = np.array((0.01, 0.05, 0.06, 0.02, 0.0, 0.01))
        np.testing.assert_allclose(
            result.cell_strain, np.tile(expected, (mesh.cell_count, 1)), atol=6.0e-14
        )
        self.assertLess(result.free_residual_norm, 1.0e-11)

    def test_uniaxial_cube_traction_matches_closed_form(self) -> None:
        mesh = unit_cube_tetrahedra(1)
        modulus, ratio, applied = 100.0, 0.2, 5.0
        result = solve_linear_elasticity_3d(
            LinearElastic3DProblem(
                mesh,
                SolidElasticMaterial(modulus, ratio),
                dirichlet=(
                    SolidDisplacementCondition("left", "x", 0.0),
                    SolidDisplacementCondition("front", "y", 0.0),
                    SolidDisplacementCondition("bottom", "z", 0.0),
                ),
                traction=(SolidTractionCondition("right", (applied, 0.0, 0.0)),),
            )
        )
        expected = np.column_stack(
            (
                applied / modulus * mesh.points[:, 0],
                -ratio * applied / modulus * mesh.points[:, 1],
                -ratio * applied / modulus * mesh.points[:, 2],
            )
        )
        np.testing.assert_allclose(result.displacements, expected, atol=4.0e-13)
        np.testing.assert_allclose(
            result.cell_stress,
            np.tile((applied, 0, 0, 0, 0, 0), (mesh.cell_count, 1)),
            atol=2.0e-11,
        )
        np.testing.assert_allclose(
            result.total_reaction, (-applied, 0, 0), atol=2.0e-12
        )
        self.assertAlmostEqual(result.strain_energy, 0.125, places=12)

    def test_cube_pure_shear_and_hydrostatic_response_are_exact(self) -> None:
        mesh = unit_cube_tetrahedra(1)
        modulus, ratio, applied = 100.0, 0.2, 5.0
        material = SolidElasticMaterial(modulus, ratio)
        shear = modulus / (2.0 * (1.0 + ratio))
        shear_result = solve_linear_elasticity_3d(
            LinearElastic3DProblem(
                mesh,
                material,
                dirichlet=(
                    SolidDisplacementCondition("front", "x", 0.0),
                    SolidDisplacementCondition("left", "y", 0.0),
                    SolidDisplacementCondition(
                        "left", "x", lambda points: applied / shear * points[:, 1]
                    ),
                    SolidDisplacementCondition("bottom", "z", 0.0),
                ),
                traction=(
                    SolidTractionCondition("back", (applied, 0.0, 0.0)),
                    SolidTractionCondition("right", (0.0, applied, 0.0)),
                ),
            )
        )
        expected_shear_displacement = np.column_stack(
            (
                applied / shear * mesh.points[:, 1],
                np.zeros(mesh.node_count),
                np.zeros(mesh.node_count),
            )
        )
        np.testing.assert_allclose(
            shear_result.displacements, expected_shear_displacement, atol=8.0e-13
        )
        np.testing.assert_allclose(
            shear_result.cell_stress,
            np.tile((0, 0, 0, applied, 0, 0), (mesh.cell_count, 1)),
            atol=4.0e-11,
        )
        np.testing.assert_allclose(
            shear_result.total_reaction, (-applied, -applied, 0), atol=5.0e-12
        )

        lame = modulus * ratio / ((1.0 + ratio) * (1.0 - 2.0 * ratio))
        dilation = applied / (3.0 * lame + 2.0 * shear)
        bulk_result = solve_linear_elasticity_3d(
            LinearElastic3DProblem(
                mesh,
                material,
                dirichlet=(
                    SolidDisplacementCondition("left", "x", 0.0),
                    SolidDisplacementCondition("front", "y", 0.0),
                    SolidDisplacementCondition("bottom", "z", 0.0),
                ),
                traction=(
                    SolidTractionCondition("right", (applied, 0.0, 0.0)),
                    SolidTractionCondition("back", (0.0, applied, 0.0)),
                    SolidTractionCondition("top", (0.0, 0.0, applied)),
                ),
            )
        )
        np.testing.assert_allclose(
            bulk_result.displacements, dilation * mesh.points, atol=8.0e-13
        )
        np.testing.assert_allclose(
            bulk_result.cell_stress,
            np.tile((applied, applied, applied, 0, 0, 0), (mesh.cell_count, 1)),
            atol=4.0e-11,
        )

    def test_underconstraint_fails_before_solve(self) -> None:
        mesh = unit_cube_tetrahedra(1)
        with self.assertRaisesRegex(ValueError, "six 3D rigid modes"):
            solve_linear_elasticity_3d(
                LinearElastic3DProblem(
                    mesh,
                    SolidElasticMaterial(1.0, 0.2),
                    dirichlet=(SolidDisplacementCondition("left", "x", 0.0),),
                )
            )


if __name__ == "__main__":
    unittest.main()
