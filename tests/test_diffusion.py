# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import unittest

import numpy as np

from agentfem_native.diffusion import (
    CellMaterial,
    DirichletCondition,
    NeumannCondition,
    SteadyDiffusionProblem,
    solve_steady_diffusion,
)
from agentfem_native.mesh import (
    TriangularMesh,
    unit_square_triangles,
    unit_square_two_triangles,
)


def _solve_unit_flux(mesh: TriangularMesh):
    return solve_steady_diffusion(
        SteadyDiffusionProblem(
            mesh=mesh,
            conductivity=1.0,
            dirichlet=(DirichletCondition("left", 0.0),),
            neumann=(NeumannCondition("right", 1.0),),
        )
    )


class DiffusionTests(unittest.TestCase):
    def test_two_triangle_mixed_boundary_problem_reproduces_u_equals_x(self) -> None:
        mesh = unit_square_two_triangles()
        result = _solve_unit_flux(mesh)
        np.testing.assert_allclose(
            result.nodal_values,
            mesh.points[:, 0],
            rtol=0.0,
            atol=3.0e-16,
        )
        self.assertLess(result.free_residual_norm, 3.0e-16)
        self.assertAlmostEqual(result.total_applied_load, 1.0)
        self.assertAlmostEqual(result.total_reaction, -1.0)
        self.assertAlmostEqual(result.total_applied_load + result.total_reaction, 0.0)

    def test_cell_order_and_orientation_do_not_change_solution(self) -> None:
        baseline = unit_square_two_triangles()
        reordered = TriangularMesh(
            points=baseline.points,
            cells=np.array(((3, 2, 0), (2, 1, 0))),
            node_sets=baseline.node_sets,
            boundary_sets=baseline.boundary_sets,
        )
        np.testing.assert_allclose(
            _solve_unit_flux(reordered).nodal_values,
            _solve_unit_flux(baseline).nodal_values,
            atol=3.0e-16,
        )

    def test_node_renumbering_does_not_change_physical_solution(self) -> None:
        baseline = unit_square_two_triangles()
        permutation = np.array((2, 0, 3, 1))  # new index -> old index
        inverse = np.empty_like(permutation)
        inverse[permutation] = np.arange(permutation.size)
        renumbered = TriangularMesh(
            points=baseline.points[permutation],
            cells=inverse[baseline.cells],
            node_sets={
                name: inverse[nodes] for name, nodes in baseline.node_sets.items()
            },
            boundary_sets={
                name: inverse[edges] for name, edges in baseline.boundary_sets.items()
            },
        )
        restored = np.empty(4)
        restored[permutation] = _solve_unit_flux(renumbered).nodal_values
        np.testing.assert_allclose(restored, _solve_unit_flux(baseline).nodal_values)

    def test_linear_dirichlet_patch_with_interior_node_is_exact(self) -> None:
        mesh = TriangularMesh(
            points=np.array(
                ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5))
            ),
            cells=np.array(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4))),
            node_sets={"boundary": np.array((0, 1, 2, 3))},
        )
        exact = lambda points: 1.0 + 2.0 * points[:, 0] - 3.0 * points[:, 1]
        result = solve_steady_diffusion(
            SteadyDiffusionProblem(
                mesh=mesh,
                conductivity=4.0,
                dirichlet=(DirichletCondition("boundary", exact),),
            )
        )
        np.testing.assert_allclose(
            result.nodal_values, exact(mesh.points), atol=5.0e-16
        )

    def test_pure_neumann_problem_is_rejected_before_linear_solve(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one Dirichlet"):
            solve_steady_diffusion(
                SteadyDiffusionProblem(
                    mesh=unit_square_two_triangles(), conductivity=1.0
                )
            )

    def test_spatially_varying_conductivity_reproduces_linear_exact_solution(
        self,
    ) -> None:
        mesh = unit_square_triangles(4)
        result = solve_steady_diffusion(
            SteadyDiffusionProblem(
                mesh=mesh,
                conductivity=lambda points: 1.0 + points[:, 0],
                source=-1.0,
                dirichlet=(DirichletCondition("left", 0.0),),
                neumann=(NeumannCondition("right", 2.0),),
            )
        )
        np.testing.assert_allclose(result.nodal_values, mesh.points[:, 0], atol=2.0e-15)

    def test_anisotropic_conductivity_reproduces_linear_exact_solution(self) -> None:
        mesh = unit_square_triangles(3)
        result = solve_steady_diffusion(
            SteadyDiffusionProblem(
                mesh=mesh,
                conductivity=np.array(((2.0, 0.0), (0.0, 5.0))),
                dirichlet=(DirichletCondition("left", 0.0),),
                neumann=(NeumannCondition("right", 2.0),),
            )
        )
        np.testing.assert_allclose(result.nodal_values, mesh.points[:, 0], atol=2.0e-15)

    def test_piecewise_material_regions_reproduce_layered_solution(self) -> None:
        base = unit_square_triangles(2, 2)
        centroids = base.points[base.cells].mean(axis=1)
        mesh = TriangularMesh(
            points=base.points,
            cells=base.cells,
            node_sets=base.node_sets,
            boundary_sets=base.boundary_sets,
            cell_sets={
                "left_material": np.flatnonzero(centroids[:, 0] < 0.5),
                "right_material": np.flatnonzero(centroids[:, 0] > 0.5),
            },
        )
        result = solve_steady_diffusion(
            SteadyDiffusionProblem(
                mesh=mesh,
                conductivity=1.0,
                dirichlet=(
                    DirichletCondition("left", 0.0),
                    DirichletCondition("right", 1.0),
                ),
                materials=(
                    CellMaterial("insulator", "left_material", 1.0),
                    CellMaterial("conductor", "right_material", 2.0),
                ),
            )
        )
        exact = np.where(
            mesh.points[:, 0] <= 0.5,
            (4.0 / 3.0) * mesh.points[:, 0],
            (2.0 / 3.0) + (2.0 / 3.0) * (mesh.points[:, 0] - 0.5),
        )
        np.testing.assert_allclose(result.nodal_values, exact, atol=8.0e-16)

    def test_overlapping_material_regions_fail_explicitly(self) -> None:
        base = unit_square_two_triangles()
        mesh = TriangularMesh(
            points=base.points,
            cells=base.cells,
            node_sets=base.node_sets,
            boundary_sets=base.boundary_sets,
            cell_sets={"all": np.array((0, 1)), "first": np.array((0,))},
        )
        problem = SteadyDiffusionProblem(
            mesh=mesh,
            conductivity=1.0,
            dirichlet=(DirichletCondition("left", 0.0),),
            materials=(
                CellMaterial("one", "all", 1.0),
                CellMaterial("two", "first", 2.0),
            ),
        )
        with self.assertRaisesRegex(ValueError, "Multiple materials"):
            solve_steady_diffusion(problem)

    def test_duplicate_material_names_fail_explicitly(self) -> None:
        base = unit_square_two_triangles()
        mesh = TriangularMesh(
            points=base.points,
            cells=base.cells,
            node_sets=base.node_sets,
            boundary_sets=base.boundary_sets,
            cell_sets={"first": np.array((0,)), "second": np.array((1,))},
        )
        problem = SteadyDiffusionProblem(
            mesh=mesh,
            conductivity=1.0,
            dirichlet=(DirichletCondition("left", 0.0),),
            materials=(
                CellMaterial("duplicate", "first", 1.0),
                CellMaterial("duplicate", "second", 2.0),
            ),
        )
        with self.assertRaisesRegex(ValueError, "Duplicate cell material name"):
            solve_steady_diffusion(problem)

    def test_manufactured_solution_has_second_order_nodal_convergence(self) -> None:
        errors = []
        for resolution in (4, 8, 16):
            mesh = unit_square_triangles(resolution)
            exact = lambda points: (
                np.sin(np.pi * points[:, 0]) * np.sin(np.pi * points[:, 1])
            )
            source = lambda points: (
                2.0
                * np.pi**2
                * np.sin(np.pi * points[:, 0])
                * np.sin(np.pi * points[:, 1])
            )
            result = solve_steady_diffusion(
                SteadyDiffusionProblem(
                    mesh=mesh,
                    conductivity=1.0,
                    source=source,
                    dirichlet=(DirichletCondition("boundary", 0.0),),
                )
            )
            errors.append(
                float(np.sqrt(np.mean((result.nodal_values - exact(mesh.points)) ** 2)))
            )
        observed_orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
        self.assertTrue(np.all(observed_orders > 1.5), (errors, observed_orders))


if __name__ == "__main__":
    unittest.main()
