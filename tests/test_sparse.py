# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import unittest

import numpy as np

from agentfem_native.sparse import (
    BSRMatrix,
    CSRAssemblyPlan,
    CSRMatrix,
    CSRPattern,
    conjugate_gradient,
)


class CSRMatrixTests(unittest.TestCase):
    def test_coo_canonicalization_sorts_and_reduces_duplicates(self) -> None:
        matrix = CSRMatrix.from_coo(
            (3, 3),
            np.array((2, 0, 2, 0, 2, 0)),
            np.array((2, 1, 0, 1, 2, 0)),
            np.array((4.0, 2.0, 3.0, -2.0, -4.0, 1.0)),
        )
        np.testing.assert_array_equal(matrix.indptr, (0, 2, 2, 4))
        np.testing.assert_array_equal(matrix.indices, (0, 1, 0, 2))
        np.testing.assert_array_equal(matrix.data, (1.0, 0.0, 3.0, 0.0))
        self.assertEqual(matrix.nnz, 4)
        self.assertEqual(matrix.storage_nbytes, 8 * (4 + 4 + 4))
        for array in (matrix.indptr, matrix.indices, matrix.data):
            self.assertFalse(array.flags.writeable)

    def test_empty_rows_and_matrix_vector_product_match_dense_oracle(self) -> None:
        matrix = CSRMatrix.from_coo(
            (4, 4),
            np.array((0, 0, 2, 3)),
            np.array((0, 2, 1, 3)),
            np.array((2.0, -1.0, 4.0, 5.0)),
        )
        vector = np.array((3.0, 2.0, -2.0, 0.5))
        expected = matrix.to_dense() @ vector
        np.testing.assert_array_equal(matrix.matvec(vector), expected)
        np.testing.assert_array_equal(matrix.residual(vector, expected), np.zeros(4))

    def test_malformed_csr_contract_fails_explicitly(self) -> None:
        invalid = (
            ((2, 2), [1, 1, 1], [], []),
            ((2, 2), [0, 2, 1], [0], [1.0]),
            ((2, 2), [0, 1, 1], [2], [1.0]),
            ((1, 3), [0, 2], [1, 1], [1.0, 2.0]),
            ((1, 1), [0, 1], [0], [np.nan]),
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                CSRMatrix(*arguments)

    def test_shape_outside_signed_64_bit_range_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "dimensions"):
            CSRMatrix.from_coo((np.iinfo(np.int64).max + 1, 0), [], [], [])

    def test_symmetric_dirichlet_transform_matches_dense_elimination(self) -> None:
        dense = np.array(((4.0, -1.0, 0.0), (-1.0, 4.0, -1.0), (0.0, -1.0, 3.0)))
        rows, columns = np.nonzero(np.ones_like(dense))
        matrix = CSRMatrix.from_coo((3, 3), rows, columns, dense.ravel())
        rhs = np.array((1.0, 2.0, 3.0))
        transformed, transformed_rhs = matrix.with_dirichlet(rhs, [1], [2.5])
        expected_matrix = dense.copy()
        expected_rhs = rhs - dense[:, 1] * 2.5
        expected_matrix[:, 1] = 0.0
        expected_matrix[1, :] = 0.0
        expected_matrix[1, 1] = 1.0
        expected_rhs[1] = 2.5
        np.testing.assert_array_equal(transformed.to_dense(), expected_matrix)
        np.testing.assert_array_equal(transformed_rhs, expected_rhs)
        np.testing.assert_array_equal(matrix.to_dense(), dense)

    def test_dirichlet_inserts_a_missing_diagonal(self) -> None:
        matrix = CSRMatrix.from_coo((2, 2), [0, 1], [1, 0], [1.0, 1.0])
        transformed, rhs = matrix.with_dirichlet([0.0, 0.0], [0], [3.0])
        np.testing.assert_array_equal(transformed.to_dense(), ((1.0, 0.0), (0.0, 0.0)))
        np.testing.assert_array_equal(rhs, (3.0, -3.0))


class CSRPatternTests(unittest.TestCase):
    def test_repeated_fill_reuses_graph_and_preserves_coo_reduction(self) -> None:
        rows = np.array((2, 0, 2, 0, 2, 0))
        columns = np.array((2, 1, 0, 1, 2, 0))
        first_values = np.array((4.0, 2.0, 3.0, -2.0, -4.0, 1.0))
        pattern = CSRPattern.from_coo((3, 3), rows, columns)
        first = pattern.fill(first_values)
        reference = pattern.fill_reference(first_values)
        second = pattern.fill(2.0 * first_values)
        expected = CSRMatrix.from_coo((3, 3), rows, columns, first_values)

        np.testing.assert_array_equal(first.indptr, expected.indptr)
        np.testing.assert_array_equal(first.indices, expected.indices)
        np.testing.assert_array_equal(first.data, expected.data)
        np.testing.assert_array_equal(first.data, reference.data)
        np.testing.assert_array_equal(second.data, 2.0 * first.data)
        self.assertIs(first.indptr, pattern.indptr)
        self.assertIs(second.indices, pattern.indices)
        self.assertFalse(first.data.flags.writeable)
        self.assertEqual(len(pattern.structure_digest), 64)
        self.assertEqual(pattern.coo_entry_count, rows.size)
        self.assertEqual(pattern.nnz, first.nnz)

    def test_element_dof_pattern_matches_explicit_local_matrix_order(self) -> None:
        cell_dofs = np.array(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
        pattern = CSRPattern.from_element_dofs(4, cell_dofs)
        values = np.arange(18, dtype=np.float64)
        actual = pattern.fill(values)
        rows = np.repeat(cell_dofs, 3, axis=1).ravel()
        columns = np.tile(cell_dofs, (1, 3)).ravel()
        expected_dense = np.zeros((4, 4))
        np.add.at(expected_dense, (rows, columns), values)
        np.testing.assert_array_equal(actual.to_dense(), expected_dense)

    def test_pattern_rejects_malformed_updates_and_dofs(self) -> None:
        pattern = CSRPattern.from_coo((2, 2), [0, 1], [0, 1])
        with self.assertRaisesRegex(ValueError, "长度"):
            pattern.fill([1.0])
        with self.assertRaisesRegex(ValueError, "finite"):
            pattern.fill([1.0, np.nan])
        with self.assertRaisesRegex(ValueError, "超出"):
            CSRPattern.from_element_dofs(2, [[0, 2]])
        with self.assertRaisesRegex(ValueError, "二维"):
            CSRPattern.from_element_dofs(2, [0, 1])

    def test_empty_pattern_has_stable_reference_and_production_semantics(self) -> None:
        empty = np.empty(0, dtype=np.int64)
        pattern = CSRPattern.from_coo((3, 4), empty, empty)
        production = pattern.fill([])
        reference = pattern.fill_reference([])
        np.testing.assert_array_equal(production.indptr, (0, 0, 0, 0))
        np.testing.assert_array_equal(production.data, reference.data)
        self.assertEqual(production.shape, (3, 4))


class CSRAssemblyPlanTests(unittest.TestCase):
    def test_plan_reuses_graph_and_matches_cold_canonicalization(self) -> None:
        cell_dofs = np.array(((0, 1), (1, 2)), dtype=np.int64)
        plan = CSRAssemblyPlan(3, cell_dofs)
        local = np.array((((2.0, -1.0), (-1.0, 2.0)),) * 2)
        prepared = plan.fill(local)
        rows = np.repeat(cell_dofs, 2, axis=1).ravel()
        columns = np.tile(cell_dofs, (1, 2)).ravel()
        cold = CSRMatrix.from_coo((3, 3), rows, columns, local.ravel())

        np.testing.assert_array_equal(prepared.indptr, cold.indptr)
        np.testing.assert_array_equal(prepared.indices, cold.indices)
        np.testing.assert_array_equal(prepared.data, cold.data)
        self.assertIs(prepared.indptr, plan.pattern.indptr)
        self.assertEqual(plan.contribution_count, 8)
        self.assertEqual(plan.nnz, 7)
        self.assertGreater(plan.storage_nbytes, plan.pattern.storage_nbytes)

    def test_plan_rejects_changed_layout_and_wrong_contribution_shape(self) -> None:
        cell_dofs = np.array(((0, 1), (1, 2)), dtype=np.int64)
        plan = CSRAssemblyPlan(3, cell_dofs)
        with self.assertRaisesRegex(ValueError, "DOF 顺序"):
            plan.validate_layout(3, cell_dofs[::-1])
        with self.assertRaisesRegex(ValueError, "局部矩阵"):
            plan.fill(np.ones((2, 2)))
        with self.assertRaisesRegex(ValueError, "超出"):
            CSRAssemblyPlan(2, cell_dofs)


class ConjugateGradientTests(unittest.TestCase):
    @staticmethod
    def _spd() -> tuple[CSRMatrix, np.ndarray, np.ndarray]:
        dense = np.array(((4.0, 1.0, 0.0), (1.0, 3.0, 1.0), (0.0, 1.0, 2.0)))
        rows, columns = np.nonzero(dense)
        matrix = CSRMatrix.from_coo((3, 3), rows, columns, dense[rows, columns])
        exact = np.array((1.0, -2.0, 3.0))
        return matrix, dense @ exact, exact

    def test_cg_jacobi_solves_spd_system(self) -> None:
        matrix, rhs, exact = self._spd()
        result = conjugate_gradient(matrix, rhs)
        self.assertTrue(result.report.converged)
        self.assertEqual(result.report.reason, "converged")
        self.assertLessEqual(result.report.iterations, 3)
        np.testing.assert_allclose(result.solution, exact, atol=2.0e-15)
        self.assertLessEqual(result.report.residual_norm, result.report.threshold)

    def test_initial_solution_and_iteration_limit_are_distinct(self) -> None:
        matrix, rhs, exact = self._spd()
        initial = conjugate_gradient(matrix, rhs, initial_guess=exact)
        self.assertEqual(initial.report.reason, "initial_residual")
        limited = conjugate_gradient(matrix, rhs, maximum_iterations=0)
        self.assertFalse(limited.report.converged)
        self.assertEqual(limited.report.reason, "iteration_limit")

    def test_non_positive_curvature_reports_breakdown(self) -> None:
        matrix = CSRMatrix.from_coo((2, 2), [0, 1], [0, 1], [1.0, -1.0])
        result = conjugate_gradient(
            matrix, [1.0, 1.0], preconditioner="none", maximum_iterations=3
        )
        self.assertFalse(result.report.converged)
        self.assertEqual(result.report.reason, "breakdown")

    def test_invalid_jacobi_diagonal_fails(self) -> None:
        matrix = CSRMatrix.from_coo((2, 2), [0, 1], [0, 1], [1.0, 0.0])
        with self.assertRaisesRegex(ValueError, "positive diagonal"):
            conjugate_gradient(matrix, [1.0, 1.0])


class BSRMatrixTests(unittest.TestCase):
    def test_scalar_csr_round_trip_and_matvec(self) -> None:
        dense = np.array(
            (
                (4.0, 1.0, -1.0, 0.0),
                (1.0, 3.0, 0.0, -2.0),
                (-1.0, 0.0, 5.0, 1.0),
                (0.0, -2.0, 1.0, 6.0),
            )
        )
        rows, columns = np.nonzero(dense)
        scalar = CSRMatrix.from_coo(dense.shape, rows, columns, dense[rows, columns])
        blocked = BSRMatrix.from_csr(scalar, 2)
        self.assertEqual(blocked.block_shape, (2, 2))
        self.assertEqual(blocked.nnzb, 4)
        np.testing.assert_array_equal(blocked.to_dense(), dense)
        vector = np.array((1.0, -2.0, 0.5, 3.0))
        np.testing.assert_allclose(blocked.matvec(vector), dense @ vector)
        np.testing.assert_array_equal(
            blocked.diagonal_blocks(),
            np.array((dense[:2, :2], dense[2:, 2:])),
        )

    def test_block_jacobi_cg_solves_vector_system(self) -> None:
        dense = np.array(
            (
                (4.0, 1.0, -1.0, 0.0),
                (1.0, 3.0, 0.0, -0.5),
                (-1.0, 0.0, 4.0, 1.0),
                (0.0, -0.5, 1.0, 3.0),
            )
        )
        rows, columns = np.nonzero(dense)
        matrix = CSRMatrix.from_coo(dense.shape, rows, columns, dense[rows, columns])
        exact = np.array((1.0, -2.0, 3.0, -4.0))
        outcome = conjugate_gradient(
            matrix,
            dense @ exact,
            preconditioner="block_jacobi",
            block_size=2,
        )
        self.assertTrue(outcome.report.converged)
        np.testing.assert_allclose(outcome.solution, exact, atol=2.0e-14)

    def test_invalid_block_contracts_fail(self) -> None:
        matrix = CSRMatrix.from_coo((3, 3), [0, 1, 2], [0, 1, 2], [1, 1, 1])
        with self.assertRaisesRegex(ValueError, "divisible"):
            BSRMatrix.from_csr(matrix, 2)
        with self.assertRaisesRegex(ValueError, "explicit block size"):
            conjugate_gradient(matrix, [1, 1, 1], preconditioner="block_jacobi")


if __name__ == "__main__":
    unittest.main()
