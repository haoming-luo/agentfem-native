# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""IC0 的独立小矩阵恒等式、SPD 失败反例、稀疏 FEM 和可选后端边界。"""

import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from agentfem_native.experimental_preconditioners import (
    IncompleteCholesky,
    IncompleteCholeskyPattern,
    IncompleteFactorizationError,
)
from agentfem_native.mesh import unit_square_triangles
from agentfem_native.native import native_kernel_available
from agentfem_native.sparse import CSRConstraintPlan, CSRMatrix, conjugate_gradient
from agentfem_native.thermoelastic import ThermoelasticOperator
from agentfem_native.volume_mesh import unit_cube_tetrahedra


def sparse(dense):
    dense = np.asarray(dense, dtype=float)
    rows, columns = np.nonzero(dense)
    return CSRMatrix.from_coo(dense.shape, rows, columns, dense[rows, columns])


class IncompleteCholeskyTests(unittest.TestCase):
    def assert_repeatable(self, first, second):
        # ADR-0033：普通 NumPy 参考归约不承诺跨独立调用逐位相同。
        scale = max(np.max(np.abs(first), initial=0), np.max(np.abs(second), initial=0))
        np.testing.assert_allclose(
            first, second, rtol=0, atol=64 * np.finfo(float).eps * scale
        )

    def test_pattern_reuses_only_structure_across_material_changes(self):
        for mesh in (unit_square_triangles(2), unit_cube_tetrahedra(1)):
            dimension = mesh.points.shape[1]
            fixed = (
                dimension * mesh.nodes("left")[:, None] + np.arange(dimension)
            ).ravel()
            matrices = []
            for young, poisson in ((100, 0.2), (300, 0.3), (70, 0.1)):
                matrix = ThermoelasticOperator(mesh, young, poisson).stiffness
                matrices.append(
                    CSRConstraintPlan.from_matrix(matrix, fixed).transform_matrix(
                        matrix
                    )
                )
            pattern = IncompleteCholeskyPattern(matrices[0])
            for matrix in matrices:
                baseline = IncompleteCholesky(matrix, shift=0.1)
                with patch.object(
                    np, "lexsort", side_effect=AssertionError("重复结构分析")
                ):
                    candidate = pattern.factor(matrix, shift=0.1)
                np.testing.assert_array_equal(
                    candidate.lower.indptr, baseline.lower.indptr
                )
                np.testing.assert_array_equal(
                    candidate.lower.indices, baseline.lower.indices
                )
                self.assert_repeatable(candidate.lower.data, baseline.lower.data)
                self.assert_repeatable(candidate.diagonal, baseline.diagonal)
                rhs = np.linspace(0.0, 1.0, matrix.shape[0])
                self.assert_repeatable(candidate.apply(rhs), baseline.apply(rhs))
                result = conjugate_gradient(
                    matrix, rhs, prepared_preconditioner=candidate
                )
                self.assertTrue(result.report.converged)
                self.assertLessEqual(
                    result.report.residual_norm, result.report.threshold
                )

    def test_pattern_failure_does_not_modify_previous_factor(self):
        bad = np.array(
            [[1, 0.6, 0.6, 0], [0.6, 1, 0, 0.6], [0.6, 0, 1, -0.6], [0, 0.6, -0.6, 1]]
        )
        initial = sparse(bad + np.eye(4))
        pattern = IncompleteCholeskyPattern(initial)
        first = pattern.factor(initial)
        before = first.apply([1, 2, 3, 4])
        with self.assertRaises(IncompleteFactorizationError):
            pattern.factor(sparse(bad))
        self.assert_repeatable(first.apply([1, 2, 3, 4]), before)
        retry = pattern.factor(sparse(bad), shift=0.2)
        self.assertFalse(np.array_equal(retry.diagonal, first.diagonal))
        with self.assertRaisesRegex(ValueError, "数值"):
            first.validate_matrix(sparse(bad))
        for values in (pattern.indices, pattern.transposed_order, pattern.lower_slots):
            with self.assertRaises(ValueError):
                values[0] = 42
        self.assertGreater(
            pattern.storage_nbytes, initial.indptr.nbytes + initial.indices.nbytes
        )

    def test_pattern_rejects_changed_graph_and_rechecks_numerical_admission(self):
        initial = sparse([[2, -1], [-1, 2]])
        pattern = IncompleteCholeskyPattern(initial)
        with self.assertRaisesRegex(ValueError, "图"):
            pattern.factor(sparse([[2, 0], [0, 2]]))
        for matrix in (sparse([[2, -1], [-0.5, 2]]), sparse([[-2, -1], [-1, 2]])):
            with self.assertRaises(ValueError):
                pattern.factor(matrix)
        # 同一显式图的零值是数值变化，不能触发删边或继承旧非零因子。
        diagonal = CSRMatrix(
            initial.shape, initial.indptr, initial.indices, [2, 0, 0, 3]
        )
        factor = pattern.factor(diagonal)
        np.testing.assert_allclose(factor.apply([2, 3]), [1, 1], atol=1e-15)
        with self.assertRaises(ValueError):
            IncompleteCholeskyPattern(sparse([[0, 1], [1, 0]]))
        with self.assertRaises(ValueError):
            IncompleteCholesky(initial, pattern="automatic")

    @unittest.skipUnless(native_kernel_available(), "原生 ABI 1.6 不可用")
    def test_native_factor_and_application_match_readable_route(self):
        for mesh in (unit_square_triangles(4), unit_cube_tetrahedra(2)):
            matrix = ThermoelasticOperator(mesh, 100, 0.25).stiffness
            d = mesh.points.shape[1]
            fixed = (d * mesh.nodes("left")[:, None] + np.arange(d)).ravel()
            matrix = CSRConstraintPlan.from_matrix(matrix, fixed).transform_matrix(
                matrix
            )
            reference = IncompleteCholesky(matrix, shift=0.1)
            native = IncompleteCholesky(matrix, shift=0.1, backend="native")
            reused = IncompleteCholeskyPattern(matrix).factor(
                matrix, shift=0.1, backend="native"
            )
            np.testing.assert_array_equal(reused.lower.data, native.lower.data)
            np.testing.assert_array_equal(reused.diagonal, native.diagonal)
            np.testing.assert_allclose(
                native.diagonal, reference.diagonal, rtol=1e-13, atol=1e-14
            )
            np.testing.assert_allclose(
                native.lower.data, reference.lower.data, rtol=1e-13, atol=1e-14
            )
            rhs = np.random.default_rng(91).normal(size=matrix.shape[0])
            np.testing.assert_allclose(
                native.apply(rhs), reference.apply(rhs), rtol=1e-12, atol=1e-12
            )
            answer = conjugate_gradient(matrix, rhs, prepared_preconditioner=native)
            self.assertTrue(answer.report.converged)
            self.assertLessEqual(answer.report.residual_norm, answer.report.threshold)
        bad = sparse(
            [[1, 0.6, 0.6, 0], [0.6, 1, 0, 0.6], [0.6, 0, 1, -0.6], [0, 0.6, -0.6, 1]]
        )
        with self.assertRaises(IncompleteFactorizationError) as caught:
            IncompleteCholesky(bad, backend="native")
        self.assertEqual(caught.exception.row, 3)

    @unittest.skipUnless(native_kernel_available(), "原生 ABI 1.6 不可用")
    def test_native_boundary_rejects_bad_graph_buffer_alias_and_nonfinite(self):
        import sys

        from agentfem_native.native import _p1_native, ic0_factor, ldlt_solve

        pointers = np.array([0, 0, 1], dtype=np.int64)
        columns = np.array([0], dtype=np.int64)
        factors, diagonal = ic0_factor(pointers, columns, [-1.0], [2.0, 2.0])
        np.testing.assert_allclose(
            ldlt_solve(pointers, columns, factors, diagonal, [1.0, 1.0]), [1, 1]
        )
        for bad in (np.array([1], dtype=np.int64), np.array([-1], dtype=np.int64)):
            with self.assertRaises(ValueError):
                ic0_factor(pointers, bad, [-1.0], [2.0, 2.0])
        with self.assertRaises(ValueError):
            ldlt_solve(pointers, columns, factors, diagonal, [1, float("nan")])
        rhs = np.ones(2)
        with self.assertRaisesRegex(ValueError, "重叠"):
            _p1_native.ldlt_solve_into(
                2, 1, pointers, columns, factors, diagonal, rhs, rhs
            )
        with self.assertRaises(ValueError):
            _p1_native.ldlt_solve_into(
                2, 1, pointers[:-1], columns, factors, diagonal, rhs, np.empty(2)
            )
        with self.assertRaises(OverflowError):
            _p1_native.ldlt_solve_into(
                sys.maxsize, 1, pointers, columns, factors, diagonal, rhs, np.empty(2)
            )
        misaligned = np.ndarray((2,), dtype=np.float64, buffer=bytearray(17), offset=1)
        with self.assertRaisesRegex(ValueError, "对齐"):
            _p1_native.ldlt_solve_into(
                2, 1, pointers, columns, factors, diagonal, misaligned, np.empty(2)
            )
        # 无下三角项也必须可用，不能要求空数组拥有非空数据指针。
        factor, pivot = ic0_factor([0, 0], [], [], [4])
        np.testing.assert_array_equal(ldlt_solve([0, 0], [], factor, pivot, [8]), [2])

    def test_tridiagonal_exact_factor_and_unit_scaling(self):
        dense = np.diag([4.0] * 5) + np.diag([-1.0] * 4, 1) + np.diag([-1.0] * 4, -1)
        exact = np.linspace(0.2, 1.0, 5)
        for factor in (1e-12, 1, 1e12):
            matrix = sparse(factor * dense)
            preconditioner = IncompleteCholesky(matrix)
            np.testing.assert_allclose(
                preconditioner.apply(matrix.matvec(exact)),
                exact,
                rtol=2e-15,
                atol=1e-15,
            )
            answer = conjugate_gradient(
                matrix,
                matrix.matvec(exact),
                prepared_preconditioner=preconditioner,
                absolute_tolerance=0,
            )
            self.assertTrue(answer.report.converged)
            self.assertLessEqual(answer.report.iterations, 1)
            np.testing.assert_allclose(answer.solution, exact, rtol=3e-15, atol=1e-15)

    def test_spd_can_break_down_and_shift_is_explicit(self):
        dense = np.array(
            [[1, 0.6, 0.6, 0], [0.6, 1, 0, 0.6], [0.6, 0, 1, -0.6], [0, 0.6, -0.6, 1]]
        )
        self.assertGreater(np.linalg.eigvalsh(dense).min(), 0.1)
        matrix = sparse(dense)
        with self.assertRaises(IncompleteFactorizationError) as caught:
            IncompleteCholesky(matrix)
        self.assertEqual(caught.exception.row, 3)
        self.assertLess(caught.exception.pivot, 0)
        prepared = IncompleteCholesky(matrix, shift=0.2)
        lower = np.eye(4) + prepared.lower.to_dense()
        rhs = np.array([1.0, 2.0, -3.0, 0.5])
        exact = prepared.scale * np.linalg.solve(
            lower.T, np.linalg.solve(lower, prepared.scale * rhs) / prepared.diagonal
        )
        np.testing.assert_allclose(prepared.apply(rhs), exact, rtol=1e-14, atol=1e-14)
        result = conjugate_gradient(matrix, rhs, prepared_preconditioner=prepared)
        self.assertTrue(result.report.converged)
        np.testing.assert_allclose(matrix.matvec(result.solution), rhs, atol=1e-12)

    def test_invalid_inputs_stale_identity_and_owned_state(self):
        matrix = sparse([[2, -1], [-1, 2]])
        prepared = IncompleteCholesky(matrix)
        before = prepared.apply([1, 2])
        with self.assertRaises(ValueError):
            prepared.diagonal[0] = 0
        self.assert_repeatable(prepared.apply([1, 2]), before)
        invalid = (
            sparse([[1, 1], [0, 1]]),
            sparse([[2, 0.5], [0.4, 2]]),
            sparse([[1, 0], [0, -1]]),
        )
        for value in invalid:
            with self.assertRaises(ValueError):
                IncompleteCholesky(value)
        for kwargs in (
            {"shift": -1},
            {"shift": float("inf")},
            {"shift": True},
            {"backend": "automatic"},
        ):
            with self.assertRaises(ValueError):
                IncompleteCholesky(matrix, **kwargs)
        for value in ([1], [1, float("nan")]):
            with self.assertRaises(ValueError):
                prepared.apply(value)
        with self.assertRaisesRegex(ValueError, "数值"):
            conjugate_gradient(
                sparse([[3, -1], [-1, 3]]), [1, 2], prepared_preconditioner=prepared
            )
        with self.assertRaisesRegex(ValueError, "图"):
            prepared.validate_matrix(sparse([[2, 0], [0, 2]]))

    def test_t3_t4_sparse_solve_equilibrium_and_energy_without_dense_conversion(self):
        for mesh in (unit_square_triangles(4), unit_cube_tetrahedra(2)):
            dimension = mesh.points.shape[1]
            operator = ThermoelasticOperator(mesh, 100, 0.25)
            fixed = (
                dimension * mesh.nodes("left")[:, None] + np.arange(dimension)
            ).ravel()
            constraint = CSRConstraintPlan.from_matrix(operator.stiffness, fixed)
            matrix = constraint.transform_matrix(operator.stiffness)
            rhs = np.zeros(matrix.shape[0])
            rhs[dimension * mesh.nodes("right")] = 1 / len(mesh.nodes("right"))
            baseline = conjugate_gradient(matrix, rhs)
            with patch.object(
                CSRMatrix, "to_dense", side_effect=AssertionError("禁止稠密化")
            ):
                prepared = IncompleteCholesky(matrix, shift=0.1)
                result = conjugate_gradient(
                    matrix, rhs, prepared_preconditioner=prepared
                )
            self.assertTrue(result.report.converged)
            self.assertLessEqual(result.report.residual_norm, result.report.threshold)
            np.testing.assert_allclose(
                result.solution, baseline.solution, rtol=1e-9, atol=1e-11
            )
            self.assertAlmostEqual(
                float(result.solution @ matrix.matvec(result.solution)),
                float(baseline.solution @ matrix.matvec(baseline.solution)),
                places=11,
            )

    @unittest.skipUnless(
        importlib.util.find_spec("scipy"), "可选 SciPy 三角提供者未安装"
    )
    def test_optional_triangular_backend_matches_reference(self):
        matrix = sparse(
            [[4, -1, 0, -0.2], [-1, 3, -0.4, 0], [0, -0.4, 2, -0.3], [-0.2, 0, -0.3, 1]]
        )
        reference = IncompleteCholesky(matrix)
        accelerated = IncompleteCholesky(matrix, backend="scipy")
        rhs = np.array([1, 2, -3, 4])
        np.testing.assert_allclose(
            accelerated.apply(rhs), reference.apply(rhs), rtol=1e-14, atol=1e-14
        )
        self.assert_repeatable(accelerated.lower.data, reference.lower.data)
        result = conjugate_gradient(matrix, rhs, prepared_preconditioner=accelerated)
        self.assertTrue(result.report.converged)
