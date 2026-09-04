# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

from math import factorial
import unittest

import numpy as np

from agentfem_native.quadrature import integrate_reference, triangle_rule


class QuadratureTests(unittest.TestCase):
    def test_weights_sum_to_reference_area(self) -> None:
        for degree in (0, 1, 2):
            with self.subTest(degree=degree):
                self.assertAlmostEqual(float(triangle_rule(degree).weights.sum()), 0.5)

    def test_degree_two_rule_integrates_monomials_exactly(self) -> None:
        rule = triangle_rule(2)
        for a, b in ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)):
            with self.subTest(a=a, b=b):
                values = rule.points[:, 0] ** a * rule.points[:, 1] ** b
                exact = factorial(a) * factorial(b) / factorial(a + b + 2)
                self.assertAlmostEqual(float(rule.integrate_values(values)), exact, places=15)

    def test_degree_one_rule_integrates_affine_vector_function(self) -> None:
        result = integrate_reference(
            lambda points: np.column_stack(
                (1.0 + points[:, 0], 2.0 - 3.0 * points[:, 1])
            ),
            exact_degree=1,
        )
        np.testing.assert_allclose(result, (2.0 / 3.0, 0.5), atol=2.0e-16)

    def test_unsupported_degrees_fail_explicitly(self) -> None:
        with self.assertRaises(ValueError):
            triangle_rule(-1)
        with self.assertRaises(NotImplementedError):
            triangle_rule(3)

    def test_non_integer_degree_is_rejected(self) -> None:
        for degree in (True, 1.5, "2"):
            with self.subTest(degree=degree), self.assertRaises(TypeError):
                triangle_rule(degree)  # type: ignore[arg-type]

    def test_value_count_must_match_rule(self) -> None:
        with self.assertRaises(ValueError):
            triangle_rule(2).integrate_values(np.ones(2))


if __name__ == "__main__":
    unittest.main()
