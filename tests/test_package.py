# SPDX-License-Identifier: LicenseRef-AgentFEM-Native-Draft

from __future__ import annotations

import unittest

import agentfem_native


class PackageTests(unittest.TestCase):
    def test_public_version_and_minimal_api(self) -> None:
        self.assertEqual(agentfem_native.__version__, "0.0.0")
        self.assertEqual(
            set(agentfem_native.__all__),
            {
                "AffineTriangleMap",
                "REFERENCE_TRIANGLE_VERTICES",
                "TriangleQuadrature",
                "inside_reference_triangle",
                "integrate_reference",
                "p1_basis",
                "p1_basis_gradients",
                "triangle_rule",
            },
        )


if __name__ == "__main__":
    unittest.main()
