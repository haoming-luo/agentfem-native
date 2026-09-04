# SPDX-License-Identifier: LicenseRef-AgentFEM-Native-Draft
"""Inspect the small Gate 0 P1 reference capability."""

from __future__ import annotations

import numpy as np

from agentfem_native import AffineTriangleMap, p1_basis, triangle_rule


triangle = AffineTriangleMap(np.array(((0.0, 0.0), (2.0, 0.0), (0.0, 1.0))))
rule = triangle_rule(2)

print("area:", triangle.area)
print("basis at reference quadrature points:")
print(p1_basis(rule.points))
print("physical basis gradients:")
print(triangle.p1_gradients())
