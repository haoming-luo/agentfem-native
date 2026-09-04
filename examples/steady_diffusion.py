# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Solve the first analytical Gate 1 diffusion slice and write VTK output."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from agentfem_native import (
    DirichletCondition,
    NeumannCondition,
    SteadyDiffusionProblem,
    solve_steady_diffusion,
    unit_square_two_triangles,
    write_legacy_vtk,
)

mesh = unit_square_two_triangles()
result = solve_steady_diffusion(
    SteadyDiffusionProblem(
        mesh=mesh,
        conductivity=1.0,
        dirichlet=(DirichletCondition("left", 0.0),),
        neumann=(NeumannCondition("right", 1.0),),
    )
)

np.testing.assert_allclose(result.nodal_values, mesh.points[:, 0], atol=3.0e-16)
output = write_legacy_vtk(
    Path("steady_diffusion.vtk"),
    mesh,
    result.nodal_values,
    field_name="temperature",
)
print("solution:", result.nodal_values)
print("load + reaction:", result.total_applied_load + result.total_reaction)
print("wrote:", output)
