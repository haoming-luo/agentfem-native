# ADR-0007: linear algebra providers

Status: accepted as design direction — 2026-09-04

Finite-element discretization and assembly belong to Native. Sparse storage,
linear/nonlinear algebra, and hardware execution sit behind capability-bearing
providers such as SciPy, PETSc, future domestic CPU/GPU solvers, or a future
native solver. Optional providers are never imported by the reference layer.

The Gate 1 release candidate implements a dense NumPy provider as the auditable
small-problem baseline and an optional SciPy COO-to-CSR direct-solve provider.
SciPy is imported only when selected, and `auto` falls back to NumPy when it is
absent. Both providers return their name, version, and matrix format.

SciPy is admitted as a general-purpose optional numerical dependency under its
BSD license. This adoption does not delegate finite-element topology,
quadrature, basis evaluation, material assignment, or assembly to SciPy.
PETSc and vendor providers remain future candidates and require separate
dependency inventories, platform evidence, and performance justification.
