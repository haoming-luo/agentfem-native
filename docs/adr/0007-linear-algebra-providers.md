# ADR-0007: linear algebra providers

Status: accepted as design direction — 2026-09-04

Finite-element discretization and assembly belong to Native. Sparse storage,
linear/nonlinear algebra, and hardware execution sit behind capability-bearing
providers such as SciPy, PETSc, future domestic CPU/GPU solvers, or a future
native solver. Optional providers are never imported by the reference layer.

The Gate 1 diffusion slice uses a dense NumPy provider solely as an auditable
small-problem baseline. It does not change the provider boundary or satisfy the
future scalable-sparse requirement.
