# ADR-0004: sparse assembly contract

Status: accepted for the serial Gate 1 slice — 2026-09-04

Element kernels return local tensors plus explicit DOF indices. Assembly owns
scatter and deterministic accumulation; linear algebra providers own sparse
storage and solve operations. Gate 1 begins with deterministic serial assembly
and tests invariance under element and node renumbering before parallelization.
The initial implementation stores duplicate deterministic COO triplets and
uses a dense NumPy conversion only inside the first linear solve provider.
