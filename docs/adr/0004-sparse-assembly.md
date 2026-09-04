# ADR-0004: sparse assembly contract

Status: proposed — 2026-09-04

Element kernels return local tensors plus explicit DOF indices. Assembly owns
scatter and deterministic accumulation; linear algebra providers own sparse
storage and solve operations. Gate 1 begins with deterministic serial assembly
and tests invariance under element and node renumbering before parallelization.
