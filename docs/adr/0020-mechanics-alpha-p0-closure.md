# ADR-0020: close the Mechanics Alpha P0 vertical chain before parallel breadth

Status: accepted

## Context

The compressed Mechanics Alpha 0.1 revision established readable T4 and linear
dynamics but left production T4/CSR execution, constrained dynamics,
interchange, and Agent plan/execute/explain semantics open. Starting MPI, GPU,
or a broad provider framework now would leave the central serial chain without
one coherent evidence boundary.

## Decision

1. Extend the standard-library C++20 ABI to minor version 1.2 with T4 volume
   assembly and canonical CSR SpMV. Keep the readable implementations as the
   mathematical and differential oracles.
2. Dispatch nontrivial CSR SpMV to the native kernel when installed; retain a
   deterministic small/extension-free path with no optional dependency.
3. Complete the linear transient vertical slice with zero-constraint
   elimination, reaction recovery, external work, and energy-balance evidence.
4. Admit a small, deterministic internal mesh/field interchange bundle. It is
   a Native artifact boundary, not AF-IR and not a public release format.
5. Make Agent operation explicit: plans are revalidated before execution,
   budgets apply before allocation, results are paired with evidence receipts,
   and explanation cannot promote scientific maturity.
6. Defer CPU threading/SIMD, MPI, and GPU implementation until this revision's
   serial semantics and Tier-1 CI are accepted.

## Consequences

The serial P0 chain is materially closer to the six-month target and now has
compiled 2D/3D assembly plus sparse operator acceleration. Gate 2 and the
linear part of Gate 3 remain `implemented`, not `verified`, until their
remaining corpus and hosted acceptance pass. The ABI major remains compatible;
the Rust comparison honestly remains at the ABI 1.0 diffusion subset.
