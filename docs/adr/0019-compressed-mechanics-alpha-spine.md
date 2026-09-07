# ADR-0019: compressed Mechanics Alpha vertical spine

- Status: accepted
- Date: 2026-09-07

## Context

The project owner requested that the one-month acceleration overlay be executed
in one sustained development pass. Calendar compression cannot erase Gate
evidence, but it can establish the complete architecture and executable
vertical slices earlier.

## Decision

1. Native BSR and block-Jacobi extend the owned scalar CSR baseline for
   node-major vector mechanics.
2. C++20 ABI 1.1 adds T3 elasticity volume assembly; readable, vectorized, and
   compiled paths remain independently comparable.
3. A readable T4 mesh-to-result path establishes 3D small-strain semantics
   before compiled T4 optimization.
4. Gate 3 begins with owned T3 mass, centered explicit dynamics, Newmark
   average acceleration, energy histories, and digest-bound restart.
5. Budgets, cancellation, progress, and structured failure are numerical
   runtime services and do not own finite-element meaning.
6. Neo-Hookean and J2 work is admitted only as experimental material-point
   candidates. No global nonlinear or Gate 4 claim follows from them.

## Consequences

The compressed pass materially advances every monthly tranche, but Gate 2 and
Gate 3 remain open until hosted evidence and their complete published benchmark
matrices pass. MPI, GPU, global Newton/cutback, and production threaded sparse
execution remain explicit future work.
