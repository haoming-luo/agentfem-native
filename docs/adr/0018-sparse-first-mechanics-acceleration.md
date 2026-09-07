# ADR-0018: sparse-first six-month mechanics acceleration

- Status: accepted
- Date: 2026-09-07

## Context

Gate 1 proves a narrow, rigorous diffusion chain and a packaged C++20 assembly
kernel. The current production solve path can still depend on optional SciPy or
densify through the NumPy provider. Gate 2 introduces vector-valued mechanics,
where dense execution is not a viable product foundation.

The project owner requested a bold six-month product plan comparable to a large
development team, with both serious high-performance ambitions and a design
suited to AgentFEM.

## Decision

1. The next performance milestone is an owned, end-to-end sparse baseline.
2. Native will own deterministic CSR/BSR structure, core sparse operations,
   CG, and Jacobi/block-Jacobi; it will not attempt to reproduce industrial AMG
   or the full solver breadth of mature libraries during this program.
3. The NumPy dense path remains a deliberately bounded mathematical oracle, not
   the production path for large problems.
4. General-purpose permissive libraries may provide stronger sparse, MPI, and
   accelerator execution behind narrow replaceable contracts. They do not own
   mesh, element, material, procedure, or evidence semantics.
5. C++20 remains the production compiled language. Execution-portability
   frameworks and GPU backends must win representative measurements before
   becoming required dependencies.
6. The six-month committed product is Mechanics Alpha: Gate 2, the linear part
   of Gate 3, and Agent-native planning/evidence behavior. Nonlinear, MPI, and
   GPU work remains maturity-labelled and gate-controlled.
7. Native will expose capability, preflight/resource planning, execution,
   cancellation/checkpoint, diagnostics, and evidence contracts without making
   current AF-IR promotion a dependency.

## Consequences

- Week 1 begins with sparse specifications and failure/evidence tests before
  optimized implementation.
- Three-platform CI remains release-blocking for committed capabilities.
- Performance comparisons include semantic equivalence, residual/balance,
  memory, setup costs, and complete environment metadata.
- PETSc/hypre, Ginkgo, Kokkos/RAJA, and GPU work are evaluated as optional
  providers or research tracks, not silently imported into the reference layer.
- Gate 2 breadth pauses if it threatens the verified sparse T3/T4 vertical
  chain.

## Rejected alternatives

- **Keep dense NumPy as the default mechanics solver:** rejected because memory
  grows quadratically and would make realistic mechanics impossible.
- **Make SciPy or PETSc the only execution path:** rejected because Native needs
  a transparent, native-three-platform baseline and replaceable providers.
- **Build a complete AMG stack now:** rejected as a poor six-month allocation;
  mature permissive providers supply this specialization.
- **Start with nonlinear breadth or GPU kernels:** rejected as the critical
  sparse/vector-DOF/linear-mechanics chain is not yet complete.
- **Couple progress to AF-IR:** rejected under ADR-0017; the Kernel Contract is
  sufficient for Agent-facing capability during this milestone.
