# Project charter

## Mission

AgentFEM Native Engine is the independently developed finite-element engine
for AgentFEM. Its success means one AgentFEM engineering model can be lowered
through a versioned contract and solved without FEniCSx performing mesh,
element, DOF, quadrature, assembly, state evolution, or result generation.

## Identity

- Repository and distribution: `agentfem-native`
- Python import: `agentfem_native`
- AgentFEM backend identifier: `native`
- Public description: “AgentFEM Native is the independently developed
  finite-element engine for AgentFEM.”

AgentFEM owns the user-facing engineering language. Native owns finite-element
discretization and computation. General linear algebra and hardware services
remain replaceable providers.

## Non-negotiable boundaries

- Native is not a FEniCSx fork, translation, rearrangement, or wrapper.
- DOLFINx, Basix, UFL, and FFCx source is not an implementation input.
- FEniCSx may be a production backend and black-box comparator, never the sole
  correctness oracle.
- General libraries such as NumPy, SciPy, PETSc, MPI, BLAS/LAPACK, and HDF5
  may be used behind approved, documented boundaries.
- Linux and native Windows are equal product targets. No OS-specific fork of
  the mathematical kernel is permitted.
- Correctness, provenance, and maintainability take priority over code volume
  or premature optimization.

## Development roles

The Kernel role works from mathematical specifications and owns independent
implementation. A separately scoped Verification role runs analytical,
manufactured, benchmark, and cross-backend comparisons without passing
third-party implementation source to the Kernel role.

## Scientific admission rule

Every admitted capability has a real consumer, formulas, typed inputs and
outputs, unit tests, analytical/manufactured evidence, golden data, failure
cases, convergence evidence, and a maturity label: `experimental`,
`implemented`, `verified`, `validated`, or `production`.

## Governance

Key decisions live in ADRs. Each release records commit, dependencies,
compiler/runtime, platform, tests, benchmarks, performance, failures,
capabilities, license inventory, and AI/provenance statements. The licensing
model remains a draft until owner and legal approval.
