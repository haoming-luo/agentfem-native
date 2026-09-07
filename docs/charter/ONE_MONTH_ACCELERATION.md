# AgentFEM Native — one-month acceleration overlay

**Window:** 2026-09-07 through 2026-10-04

**Intent:** compress the six-month Mechanics Alpha program into four concurrent
delivery tranches without weakening scientific admission or clean-room rules.

## Compression rule

Calendar compression changes concurrency, automation, and scope priority. It
does not turn prototypes into verified capabilities. Every item retains its
roadmap Gate and maturity label.

The fastest credible path is to preserve one expanding vertical spine:

```text
sparse scalar solve
  -> vector DOFs and T3
  -> C++20 vector/sparse acceleration
  -> T4 and 3D
  -> mass/time/state
  -> limited nonlinear and scale experiments
```

Disconnected element, GPU, or material demos do not outrank this spine.

## Tranche 1 — sparse foundation and T3 reference

**Nominal Week 1, locally completed on 2026-09-07**

- Independent sparse and T3 mathematical specifications.
- Immutable deterministic scalar CSR and canonical COO reduction.
- SpMV, diagonal, residual, exact storage reporting, and symmetric sparse
  Dirichlet transformation.
- CG/Jacobi with explicit convergence, limit, and breakdown evidence.
- Native sparse becomes the dependency-free default; dense NumPy is an
  explicitly selected bounded oracle and SciPy remains optional.
- Vector DOF numbering and readable T3 plane-stress/plane-strain mechanics.
- Body force, boundary traction, component constraints, reactions, energy,
  cell strain/stress recovery, and rigid-mode constraint checks.
- Execution-free capability and resource plans with deterministic digests.
- Local analytical, patch, provider, independence, and performance evidence.

Acceptance closed: GitHub Actions run `34091224291` established native Windows,
Linux, and macOS evidence for commit `5230970`. T3 remained `implemented`, not
yet complete Gate 2, at that revision.

## Compressed implementation pass — 2026-09-07

The owner authorized an immediate full-month acceleration pass. It delivered
the architecture and executable vertical slices from all four tranches:

- Tranche 2: BSR/block-Jacobi, C++20 ABI 1.1 T3 assembly, material regions,
  vector/tensor VTK, and expanded 2D response/convergence evidence;
- Tranche 3: owned T4/3D mesh-to-result mechanics, T3 mass, explicit and
  implicit linear dynamics, energy history, and deterministic restart;
- Tranche 4: resource budgets, progress/cancellation, structured errors, and
  experimental Neo-Hookean/J2 material-point transactions.

Compression created implemented candidates early; it did not automatically
satisfy every tranche exit. Threaded/SIMD sparse execution, full engineering
benchmark coverage, constrained wave dynamics, global Newton/cutback, and
MPI/GPU experiments remain explicit acceptance work. Commit `db33958` passed
all 21 Windows, Linux, and macOS jobs in hosted run `34099397789`.

## Tranche 2 — production 2D mechanics

**Nominal Week 2**

- Add bounded vectorized and C++20 T3 batch assembly with exact reference
  equivalence and a versioned ABI extension.
- Add deterministic BSR or block-graph construction and block-Jacobi.
- Add material regions, spatially variable loads, recovery policies, and VTK
  vector/tensor output.
- Complete constant-strain, uniaxial, shear, bulk, cantilever, convergence,
  orientation, numbering, balance, and failure suites.
- Add thread-safe CPU parallel assembly/operator execution and record scaling.
- Extend the Kernel Contract to the admitted 2D elasticity subset only after
  its Python API evidence is stable.

Exit target: verified 2D T3 linear elasticity on all Tier-1 platforms.

## Tranche 3 — T4, 3D, and linear time

**Nominal Week 3**

- Add independently specified tetrahedral geometry, quadrature, vector DOFs,
  isotropic T4 stiffness, loads, constraints, recovery, and 3D rigid modes.
- Add 3D patch, uniaxial/shear/bulk, energy, reaction, orientation, and
  convergence evidence.
- Add consistent/lumped mass and one explicit and one implicit linear
  integrator.
- Implement begin/commit/rollback, cancellation, checkpoint/restart, and
  energy/external-work ledgers.
- Benchmark Native, SciPy, Ginkgo, and PETSc/hypre candidates on identical
  matrices and accept provider decisions separately.

Exit target: Gate 2 candidate and a linear Gate 3 vertical slice. Gate
acceptance still requires the full published evidence matrix.

## Tranche 4 — Alpha integration and frontier tracks

**Nominal Week 4**

- Harden cross-platform install/test artifacts, fuzzing, sanitizers, malformed
  input, resource budgets, cancellation, and machine-readable diagnostics.
- Add a narrow Newton/line-search/cutback/state-rollback procedure.
- Qualify Neo-Hookean and small-strain J2 first at material-point level.
- Run MPI 1/2/4/8-rank and one GPU/matrix-free operator experiment with setup,
  communication, and transfer costs included.
- Freeze an internal Mechanics Alpha capability contract and known-limitations
  record. Do not publish packages without separate authorization.

Exit target: Gate 2 and linear Gate 3 accepted if evidence permits; nonlinear,
MPI, and GPU remain candidate/experimental unless their own Gate criteria pass.

## Daily operating loop

Each compressed day must close a small scientific loop:

1. specification and maturity target;
2. failure and identity tests;
3. readable implementation;
4. optimized implementation or provider path;
5. analytical/convergence evidence;
6. performance and memory record;
7. independence, formatting, package, and hosted-platform checks;
8. roadmap/provenance update.

The program stops adding breadth whenever the latest admitted path is not green.
This is the mechanism that makes the compressed schedule aggressive without
making its claims fictional.
