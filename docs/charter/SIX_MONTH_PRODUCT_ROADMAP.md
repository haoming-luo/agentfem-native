# AgentFEM Native — six-month product roadmap

**Execution window:** 2026-09-07 through 2027-03-07

**Planning model:** 28–36 experienced contributors, approximately 15 person-years

**Target:** internal **Mechanics Alpha**, not a public package release

## Executive decision

Six months is enough to turn the current rigorous diffusion slice into the
foundation of a serious mechanics product. It is not enough to reproduce the
breadth, field history, or ecosystem of a mature 5–10-year finite-element
kernel. The program therefore optimizes for a narrow, defensible product core:

1. an owned, end-to-end sparse execution spine that never silently densifies;
2. verified 2D and 3D small-strain solid mechanics;
3. verified linear transient mechanics and trustworthy state/restart behavior;
4. a deliberately limited nonlinear mechanics candidate;
5. high-performance CPU execution, with measured MPI and GPU research tracks;
6. an Agent-native control plane that can plan, bound, execute, explain, and
   audit a computation without owning its numerical semantics.

At the end of the window, Gate 2 and Gate 3 should be complete. Selected Gate 4
capabilities may be experimental or candidate maturity. MPI and GPU work may
produce strong evidence, but do not advance Gate 6 or Gate 7 by schedule alone.

## Product position

AgentFEM Native is not primarily another weak-form language. Its product is a
fast, inspectable, machine-operable mechanics engine whose numerical meaning is
owned from mesh topology through result evidence.

The distinguishing combination is:

- **native Windows, macOS, and Linux**, with no WSL substitution;
- **owned finite-element semantics** behind a stable Kernel Contract;
- **a permanent readable oracle** beside optimized C++20 execution;
- **sparse-first performance** on workstations and replaceable scale providers;
- **machine-readable trust** for agents: capabilities, estimates, assumptions,
  progress, failures, provenance, and evidence;
- **no AF-IR dependency** in the six-month critical path.

## Six-month product acceptance

### P0 — commitments

These define the Mechanics Alpha. Missing any one makes the program incomplete.

- A Native CSR/BSR algebra boundary with deterministic graph construction,
  canonical duplicate reduction, sparse boundary treatment, SpMV, residual
  evaluation, and no hidden dense solve in the production path.
- An owned serial solver baseline: CG plus Jacobi/block-Jacobi for symmetric
  positive-definite systems, with explicit convergence and failure reports.
- 2D plane-stress and plane-strain elasticity on P1 triangles, followed by 3D
  elasticity on P1 tetrahedra.
- Vector DOF ownership, displacement/traction/body-force conditions, reactions,
  strain energy, and stress/strain recovery.
- Linear transient mechanics with mass operators, at least one explicit and one
  implicit time integrator, checkpoint/restart, and energy/work evidence.
- One contract and scientific acceptance suite on native Windows x86_64,
  macOS arm64/x86_64, and Linux x86_64.
- Agent-facing `capabilities`, `plan`, `execute`, and evidence/result envelopes,
  including resource estimates and structured unsupported-path failures.

### P1 — strong product objectives

- Threaded and SIMD-aware C++20 assembly/operator execution.
- One optional single-node sparse provider and one optional distributed/HPC
  provider selected through reproducible measurements and license/platform
  review.
- A limited Newton/line-search/cutback procedure with begin/commit/rollback.
- Material-point-qualified Neo-Hookean elasticity and small-strain J2
  plasticity; element-level maturity is reported separately.
- Import/export workflows sufficient for internal mechanics studies without
  coupling core semantics to a third-party FEM library.
- Cancellation, compute budgets, structured progress, and deterministic plan
  and result digests for long-running agent-driven work.

### Stretch — evidence tracks, not promises

- Q4 and H8 element candidates after T3/T4 evidence is complete.
- Matrix-free or partial-assembly operator application for selected mechanics
  paths, independently specified from public mathematics and papers.
- MPI consistency and weak-scaling studies through 1/2/4/8 ranks.
- One GPU operator-application prototype with transfer costs included.
- Linux aarch64 and Windows arm64 candidate CI when reliable runners exist.

### Explicitly outside the six-month alpha

Contact, fracture, remeshing, adaptive mesh refinement, arbitrary high order,
general mixed formulations, multiphysics breadth, production AMG development,
full constitutive-model catalogs, certified validation, and public binary
release are not six-month commitments.

## Architecture decisions for speed and autonomy

### Own the sparse baseline

The default production path must remain useful without SciPy, PETSc, MPI, or a
GPU runtime. Native therefore owns the minimum sparse substrate required for
correct execution and diagnosis:

- immutable CSR and block-CSR structures with 64-bit-safe sizing;
- deterministic sparsity preallocation from owned DOF connectivity;
- canonical COO-to-CSR construction for compatibility and evidence;
- SpMV, diagonal/block-diagonal extraction, norms, residuals, and constraints;
- CG and Jacobi/block-Jacobi as transparent baseline algorithms;
- a narrow provider interface for stronger solvers and preconditioners.

Native should not spend the six months reimplementing industrial AMG or every
Krylov method. Those belong behind optional provider contracts.

### Keep providers replaceable

The evaluation order is:

1. **owned C++20 sparse CPU core** — mandatory Tier-1 baseline;
2. **SciPy** — optional Python development and verification provider;
3. **Ginkgo** — candidate single-node CPU/GPU sparse provider, subject to
   native-Windows and packaging evidence;
4. **PETSc with optional hypre** — candidate MPI/HPC provider, with every
   transitive component audited and Windows capability reported honestly;
5. **Kokkos or RAJA** — measured execution-portability experiments, not an
   architecture-wide dependency before they win representative benchmarks;
6. vendor and domestic-computing providers — later adapters behind the same
   contracts.

Public libCEED operator decomposition is useful research input for an
independently written matrix-free specification. No external FEM implementation
source is part of this program.

### Make Agent behavior a first-class product surface

The Kernel Contract remains backend-neutral. AgentFEM integration must not
require AF-IR promotion during this program. The engine should expose:

- `capabilities()` — exact element, procedure, provider, platform, and maturity;
- `plan(request)` — no execution; reports DOFs, estimated nonzeros, memory,
  provider choice, expected artifacts, assumptions, and unsupported features;
- `execute(plan)` — executes an immutable plan digest and emits progress,
  cancellation, checkpoint, and budget events;
- `explain(result)` — reports what ran and the evidence attached to it, without
  upgrading `successful` to `verified` or `validated`;
- structured errors containing a stable code, path, cause, retryability, and
  safe remediation choices;
- an evidence manifest connecting request, plan, binary/provider identities,
  tolerances, convergence history, balances, artifacts, and source revision.

Automatic provider choice may improve speed, but must never silently change
the mathematical model, precision, constraint method, or evidence threshold.

## Organization and parallel workstreams

One plausible 31-person allocation is:

| Workstream | People | Primary responsibility |
|---|---:|---|
| Elements and numerical foundations | 6 | DOFs, cells, quadrature, T3/T4, recovery |
| Mechanics, materials, procedures | 6 | elasticity, dynamics, nonlinear state |
| Sparse algebra and solvers | 5 | CSR/BSR, Krylov, preconditioners, providers |
| Runtime and HPC | 5 | C++20, threading/SIMD, MPI/GPU experiments |
| Verification and reliability | 5 | independent tests, benchmarks, fuzzing, evidence |
| Platform, product, and Agent interface | 4 | three-OS CI, packaging, contracts, diagnostics |

Verification ownership remains sufficiently independent to challenge
scientific and performance claims. Architecture, product, and delivery leads
are embedded across workstreams rather than added as a separate handoff layer.

The streams run concurrently. Gate discipline constrains scientific claims;
it does not require all research to wait in a waterfall.

## Monthly execution map

### Month 1 — Sparse Spine 0.1 and T3 oracle

**2026-09-07 through 2026-10-04**

- Specify sparse invariants, resource limits, vector DOFs, and 2D elasticity.
- Build owned deterministic CSR/BSR construction, SpMV, residual/norm
  operations, and sparse Dirichlet treatment.
- Build transparent CG plus Jacobi baseline and explicit failure semantics.
- Migrate diffusion production execution away from dense fallback above a
  deliberately small oracle threshold.
- Implement the readable T3 plane-stress/plane-strain mathematical oracle.
- Establish machine-readable memory, convergence, and provider evidence.

Exit: the diffusion production path is sparse end to end; no hidden
densification exists; the T3 oracle has rigid-mode and constant-strain evidence.

### Month 2 — Gate 2, 2D mechanics

**2026-10-05 through 2026-11-01**

- Complete T3 loads, constraints, reactions, energy, and recovery.
- Add C++20 vector-field assembly and block-sparse execution.
- Add bounded CPU threading and initial SIMD/data-layout work.
- Complete uniaxial, shear, bulk, cantilever, orientation, renumbering, and
  convergence evidence.
- Deliver Agent preflight planning and solver/convergence reports.
- Keep Q4 as an experimental stretch path only.

Exit: verified 2D small-strain elasticity on all Tier-1 platforms.

### Month 3 — Gate 2, 3D mechanics and provider decision

**2026-11-02 through 2026-12-06**

- Implement and verify P1 tetrahedral 3D elasticity and recovery.
- Complete the required internal mesh/result interchange path.
- Benchmark owned sparse, SciPy, Ginkgo, and PETSc/hypre candidates on exactly
  matching matrices; select providers in a separate ADR.
- Prototype matrix-free operator application without making it the only path.
- Harden million-DOF workstation behavior and three-platform installation.

Exit: Gate 2 accepted; provider choice justified by evidence, not reputation.

### Month 4 — Gate 3, linear time and state

**2026-12-07 through 2027-01-10**

- Add consistent/lumped mass and verified explicit central difference.
- Add one verified implicit linear integrator.
- Implement state transaction, checkpoint, restart, cancellation, and budgets.
- Verify SDOF response, wave propagation, time convergence, energy behavior,
  and restart equivalence.
- Make long-running procedure progress and failure machine-readable.

Exit: the linear dynamics and state-lifecycle portion of Gate 3 is accepted.

### Month 5 — Gate 4 candidate and scale experiments

**2027-01-11 through 2027-02-07**

- Add Newton residual/tangent, line search, adaptive cutback, and rollback.
- Qualify Neo-Hookean and small-strain J2 at material-point level; admit only
  separately evidenced element paths.
- Add tangent directional-derivative, objectivity, path, and rollback tests.
- Run MPI 1/2/4/8-rank consistency/scaling experiments on Linux.
- Run one GPU operator prototype and advance it only if end-to-end evidence wins.

Exit: nonlinear and scale capabilities receive honest maturity labels; no Gate
is advanced merely because a prototype ran.

### Month 6 — Mechanics Alpha hardening

**2027-02-08 through 2027-03-07**

- Freeze the internal Alpha contract and eliminate integration seams.
- Expand differential fuzzing, sanitizers, malformed-input tests, benchmark
  corpus, capability matrices, and reproducible evidence bundles.
- Tune CPU sparse assembly/operator/solver paths without weakening the oracle.
- Finish Agent planning, explanation, diagnostic, cancellation, and artifact
  workflows.
- Publish internal documentation and known-limitations records; do not create a
  public package release unless separately authorized.

Exit: Gate 2 and the linear portion of Gate 3 are complete; experimental Gate 4,
MPI, and GPU capabilities are clearly separated from production claims.

## Performance objectives

These are targets to test, not claims that already hold.

| Objective | Six-month target | Evidence rule |
|---|---:|---|
| Sparse memory | O(nnz), no production densification | allocation guard plus peak-memory record |
| Workstation scale | 1 million displacement DOFs within 16 GiB peak | complete solve, residual, balance, metadata |
| Native assembly | at least 10× readable NumPy oracle on admitted T3/T4 cases | identical semantics and interleaved repetitions |
| CPU scaling | at least 6× on 8 physical cores for assembly/operator apply | fixed problem, affinity and hardware recorded |
| Sparse competitiveness | within 1.5× selected mature provider on three canonical matrices | same matrix, stopping rule, precision, and output checks |
| Python overhead | below 2% for admitted workloads above 100k elements | end-to-end and native-only timings |
| MPI research | at least 70% weak-scaling efficiency through 8 ranks | partition consistency and communication metadata |
| GPU stretch | at least 3× one CPU socket at 5M+ DOFs | transfer and setup costs included |

Failure to meet a stretch target creates evidence and a design decision; it does
not justify a benchmark shortcut. If an owned solver misses the competitiveness
target, an optional mature provider may become the high-performance default
while the owned baseline remains the transparent fallback.

## Scientific acceptance matrix

| Capability | Minimum evidence before acceptance |
|---|---|
| Sparse algebra | exact graph/canonicalization identities, malformed structures, residual checks, provider equivalence |
| 2D/3D elasticity | rigid modes, constant-strain patch, uniaxial/shear/bulk, cantilever, energy/reactions, convergence |
| Dynamics | SDOF, wave propagation, time convergence, stability/energy behavior, restart equivalence |
| Nonlinear procedure | residual/tangent directional checks, forced cutback, rollback, convergence/failure paths |
| Materials | analytical material-point paths, objectivity where applicable, consistent tangent, state rollback |
| Optimized paths | oracle equivalence, thread/provider/platform invariance, sanitizers, bounded memory |
| MPI/GPU | partition/device invariance within specified reduction tolerance, setup/transfer included in claims |

Floating reduction order may differ across providers, threads, and ranks. Graph
and identity data remain exact; numerical tolerances are justified per quantity
and never widened merely to make a platform green.

## First seven days — Sparse Foundation Sprint

**2026-09-07 through 2026-09-13**

### Day 1 — contracts before kernels

- Accept the sparse-first architecture decision and this roadmap.
- Write CSR/BSR invariants, ownership, index-width, duplicate, and failure specs.
- Capture current dense/SciPy/native assembly baselines and add failing
  no-densification and resource-report tests.

### Day 2 — deterministic structure

- Implement validated immutable CSR graph/data types in the readable layer.
- Implement canonical COO reduction and deterministic DOF-connectivity
  preallocation, including duplicate and empty-row cases.

### Day 3 — executable sparse algebra

- Implement SpMV, diagonal extraction, norms, residuals, and symmetric sparse
  Dirichlet treatment.
- Add algebraic identities, invalid-input tests, and dense-oracle comparisons.

### Day 4 — transparent solver baseline

- Implement CG and Jacobi with explicit stopping, breakdown, non-finite,
  iteration-limit, and initial-residual behavior.
- Freeze the narrow C ABI needed for C++20 sparse kernels.

### Day 5 — end-to-end diffusion migration

- Route nontrivial diffusion through Native sparse execution without
  densification; retain dense NumPy only as a small-problem oracle.
- Compare owned and optional SciPy providers on solution, residual, reactions,
  energy, failure semantics, memory, and time.

### Day 6 — mechanics foundation

- Specify and implement vector DOF mapping and the readable T3 elasticity
  oracle for plane stress and plane strain.
- Add rigid-mode, symmetry, and constant-strain patch evidence.

### Day 7 — hardening and decision review

- Run macOS evidence locally and the full Windows/Linux/macOS hosted matrix.
- Run sanitizer, independence, memory, and scale checks; record raw artifacts.
- Review whether Week 2 begins C++ sparse acceleration or closes correctness
  gaps first. Evidence, not the calendar, decides.

The sprint is done only when sparse structures cannot be silently densified,
solver outcomes are machine-readable, oracle/provider equivalence is recorded,
the independence check passes, and all Tier-1 CI jobs are green.

## Program control

Every Friday, the program reviews five ledgers:

1. **scientific maturity** — what is implemented, tested, verified, or validated;
2. **performance** — comparable throughput, memory, scaling, and regressions;
3. **platform/provider capability** — exact tested combinations and limitations;
4. **Agent operability** — planning accuracy, diagnosability, budgets, recovery;
5. **scope and risk** — P0 threats, P1 tradeoffs, and stretch kill decisions.

The primary kill rule is simple: when breadth threatens the sparse T3/T4
mesh-to-result chain, breadth stops. A fast trustworthy spine compounds for the
next five years; disconnected prototypes do not.

## Research inputs and clean-room boundary

Only official public feature, platform, license, and architecture documentation
was used to evaluate general-purpose infrastructure:

- PETSc features, installation, and license documentation;
- Ginkgo official documentation and repository/license metadata;
- Kokkos and Kokkos Kernels documentation and release/license information;
- hypre official documentation;
- RAJA official repository/documentation;
- libCEED public user documentation describing operator decomposition.

No DOLFINx, Basix, UFL, FFCx, or other third-party FEM implementation source was
read or used. All Native mathematical and implementation specifications remain
independently derived and provenance-recorded.
