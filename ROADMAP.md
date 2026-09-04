# AgentFEM Native roadmap

This roadmap is gate-based. A gate advances only when its claims have evidence;
calendar pressure does not lower mathematical or independence requirements.
Native Windows, macOS, and Linux are required from Gate 0 onward. WSL may be offered as an
additional route but never substitutes for native Windows acceptance.

## Gate 0 — charter and independent lineage

Exit criteria:

- approved project boundary, clean-room policy, provenance process, and
  license decision;
- versioned Kernel Contract draft and recorded architecture decisions;
- P1 reference triangle, affine map, and degree-1/degree-2 quadrature covered
  by mathematical tests;
- isolated tests prove no import of DOLFINx, UFL, Basix, or FFCx;
- source package and wheel build on Linux, native Windows, and macOS;
- the same test suite passes on all three systems.

Current state: complete. Hosted native Windows, macOS, and Linux tests and
installability checks passed in GitHub Actions run `33834316684` at commit
`f005c8d`.

## Gate 1 — serial scalar reference kernel

Scope: 2D triangular meshes, named sets, P1 scalar space, global DOF numbering,
Dirichlet and Neumann conditions, sparse assembly, steady diffusion/heat,
nodal results, VTK output, and an external AgentFEM adapter prototype.

Required evidence: constant and linear reproduction, patch tests, analytical
1D-equivalent solution, manufactured solution, mesh convergence, boundary
integration, reversed element orientation, and node-renumbering invariance on
native Windows, macOS, and Linux.

First vertical slice: solve `-div(k grad u) = f` on the unit square using two
P1 triangles, then generalize only after the element-to-result evidence chain
is complete.

Current state: a local Gate 1 release candidate is implemented. It includes
analytical, patch, convergence, orientation, renumbering, balance, failure,
and VTK evidence; cell material regions; variable and anisotropic
conductivity; NumPy and optional SciPy providers; Kernel Contract 0.1 JSON
request/result envelopes; a cross-platform CLI; and an external public-AF-IR
lowering prototype. The same scientific suite passes on hosted native Windows,
macOS, and Linux. Gate 1 remains open only for integration of the portable
extension into AgentFEM's public exported IR.

## Performance milestone P1 — bounded serial throughput (complete locally)

- Completed: vectorized structured-mesh generation and validation; bounded
  vectorized static P1 assembly; conservative auto-dispatch; reference/fast
  path equivalence and scale benchmarks.
- Completed locally: standard-library-only C++20/C ABI P1 assembly spike and
  exact comparison with vectorized NumPy.
- Measured local direction: hundreds-fold vectorized speedup over the readable
  element loop, followed by additional tens-fold C++ kernel headroom. These are
  local kernel benchmarks, not cross-platform end-to-end claims.
- Completed locally: sanitizer builds, compiled-wheel packaging, a dependency-
  free Rust equivalent, material-region ABI, and sparse end-to-end timing.

## Performance milestone P2 — packaged compiled kernel (complete)

- Completed locally: a hand-written CPython stable-ABI adapter packages the
  C++20 kernel in a `cp311-abi3` wheel without pybind11, nanobind, Cython, or
  the NumPy C API.
- Completed locally: exact C++/Rust/NumPy COO and load-vector equivalence,
  interleaved raw timings, public-runtime timing, and identical SciPy solution,
  reaction, and energy evidence.
- Decision: C++20 is the primary production kernel language. Rust remains a
  valuable safety and design cross-check but is not a second shipping runtime
  or a release blocker.
- Completed locally: SciPy 1.18.1 was installed from an official wheel after
  SHA-256 verification and passes provider equivalence; SciPy remains optional
  infrastructure and owns no finite-element semantics.
- Completed in CI: native installation wheels build, pass strict Stable ABI
  auditing, install, and pass the scientific suite on Windows x86_64, Linux
  x86_64, macOS x86_64, and macOS arm64. These are ephemeral private CI test
  artifacts, not a PyPI or GitHub Release publication.
- Deferred until an actual release milestone: public artifact publication,
  signing, binary SBOM policy, and broader field/operator coverage.

Execution policy: the local development machine runs macOS builds and tests.
GitHub CI is the required execution environment for native Windows and Linux;
local emulation is diagnostic only and never substitutes for those runners.

Before Gate 2, keep the hosted P2 installability evidence green. Prefer permissive
general-purpose dependencies; do not use a third-party FEM implementation to
substitute for Native discretization.

## Gate 2 — basic solid mechanics

Scope: 2D small-strain isotropic elasticity, plane stress/strain, displacement
constraints, traction/body force, reactions, strain energy, stress/strain
recovery, P1 triangles, followed by P1 tetrahedra and basic 3D.

Evidence: rigid-body modes, constant-strain patch tests, uniaxial/shear/bulk
responses, cantilever, symmetry, energy, reaction balance, and convergence.

## Gate 3 — time and nonlinear lifecycle

Scope: consistent and lumped mass, central difference, implicit increments,
Newton residual/tangent, begin/commit/rollback, adaptive cutback,
checkpoint/restart, and energy/external-work ledgers.

Evidence: SDOF dynamics, wave propagation, time convergence, energy behavior,
forced cutback, and restart equivalence across supported platforms.

## Gate 4 — nonlinear solids and material state

Scope: finite-strain kinematics, Neo-Hookean and Mooney–Rivlin models,
quadrature state, J2 plasticity, creep, multiple material regions, and
consistent tangents.

Evidence: material-point and element paths, uniaxial/biaxial loading,
objectivity, volume response, rollback, and public benchmarks.

## Gate 5 — higher order, mixed, and near-incompressible

Scope: P2 fields and geometry, DOF transformations, mixed displacement-pressure
spaces, near-incompressibility, C3D10H-equivalent capability targets, and
multiple cell-block topologies. Naming equivalence alone is never acceptance.

## Gate 6 — MPI and scale

Scope: partitioning, ghost entities, distributed DOFs and assembly, global
state identity, parallel checkpoint/output, and replaceable partitioners.

Evidence: 1/2/4/8-rank consistency, partition independence, restart across
partition layouts, global energy/balance, and scaling studies on Linux.
Windows capability is reported explicitly per MPI/provider combination.

## Gate 7 — GPU and domestic computing ecosystem

Scope: replaceable vectorized CPU and GPU kernels, domestic CPU/GPU,
domestic Linux distributions, compilers, MPI, and sparse solvers. The goal is
independent interfaces, builds, and critical paths—not rejection of every
general-purpose open-source dependency.

## Platform release tiers

- Tier 1: native Windows x86_64, macOS arm64/x86_64, and Linux x86_64;
  release-blocking.
- Candidate: Linux aarch64 and Windows arm64 after reliable hosted or
  self-hosted runners exist.
- Provider-specific capabilities (MPI, PETSc, GPU) are reported separately;
  an unavailable provider must not make the serial NumPy reference kernel
  unavailable.
