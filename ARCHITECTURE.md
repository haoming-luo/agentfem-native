# Architecture

## Dependency direction

```text
AgentFEM engineering language
          |
          v
backend-neutral lowering
          |
          v
Kernel Model + Solve Request
          |
          v
Mesh -> Space -> Element -> Operator
          |
          v
Assembly -> State -> Procedure
          |
          v
LinearAlgebraProvider
          |
          v
Simulation Result + Evidence
```

Dependencies point downward. Materials do not know solver implementations;
solvers do not know agents or GUIs; output and verification do not mutate the
numerical solve. Public contracts expose owned arrays and typed records rather
than FEniCSx objects.

## Layers

- `reference`: small, readable NumPy executable mathematics used as an
  independent oracle for future optimized kernels.
- `kernel`: triangle/tetrahedron topology, scalar/vector DOFs, P1 diffusion,
  T3/T4 linear-elasticity operators, deterministic assembly, and linear state
  owned by AgentFEM Native.
- `fast path`: a packaged standard-library C++20 kernel for supported static
  volume data, bounded vectorized NumPy fallback, and the readable oracle;
  every optimized path reproduces the oracle without becoming the source of
  mathematical meaning.
- `providers`: an owned dependency-free CSR/BSR/CG/Jacobi/block-Jacobi baseline,
  C++20 CSR SpMV, replaceable dense/sparse linear algebra, and future parallel runtime and hardware
  implementations. Dense NumPy is a bounded oracle, not the production default.
- `contract`: versioned JSON request/result execution and public AgentFEM
  AF-IR lowering boundary.
- `verification`: tests and evidence that consume public results without
  becoming part of the solver.

The procedure layer now owns centered explicit and Newmark average-acceleration
linear dynamics, fixed-DOF elimination, reaction and external-work/energy
ledgers, immutable histories, digest-bound restart, progress, cancellation, and
budgets. The Agent control surface revalidates plans before execution and emits
compact evidence receipts without coupling Native to AF-IR. A deterministic
internal JSON bundle carries owned meshes, named sets, and result fields.
Experimental nonlinear material points remain
separate from admitted element and procedure capabilities.

The current implementation deliberately keeps these small enough to audit.
C++20 is the selected production compiled language; Rust remains a
non-shipping comparison track, and both must reproduce the same reference.

## Cross-platform rule

Platform differences are isolated behind providers or packaging. Scientific
semantics, numbering rules, file schemas, tolerance policies, and the Kernel
Contract remain identical across native Windows, macOS, and Linux. Code uses Python APIs and
`pathlib`, does not require a POSIX shell, treats filesystem case differences
explicitly, and never embeds absolute developer paths in artifacts.

The package depends only on Python, NumPy, and the platform C/C++ runtime. Its
C++20 extension exposes C ABI 1.2 for P1 diffusion, T3/T4 volume assembly, and
CSR SpMV, and uses CPython's 3.11+ stable ABI;
it does not use a third-party binding framework. PETSc, SciPy, MPI, GPU, and
vendor solvers are optional providers and cannot become import-time
requirements of the reference path.

See `docs/adr/` for decisions and `docs/specifications/KERNEL_CONTRACT_V0.md`
for the boundary presented to AgentFEM.
