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
- `kernel`: mesh, topology, P1 elements, diffusion operators, deterministic
  assembly, and future state semantics owned by AgentFEM Native.
- `fast path`: bounded vectorized CPU batches today; future compiled kernels
  reproduce the oracle without becoming the source of mathematical meaning.
- `providers`: replaceable dense/sparse linear algebra now, and future
  parallel runtime and hardware implementations.
- `contract`: versioned JSON request/result execution and public AgentFEM
  AF-IR lowering boundary.
- `verification`: tests and evidence that consume public results without
  becoming part of the solver.

The current implementation deliberately keeps these small enough to audit;
future optimized C++20/Rust/compiled kernels must reproduce this reference.

## Cross-platform rule

Platform differences are isolated behind providers or packaging. Scientific
semantics, numbering rules, file schemas, tolerance policies, and the Kernel
Contract remain identical across native Windows, macOS, and Linux. Code uses Python APIs and
`pathlib`, does not require a POSIX shell, treats filesystem case differences
explicitly, and never embeds absolute developer paths in artifacts.

The Python layer depends only on Python and NumPy wheels. The experimental
C++20 spike depends only on the C++ standard library and exposes a narrow C
ABI; it is not yet part of release packages. PETSc, MPI, GPU,
and vendor solvers are optional future providers and cannot become import-time
requirements of the reference package.

See `docs/adr/` for decisions and `docs/specifications/KERNEL_CONTRACT_V0.md`
for the boundary presented to AgentFEM.
