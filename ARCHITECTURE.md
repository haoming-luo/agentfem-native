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
- `kernel`: future mesh, topology, spaces, elements, operators, assembly, and
  state semantics owned by AgentFEM Native.
- `providers`: future replaceable sparse linear algebra, parallel runtime, and
  hardware implementations.
- `adapters`: future versioned lowering from AgentFEM and result conversion.
- `verification`: tests and evidence that consume public results without
  becoming part of the solver.

Only the reference layer exists in code today.

## Cross-platform rule

Platform differences are isolated behind providers or packaging. Scientific
semantics, numbering rules, file schemas, tolerance policies, and the Kernel
Contract remain identical across Linux and Windows. Code uses Python APIs and
`pathlib`, does not require a POSIX shell, treats filesystem case differences
explicitly, and never embeds absolute developer paths in artifacts.

The reference layer depends only on Python and NumPy wheels. PETSc, MPI, GPU,
and vendor solvers are optional future providers and cannot become import-time
requirements of the reference package.

See `docs/adr/` for decisions and `docs/specifications/KERNEL_CONTRACT_V0.md`
for the boundary presented to AgentFEM.
