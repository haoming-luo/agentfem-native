# Gate 1 steady-diffusion slice evidence

Date: 2026-09-04. Maturity: `verified` locally on macOS; three-platform CI
evidence pending.

## Designed

- Owned zero-based triangle mesh and named node/boundary sets.
- Provider-neutral deterministic COO assembly.
- Scalar steady-diffusion weak form, essential/natural boundary conventions,
  result balance fields, and portable VTK output.

## Implemented

- Structured unit-square triangle meshes.
- P1 element stiffness, source, and boundary-flux integration.
- Global COO assembly with duplicate triplets.
- Dirichlet elimination and initial dense NumPy linear provider.
- Nodal solution, residual, reaction, energy, and load-balance result data.
- Cross-platform-safe atomic ASCII VTK writer.

## Tested

- Environment: isolated Python 3.12.13 with NumPy 2.3.5 on macOS arm64;
  DOLFINx, UFL, Basix, and FFCx absent.
- 42 source and installed-wheel tests passed in the final local run.
- Manufactured nodal RMS errors for 4/8/16 subdivisions were
  `1.97709e-2`, `5.72920e-3`, and `1.53241e-3`, giving observed orders
  `1.78697` and `1.90253`.
- Built and independently installed
  `agentfem_native-0.1.0a1-py3-none-any.whl`; SHA-256
  `505483cc480184e8c09f4c24ea59776f2ba83a89f9624c99c9d4f6878b3be4ba`.
- The installed artifact passed all 42 tests and the forbidden-import scan.

## Cross-validated

- Analytical exact solution `u=x` for the mixed-boundary unit-square problem.
- Manufactured `sin(pi x) sin(pi y)` solution demonstrates the expected nodal
  convergence trend.
- No FEniCSx comparison was needed or used as an oracle for this slice.

## Not yet resolved

- Remote native Windows, macOS, and Linux CI evidence.
- Variable/tensor conductivity and material-region assignment.
- Scalable SciPy/PETSc provider implementation.
- AgentFEM external lowering adapter and versioned serialized request schema.

This is the first Gate 1 vertical slice, not completion of all Gate 1 scope.
