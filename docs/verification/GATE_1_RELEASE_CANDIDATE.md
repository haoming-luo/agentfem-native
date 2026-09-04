# Gate 1 release-candidate evidence

Date: 2026-09-04. Version: `0.2.0a1`. Maturity: locally verified on native
macOS arm64; hosted native Windows and Linux evidence pending.

## Implemented scope

- Owned zero-based 2D P1 triangle meshes and named node, boundary, and cell
  sets.
- Scalar, spatially varying scalar, and symmetric positive-definite tensor
  conductivity, including nonoverlapping material-region overrides.
- Degree-2 domain and two-point boundary quadrature, deterministic COO
  assembly, Dirichlet elimination, reactions, balance, and potential energy.
- Replaceable dense NumPy and optional SciPy COO-to-CSR direct-solve providers
  with runtime provider identity and explicit unavailable-provider failures.
- Kernel Contract 0.1 request/result envelopes, two bundled JSON Schema
  2020-12 descriptions, addressable errors, safe relative artifact paths,
  SHA-256 artifact identity, and a cross-platform JSON CLI.
- External AgentFEM public-AF-IR lowering prototype using an explicit
  `root.native_kernel` executable extension.
- Repository-local `.venv` bootstrap supporting Windows, macOS, and Linux.

## Scientific evidence

- Exact `u=x` analytical mixed-boundary solution and affine patch tests.
- Spatial `k=1+x` exact solution with consistent source/flux.
- Anisotropic `K=diag(2,5)` exact linear solution.
- Two-region `k=1/2` layered solution with exact interface value `2/3`.
- Manufactured `sin(pi x)sin(pi y)` nodal convergence with previously
  recorded orders `1.78697` and `1.90253` for 4/8/16 subdivisions.
- Orientation, cell-order, node-renumbering, constant-null-mode, balance,
  failure, and artifact tests.
- SciPy 1.17.1 sparse-provider result matched the NumPy provider in the
  independent provider test run (`5 passed, 1 intentionally skipped`).

## Local build and execution evidence

- Environment: native macOS arm64, Python 3.12.13, NumPy 2.3.5.
- Source run: 69 tests passed; the SciPy-only test was skipped in the minimal
  environment by design.
- Installed-wheel run: 69 tests passed with the same intentional skip, and
  the forbidden-import scan passed.
- Wheel: `agentfem_native-0.2.0a1-py3-none-any.whl`, SHA-256
  `6f8e4ed602e4ef76d22e8769787c617799dc217d07b8338de4f7042eae82fab8`.
- Source distribution SHA-256:
  `360b8dd7f7da0e096690b91049e7f9eb323c2d9199ef304c3604ccc0e76ec9b9`.
- Installed CLI example returned nodal values `[0.0, 1.0, 1.0, 0.0]` and a
  content-addressed VTK artifact.
- Current NumPy reference assembly baseline at resolution 32: 1,089 nodes,
  2,048 cells, 18,432 duplicate COO entries, and median 0.1841 seconds over
  three local runs. This is a reproducible comparison point, not a production
  performance claim.

## Environment integration evidence

The project `.venv` contains editable AgentFEM Native 0.2.0a1, editable
AgentFEM 0.3.1 from clean commit `5dbeee0`, h5py 3.16.0, and mpi4py 4.1.2;
`pip check` reports no broken requirements. Importing AgentFEM at the public
package boundary did not import DOLFINx, UFL, Basix, or FFCx. The Native
runtime itself still depends only on NumPy; SciPy and AgentFEM remain optional
and do not alter independent discretization ownership. MPI execution is not a
Gate 1 claim and was not validated in the restricted local sandbox.

## Remaining release blockers

- Run the configured workflow on hosted native Windows, macOS, and Linux and
  preserve the job URLs/artifact digests. WSL cannot substitute for Windows.
- Add the executable portable extension to AgentFEM's public AF-IR export and
  test end-to-end lowering in the AgentFEM repository.
- Benchmark and package the C++20, Rust, and compiled-kernel candidates before
  selecting a production implementation language.

Until the first two items close, this is a Gate 1 release candidate rather
than a cross-platform Gate 1 completion claim.
