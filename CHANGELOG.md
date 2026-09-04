# Changelog

## Unreleased — Gate 0 foundation

- Established the project charter, architecture, Kernel Contract draft,
  provenance policy, verification policy, and gate-based roadmap.
- Established the initial cross-platform CI verification matrix.
- Added an independent NumPy reference implementation for the P1 triangle,
  affine geometry mapping, and degree-1/degree-2 triangle quadrature.
- Added mathematical, orientation, independence, and packaging tests.

## 0.2.0a1 — Gate 1 release candidate

- Added named cell material regions, spatially varying scalar conductivity,
  and symmetric positive-definite anisotropic conductivity.
- Added replaceable NumPy dense and optional SciPy sparse providers with
  runtime provider identity.
- Implemented Kernel Contract 0.1, bundled JSON Schema 2020-12 description,
  structured failures, portable artifact digests, and cross-platform CLI.
- Added a public AgentFEM AF-IR lowering prototype with an explicit portable
  extension and no AgentFEM/FEniCSx runtime dependency.
- Added one-command repository-local environment bootstrapping for native
  Windows, macOS, and Linux.

## 0.1.0a1 — Gate 1 diffusion slice

- Accepted PolyForm Noncommercial 1.0.0 plus a separate commercial-license
  requirement.
- Defined native Windows, macOS, and Linux as equal first-class targets.
- Added owned triangle meshes, named boundary sets, P1 diffusion/source/flux
  integration, deterministic COO assembly, Dirichlet elimination, a dense
  NumPy solve provider, reaction/balance evidence, and portable VTK output.
- Added analytical, patch, reordering, manufactured-convergence, failure, and
  output tests.
