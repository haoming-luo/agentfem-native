# AgentFEM public API audit

Audit date: 2026-09-03/04 (Asia/Shanghai). This was a read-only, clean-room
audit of public declarations and metadata, not FEniCSx implementation details.

## Repository identity

- `/Users/luo/Documents/Codex/agentfem-main-worktree` was clean on `main`, at
  `5dbeee0e8827cebefe10eb8dc979ed3c252e6fc7`, matching `origin/main`.
- `/Users/luo/Desktop/开发环境/fenicsx-code/agentfem` was clean on
  `feat/native-windows-runtime`, at
  `3ce643d5447cb1a6b4465ec038d7cd5df9b26bdf`.
- The feature branch adds Windows runtime/portability work to current
  AgentFEM. Its “native Windows runtime” terminology must not be confused with
  the autonomous AgentFEM Native Engine backend.
- System `python3` did not resolve `agentfem`.
- Existing `fenicsx-env` resolved AgentFEM 0.2.5 from its site-packages.
- `origin/main` package metadata declared AgentFEM 0.3.1.

## Engineering-language concepts retained above the backend boundary

- `Study`: analysis, physics, dimension, assumptions, time domain, linearity,
  and preferred procedure.
- `Model`: registry and lifecycle for fields, materials, named regions,
  constraints, loads, boundary models, engineering steps, output requests,
  validation, summaries, and neutral IR.
- Mesh intent: geometry/input mesh identity, regions, selectors, node/element
  sets, and user labels.
- Material intent: physical identity, behavior roles, compatibility with the
  Study, source metadata, and user parameters.
- `EngineeringStep`: inherited activation/deactivation and predefined fields,
  distinct from numerical solver controls.
- `SimulationResult`: quantities, fields, histories, artifacts, checkpoints,
  scientific inputs, verification reports, and trust level.

## Native-owned concepts below the contract

Concrete topology/indexing, reference cells, basis and quadrature, geometric
maps, FE spaces and DOFs, local kernels, sparse assembly, constraint
elimination, state transactions, time/nonlinear procedures, provider calls,
and runtime provenance belong to AgentFEM Native.

## Compatibility observations

AgentFEM already exposes backend descriptors/adapters and neutral result
concepts, but some public objects remain DOLFINx-like or provider-specific.
Native integration should therefore begin with an external lowering adapter
and a small versioned Kernel Contract. No large change to AgentFEM main is
justified before the Gate 1 diffusion slice proves the boundary.

## Implemented external prototype

Kernel Contract 0.1 and an external lowering adapter now exist in Native.
The adapter calls only public `to_ir()`/`as_dict()` methods. AF-IR 0.1's
current runtime mesh summary declares itself non-reconstructable, so the
prototype requires `root.native_kernel` to carry executable portable arrays
and conditions. The current AgentFEM repository was not modified; promotion
of that extension into the public AgentFEM export is the next integration
change and requires its own tests in the AgentFEM repository.
