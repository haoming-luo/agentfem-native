# Kernel Contract 0.1

Status: implemented experimental Gate 1 boundary. Compatibility changes before
1.0 require an explicit version change.

## Identity and transport

Requests use `contract = "agentfem.native-kernel-request"` and
`contract_version = "0.1.0"`. The representation is JSON-safe, owns no
FEniCSx objects, and uses zero-based indices and portable forward-slash
artifact paths. Machine-readable request and result schemas are bundled as
`agentfem_native/schemas/kernel-request-0.1.schema.json` and
`agentfem_native/schemas/kernel-result-0.1.schema.json` using JSON Schema
2020-12.

The `agentfem-native` command accepts a UTF-8 request file and returns one JSON
result. Success exits with status 0; any input, capability, provider, or model
failure exits with status 2. Python callers use `run_kernel_request`.

## Implemented request subset

- study: 2D, linear-static heat transfer;
- mesh: owned points, P1 triangle connectivity, named node/boundary/cell sets;
- physics: constant scalar or 2-by-2 conductivity and constant source;
- conditions: constant Dirichlet values and outward Neumann fluxes;
- materials: named cell-set conductivity overrides;
- procedure: steady diffusion with `numpy`, `scipy`, or `auto` provider, and
  `auto`, `reference`, `vectorized`, or `native` assembly selection;
- outputs: optional nodal legacy VTK artifact beneath an explicit root.

Python's direct problem API additionally accepts spatial conductivity and
source callables. They are intentionally excluded from JSON because executable
code is not a portable data contract.

`procedure.assembly = "auto"` is conservative: static scalar/tensor volume
data use the packaged native kernel when its ABI matches, then bounded
vectorized assembly as the fallback; callable volume fields remain on the
element-by-element reference path. Explicit optimized selection rejects
unsupported data instead of silently changing its meaning. The result runtime
records the actual assembly path and native ABI identity.

## Result and failures

A successful result contains contract and backend identities, request ID,
capabilities, balance/energy quantities, nodal field values, artifact relative
paths and SHA-256 digests, runtime OS/architecture/dependency/provider identity,
and warnings. A failed result contains a stable error `code`, human-readable
`message`, and JSON-style `path`.

The result reports computation, not automatic scientific validation.
`verified_local` means the capability has local repository evidence; it does
not claim that a user's individual model is validated.

## AgentFEM lowering prototype

`lower_agentfem_ir` consumes the public `agentfem.af-ir` 0.1 model envelope.
The current AgentFEM runtime mesh summary is not reconstructable, so executable
data must be present in `root.native_kernel`; absence fails explicitly at that
path. `lower_agentfem_model` calls only the public `to_ir()` and optional
`as_dict()` methods and never imports AgentFEM or a FEniCSx backend.

The next integration step is to make the portable extension an official
AgentFEM export without weakening the independent Native ownership boundary.
