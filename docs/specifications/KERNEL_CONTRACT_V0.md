# Kernel Contract v0 — draft

Status: design draft; no compatibility guarantee before Gate 1.

## Purpose

The contract separates AgentFEM engineering intent from a numerical backend.
It must be serializable, typed, versioned, deterministic, and free of FEniCSx
objects. Version 0 intentionally describes only the shape of the boundary.

## Request envelope

```text
KernelSolveRequest
  contract_version: "0.1-draft"
  request_id: stable string
  study: StudySpec
  mesh: MeshSpec
  fields: [FieldSpec]
  materials: [MaterialAssignment]
  constraints: [ConstraintSpec]
  loads: [LoadSpec]
  procedure: ProcedureSpec
  outputs: OutputSpec
  provenance: RequestProvenance
```

Arrays use explicit dtype, shape, index base (always zero internally), entity
dimension, and ordering. User labels are preserved separately from internal
indices. Units and coordinate-system semantics are mandatory at the lowering
boundary, even if the kernel operates on normalized numbers.

## Response envelope

```text
KernelSolveResult
  contract_version
  request_id
  status: success | failed | interrupted
  backend: BackendIdentity(name="native", version, revision)
  capabilities: exact maturity statements
  quantities: typed scalar/tensor values
  fields: values + association + component convention
  histories: abscissa + values
  artifacts: portable relative paths + digest
  convergence: iterations, residuals, reasons
  verification: evidence references, never an automatic validation claim
  runtime: OS, architecture, Python, dependencies, provider identities
  warnings: structured records
```

## Error contract

Invalid model intent, unsupported capability, numerical nonconvergence,
provider failure, and internal error are distinct machine-readable categories.
Unsupported capability fails before assembly. Partial results are labeled and
cannot be mistaken for successful results.

## Capability negotiation

A backend publishes capability name, maturity, supported cell/element,
dimension, scalar type, provider requirements, platform evidence, and known
limitations. AgentFEM lowers only when every required capability matches.

## Ownership and determinism

The request owns immutable input buffers during a solve. The result owns its
buffers. State mutation occurs only inside an explicit transaction. Stable
entity identities survive reordering and partitioning; storage order does not
become scientific identity. Deterministic serial assembly is the Gate 1
baseline.

## Initial supported subset

No solve request is implemented yet. The executable reference package exposes
only the P1 triangle and quadrature mathematics needed to implement Gate 1.
