# ADR-0017: decouple AF-IR promotion from Native scientific gates

Status: accepted — 2026-09-04

## Context

AgentFEM Native already owns an executable, versioned Kernel Contract, a
standalone CLI, and an external adapter prototype that consumes public AF-IR.
The current AF-IR mesh summary is not reconstructable without the explicit
`root.native_kernel` extension. Promoting that extension into AgentFEM would
require upstream work, while AF-IR is not a current AgentFEM development
priority.

Gate 1's numerical scope and evidence are independently complete across native
Windows, macOS, and Linux. Making its completion depend on a non-prioritized
upstream interface would couple Native scientific progress to AgentFEM release
planning rather than to finite-element capability and evidence.

## Decision

Public AF-IR promotion is an optional AgentFEM integration track and is not an
exit criterion for Native Gate 1 or an entry condition for Gate 2. The external
prototype remains supported as experimental boundary evidence. Native must not
reach into private AgentFEM objects to compensate for missing public data.

Integration resumes when AgentFEM prioritizes a reconstructable, versioned
public export and can carry its own end-to-end tests. The Kernel Contract stays
backend-neutral and independently executable in the meantime.

## Consequences

- Gate 1 is complete on its owned scientific and cross-platform evidence.
- Gate 2 basic solid mechanics is the next active Native milestone.
- No mathematical test, platform requirement, or independence rule is relaxed.
- AF-IR compatibility remains experimental and may evolve independently until
  both projects deliberately converge on a public integration contract.
