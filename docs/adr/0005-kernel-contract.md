# ADR-0005: AgentFEM lowering contract

Status: proposed — 2026-09-04

AgentFEM communicates through a versioned, typed, serializable Kernel Contract.
It contains engineering intent and owned neutral data, never FEniCSx objects.
Native returns structured results, evidence, runtime provenance, warnings, and
failure categories. Integration remains external until Gate 1 proves the slice.
