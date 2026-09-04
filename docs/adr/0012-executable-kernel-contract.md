# ADR-0012: executable Kernel Contract and AgentFEM extension

Status: accepted experimentally — 2026-09-04

AgentFEM Native exposes a versioned JSON boundary before binding to a compiled
production language. This prevents Python object identity, a particular sparse
library, or FEniCSx types from becoming the backend contract.

Contract 0.1 implements one narrow, executable steady-diffusion subset. It has
a bundled JSON Schema, deterministic JSON result envelope, addressable errors,
portable artifact paths, runtime/provider identity, and a cross-platform CLI.
Contract versions change when compatibility breaks.

The external AgentFEM adapter calls the public `to_ir()` boundary. Because
AF-IR 0.1's current mesh summary is not sufficient to reconstruct a numerical
model, the adapter requires an explicit `root.native_kernel` extension. It
must fail clearly when that extension is absent; it must not reach into private
AgentFEM or FEniCSx implementation objects.
