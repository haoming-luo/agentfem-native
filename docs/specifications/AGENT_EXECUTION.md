# Agent plan / execute / explain protocol

Status: implemented, not a maturity upgrade for the underlying physics.

## Purpose

AgentFEM Native exposes an explicit separation between resource planning,
execution, and explanation. An agent can inspect an `ExecutionPlan`, apply a
resource policy before allocating the finite-element system, execute only that
accepted plan, and retain an `ExecutionReceipt` with compact numerical evidence.

This protocol does not introduce a second model language and does not make
AF-IR a prerequisite for the native kernel.

## Contract

1. `plan_*` performs validation needed for path selection but does not assemble
   or solve.
2. `execute_plan(plan, request, context=...)` recomputes the plan and rejects a
   changed geometry, topology, named-set identity, provider, assembly path,
   maturity, or warning set before execution. Preassembled dynamics additionally
   bind sparse operators and initial state.
3. The execution context enforces conservative DOF and peak-memory budgets.
   Dynamics additionally enforces step budgets and cancellation at accepted
   state boundaries.
4. A successful immutable receipt holds the full typed numerical result
   separately from canonical JSON evidence. Evidence access returns a fresh
   object, so callers cannot silently invalidate its digest. The evidence
   records engine/native identities, residual or energy-balance measures,
   provider identity, and the unchanged maturity.
5. `explain_execution` is a pure projection of the receipt. It never re-runs a
   solve and never promotes `implemented` evidence to `verified` or `validated`.

## Digest boundary

The execution-plan digest binds resource shape, structural identity, provider,
assembly path, maturity, and warnings. It is not a full identity of material,
load, or callable physics values, because arbitrary Python callables cannot be
portably serialized. The evidence digest binds the accepted plan and compact
evidence for one completed execution. Persistent identity of a complete model
belongs to a future explicit, versioned model serialization contract.

## Failure semantics

A changed request is rejected as `execution.plan_mismatch`. Resource limits and
cancellation use the structured errors defined by the runtime-control
specification. Scientific input and solver failures retain their typed Python
exceptions and are not disguised as successful receipts.
