# ADR-0008: MPI identity and partitioning

Status: proposed — 2026-09-04

Global mesh entities, DOFs, and committed material state have stable identities
independent of rank count or partition layout. Partition-local indices and
ghost ownership are runtime views. Parallel checkpoint/restart must support a
different partition layout. Implementation is deferred to Gate 6.
