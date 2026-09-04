# ADR-0006: material state transactions

Status: proposed — 2026-09-04

History-dependent state uses explicit `begin`, trial update, `commit`, and
`rollback` semantics. Rejected increments cannot leak trial state. Checkpoints
contain only committed state plus versioned identity and provenance. This is a
future Gate 3/4 contract and has no implementation claim today.
