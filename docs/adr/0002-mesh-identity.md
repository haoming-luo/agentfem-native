# ADR-0002: mesh ownership and indexing

Status: accepted as design direction — 2026-09-04

Native will own topology and geometry buffers. Internal indices are zero-based,
contiguous storage locations; imported user labels remain separate stable
identities. Entity identity must not depend on node order, partition, memory
address, or provider type. Exact schemas wait for the Gate 1 mesh slice.
