# Native internal interchange 0.1

Status: implemented for Mechanics Alpha internal studies.

The Native bundle is deterministic UTF-8 JSON identified by
`agentfem-native.bundle/0.1`. It reconstructs one owned `triangle3` or
`tetrahedron4` mesh, named node/boundary/cell sets, finite point and cell
fields, and JSON metadata. Reading always passes through the same mesh and
field validation as in-memory construction; a file is never trusted merely
because its JSON parses.

Keys are sorted, whitespace is canonicalized, non-finite JSON numbers are
forbidden, and files above 64 MiB are rejected. Writes use a sibling temporary
file followed by platform-native atomic replacement. The format uses no POSIX
shell behavior and has identical semantics on Windows, macOS, and Linux.

This is an internal reconstructable evidence format, not a long-term public
mesh standard. Large-scale binary/HDF5 interchange, schema migration, units,
coordinate systems, distributed ownership, and public compatibility policy
remain separate admission work.
