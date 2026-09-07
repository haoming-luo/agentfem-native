# Native P1/T3 C ABI 1.1

Status: ABI 1.1 is implemented in the packaged C++20 accelerator. The Rust
comparison spike implements only the ABI 1.0 diffusion subset.

## Version and ownership

`afn_p1_abi_version()` returns `0x00010001`, encoded as major 1 and minor 1.
An incompatible major version must not be called. The caller owns every input
and output buffer for the full duration of a call; the kernel retains no
pointer, allocates no output, invokes no callback, and throws no exception
across the boundary.

All scalar values are IEEE-754 binary64, indices are signed 64-bit integers,
and counts use the platform C `size_t`. Buffers must be naturally aligned,
contiguous, non-overlapping, and at least as long as the counts imply:

- points: `node_count * 2` values in node-major `(x, y)` order;
- cells: `cell_count * 3` zero-based node indices;
- conductivity: four row-major values globally, or `cell_count * 4` values
  for the per-cell function;
- rows, columns, data: `cell_count * 9` values;
- load: `node_count` values.

ABI 1.1 adds `afn_t3_elasticity_assemble_cells`. Its points and cells use the
same layouts. Constitutive data contains `cell_count * 9` row-major values;
body force contains two global values; COO outputs contain `cell_count * 36`
entries using six node-major vector DOFs per cell; load contains
`node_count * 2` values. Thickness is one positive binary64 scalar.

The Rust spike participates only in diffusion-kernel language comparisons. It
does not advertise or provide the ABI 1.1 T3 entry point; production C++ is the
sole shipping native implementation at this milestone.

## Numerical contract

For each affine P1 triangle, the kernel computes analytical physical basis
gradients, area, `area * G K G^T`, and the exact constant-source load. COO
output order is cell-major, then local row, then local column. Reversed valid
orientation changes neither physical bilinear form nor load.

Conductivity must be finite, symmetric within the documented floating-point
tolerance, and positive definite. Coordinates and source must be finite;
connectivity must be in range and each triangle nondegenerate.

T3 computes `thickness * area * B^T D B` and the exact constant body-force
load. Every per-cell constitutive matrix must be finite, symmetric, and positive
definite. T3 COO ordering is cell, local vector row, then local vector column.

## Failure contract

Status codes are:

| Code | Meaning |
|---:|---|
| 0 | success |
| 1 | null pointer, empty mesh, or empty cell set |
| 2 | invalid conductivity |
| 3 | invalid connectivity, size range, or degenerate geometry |
| 4 | non-finite coordinate or source |

Any nonzero status invalidates every output buffer, including entries written
before the failure. Callers must discard them. Python converts these statuses
to stable exceptions before constructing a public matrix result.

## Compatibility rule

Minor versions may add functions or diagnostics without changing existing
layout or meaning. A caller may use an older minor version only for functions
already present in that version. Any buffer-layout, scalar-width, ordering,
status, or mathematical-semantics change requires a new major version. Both
C++20 and Rust implementations consume this specification; neither
implementation defines it alone.
