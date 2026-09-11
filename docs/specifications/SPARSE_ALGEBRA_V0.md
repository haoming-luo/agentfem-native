# Native sparse algebra specification 0.1

Status: implementation target for performance milestone P3.

## Purpose and ownership

Native sparse algebra is the dependency-free production baseline between
finite-element assembly and replaceable high-performance providers. It owns
enough storage, constraint, residual, and iterative-solve behavior to execute
real problems without SciPy and without dense conversion. It is deliberately
not an attempt to reproduce industrial direct solvers or algebraic multigrid.

NumPy supplies array primitives only. It does not own the sparse data model or
solver semantics.

## Scalar CSR contract

For a matrix with shape `(m, n)`, CSR is represented by:

```text
indptr:  int64, shape (m + 1,)
indices: int64, shape (nnz,)
data:    float64, shape (nnz,)
```

The invariants are:

- dimensions are nonnegative and indexable by signed 64-bit integers;
- `indptr[0] == 0`, `indptr[-1] == nnz`, and `indptr` is nondecreasing;
- every column index is in `[0, n)`;
- column indices are strictly increasing within every row;
- every stored value is finite;
- arrays are copied on construction and exposed read-only;
- explicit numerical zeros are permitted because structural identity is
  separate from numerical cancellation.

## Deterministic COO canonicalization

COO entries are ordered lexicographically by `(row, column)` using a stable
ordering. Entries with equal coordinates are reduced in that order using
float64 addition. The canonical result contains one stored entry per coordinate
and preserves an exactly zero result as an explicit structural entry.

Input ordering therefore cannot change the CSR graph. Floating values may
differ at roundoff level when the summation order inside an equal-coordinate
group differs; finite-element assembly supplies deterministic cell/local order
and provider comparisons specify tolerances accordingly.

## Operations

For CSR matrix `A` and finite vector `x`, matrix-vector multiplication returns
`y_i = sum_j A_ij x_j`. Residual is `r = b - A x`; Euclidean norm uses float64.
Diagonal extraction requires every requested diagonal location to be present.

The readable NumPy reduction is the permanent differential oracle. Canonical
non-empty CSR matrices with at least 2,048 stored values automatically use the
C++20 ABI 1.2 SpMV when installed; smaller or extension-free cases use the
oracle. Both accumulate each row in stored-column order and must agree within
documented floating-point tolerances without densification.

Memory reporting is exact for owned CSR arrays:

```text
storage_bytes = indptr.nbytes + indices.nbytes + data.nbytes
```

It excludes Python object headers and temporary solver work vectors, which are
reported separately by benchmarks.

## Strong Dirichlet transformation

Given distinct constrained indices `C` and values `g`, form an algebraically
equivalent symmetric system:

```text
b' = b - A[:, C] g
A'[C, :] = 0
A'[:, C] = 0
A'[c, c] = 1 and b'[c] = g_c for every c in C.
```

Missing diagonal entries are inserted structurally before transformation. The
operation returns a new CSR matrix and right-hand side; the input is immutable.
For symmetric positive-definite `A` on the free subspace, `A'` is symmetric
positive definite.

## Conjugate gradient and Jacobi

The baseline CG algorithm solves symmetric positive-definite systems. Optional
Jacobi preconditioning uses `M^-1 r = r / diag(A)` and requires finite strictly
positive diagonal entries.

With right-hand side `b`, the stopping threshold is:

```text
max(atol, rtol * ||b||_2).
```

The report records convergence, reason, iteration count, initial residual,
final residual, threshold, and configured tolerances. The following paths are
distinct and machine-readable:

- initial residual already satisfies the threshold;
- converged after one or more iterations;
- iteration limit reached;
- non-positive/non-finite curvature indicates breakdown or an inadmissible
  matrix;
- non-finite input, iterate, or residual is rejected.

The production provider raises on nonconvergence and includes the report in a
successful outcome. Lower-level CG may return a nonconverged report so tests and
future procedures can inspect controlled failure.

## BSR contract

Mechanics uses node-major, component-minor interleaved DOFs. BSR groups a
canonical scalar CSR matrix into square dense blocks with ordered block columns,
immutable float64 block data, and int64 pointers/indices. Scalar dimensions must
be divisible by block size. BSR matvec, diagonal-block extraction, scalar-CSR
round trip, exact storage reporting, and missing-diagonal failures are owned.

Block-Jacobi inverts only the small symmetric positive-definite diagonal blocks
and applies them inside CG. A singular or non-positive diagonal block fails
before iteration. Scalar CSR remains the canonical constraint representation;
BSR is not claimed merely from interleaved numbering.

## Acceptance evidence

- malformed shapes, pointers, columns, ordering, and non-finite data fail;
- COO duplicates and empty rows canonicalize exactly;
- matvec, diagonal, residual, and constraint transformation match small dense
  calculations without using dense conversion in implementation;
- CG solves independently constructed SPD systems and reports initial,
  convergence, limit, and breakdown paths;
- diffusion solution, reactions, energy, and residual match dense NumPy and
  optional SciPy providers;
- a no-densification guard proves Native production solve does not call
  `COOMatrix.to_dense`;
- memory growth is O(nnz), with raw evidence for representative meshes.
