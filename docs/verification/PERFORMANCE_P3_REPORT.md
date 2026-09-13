# Performance P3 sparse-foundation report

**Date:** 2026-09-14

**Maturity:** implemented and locally verified; hosted-platform evidence pending

## Delivered scope

- deterministic immutable scalar CSR with canonical COO duplicate reduction;
- dependency-free CSR SpMV, residual, diagonal, and storage reporting;
- symmetric strong-Dirichlet transformation without dense conversion;
- CG with Jacobi, explicit convergence/limit/breakdown reports;
- `native_sparse` as the default solve provider;
- a no-densification test that makes any production call to
  `COOMatrix.to_dense` fail;
- execution-free conservative resource planning and capability reporting.

Scalar CSR is complete for this reference foundation. BSR/block-Jacobi and
C++20 CSR SpMV were subsequently implemented. The ABI 1.3 deterministic T3/T4
CPU-parallel candidate and its honest local scaling boundary are recorded in
[`CPU_PARALLEL_P3_REPORT.md`](CPU_PARALLEL_P3_REPORT.md). Direct element-to-CSR
memory reduction, SIMD, and stronger preconditioners remain P3 work. ABI 1.3 passed hosted Tier-1
acceptance in run `34752687326`; P3 as a whole is still active.

The subsequent reusable sparse-pattern reference candidate is recorded in
[`SPARSE_PATTERN_P3_REPORT.md`](SPARSE_PATTERN_P3_REPORT.md). It separates
one-time canonical graph construction from repeated numeric fill; C++ direct
fill subsequently passed ABI 1.4 Tier-1 acceptance. T3/T4 prepared assembly and
fixed-matrix repeated constrained solve are now implemented locally; their
latest evidence is recorded in
[`PREPARED_CONSTRAINED_SOLVE_REPORT.md`](PREPARED_CONSTRAINED_SOLVE_REPORT.md).
Direct element-to-CSR memory reduction remains open.

## Local numerical evidence

Host: Darwin arm64, Python 3.11.15, NumPy 2.4.6. Each local timing is the median
of five warm-process wall-time repetitions.

| Grid | DOFs | cells | CSR nnz | CSR bytes | equivalent dense bytes | dense/CSR | solve median |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32² | 1,089 | 2,048 | 7,361 | 126,496 | 9,487,368 | 75.0× | 0.00833 s |
| 64² | 4,225 | 8,192 | 29,057 | 498,720 | 142,805,000 | 286.3× | 0.03204 s |
| 128² | 16,641 | 32,768 | 115,457 | 1,980,448 | 2,215,383,048 | 1,118.6× | 0.15474 s |

The benchmark problem is the analytical unit-square `u=x` diffusion case. At
all three sizes, the recorded maximum analytical error, free residual norm, and
global load-plus-reaction balance are zero at the serialized precision. CG took
32, 64, and 128 iterations respectively. That unusually exact behavior belongs
to this structured analytical case and is not generalized to arbitrary systems.

The exact raw repetitions, environment, memory counts, and caveats are in
`benchmarks/results/2026-09-07-darwin-arm64-sparse-foundation.json`.

## Claim boundary

- CSR storage numbers are exact array payload sizes, excluding Python object
  headers and temporary sorting/solver allocations.
- Plan peak bytes are conservative formula estimates, not measured process RSS.
- Current conversion and iterative algebra use NumPy operations orchestrated by
  readable Python; this report is correctness and memory foundation evidence,
  not the final C++20 throughput claim.
- No Windows or Linux performance is inferred from macOS measurements.
- Dense conversion remains available only as a diagnostic/oracle operation.
