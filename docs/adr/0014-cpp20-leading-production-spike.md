# ADR-0014: C++20 leads the production-kernel experiments

Status: superseded by ADR-0016 — 2026-09-04

C++20 becomes the leading compiled-kernel experiment because the first
standard-library-only P1 assembly spike demonstrates a direct stable C ABI,
native compiler availability, exact agreement with the NumPy fast path, and a
large measured throughput margin. CMake supplies one build description for
native Windows, macOS, and Linux; no POSIX runtime is required.

The spike is intentionally small. It does not yet enter the Python wheel or
claim production safety. Before promotion it needs:

1. hosted three-OS build/test evidence and sanitizer runs;
2. material-region and callable/tabulated-field semantics;
3. ownership-safe buffers, structured diagnostics, and ABI versioning;
4. packaging through a thin binding or stable extension boundary;
5. end-to-end sparse assembly/solve benchmarks, not kernel-only timing;
6. reproducible comparisons against a Rust implementation.

Rust remains the second full candidate because its official Tier-1 host tools
cover native Windows, macOS, and Linux and its safety model is attractive for
stateful nonlinear and parallel code. The final language decision follows
evidence; the Kernel Contract, tests, and mathematical oracle remain language
neutral.
