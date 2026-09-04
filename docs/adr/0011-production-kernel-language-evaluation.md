# ADR-0011: benchmark before selecting the production-kernel language

Status: superseded by ADR-0016 — 2026-09-04

Python/NumPy remains the executable mathematical reference, not a commitment
to the final high-performance kernel. Before Gate 2 implementation grows, use
the same P1 diffusion element and assembly workloads to benchmark:

1. C++20 with a thin Python binding;
2. Rust with a thin Python binding;
3. Python orchestration with an independently compiled numerical kernel;
4. portable performance layers only where they improve measured CPU/GPU and
   domestic-platform coverage.

The decision uses measured throughput, memory, deterministic equivalence,
native Windows/macOS/Linux wheel construction, MPI/GPU integration, compiler
availability, domestic CPU/GPU support, binding complexity, sanitizer/tooling
quality, maintenance cost, and AI-assisted code-review reliability.

C++20 is the initial performance-spike baseline because of mature scientific
and HPC interoperability. Rust remains a full candidate because of memory
safety and first-class native platform tooling. Neither is selected until the
benchmark evidence and packaging prototypes are recorded.

Third-party components are preferred under permissive licenses such as BSD,
MIT, or Apache-2.0. Every transitive dependency is inventoried. Copyleft,
source-available, proprietary, or unclear dependencies require an explicit ADR
and cannot enter the default Native kernel accidentally.
