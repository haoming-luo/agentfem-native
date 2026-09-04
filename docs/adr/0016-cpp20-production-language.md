# ADR-0016: select C++20 as the production kernel language

Status: accepted — 2026-09-04

C++20 is selected as AgentFEM Native's primary production-kernel language.
Rust is retained as a non-shipping cross-check and research path, but is no
longer a language-selection gate for the next mechanics milestones.

The decision follows executable evidence on the same P1 C ABI. C++20, Rust,
and vectorized NumPy emitted identical COO indices, matrix values, and loads at
all recorded sizes. On the local macOS arm64 host, C++20 was approximately 20%
to 21% faster than Rust at 8,192, 131,072, and 524,288 cells. The packaged C++
runtime was 13.85 times faster than vectorized NumPy at 524,288 cells, while an
end-to-end 32,768-cell SciPy sparse case was 1.28 times faster with identical
solution, reaction, and energy.

Performance alone is not decisive. C++20 already has a dependency-free
`cp311-abi3` wheel path, a true versioned C ABI, sanitizer evidence, direct
interoperability with mature sparse/MPI/GPU/vendor ecosystems, and native
compiler support across all Tier-1 systems. Selecting it removes the cost and
risk of maintaining two production languages.

The risks of C++ are controlled by narrow buffer APIs, checked sizes and
indices, no ownership transfer, no exceptions across the ABI, sanitizers,
strict compiler warnings, deterministic full-output differential tests, and a
permanent readable NumPy oracle. Rust can be reconsidered for an isolated
stateful subsystem only if measured safety or performance evidence exceeds the
cost of a second toolchain and implementation.
