# Production-kernel language and library evaluation

## Principle

Keep one readable NumPy oracle and compare optimized implementations against
it bit-for-bit where appropriate and tolerance-by-tolerance where floating-
point reordering is intentional. Performance code may be replaced; the Kernel
Contract and mathematical evidence remain stable.

## Candidate tracks

| Track | Strengths to measure | Risks to measure |
|---|---|---|
| C++20 + thin binding | HPC, MPI, vendor compiler and solver ecosystem | memory safety, ABI/build complexity |
| Rust + thin binding | memory/thread safety, Cargo tooling, native OS targets | scientific/MPI/GPU ecosystem maturity |
| Python + compiled kernels | fastest iteration and close oracle comparison | tool lock-in, packaging and accelerator fragmentation |
| Portable layer such as Kokkos | CPU/GPU portability and vendor backends | abstraction cost, toolchain size, platform coverage |

## Benchmark ladder

1. P1 basis/geometry throughput over large batches.
2. Diffusion element stiffness and load throughput.
3. Deterministic COO emission and duplicate reduction.
4. Structured-mesh assembly throughput and peak memory.
5. Repeated linear solves through provider interfaces.
6. Native wheel build/install/test on Windows, macOS, and Linux.
7. Compiler sanitizers, failure diagnostics, and reproducible build metadata.
8. MPI and GPU spike only after the serial results match the NumPy oracle.

Each benchmark stores hardware, OS, architecture, compiler, flags, dependency
versions, commit, numerical digest, wall time, and peak memory. A result without
reproducibility metadata does not influence the decision.

`benchmarks/reference_diffusion.py` is the first machine-readable NumPy
assembly baseline. Optimized spikes must consume the same mesh and reproduce
the same algebraic checks before performance comparisons are accepted.

## Permissive-library policy

Preferred inbound licenses are BSD-2-Clause, BSD-3-Clause, MIT, Apache-2.0,
and similarly permissive terms compatible with commercial relicensing of
AgentFEM Native. Examples for evaluation—not automatic dependencies—include:

- PETSc, currently BSD-2-Clause: https://petsc.org/main/install/license/
- Kokkos/Kokkos Kernels, currently Apache-2.0 with LLVM exception:
  https://kokkos.org/about/releases/
- nanobind for a possible C++/Python boundary, distributed as BSD:
  https://nanobind.readthedocs.io/en/latest/packaging.html
- Rust's native target support is evaluated from its official platform list:
  https://doc.rust-lang.org/rustc/platform-support.html

Using a permissive finite-element library to perform Native discretization or
assembly would undermine the autonomous-kernel mission even if legally
allowed. Permissive general-purpose runtime, binding, portability, sparse
algebra, and hardware libraries may be approved after provenance and benchmark
review.

## Decision point

Select the production language in a later ADR after the serial diffusion
benchmark suite and three-platform packaging spikes exist. Until then, no
optimized track may replace the NumPy oracle or expand public semantics.
