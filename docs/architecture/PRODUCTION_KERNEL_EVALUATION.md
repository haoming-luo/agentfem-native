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

The benchmark suite contains the oracle baseline, reference/vectorized
comparison, large-scale vectorized guard, interleaved C++20/Rust shared-library
comparison, packaged-runtime comparison, and end-to-end sparse solve. Optimized
paths consume the same owned mesh and reproduce complete COO/load data before
their timings are admitted.

## Current evidence and direction

On the local macOS arm64 development machine with Python 3.12.13 and NumPy
2.3.5:

- 8,192 anisotropic P1 cells: bounded vectorized assembly was approximately
  385 times faster than the element-by-element oracle; COO indices were
  identical, maximum matrix difference was `6.67e-16`, and maximum load
  difference was `1.63e-19`.
- 524,288 cells: vectorized mesh construction/validation took approximately
  0.113 seconds and assembly took approximately 0.199 seconds, producing
  4,718,592 COO entries at about 2.63 million cells/second with exact constant
  null mode and total load.
- At 8,192, 131,072, and 524,288 anisotropic cells, C++20 was respectively
  1.25, 1.27, and 1.26 times as fast as the dependency-free Rust equivalent;
  both emitted exactly identical COO indices, matrix data, and load data.
- At 524,288 cells, raw C++20 was 31.2 times as fast as vectorized NumPy, while
  the packaged public native path—including owned outputs—was 13.85 times as
  fast as the public vectorized path.
- At 32,768 cells, native assembly plus the same SciPy sparse solve was 1.28
  times as fast end to end, with identical solution, reaction, and energy.

These are local macOS arm64 measurements and do not establish cross-platform
superiority. Together with the packaged stable-ABI prototype, sanitizer tests,
and HPC/toolchain considerations, they select C++20 as the production compiled
language. Rust remains a useful non-shipping safety/design comparator.

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
- OpenBLAS is a BSD-3-Clause candidate for portable optimized BLAS across
  several CPU architectures: https://www.openmathlib.org/OpenBLAS/docs/
- CMake publishes native distributions/toolchain support for Windows, macOS,
  and Linux: https://cmake.org/download/

Using a permissive finite-element library to perform Native discretization or
assembly would undermine the autonomous-kernel mission even if legally
allowed. Permissive general-purpose runtime, binding, portability, sparse
algebra, and hardware libraries may be approved after provenance and benchmark
review.

## Decision

ADR-0016 selects C++20. The language decision is closed; release acceptance is
not. Hosted three-platform wheel builds, binary provenance, and broader kernel
coverage remain required. No optimized track may replace the NumPy oracle or
independently expand public semantics.
