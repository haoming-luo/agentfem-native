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

The benchmark suite now contains the oracle baseline, reference/vectorized
comparison, large-scale vectorized guard, and C++20 shared-library comparison.
Optimized spikes consume the same owned mesh and reproduce complete COO/load
data before their timings are admitted.

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
- 131,072 anisotropic cells: the standard-library C++20/C ABI spike was about
  33 times faster than vectorized NumPy and emitted identical COO indices,
  matrix data, and load data on this structured case.

These are local kernel measurements and do not establish end-to-end solver or
cross-platform superiority. They do establish enough headroom to make C++20
the leading compiled experiment. Rust remains required comparative evidence
before final lock-in.

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

## Decision point

C++20 leads the current experiment under ADR-0014. Final selection still waits
for hosted three-platform builds, sanitizers, compiled-wheel packaging, a Rust
comparison, and end-to-end sparse solve evidence. No optimized track may
replace the NumPy oracle or independently expand public semantics.
