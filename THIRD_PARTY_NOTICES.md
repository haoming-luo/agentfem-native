# Third-party notices

## Runtime dependency

- NumPy — array operations. License and version are resolved from the installed
  distribution and must be captured in each release manifest.

## Optional runtime dependency

- SciPy — sparse COO/CSR representation and direct linear solve. SciPy is
  distributed under a BSD-style license. It is imported only when its provider
  is requested and does not implement Native finite-element discretization.
  Official license: https://projects.scipy.org/scipylib/license.html
  The local P2 evidence used SciPy 1.18.1; the official wheel digest and
  bundled-component license inventory are recorded in the benchmark evidence.

## Development dependencies

- pytest — test execution.
- setuptools and wheel — Python package construction.
- CPython 3.11+ Limited API — the packaged native binding boundary; no
  pybind11, nanobind, Cython, or NumPy C API is used.
- CMake and a C++20 compiler — source-build and native-test tools. The shipped
  P1 kernel itself uses only the C++ standard library and platform runtime.
- Rust 1.98.1 — development-only exact comparison implementation. It has no
  Cargo dependencies and is not shipped in the runtime wheel.

No code from FEniCSx, DOLFINx, Basix, UFL, or FFCx is included or required.
This inventory is provisional and must be regenerated for each release.

## Evaluated but not included

PETSc, Kokkos, nanobind, PyO3, OpenBLAS, and other provider or binding options
may be benchmarked later. Listing a candidate does not add it as a
dependency; its exact version, license, transitive inventory, and platform
evidence are required before adoption.
