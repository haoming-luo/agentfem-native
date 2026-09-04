# Third-party notices

## Runtime dependency

- NumPy — array operations. License and version are resolved from the installed
  distribution and must be captured in each release manifest.

## Optional runtime dependency

- SciPy — sparse COO/CSR representation and direct linear solve. SciPy is
  distributed under a BSD-style license. It is imported only when its provider
  is requested and does not implement Native finite-element discretization.
  Official license: https://projects.scipy.org/scipylib/license.html

## Development dependencies

- pytest — test execution.
- setuptools and wheel — Python package construction.

No code from FEniCSx, DOLFINx, Basix, UFL, or FFCx is included or required.
This inventory is provisional and must be regenerated for each release.

## Evaluated but not included

PETSc, Kokkos, nanobind, Rust/PyO3, and other production-kernel options
may be benchmarked later. Listing a candidate does not add it as a dependency;
its exact version, license, transitive inventory, and platform evidence are
required before adoption.
