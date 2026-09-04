# ADR-0015: package C++20 through a CPython Stable ABI buffer boundary

Status: accepted — 2026-09-04

AgentFEM Native ships the owned C++20 P1 accelerator inside platform wheels.
The binding uses only CPython's Limited API at the Python 3.11 floor and the
standard buffer protocol. It does not add pybind11, nanobind, Cython, or the
NumPy C API as a build/runtime dependency.

Python owns and validates contiguous NumPy input/output arrays. The extension
acquires typed buffers, checks all byte counts with overflow protection, holds
the exporting objects alive, releases the GIL during the non-throwing C ABI
call, and releases every buffer afterward. The C++ kernel never owns Python
objects. ABI version `1.0` is queried at runtime and recorded in result
evidence.

Static scalar/tensor P1 volume data, including per-cell material tensors, may
use `native`. `auto` prefers the compiled kernel when its ABI matches, then
falls back to bounded vectorized NumPy; callable fields stay on `reference`.
The oracle and vectorized path remain distributable source and verification
implementations, so loss of an accelerator cannot change mathematical meaning.

Platform wheels are tagged `cp311-abi3` and remain OS/architecture specific.
CI must build and import them natively on Windows, macOS, and Linux and compare
their complete outputs, not merely confirm that compilation succeeded.
