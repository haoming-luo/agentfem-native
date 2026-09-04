# Cross-platform policy

Native Windows, macOS, and Linux are core differentiators and first-class
targets.

## Release acceptance

For every release candidate, `ubuntu-latest`, `windows-latest`, and
`macos-latest` must:

1. install from the source distribution and wheel in a clean environment;
2. run the identical mathematical and public-contract tests;
3. pass forbidden-dependency and SPDX scans;
4. create and reload portable evidence using relative paths;
5. report numerical tolerances and provider capabilities explicitly.

Architecture-specific and provider-specific claims require their own runners
and evidence, but an optional unavailable provider cannot remove the serial
Native engine from any Tier-1 operating system.

## Engineering constraints

- No shell scripts are required for install, tests, or core operation.
- Paths use `pathlib`; serialized paths use portable relative POSIX spelling.
- Do not depend on case-sensitive filenames, symlinks, executable bits, fork,
  Unix signals, or `/tmp` semantics in public behavior.
- File replacement and locking behavior must be tested on Windows.
- Floating-point comparisons are tolerance-based and justified scientifically;
  failures are not hidden by widening tolerances per operating system.
- Optional provider discovery fails closed with an actionable capability
  report and never changes the mathematical contract.

## Packaging stages

Gate 0 began with a pure Python/NumPy wheel. ADR-0016 now selects C++20 and the
P2 candidate builds `cp311-abi3` platform wheels for manylinux, native Windows,
and macOS from the same source revision. CI must install and test the produced
wheel—not only the checkout—and record toolchain metadata. Reproducible binary
provenance and an SBOM remain release-gate work.
