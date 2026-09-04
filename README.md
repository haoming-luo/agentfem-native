# AgentFEM Native Engine

AgentFEM Native is the independently developed finite-element engine for
AgentFEM. It is an official autonomous computation backend, not a FEniCSx
fork and not a second end-user product.

The repository is intentionally at **Gate 0 / reference-kernel maturity**.
Its implemented scientific scope is currently limited to the two-dimensional
reference triangle, P1 Lagrange basis functions, affine triangle maps, and
degree-1/degree-2 triangle quadrature. No PDE solver is claimed yet.

## Stable direction

```text
AgentFEM engineering model
        -> versioned Kernel Contract
        -> AgentFEM Native
        -> replaceable linear-algebra and hardware providers
```

The same AgentFEM engineering model is intended eventually to support:

```python
study.solve()
study.solve(backend="fenicsx")
study.solve(backend="native")
```

Linux and native Windows are first-class release targets. macOS remains a
supported development and verification target. Platform support is defined by
the same public contract, scientific tests, package build, and independence
checks passing on each operating system; it is not implemented as separate
physics forks.

## Current evidence

Run the reference tests without FEniCSx:

```text
python -m unittest discover -v
python tools/check_independence.py
```

See [ROADMAP.md](ROADMAP.md), [ARCHITECTURE.md](ARCHITECTURE.md), the
[project charter](docs/charter/PROJECT_CHARTER.md), and the
[P1 mathematical specification](docs/specifications/P1_TRIANGLE.md).

## Licensing status

This pre-release repository uses a draft license boundary. The intended
noncommercial and commercial terms require project-owner confirmation and
legal review before distribution. See [LICENSE](LICENSE) and
[LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md). Do not describe the project as
OSI-approved open source.
