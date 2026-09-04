# AgentFEM Native Engine

AgentFEM Native is the independently developed finite-element engine for
AgentFEM. It is an official autonomous computation backend, not a FEniCSx
fork and not a second end-user product.

The repository is at **Gate 1 reference-kernel maturity**. Its first verified
local vertical slice solves serial two-dimensional steady scalar diffusion on
affine P1 triangles, with named boundaries, deterministic COO assembly,
Dirichlet/Neumann conditions, a dense NumPy provider, balance evidence, and
portable VTK output. The API remains experimental.

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

Native Windows, macOS, and Linux are all first-class release targets. Platform
support is defined by the same public contract, scientific tests, package
build, and independence
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
[steady-diffusion specification](docs/specifications/STEADY_DIFFUSION.md).

## Licensing

The kernel is source-available under PolyForm Noncommercial 1.0.0. Others may
use it only for purposes permitted by that license; commercial use requires a
separate written AgentFEM commercial license. See [LICENSE](LICENSE) and
[LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md). It is not OSI-approved open
source.
