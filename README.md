# AgentFEM Native Engine

AgentFEM Native is the independently developed finite-element engine for
AgentFEM. It is an official autonomous computation backend, not a FEniCSx
fork and not a second end-user product.

The repository is at **Gate 1 performance-candidate maturity**. The locally
verified serial kernel solves two-dimensional steady scalar diffusion on
affine P1 triangles with named node, boundary, and material regions;
scalar, spatially varying, or symmetric positive-definite tensor
conductivity; deterministic COO assembly; replaceable NumPy/SciPy linear
algebra providers; balance evidence; and portable VTK output. Static problems
automatically use bounded vectorized assembly while callable fields retain the
element-by-element oracle. A standard-library C++20/C ABI spike is the leading
compiled-kernel experiment. The API remains experimental until native
three-platform CI evidence closes the gate.

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
build, and independence checks passing on each operating system; it is not
implemented as separate physics forks.

## Current evidence

Create the repository-local environment on Windows, macOS, or Linux:

```text
python tools/bootstrap_env.py
python tools/bootstrap_env.py --with-scipy  # optional sparse provider
python tools/bootstrap_env.py --agentfem /path/to/agentfem
```

For this development checkout, the active `.venv` contains editable
AgentFEM 0.3.1 from the clean local `agentfem-main-worktree` and editable
AgentFEM Native 0.3.0a1. Installing both does not make AgentFEM or FEniCSx a
Native runtime dependency; their communication remains the public contract.

Then run the reference tests without FEniCSx:

```text
python -m unittest discover -v
python tools/check_independence.py
```

Execute a portable Kernel Contract request:

```text
agentfem-native examples/steady_diffusion_request.json \
  --artifact-directory work/example --result work/example-result.json
```

On Windows PowerShell, enter the same command on one line. The command and
the Python API return a versioned, JSON-safe result envelope; failures are
structured and the command exits nonzero.

See [ROADMAP.md](ROADMAP.md), [ARCHITECTURE.md](ARCHITECTURE.md), the
[project charter](docs/charter/PROJECT_CHARTER.md), and the
[steady-diffusion specification](docs/specifications/STEADY_DIFFUSION.md).
Environment details are in
[docs/development/ENVIRONMENT.md](docs/development/ENVIRONMENT.md).
The long-term technical thesis is recorded in
[docs/charter/NATIVE_VISION_2035.md](docs/charter/NATIVE_VISION_2035.md).

## Licensing

The kernel is source-available under PolyForm Noncommercial 1.0.0. Others may
use it only for purposes permitted by that license; commercial use requires a
separate written AgentFEM commercial license. See [LICENSE](LICENSE) and
[LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md). It is not OSI-approved open
source.
