# Development environment

AgentFEM Native uses one repository-local `.venv` on native Windows, macOS,
and Linux. The directory is ignored by Git and can always be rebuilt.

## Create or refresh

From the repository root:

```text
python tools/bootstrap_env.py
```

Add the optional SciPy sparse provider:

```text
python tools/bootstrap_env.py --with-scipy
```

Install release and lint tooling when developing the package itself:

```text
python -m pip install --editable ".[dev,scipy]"
```

Install AgentFEM into the same environment from a local checkout:

```text
python tools/bootstrap_env.py --agentfem /path/to/agentfem
```

`--agentfem` also accepts a package requirement instead of a path. A local
path is installed editable so changes in both repositories are visible
without reinstalling.

The repository's own `agentfem-native` package is always installed editable
into `.venv`; the command-line entry point and imports therefore exercise the
current checkout, including the C++20 extension when a supported compiler is
available. Passing `--agentfem` adds the separate public AgentFEM package to
that same environment.

## Activate

Windows PowerShell:

```text
.venv\Scripts\Activate.ps1
```

macOS or Linux:

```text
source .venv/bin/activate
```

Activation is optional; automation may call `.venv`'s Python directly. Never
commit `.venv`, hard-code a developer's absolute path in project data, or
assume WSL is the Windows runtime.

## Verify

```text
python -m unittest discover -v
python tools/check_independence.py
python -m agentfem_native.cli examples/steady_diffusion_request.json --artifact-directory work/example --result work/example-result.json
```

The regular editable install builds the packaged C++20 extension. The separate
CMake workflow in `native_spikes/cpp20/README.md` runs the C ABI, C-header, and
sanitizer tests. A C++20 compiler is therefore required when building from
source; release wheels carry the compiled extension.

The minimal Native runtime requires only NumPy. AgentFEM and SciPy are
integration/provider dependencies; Native source must remain importable and
testable without either one.

The local 2026-09-04 milestone environment contains Python 3.12.13, NumPy
2.3.5, SciPy 1.18.1, editable AgentFEM Native 0.4.0a1, and editable AgentFEM
0.3.1. SciPy was installed from its official macOS arm64 wheel only after its
published SHA-256 digest was verified; `pip check` and provider-equivalence
tests pass.
