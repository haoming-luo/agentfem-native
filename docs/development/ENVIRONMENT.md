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

Install AgentFEM into the same environment from a local checkout:

```text
python tools/bootstrap_env.py --agentfem /path/to/agentfem
```

`--agentfem` also accepts a package requirement instead of a path. A local
path is installed editable so changes in both repositories are visible
without reinstalling.

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

The minimal Native runtime requires only NumPy. AgentFEM and SciPy are
integration/provider dependencies; Native source must remain importable and
testable without either one.
