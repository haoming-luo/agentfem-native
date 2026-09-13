# Development and CI governance

Status: active from 2026-09-11.

## Objective

Preserve native Windows, macOS, and Linux acceptance while avoiding repeated
full matrices for small increments. The target is 70–90% lower private-runner
usage under comparable development activity; public visibility is not a reason
to waste shared runner capacity.

## Three verification tiers

Scientific test selection inside each tier follows
`docs/verification/SCRUM_VERIFICATION.md`. The CI tiers decide where and when
checks run; the Scrum policy decides the smallest evidence appropriate to the
change.

### Tier L — local development

- Run focused tests and lint after each small change.
- Run the complete local suite, independence scan, and applicable compiled
  contracts before a milestone commit.
- Keep benchmark sweeps and large scientific cases local or on explicitly
  assigned compute resources.
- Batch pushes around reviewable capabilities. Fine-grained local commits are
  welcome; each local commit does not need its own remote matrix.

### Tier F — fast remote development

Ordinary code pushes and pull requests run one cached Ubuntu/Python 3.12 job.
It installs the project and optional SciPy provider, checks formatting and
independence, runs the complete lightweight test suite, and executes the
portable contract smoke case. New pushes cancel superseded fast runs.

Markdown/documentation-only changes do not start CI. Benchmark JSON, source,
tests, packaging, dependency, tool, example, and workflow changes are not
classified as documentation-only.

### Tier A — Tier-1 acceptance

The full workflow runs only from an explicit `workflow_dispatch` with a reason,
or from a version tag. It builds and tests native wheels for Linux x86_64,
Windows x86_64, macOS x86_64, and macOS arm64; installs the CPython 3.11 Stable-
ABI wheels on Python 3.13; compiles C/C++ contracts on all three operating
systems; runs Linux sanitizers and the non-shipping Rust comparator; and emits
one digest manifest. Acceptance runs are not automatically cancelled.

Use Tier A for a scientific gate candidate, native ABI/build/platform changes,
a release candidate, or an explicit platform investigation. Cite the tested
source commit and run URL in later evidence documentation; that documentation
update does not need to rerun the same matrix.

## Failure and spending behavior

- A billing/spending-limit failure means `not executed`, not a numerical test
  failure and not cross-platform acceptance.
- Do not rerun unchanged work after a billing limit, infrastructure outage, or
  deterministic test failure. Diagnose locally and rerun only after the cause
  or source revision changes.
- Keep the account's zero-overage/stop-usage protection unless the owner
  explicitly changes it.
- CI artifacts are ephemeral evidence, retained for five days. They are not a
  package publication or compatibility promise.

## Temporary public visibility

The owner explicitly authorized making `haoming-luo/agentfem-native` public for
the remainder of September 2026 so development can continue without private
Actions-minute pressure. Review visibility on **2026-10-01**; it does not change
automatically. Public visibility does not change the PolyForm Noncommercial
license, authorize third-party commercial use, or publish a release package.

While public, never commit credentials, private datasets, customer models,
machine-specific paths, or proprietary benchmark inputs. Pull requests from
forks receive read-only permissions and no project secrets; workflows must not
use `pull_request_target` for untrusted code.

## Metrics to review after one week

- workflow runs and job count per accepted capability;
- cancelled superseded fast runs;
- full acceptance frequency and duplicated source SHAs;
- runner minutes by operating system;
- failures caused by code, infrastructure, or billing;
- platform defects that escaped Tier F and were caught by Tier A.
