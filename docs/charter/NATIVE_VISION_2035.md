# AgentFEM Native — 2035 vision

## Thesis

The future finite-element bottleneck will not be writing one more weak form.
Agents will generate models, materials, parameter studies, and optimization
loops at a scale that makes opaque execution dangerous. The scarce capability
will be fast computation whose meaning, provenance, failure mode, and evidence
remain inspectable by humans and machines.

AgentFEM Native exists to own that capability. It is not a wrapper around a
foreign finite-element object model and not a clone of an existing research
framework. AgentFEM owns engineering intent; Native owns numerical meaning from
topology through results. The versioned Kernel Contract connects them.

## Durable advantages

1. **Native three-platform engineering.** Windows, macOS, and Linux execute the
   same model contract and scientific suite. Windows is not redirected to WSL.
2. **Independent numerical ownership.** Reference cells, geometry, quadrature,
   DOFs, operators, materials, assembly, state, procedures, and result evidence
   are Native concepts rather than leaked backend objects.
3. **Rigor before feature count.** Every capability earns maturity through
   algebraic identities, failure tests, analytical/manufactured solutions,
   convergence, conservation, energy, restart, and platform evidence.
4. **Two-speed implementation.** A compact executable oracle defines meaning;
   vectorized, C++20, Rust, GPU, and vendor paths compete to reproduce it faster.
5. **Machine-readable trust.** Requests, results, capabilities, errors,
   provider identities, artifact hashes, and verification evidence are data,
   enabling agents to reason about what ran and what remains unproven.
6. **Replaceable infrastructure.** Sparse solvers, MPI, partitioners, BLAS,
   accelerators, and domestic computing providers sit behind narrow contracts.
   No optional library owns the finite-element semantics.
7. **Controlled commercial kernel.** The source-available noncommercial core
   can be inspected and researched, while commercial rights remain separately
   granted by the project owner.

## Performance doctrine

Speed is a scientific property only when the compared computations are the
same. Every optimized path therefore reports:

- exact topology and ordering identity;
- maximum matrix, vector, solution, energy, and balance differences;
- problem size, precision, provider, OS, architecture, compiler, flags, and
  dependency versions;
- wall time, throughput, memory peak, warm-up policy, and raw repetitions;
- a content digest and the source revision that produced it.

The execution ladder is:

```text
readable NumPy oracle
    -> bounded vectorized CPU batches
    -> C++20 production kernels behind a stable C ABI
    -> threaded/SIMD kernels
    -> distributed-memory providers
    -> GPU and domestic accelerator providers
```

Optimization proceeds by measurement: eliminate Python object traffic, choose
data layouts deliberately, batch affine geometry, preallocate sparse emission,
bound temporary memory, fuse kernels when evidence supports it, and keep
solver/provider costs separate from discretization costs. No benchmark-only
shortcut may enter scientific execution unseen.

## Production-language judgment

C++20 is the selected production compiled language because scientific/HPC
interoperability, vendor compilers, MPI, GPU toolchains, and stable C linkage
are mature. The packaged P1 kernel proves that a dependency-free compiled loop
can reproduce Native COO data exactly and retain substantial headroom beyond
vectorized NumPy.

Rust remains strategically useful as a non-shipping comparison and research
track. Its ownership model offers lessons for nonlinear material state,
rollback, concurrency, and long-lived services. It is not a second production
runtime: mixed implementation would require a later evidence-backed ADR,
remain behind one stable ABI, and never duplicate scientific semantics.

Permissive general-purpose libraries are leverage, not weakness. BSD/MIT/
Apache components such as SciPy, PETSc, OpenBLAS, Kokkos, or binding tools may
be adopted after transitive-license, platform, determinism, and performance
review. A third-party FEM library may not replace Native discretization.

## Scientific confidence system

The long-term verification pyramid contains:

- exact basis, quadrature, mapping, symmetry, null-space, and conservation
  identities;
- material-point tests and consistent-tangent checks;
- patch, analytical, manufactured, convergence, and locking tests;
- orientation, numbering, partition, thread-count, and provider invariance;
- energy/external-work ledgers and reaction balance;
- transactional begin/commit/rollback and restart equivalence;
- differential fuzzing between oracle and optimized paths;
- cross-platform artifact manifests and reproducible capability matrices;
- independent public benchmark suites and explicit known limitations.

“Successful” means the computation finished. “Verified” means defined evidence
passed. “Validated” requires comparison to the physical world or accepted
experimental truth. The product must never collapse these meanings.

## Horizon roadmap

### 2026–2027: trusted serial core

Close Gate 1 on all three operating systems; productionize the compiled ABI;
add 2D/3D linear solids, recovery, mesh import, robust constraints, sparse
provider selection, and end-to-end AgentFEM lowering.

### 2027–2029: nonlinear stateful mechanics

Add time integration, Newton/cutback procedures, checkpoint/restart, finite
strain, hyperelasticity, J2 plasticity, creep, state transactions, automatic
differentiation experiments, and industrial verification packs.

### 2029–2031: higher order and scale

Add P2/hierarchical/mixed spaces, near-incompressibility, contact foundations,
distributed identity/assembly, deterministic parallel evidence, and scalable
PETSc/vendor providers. Establish CPU SIMD and GPU kernel generation from the
same operator semantics.

### 2031–2035: autonomous trustworthy simulation

Agents construct, critique, refine, execute, compare, and explain simulations
through the engineering language. Native supplies fast sovereign computation,
capability negotiation, uncertainty/reliability workflows, multi-fidelity
campaigns, and auditable evidence across international and domestic hardware.

## Non-negotiables

- Never trade mathematical meaning for a headline benchmark.
- Never claim an operating system, provider, element, or material without its
  evidence.
- Never let an optimized implementation become the only oracle.
- Never hide nonconvergence, unsupported intent, partial results, or provenance.
- Never couple the engineering language to one numerical vendor.
- Never confuse a grand vision with permission to skip gates.

The ambition is large: an AI-native engineering system should be able to trust
its own calculations. That trust will be earned line by line, identity by
identity, benchmark by benchmark, and platform by platform.
