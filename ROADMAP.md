# AgentFEM Native roadmap

2026-09-13 形成的下一阶段架构收敛、功能优先级和四周执行计划见
`docs/charter/NEXT_DEVELOPMENT_PLAN.md`。该计划以关闭 P3、验证 Gate 2 和
Gate 3 线性部分为主线，不改变本文件的 Gate 证据要求。

快速 Scrum 阶段采用 `docs/verification/SCRUM_VERIFICATION.md` 的四层验证环。
日常增量只完成风险匹配的最小数学闭环；收敛、性能扫描、第三方黑盒对照和完整
Tier-1 平台矩阵集中在里程碑，不以重复大工程拖慢主线。

本路线图的唯一首要产品方向是服务 AgentFEM 的未来使用。Native 的独立入口
用于开发、验证与稳定集成，不代表建设第二个终端产品。新增能力必须先说明其
AgentFEM 工程消费场景，再进入相应 Gate。

本路线图同时服务国产替代和自主掌握目标。每个 Gate 除功能正确外，还必须证明
相应数学规格、参考实现、生产路径、平台构建、诊断和验证证据由项目掌握；仅把
计算转交给不可替换的第三方有限元内核不构成完成。许可清晰的通用计算库可以
通过提供者边界加速交付，但 Tier-1 串行主干必须在它们缺席时仍然成立。

This roadmap is gate-based. A gate advances only when its claims have evidence;
calendar pressure does not lower mathematical or independence requirements.
Native Windows, macOS, and Linux are required from Gate 0 onward. WSL may be offered as an
additional route but never substitutes for native Windows acceptance.

## Gate 0 — charter and independent lineage

Exit criteria:

- approved project boundary, clean-room policy, provenance process, and
  license decision;
- versioned Kernel Contract draft and recorded architecture decisions;
- P1 reference triangle, affine map, and degree-1/degree-2 quadrature covered
  by mathematical tests;
- isolated tests prove no import of DOLFINx, UFL, Basix, or FFCx;
- source package and wheel build on Linux, native Windows, and macOS;
- the same test suite passes on all three systems.

Current state: complete. Hosted native Windows, macOS, and Linux tests and
installability checks passed in GitHub Actions run `33834316684` at commit
`f005c8d`.

## Gate 1 — serial scalar reference kernel

Scope: 2D triangular meshes, named sets, P1 scalar space, global DOF numbering,
Dirichlet and Neumann conditions, sparse assembly, steady diffusion/heat,
nodal results, VTK output, and an external AgentFEM adapter prototype.

Required evidence: constant and linear reproduction, patch tests, analytical
1D-equivalent solution, manufactured solution, mesh convergence, boundary
integration, reversed element orientation, and node-renumbering invariance on
native Windows, macOS, and Linux.

First vertical slice: solve `-div(k grad u) = f` on the unit square using two
P1 triangles, then generalize only after the element-to-result evidence chain
is complete.

Current state: complete. Gate 1 includes
analytical, patch, convergence, orientation, renumbering, balance, failure,
and VTK evidence; cell material regions; variable and anisotropic
conductivity; NumPy and optional SciPy providers; Kernel Contract 0.1 JSON
request/result envelopes; a cross-platform CLI; and an external public-AF-IR
lowering prototype. The same scientific suite passes on hosted native Windows,
macOS, and Linux. Promotion of the optional adapter into AgentFEM's public
export is not a Native gate requirement and resumes only when the upstream
AgentFEM roadmap needs that integration.

## Performance milestone P1 — bounded serial throughput (complete locally)

- Completed: vectorized structured-mesh generation and validation; bounded
  vectorized static P1 assembly; conservative auto-dispatch; reference/fast
  path equivalence and scale benchmarks.
- Completed locally: standard-library-only C++20/C ABI P1 assembly spike and
  exact comparison with vectorized NumPy.
- Measured local direction: hundreds-fold vectorized speedup over the readable
  element loop, followed by additional tens-fold C++ kernel headroom. These are
  local kernel benchmarks, not cross-platform end-to-end claims.
- Completed locally: sanitizer builds, compiled-wheel packaging, a dependency-
  free Rust equivalent, material-region ABI, and sparse end-to-end timing.

## Performance milestone P2 — packaged compiled kernel (complete)

- Completed locally: a hand-written CPython stable-ABI adapter packages the
  C++20 kernel in a `cp311-abi3` wheel without pybind11, nanobind, Cython, or
  the NumPy C API.
- Completed locally: exact C++/Rust/NumPy COO and load-vector equivalence,
  interleaved raw timings, public-runtime timing, and identical SciPy solution,
  reaction, and energy evidence.
- Decision: C++20 is the primary production kernel language. Rust remains a
  valuable safety and design cross-check but is not a second shipping runtime
  or a release blocker.
- Completed locally: SciPy 1.18.1 was installed from an official wheel after
  SHA-256 verification and passes provider equivalence; SciPy remains optional
  infrastructure and owns no finite-element semantics.
- Completed in CI: native installation wheels build, pass strict Stable ABI
  auditing, install, and pass the scientific suite on Windows x86_64, Linux
  x86_64, macOS x86_64, and macOS arm64. These are ephemeral private CI test
  artifacts, not a PyPI or GitHub Release publication.
- Deferred until an actual release milestone: public artifact publication,
  signing, binary SBOM policy, and broader field/operator coverage.

Execution policy: the local development machine runs macOS builds and tests.
GitHub CI is the required execution environment for native Windows and Linux;
local emulation is diagnostic only and never substitutes for those runners.
Ordinary pushes now use one cancellable Linux fast job; explicit Tier-1
acceptance retains native Windows x86_64, Linux x86_64, macOS x86_64/arm64,
Stable ABI, C/C++, sanitizer, and artifact evidence. Documentation-only changes
reuse the accepted source revision instead of repeating its matrix. The full
policy is `docs/development/CI_GOVERNANCE.md`.

Before Gate 2, keep the hosted P2 installability evidence green. Prefer permissive
general-purpose dependencies; do not use a third-party FEM implementation to
substitute for Native discretization.

## Six-month acceleration program — active

The detailed execution plan for 2026-09-07 through 2027-03-07 is
[the six-month product roadmap](docs/charter/SIX_MONTH_PRODUCT_ROADMAP.md).
It assumes a 28–36-person large-team equivalent and targets an internal
Mechanics Alpha rather than a public package release.

Committed end state:

- Gate 2 verified in 2D and 3D on native Windows, macOS, and Linux;
- the linear dynamics and state/restart portion of Gate 3 verified;
- owned end-to-end CSR/BSR sparse execution and a transparent solver baseline;
- production CPU threading/SIMD, with optional provider comparisons;
- Agent-native capability, resource-planning, diagnostics, cancellation, and
  evidence surfaces independent of current AF-IR promotion;
- limited nonlinear, MPI, matrix-free, and GPU work reported as candidates or
  experiments until their actual gates pass.

Priority order is P0 sparse/vector DOF/T3/T4/three-platform/Agent preflight;
P1 dynamics/state/threading/providers; then stretch elements, nonlinear paths,
MPI, and GPU. Breadth stops when it threatens the verified mesh-to-result spine.

The project owner subsequently requested calendar compression. The active
[one-month acceleration overlay](docs/charter/ONE_MONTH_ACCELERATION.md)
compresses this program into four concurrent tranches while preserving every
Gate and maturity boundary.

## Performance milestone P3 — end-to-end sparse spine (active, expanded)

- Completed locally: deterministic scalar CSR and canonical COO conversion;
  SpMV, residual/norm operations, sparse constraints, CG, and Jacobi with
  explicit convergence and failure reports.
- Completed locally: dependency-free `native_sparse` is the default provider;
  tests prove it does not call dense conversion. Dense NumPy is now an explicit
  bounded oracle, while SciPy remains optional.
- Completed locally: execution-free resource plans, deterministic plan digests,
  capability reporting, and a reproducible sparse-memory/performance record.
- Completed locally: deterministic BSR grouping/round-trip/matvec and
  block-Jacobi for vector mechanics; C++20 ABI 1.1 T3 volume assembly; hosted
  three-platform evidence for the scalar sparse/T3 reference foundation at
  commit `5230970` in run `34091224291`.
- Completed hosted: commit `db33958` passed all 21 Windows, Linux, and macOS
  jobs in run `34099397789`, including native installation wheels.
- Completed and accepted: C++20 ABI 1.2 T4 volume assembly and canonical CSR
  SpMV, with permanent readable differential oracles. Commit `4f3da30` passed
  the 14-job explicit Tier-1 Windows/Linux/macOS acceptance in run
  `34566488847`, including Stable ABI reuse on Python 3.13.
- 本地候选已完成：C++20 ABI 1.3 为 T3/T4 增加显式线程数、静态连续分块、
  独占 COO 写入和确定性载荷归并；线程临时内存进入 Agent 执行计划与收据。
  macOS arm64 代表规模记录中，T3 四线程为串行的 2.53×，T4 八线程为 4.00×；
  这不是跨平台速度声明。源提交 `da92dbd` 已在运行 `34752687326` 中通过 14 项
  Windows/Linux/macOS Tier-1 验收，ABI 1.3 平台边界关闭。
- 已完成并验收：不可变 `CSRPattern` 分离规范图和多次数值回填，C++20 ABI 1.4
  按原贡献顺序确定性生成 CSR 数值。代表 T3 结构的生产回填比重新排序/归并
  COO 快 134×–146×，但相对 NumPy 参考回填仅快 1.02×–1.07×，因此不宣称
  端到端巨大加速。源提交 `d172cf6` 的 Linux 快检 `34757574303` 与 14 项完整
  Tier-1 验收 `34757610403` 均通过。
- Remaining: reusable-pattern integration into repeated engineering workflows,
  end-to-end memory/time evidence, SIMD, and stronger preconditioners.
- Admit optional Ginkgo, PETSc/hypre, or other permissive providers only behind
  narrow contracts and after license, platform, determinism, and performance
  evidence.
- Begin vector DOFs and the readable T3 plane-stress/plane-strain oracle in the
  same sprint so the sparse design serves mechanics rather than a scalar demo.

## AgentFEM integration track — deferred, non-blocking

- Retain the external Kernel Contract/AF-IR lowering prototype as boundary
  evidence; do not couple Native internals to the current AF-IR shape.
- Do not spend core-kernel milestone capacity promoting `root.native_kernel`
  into AgentFEM while AF-IR is not an AgentFEM development priority.
- Resume end-to-end AgentFEM integration when the upstream public export is a
  prioritized, reconstructable, versioned contract. This track does not block
  Gate 2 or later Native scientific work.

## Gate 2 — basic solid mechanics (active after P3 foundation)

Scope: 2D small-strain isotropic elasticity, plane stress/strain, displacement
constraints, traction/body force, reactions, strain energy, stress/strain
recovery, P1 triangles, followed by P1 tetrahedra and basic 3D.

Evidence: rigid-body modes, constant-strain patch tests, uniaxial/shear/bulk
responses, cantilever, symmetry, energy, reaction balance, and convergence.

Current state: T3 now has readable, vectorized, and C++20 volume assembly,
material regions, BSR/block-Jacobi, vector/tensor VTK, pure-shear, dilation, and
second-order manufactured evidence in addition to the original reference
suite. Readable and C++20 ABI 1.2 T4/3D paths now own tetrahedral topology,
volume assembly, loads, constraints,
reactions, energy, recovery, uniaxial/patch/orientation/provider evidence, and a
second-order manufactured trend. ABI 1.2 has passed hosted Tier-1 acceptance.
Gate 2 remains `implemented`, not verified: wider engineering/convergence
evidence, contract admission, and production CPU parallel evidence are still
required.

2026-09-13 本地科学验收候选已把分散证据收敛为十二项矩阵，并强化 T3/T4 实际
求解收敛、合法畸变 patch、尺度化残差/平衡和 Agent 可读分析摘要。当前科学
矩阵 G2-01 至 G2-10 本地通过；力学 JSON Contract 0.2 和最终 Tier-1 Gate
候选仍未关闭。P3 CPU 并行与 ABI 1.4 可复用图数值回填已完成本地与三平台
验收，但真实重复装配生命周期和力学 Contract 仍未关闭，因此 Gate 2 继续保持
`implemented`。

## Gate 3 — time and nonlinear lifecycle

Scope: consistent and lumped mass, central difference, implicit increments,
Newton residual/tangent, begin/commit/rollback, adaptive cutback,
checkpoint/restart, and energy/external-work ledgers.

Evidence: SDOF dynamics, wave propagation, time convergence, energy behavior,
forced cutback, and restart equivalence across supported platforms.

Current state: the linear foundation is implemented locally: T3 consistent and
lumped mass, centered explicit integration, Newmark average acceleration,
energy histories, accepted-state progress/cancellation/budgets, and digest-bound
restart, zero fixed-DOF elimination, reactions, external-work/energy ledgers,
and a constrained T3 transient. SDOF order/energy and exact restart evidence
pass. Wave propagation, implicit checkpoint coverage, time-refinement evidence,
and wider constrained-FEM platform evidence remain before Gate 3 verification.

## Gate 4 — nonlinear solids and material state

Scope: finite-strain kinematics, Neo-Hookean and Mooney–Rivlin models,
quadrature state, J2 plasticity, creep, multiple material regions, and
consistent tangents.

Evidence: material-point and element paths, uniaxial/biaxial loading,
objectivity, volume response, rollback, and public benchmarks.

## Gate 5 — higher order, mixed, and near-incompressible

Scope: P2 fields and geometry, DOF transformations, mixed displacement-pressure
spaces, near-incompressibility, C3D10H-equivalent capability targets, and
multiple cell-block topologies. Naming equivalence alone is never acceptance.

## Gate 6 — MPI and scale

Scope: partitioning, ghost entities, distributed DOFs and assembly, global
state identity, parallel checkpoint/output, and replaceable partitioners.

Evidence: 1/2/4/8-rank consistency, partition independence, restart across
partition layouts, global energy/balance, and scaling studies on Linux.
Windows capability is reported explicitly per MPI/provider combination.

## Gate 7 — GPU and domestic computing ecosystem

Scope: replaceable vectorized CPU and GPU kernels, domestic CPU/GPU,
domestic Linux distributions, compilers, MPI, and sparse solvers. The goal is
independent interfaces, builds, and critical paths—not rejection of every
general-purpose open-source dependency.

## Platform release tiers

- Tier 1: native Windows x86_64, macOS arm64/x86_64, and Linux x86_64;
  release-blocking.
- Candidate: Linux aarch64 and Windows arm64 after reliable hosted or
  self-hosted runners exist.
- Provider-specific capabilities (MPI, PETSc, GPU) are reported separately;
  an unavailable provider must not make the serial NumPy reference kernel
  unavailable.
