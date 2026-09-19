# Known limitations

- The Kernel Contract and optional AgentFEM public-AF-IR adapter are
  experimental 0.x interfaces. Current AF-IR requires an explicit executable
  extension because its mesh summary is not reconstructable. Public AF-IR
  promotion is deferred and does not block Native kernel milestones.
- 仿射二维三角形和仿射三维四面体已经进入主线。稳态标量 P1 扩散与 Gate 2
  验收矩阵限定的 T3/T4 基础线性静力范围均为 `verified`；这不扩展到高阶单元、
  几何非线性或材料非线性。
- The bounded vectorized path currently supports static scalar/tensor
  conductivity and static scalar sources, including static cell-material
  overrides. Spatial callables deliberately use the reference path.
- Native owns CSR/BSR, CG, Jacobi, and block-Jacobi. C++20 accelerates canonical
  CSR SpMV plus static diffusion and T3/T4 volume assembly. Advanced
  preconditioners, compiled constraints/solvers, and threaded execution remain
  open. SciPy provides optional sparse storage and a direct solve. Boundary
  integration and callable fields remain in Python.
- Quadrature supports exact polynomial degrees 1 and 2 only.
- 不声明跨平台性能优势、MPI、GPU 或全局非线性过程。线性时间积分、重启和非线性
  材料点仍处于明确受限的成熟度，不能据此声称 Gate 3 或 Gate 4 完成；本地性能
  记录与 Gate 2 科学验证是不同证据。
- Dynamic constraints currently admit zero fixed values only. Linear dynamics
  has no damping, general prescribed motion, wave benchmark, or adaptive time
  stepping. The internal interchange format has a 64 MiB limit and is not a
  public long-term compatibility promise.
- Plan digests cover resource shape and selected execution path, not a complete
  serialization of callable physics. Execution receipts keep typed numerical
  arrays in memory and expose only compact evidence as JSON.
- CI installation wheels are ephemeral private validation artifacts. No public
  package, signed release, long-term binary archive, or compatibility promise
  exists during the current rapid-iteration phase.
