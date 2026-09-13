# Architecture decision records

Accepted means accepted for the private Gate 0 baseline, subject to project-
owner review. Superseded decisions remain in history.

- [ADR-0001](0001-language-and-runtime.md): Python/NumPy reference layer
- [ADR-0002](0002-mesh-identity.md): owned mesh identity and zero-based indexing
- [ADR-0003](0003-element-and-quadrature.md): explicit reference-cell contracts
- [ADR-0004](0004-sparse-assembly.md): deterministic assembly boundary
- [ADR-0005](0005-kernel-contract.md): versioned backend-neutral contract
- [ADR-0006](0006-material-state-transactions.md): transactional state
- [ADR-0007](0007-linear-algebra-providers.md): replaceable providers
- [ADR-0008](0008-mpi-identity.md): identity independent of partition layout
- [ADR-0009](0009-licensing-boundary.md): source-available dual-license boundary
- [ADR-0010](0010-cross-platform-baseline.md): Windows, macOS, and Linux first-class
- [ADR-0011](0011-production-kernel-language-evaluation.md): measured
  production-language and permissive-library selection
- [ADR-0012](0012-executable-kernel-contract.md): executable JSON contract and
  external AgentFEM lowering extension
- [ADR-0013](0013-bounded-vectorized-assembly.md): bounded vectorized assembly
  beside the mathematical oracle
- [ADR-0014](0014-cpp20-leading-production-spike.md): C++20 leads compiled
  production-kernel experiments without final language lock-in
- [ADR-0015](0015-packaged-cpp20-stable-abi.md): package the C++20 kernel with
  a hand-written CPython stable-ABI adapter
- [ADR-0016](0016-cpp20-production-language.md): select C++20 as the production
  compiled-kernel language after an exact Rust comparison
- [ADR-0017](0017-decouple-af-ir-from-native-gates.md): keep AF-IR promotion as
  an optional AgentFEM integration track rather than a Native gate blocker
- [ADR-0018](0018-sparse-first-mechanics-acceleration.md): make the owned sparse
  spine and Agent-native Mechanics Alpha the six-month execution priority
- [ADR-0019](0019-compressed-mechanics-alpha-spine.md): implement the compressed
  BSR/T3/T4/dynamics/control spine while preserving Gate maturity boundaries
- [ADR-0020](0020-mechanics-alpha-p0-closure.md): close native T4/CSR,
  constrained dynamics, interchange, and Agent execution before parallel breadth
- [ADR-0021](0021-deterministic-cpu-parallel-assembly.md)：使用显式线程数、静态
  分块和确定性载荷归并实现 T3/T4 CPU 并行装配
- [ADR-0022](0022-reusable-sparse-pattern.md)：把规范 CSR 图与重复数值回填分离，
  为多载荷、隐式动力和 Newton 复用符号工作
- [ADR-0023](0023-cpp20-deterministic-csr-fill.md)：通过 ABI 1.4 按原 COO
  贡献顺序确定性回填 CSR 数值
