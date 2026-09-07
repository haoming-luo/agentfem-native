# Known limitations

- The Kernel Contract and optional AgentFEM public-AF-IR adapter are
  experimental 0.x interfaces. Current AF-IR requires an explicit executable
  extension because its mesh summary is not reconstructable. Public AF-IR
  promotion is deferred and does not block Native kernel milestones.
- Only affine 2D triangles exist. Steady scalar P1 diffusion is verified; the
  first vector T3 plane-stress/plane-strain elasticity slice is implemented but
  has not completed Gate 2.
- The bounded vectorized path currently supports static scalar/tensor
  conductivity and static scalar sources, including static cell-material
  overrides. Spatial callables deliberately use the reference path.
- Native owns a scalar CSR/CG/Jacobi baseline; BSR, compiled sparse operations,
  advanced preconditioners, and threaded execution remain open. SciPy provides
  optional sparse storage and a direct solve. The packaged C++20 kernel
  currently accelerates only static diffusion P1 volume assembly; T3,
  boundaries, and unsupported/callable fields remain in Python.
- Quadrature supports exact polynomial degrees 1 and 2 only.
- No cross-platform performance superiority, MPI, GPU, 3D mechanics, time
  integration, nonlinear material, or checkpoint capability is claimed. The T3
  slice is not a claim of Gate-2 completion. Local benchmarks are recorded
  separately.
- CI installation wheels are ephemeral private validation artifacts. No public
  package, signed release, long-term binary archive, or compatibility promise
  exists during the current rapid-iteration phase.
