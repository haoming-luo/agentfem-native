# Known limitations

- The Kernel Contract and optional AgentFEM public-AF-IR adapter are
  experimental 0.x interfaces. Current AF-IR requires an explicit executable
  extension because its mesh summary is not reconstructable. Public AF-IR
  promotion is deferred and does not block Native kernel milestones.
- Affine 2D triangles and affine 3D tetrahedra exist. Steady scalar P1 diffusion
  is verified; T3/T4 elasticity is implemented but has not completed Gate 2.
- The bounded vectorized path currently supports static scalar/tensor
  conductivity and static scalar sources, including static cell-material
  overrides. Spatial callables deliberately use the reference path.
- Native owns CSR/BSR, CG, Jacobi, and block-Jacobi. Compiled sparse operations,
  advanced preconditioners, and threaded execution remain open. SciPy provides
  optional sparse storage and a direct solve. C++20 accelerates static diffusion
  and static T3 volume assembly; T4, boundaries, and callable fields remain in
  Python.
- Quadrature supports exact polynomial degrees 1 and 2 only.
- No cross-platform performance superiority, MPI, GPU, or nonlinear global
  procedure is claimed. T4/3D, linear time integration, restart, and nonlinear
  material points are implemented at explicitly limited maturity; they do not
  imply Gate 2, Gate 3, or Gate 4 completion. Local benchmarks are separate.
- CI installation wheels are ephemeral private validation artifacts. No public
  package, signed release, long-term binary archive, or compatibility promise
  exists during the current rapid-iteration phase.
