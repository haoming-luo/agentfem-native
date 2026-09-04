# Known limitations

- The Kernel Contract and AgentFEM public-AF-IR adapter are experimental 0.x
  interfaces. Current AF-IR requires an explicit executable extension because
  its mesh summary is not reconstructable.
- Only affine 2D triangles, nodal scalar P1 fields, steady diffusion, and
  stateless conductivity materials exist.
- The bounded vectorized path currently supports static scalar/tensor
  conductivity and static scalar sources, including static cell-material
  overrides. Spatial callables deliberately use the reference path.
- SciPy provides optional sparse storage and a direct solve. The packaged
  C++20 kernel currently accelerates only static P1 volume assembly; boundary
  flux integration and unsupported/callable fields remain in Python. C++20 is
  selected; hosted Tier-1 correctness and installability evidence passes.
- Quadrature supports exact polynomial degrees 1 and 2 only.
- No cross-platform performance superiority, MPI, GPU, solid
  mechanics, time integration, nonlinear material, or checkpoint capability is
  claimed. Local assembly microbenchmarks are recorded separately.
- CI installation wheels are ephemeral private validation artifacts. No public
  package, signed release, long-term binary archive, or compatibility promise
  exists during the current rapid-iteration phase.
