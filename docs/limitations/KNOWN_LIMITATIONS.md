# Known limitations

- The Kernel Contract and AgentFEM public-AF-IR adapter are experimental 0.x
  interfaces. Current AF-IR requires an explicit executable extension because
  its mesh summary is not reconstructable.
- Only affine 2D triangles, nodal scalar P1 fields, steady diffusion, and
  stateless conductivity materials exist.
- SciPy provides optional sparse storage and a direct solve, but production
  language/provider selection still requires benchmarks and native packaging.
- Quadrature supports exact polynomial degrees 1 and 2 only.
- No performance, MPI, GPU, solid mechanics, time integration, nonlinear
  material, or checkpoint capability is claimed.
- Cross-platform CI is configured but cannot be claimed as passing until it
  runs remotely on native Windows, macOS, and Linux.
