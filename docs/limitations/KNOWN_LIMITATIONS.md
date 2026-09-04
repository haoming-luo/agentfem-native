# Known limitations

- The Kernel Contract and AgentFEM public-AF-IR adapter are experimental 0.x
  interfaces. Current AF-IR requires an explicit executable extension because
  its mesh summary is not reconstructable.
- Only affine 2D triangles, nodal scalar P1 fields, steady diffusion, and
  stateless conductivity materials exist.
- The bounded vectorized path currently supports static scalar/tensor
  conductivity and static scalar sources, including static cell-material
  overrides. Spatial callables deliberately use the reference path.
- SciPy provides optional sparse storage and a direct solve. The standard-
  library C++20 kernel is still an unbound experiment, not wheel/runtime code;
  final production language/provider selection requires cross-platform builds,
  packaging, sanitizers, a Rust comparison, and end-to-end solver benchmarks.
- Quadrature supports exact polynomial degrees 1 and 2 only.
- No cross-platform or end-to-end performance superiority, MPI, GPU, solid
  mechanics, time integration, nonlinear material, or checkpoint capability is
  claimed. Local assembly microbenchmarks are recorded separately.
- Cross-platform CI is configured but cannot be claimed as passing until it
  runs remotely on native Windows, macOS, and Linux.
