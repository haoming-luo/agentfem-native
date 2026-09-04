# Known limitations

- An experimental serial scalar diffusion slice exists, but there is no
  AgentFEM adapter or stable serialized Kernel Contract yet.
- Only affine 2D triangles, nodal scalar P1 fields, constant isotropic
  conductivity, and a dense NumPy solve provider exist.
- Quadrature supports exact polynomial degrees 1 and 2 only.
- No performance, MPI, GPU, solid mechanics, time integration, nonlinear
  material, or checkpoint capability is claimed.
- Cross-platform CI is configured but cannot be claimed as passing until it
  runs remotely on native Windows, macOS, and Linux.
