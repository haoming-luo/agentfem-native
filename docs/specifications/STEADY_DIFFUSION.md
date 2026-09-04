# Steady scalar diffusion specification

Status: Gate 1 release candidate implemented and locally verified.

## Strong and weak forms

For scalar field `u`, symmetric positive-definite conductivity tensor `K(x)`,
source `f`, outward normal `n`, Dirichlet boundary `Gamma_D`, and Neumann
boundary `Gamma_N`:

```text
-div(K grad u) = f                in Omega
u = u_D                           on Gamma_D
K grad u dot n = g                on Gamma_N
```

The P1 Galerkin problem is to find `u_h` satisfying the essential conditions
and, for every zero-valued test function `v_h` on `Gamma_D`,

```text
integral_Omega grad(v_h) dot K grad(u_h) dOmega
  = integral_Omega v_h f dOmega + integral_Gamma_N v_h g dGamma.
```

For one affine triangle with basis-gradient matrix `G`,

```text
K_e = integral_element G K(x) G^T dOmega.
```

The source is integrated by the degree-2 triangle rule. Boundary flux uses
two-point Gauss integration on each straight edge. Element entries are emitted
as deterministic COO triplets. The auditable dense NumPy provider and optional
SciPy CSR direct provider solve the same constrained system behind one
interface. Provider selection does not change discretization or assembly.

For static scalar/tensor conductivity and scalar source, `auto` selects a
bounded vectorized implementation. It computes affine Jacobian inverses
analytically, emits the same cell-major/local-row/local-column COO ordering,
and accumulates loads in the same cell order. `reference` always executes the
quadrature-based element oracle. Callable fields select `reference`; requesting
`vectorized` for unsupported field semantics fails instead of silently
changing behavior. Chunk size is a performance control and cannot alter
scientific results.

Conductivity may be a positive scalar, a spatial scalar callable, or a
symmetric positive-definite 2-by-2 tensor. Named cell sets may override the
default conductivity. Material assignments must not overlap; unassigned cells
retain the default.

## Boundary-condition convention

`NeumannCondition.flux` is the outward physical flux `K grad(u) dot n` in the
weak-form sign above. Unspecified boundaries are homogeneous natural
boundaries. At least one Dirichlet node is currently required; pure Neumann
null-space handling is not implemented.

## First analytical slice

On the unit square, set `k=1`, `f=0`, `u=0` on the left edge, `g=1` on the
right edge, and homogeneous natural conditions on top and bottom. The exact
solution is `u=x`. The two-triangle mesh must reproduce nodal values exactly,
have zero free residual within roundoff, and satisfy total applied flux plus
total reaction equals zero.

## Acceptance evidence

- independent closed-form element stiffness matrix;
- constant source and edge-flux integrals;
- symmetry and constant null mode before constraints;
- exact linear patch with an interior free node;
- analytical mixed-boundary solution on two triangles;
- element orientation, cell order, and node-renumbering invariance;
- manufactured solution `sin(pi x) sin(pi y)` with observed nodal convergence
  order greater than 1.5 over 4/8/16 subdivisions;
- explicit rejection of degenerate mesh entities and unsupported pure-Neumann
  solve;
- portable ASCII VTK nodal output;
- variable-conductivity exact solution, anisotropic linear exact solution,
  and a two-region layered-material exact solution;
- numerical identity between installed NumPy and SciPy provider paths.
- triplet-by-triplet reference/vectorized comparison for scalar, anisotropic,
  material-region, boundary-flux, orientation, and multiple chunk-size cases;
- standard-library C++20 spike comparison against vectorized NumPy.

## Current limitations

Geometry is two-dimensional and affine; source and boundary values in the
serialized contract are constant scalars; and materials are stateless
conductivity records. The SciPy path is a scalable sparse storage/direct-solve
bridge, not the final production solver. Nonlinear, transient, 3D, higher-order,
and stateful material behavior remain later-gate work.
