# Reference-kernel scope

The executable reference layer is a mathematical specification, not a
high-performance production kernel. Its purpose is to keep formulas readable,
provide an oracle independent of future compiled code, and expose small
invariants that fail locally.

The minimum chain for the first diffusion solve is:

```text
mesh topology -> affine cell geometry -> P1 basis/quadrature
-> scalar DOF map -> element stiffness/load -> sparse assembly
-> boundary conditions -> linear provider -> nodal result -> evidence
```

That chain is complete and verified for Gate 1. The Gate 2 reference extension
is:

```text
triangle mesh -> node-major vector DOFs -> T3 B and isotropic D
-> element stiffness/vector loads -> deterministic COO -> owned CSR
-> sparse constraints and CG/Jacobi -> displacement/stress/reaction evidence
```

This second chain is implemented with local analytical and patch evidence. It
is not yet Gate-2 verified and remains the oracle for future C++20/BSR paths.
