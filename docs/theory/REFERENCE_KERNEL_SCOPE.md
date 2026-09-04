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

The current implementation ends after basis/quadrature/geometry. Mesh,
assembly, boundary conditions, solve, and result writing remain Gate 1 work.
