# P1 triangle mathematical specification

Status: implemented and unit-tested; not yet validated as a PDE solver.

## Reference cell

Use the closed triangle

```text
T_hat = {(xi, eta) : xi >= 0, eta >= 0, xi + eta <= 1}
```

with vertices `(0,0)`, `(1,0)`, `(0,1)` and area `1/2`.

## Lagrange basis

```text
N0 = 1 - xi - eta
N1 = xi
N2 = eta
```

The reference gradients are the constant row vectors

```text
grad_hat N = [[-1,-1], [1,0], [0,1]].
```

Acceptance identities are the Kronecker property at vertices, partition of
unity, zero sum of gradients, nonnegativity inside the cell, and exact
interpolation of all affine scalar functions.

## Affine physical map

For physical vertices `x0,x1,x2`,

```text
x(xi,eta) = x0 + J [xi,eta]^T
J = [x1-x0, x2-x0]
dOmega = abs(det J) dT_hat
grad_x N = grad_hat N J^{-1}
area = abs(det J)/2.
```

Reversing vertex orientation changes the determinant sign but not area or a
scalar volume integral. A zero or numerically singular determinant is rejected.

## Quadrature

- Degree 1: centroid `(1/3,1/3)`, weight `1/2`.
- Degree 2: points `(1/6,1/6)`, `(2/3,1/6)`, `(1/6,2/3)`, each weight `1/6`.

Weights integrate over the reference triangle directly and therefore sum to
`1/2`. Degree-2 acceptance integrates `1, xi, eta, xi^2, xi*eta, eta^2`
against analytical monomial moments exactly within floating-point tolerance.

For nonnegative integers `a,b`,

```text
integral_T_hat xi^a eta^b dA = a! b! / (a+b+2)!.
```

## Sources and independent derivation

- Imperial College London, *Finite element course*, introduction and numerical
  quadrature: https://finite-element.github.io/
- P. G. Ciarlet, *The Finite Element Method for Elliptic Problems*, SIAM
  Classics in Applied Mathematics.
- S. C. Brenner and L. R. Scott, *The Mathematical Theory of Finite Element
  Methods*, Springer.

These are mathematical references. No finite-element library implementation
source was used.
