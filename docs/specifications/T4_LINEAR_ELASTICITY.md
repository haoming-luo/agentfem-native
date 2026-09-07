# T4 small-strain linear elasticity specification

Status: Gate 2 three-dimensional implementation target.

## Geometry, kinematics, and constitutive response

An affine four-node tetrahedron maps reference coordinates through the three
edge columns `[x1-x0, x2-x0, x3-x0]`. Its positive physical volume is
`abs(det(J))/6`, and physical P1 gradients are the reference gradients
`[-1,-1,-1], [1,0,0], [0,1,0], [0,0,1]` multiplied by `inv(J)`.

Displacement DOFs are node-major `[ux, uy, uz]`. Engineering strain is ordered
`[exx, eyy, ezz, gxy, gyz, gxz]`. For isotropic elasticity,
`mu=E/(2(1+nu))` and `lambda=E nu/((1+nu)(1-2nu))`; the normal block has
diagonal `lambda+2mu`, off-diagonal `lambda`, and the engineering-shear
diagonal is `mu`. The element stiffness is `V B^T D B`.

## Loads, conditions, and results

Constant body force contributes `V b/4` to every node. A triangular-face
traction is integrated by the symmetric three-point degree-2 triangle rule.
Component or vector displacement conditions must remove three translations and
three rotations. The result owns displacement, residual, constrained reactions,
cell strain/stress, applied/reaction resultants, strain energy, provider, and
solver convergence evidence.

## Acceptance evidence

- six rigid-body modes and six positive deforming modes for a single T4;
- affine constant-strain patch with a free interior node;
- uniaxial cube traction with Poisson contraction and exact stress;
- volume/face load resultants, orientation, numbering, material-region,
  provider, energy, and reaction balance checks;
- explicit invalid geometry, material, field, conflict, and underconstraint
  failures;
- native Windows, macOS, and Linux execution of the same suite.

The readable T4 is an implementation milestone. Gate 2 is verified only after
the full convergence and optimized-path evidence is recorded.
