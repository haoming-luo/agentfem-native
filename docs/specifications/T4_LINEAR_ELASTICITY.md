# T4 小应变线弹性规格

**状态：** 已实现可读路径、C++20 ABI 1.2 串行体积装配和 ABI 1.3 确定性并行
候选；Gate 2 验证仍未关闭。

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

可读与 Native T4 路径经过规范 COO 归并后必须一致。ABI 1.3 并行路径的 COO
顺序必须与串行逐位相同，载荷只允许固定归并导致的舍入级差异。Native 体积装配
不会把面力积分或本构含义移出项目拥有的数学层。只有完整工程算例、收敛和原生
三平台证据均被记录后，Gate 2 才能标记为已验证。
