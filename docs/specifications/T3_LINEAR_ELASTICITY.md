# T3 small-strain linear elasticity specification

Status: Gate 2 reference-slice implementation target.

## Kinematics and ordering

The two displacement components are ordered node-major and component-minor:

```text
dof(node i, x) = 2 i
dof(node i, y) = 2 i + 1.
```

For a three-node triangle, the local vector is
`[u1x, u1y, u2x, u2y, u3x, u3y]`. With physical P1 gradients
`(dN_i/dx, dN_i/dy)`, engineering strain is
`epsilon = [epsilon_xx, epsilon_yy, gamma_xy]` and:

```text
    [ dN1/dx       0 dN2/dx       0 dN3/dx       0 ]
B = [       0 dN1/dy       0 dN2/dy       0 dN3/dy ]
    [ dN1/dy dN1/dx dN2/dy dN2/dx dN3/dy dN3/dx ].
```

P1 gradients and therefore strain/stress are constant within one affine T3.

## Isotropic constitutive matrices

Young's modulus satisfies `E > 0`; Poisson's ratio satisfies `-1 < nu < 0.5`.
For plane stress:

```text
D = E/(1 - nu^2) *
    [[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu)/2]].
```

For plane strain:

```text
D = E/((1 + nu)(1 - 2 nu)) *
    [[1 - nu, nu, 0], [nu, 1 - nu, 0], [0, 0, (1 - 2 nu)/2]].
```

The element stiffness for positive thickness `t` and area `A` is
`K_e = t A B^T D B`. Reversing element orientation changes neither `A` nor the
physical result.

## Loads

A body force `b = [b_x, b_y]` is force per unit volume and contributes:

```text
f_e = t * integral_element N^T b dA.
```

The readable implementation uses the existing degree-2 triangle quadrature.
Boundary traction `q = [q_x, q_y]` contributes
`t * integral_edge N^T q ds` using two-point Gauss integration. Static vectors
and callables evaluated at physical points are admitted; values must be finite.

## Conditions, solve, and evidence

Displacement conditions constrain one named component or both components on a
named node set. Conflicting assignments fail. At least three independent scalar
constraints must remove the two translations and one in-plane rotation; the
sparse solver reports singular or breakdown behavior when they do not.

After solving `K u = f` with symmetric strong constraints:

```text
residual = K u - f
reaction = residual on constrained DOFs
strain_e = B_e u_e
stress_e = D strain_e
strain_energy = 0.5 u^T K u.
```

Results expose nodal displacements, the full residual, constrained DOFs,
cell-wise engineering strain and Cauchy stress in `[xx, yy, xy]` order, total
applied force, total reaction, energy, provider identity, and solver report.

## Acceptance evidence

- element symmetry and three rigid-body null modes;
- exact constant-strain patch with an interior node;
- plane-stress uniaxial traction solution and Poisson contraction;
- plane-strain constitutive response checked against its closed form;
- force/reaction and energy consistency;
- cell orientation, cell order, and node renumbering invariance;
- explicit failures for invalid material, thickness, field shapes, conflicting
  constraints, and under-constrained systems;
- dense-oracle and Native sparse provider equivalence on small problems.

This is an implemented reference slice only. It does not claim Gate 2 closure,
3D mechanics, heterogeneous elasticity, Q4/H8, dynamics, or nonlinear state.
