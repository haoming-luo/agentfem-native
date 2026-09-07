# Linear second-order dynamics specification

Status: Gate 3 linear lifecycle implementation target.

## Governing system

The admitted system is `M a(t) + K u(t) = f(t)` with finite symmetric positive-
definite mass and stiffness on the active DOFs. The state is time, displacement,
velocity, and acceleration. Every state transition creates new owned arrays;
checkpoint data never aliases live integration storage.

T3 consistent mass is `rho t A/12 * [[2,1,1],[1,2,1],[1,1,2]]` independently
for each displacement component. Row-sum lumping gives `rho t A/3` per nodal
component.

## Procedures

The explicit procedure is velocity-Verlet, algebraically equivalent to a
centered second-order displacement update. It requires an explicitly diagonal,
strictly positive mass. The implicit procedure is Newmark average acceleration
with `beta=1/4`, `gamma=1/2`, solving
`(M + beta dt^2 K) a[n+1] = f[n+1] - K u_predict`.

Both procedures record displacement, velocity, acceleration, kinetic energy,
strain energy, and total mechanical energy at every accepted state. Invalid or
non-finite inputs, non-positive mass, solver nonconvergence, and inconsistent
restart state fail explicitly.

## Checkpoint and evidence

A checkpoint contains method, time, step, state arrays, and a deterministic
SHA-256 digest over canonical numeric bytes and metadata. Restart must reproduce
an uninterrupted trajectory. Minimum evidence is an undamped SDOF oscillator,
time-convergence trend, bounded energy behavior, explicit/implicit comparison,
and exact restart equivalence.
