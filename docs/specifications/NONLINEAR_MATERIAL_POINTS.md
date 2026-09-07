# Nonlinear material-point candidate specification

Status: Gate 4 experimental candidate; no element-level admission.

The compressible Neo-Hookean candidate uses
`W = mu/2 (F:F-3) - mu ln(J) + lambda/2 ln(J)^2` for positive `J=det(F)` and
`P = mu(F-F^-T) + lambda ln(J) F^-T`. It must give zero energy/stress for any
proper rigid rotation and reject non-positive Jacobians.

The small-strain J2 candidate stores a symmetric plastic-strain tensor and
equivalent plastic strain. Its radial return uses
`q=sqrt(3/2 s:s)`, yield stress `sigma_y + H alpha`, plastic increment
`dgamma=f_trial/(3 mu + H)`, and associative flow
`dep=dgamma 3/2 s_trial/q_trial`. A material point has one committed state and
at most one trial state; commit replaces the committed state, while rollback
discards the trial exactly.

Acceptance at this stage covers rotation objectivity, finite-difference energy
derivative for Neo-Hookean response, elastic/plastic J2 paths, yield-surface
consistency, and commit/rollback. Consistent tangents, finite-strain elements,
and nonlinear global procedures remain outside this candidate.
