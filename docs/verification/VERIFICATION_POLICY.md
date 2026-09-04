# Verification policy

Evidence advances in this order:

```text
algebraic identity -> quadrature -> element matrix -> patch test
-> small structure -> mesh/time convergence -> public benchmark
-> cross-backend comparison -> MPI/restart -> engineering case
```

Every claim records its tolerance, precision, platform, dependency versions,
and failure modes. FEniCSx may be one black-box comparator but never the only
oracle. Agreement among programs does not replace analytical and convergence
evidence.

Maturity labels mean:

- `experimental`: interface and mathematics may change;
- `implemented`: code path exists and basic tests run;
- `verified`: specified identities and numerical evidence pass;
- `validated`: accepted external benchmark or experimental evidence supports
  the declared use domain;
- `production`: validated capability also meets reliability, performance,
  platform, documentation, and release-governance gates.

Cross-platform consistency requires equal scientific assertions. Small
floating-point differences may use one justified tolerance; OS-specific
tolerance inflation requires a written numerical diagnosis and ADR.
