# Gate 2 T3 reference-slice report

**Date:** 2026-09-07

**Disposition:** implemented; Gate 2 remains open

## Implemented capability

- deterministic node-major two-component DOF numbering;
- isotropic plane stress and plane strain;
- affine P1 triangle stiffness with orientation-independent area;
- finite vector body force and boundary traction integration;
- component-wise or vector displacement constraints;
- rigid-mode constraint-rank and conflicting-value failures;
- Native sparse and dense-oracle solves;
- nodal displacement, full residual, reactions, cell engineering strain,
  cell stress, force balance, strain energy, provider, and convergence evidence;
- execution-free resource planning with an honest `implemented` maturity label.

## Evidence passed locally

- stiffness symmetry, three rigid-body null modes, and three positive deforming
  modes on the reference triangle;
- exact affine strain recovery;
- orientation reversal with the corresponding local DOF permutation;
- node-renumbering invariance of displacement and cell stress;
- exact constant body-force and traction resultants;
- independently written plane-stress and plane-strain closed forms;
- constant-strain patch with an interior free node;
- plane-stress uniaxial traction, exact displacement/Poisson contraction,
  stress, energy, and reaction balance;
- Native sparse versus NumPy dense solution and stress equivalence;
- invalid material, insufficient rigid-mode constraints, and conflicting
  displacement failures.

The local scale record used 1,250 displacement DOFs and 1,152 cells. Five
end-to-end Native sparse repetitions had a 0.09640 s median, maximum analytical
displacement error `3.07e-15`, free residual norm `1.87e-12`, and force-balance
norm `1.18e-12`.

## Why Gate 2 remains open

T3 currently has only the readable assembly path. The complete Gate requires
the broader shear/bulk/cantilever and mesh-convergence packs, material regions,
recovery policy, optimized C++20 execution, 3D T4, output/contract integration,
and native Windows/macOS/Linux evidence. No 2D result in this report implies 3D
or general industrial mechanics capability.
