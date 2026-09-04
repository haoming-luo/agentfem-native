# ADR-0003: element and quadrature abstraction

Status: accepted for Gate 0 — 2026-09-04

Reference cell, basis, quadrature, and geometry map are explicit independent
concepts. Quadrature weights include reference-cell measure and declare exact
polynomial degree. Geometry mappings supply signed orientation and positive
integration measure separately. Gate 0 implements only affine P1 triangles.
