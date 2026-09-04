# ADR-0001: language and runtime strategy

Status: accepted for Gate 0 — 2026-09-04

Use Python 3.11+ and NumPy for a small executable mathematical reference layer.
Do not choose the production language until profiling and provider prototypes
measure performance, MPI, binding, GPU, packaging, domestic-platform, staffing,
maintenance, and review tradeoffs. This prevents premature multi-language
complexity while preserving an independent oracle.
