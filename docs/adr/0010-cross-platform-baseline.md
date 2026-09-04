# ADR-0010: Linux and Windows are first-class platforms

Status: accepted as design direction — 2026-09-04

Tier-1 release targets are Linux x86_64 and native Windows x86_64. The same
scientific kernel and contract serve both; WSL is additional, not a substitute.
macOS remains supported for reference-layer development. Every Gate starts
cross-platform rather than postponing portability until after implementation.
OS differences are isolated in packaging/providers and exposed as capabilities.
