# ADR-0010: Windows, macOS, and Linux are first-class platforms

Status: accepted as design direction — 2026-09-04

Tier-1 release targets are native Windows x86_64, macOS arm64/x86_64, and Linux
x86_64. The same scientific kernel and contract serve all three; WSL is
additional, not a substitute for native Windows. Every Gate starts cross-
platform rather than postponing portability until after implementation. OS
differences are isolated in packaging/providers and exposed as capabilities.
