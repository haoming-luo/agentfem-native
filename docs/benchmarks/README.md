# Benchmarks

Benchmarks are accepted only after analytical, manufactured, algebraic, and
failure tests establish equivalent work. Timings never replace verification.

- `reference_diffusion.py`: element-by-element NumPy oracle baseline.
- `assembly_comparison.py`: reference versus bounded vectorized NumPy,
  including raw repetitions, memory peak, digests, topology identity, and
  maximum matrix/load differences.
- `vectorized_scale.py`: large mesh generation and assembly with constant-null
  mode and total-load guards.
- `cpp20_batch_comparison.py`: standard-library C++20 shared-library spike
  versus vectorized NumPy, including complete COO/load comparison.
- `language_kernel_comparison.py`: interleaved C++20/Rust C-ABI timing with
  complete cross-language and NumPy output comparison.
- `runtime_native_comparison.py`: packaged public native path versus the public
  bounded-vectorized path, including output ownership cost.
- `end_to_end_sparse_comparison.py`: native versus vectorized assembly through
  the same optional SciPy sparse solve with solution/reaction/energy checks.
- `sparse_foundation.py`: owned CSR memory, COO canonicalization, Native
  CG/Jacobi end-to-end diffusion, and T3 uniaxial mechanics evidence.
- `mechanics_alpha.py`: T3/T4 readable/native assembly equivalence and timing,
  CSR/BSR operator evidence, a T4 solve, and long-step linear dynamics.
- `sparse_pattern_reuse.py`：比较重复 COO 规范化、NumPy 参考回填与 ABI 1.4
  生产回填，并明确不代表端到端求解。
- `prepared_mechanics_assembly.py`：比较 T3/T4 从元素装配到规范 CSR 的冷路径与
  显式预备图路径，验证结构和数值逐位一致。

Run from the repository environment. Raw-language comparison additionally
requires the C++ library built from `native_spikes/cpp20` and the Rust library
built from `native_spikes/rust`.

Performance numbers are machine-specific. A committed evidence snapshot must
record OS, architecture, Python/NumPy or compiler version, flags, source
revision, problem size, repetitions, memory method, raw values, and numerical
differences. Cross-platform superiority is claimed only after all Tier-1 jobs
produce comparable evidence.
