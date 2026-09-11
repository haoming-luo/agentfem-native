# C++20 Native kernel

This is an independently implemented, standard-library-only performance
kernel. It ships inside private validation wheels through a hand-written
CPython Stable-ABI adapter and does not replace the readable NumPy oracle. Its
public boundary is a true C ABI so the binding remains thin and replaceable.

ABI 1.2 emits deterministic per-cell COO entries and nodal loads for static
anisotropic P1 diffusion and T3/T4 linear-elastic volume terms. It also applies
canonical CSR matrices. A nonzero status invalidates all output buffers;
callers must not consume partial output. See
`docs/specifications/NATIVE_P1_ABI.md` for layouts and compatibility.

## Build and test

The same commands are intended for native Windows, macOS, and Linux:

```text
cmake -S native_spikes/cpp20 -B work/cpp20-build -DCMAKE_BUILD_TYPE=Release
cmake --build work/cpp20-build --config Release
ctest --test-dir work/cpp20-build -C Release --output-on-failure
```

On a Clang or GCC development host, enable AddressSanitizer and
UndefinedBehaviorSanitizer with:

```text
cmake -S native_spikes/cpp20 -B work/cpp20-sanitize -DAFN_ENABLE_SANITIZERS=ON
cmake --build work/cpp20-sanitize
ctest --test-dir work/cpp20-sanitize --output-on-failure
```

Every ABI extension requires native three-platform CI, sanitizer evidence,
compiled-wheel installation, versioned metadata, and continued full-output
comparison with the NumPy oracle. No public binary release is implied.
