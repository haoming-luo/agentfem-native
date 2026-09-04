# C++20 P1 batch-kernel spike

This is an independently implemented, standard-library-only performance
experiment. It is not yet part of the Python wheel and does not replace the
readable NumPy oracle. Its public boundary is a true C ABI so that a future
binding can be thin and replaceable.

The current function emits deterministic per-cell COO entries and a nodal
load for static anisotropic P1 diffusion. A nonzero status invalidates all
output buffers; callers must not consume partial output.

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

Promotion into the production package requires native three-platform CI,
sanitizer evidence, compiled-wheel packaging, versioned ABI metadata, and
continued full-output comparison with the NumPy oracle.
