# C++20 Native 内核

这是独立实现、只依赖标准库的生产性能内核。它通过手写 CPython Stable ABI
适配器进入验证 wheel，但不替代可读 NumPy 数学参考。公开边界是真正的 C ABI，
因此绑定层保持轻薄且可替换。

ABI 1.3 为静态各向异性 P1 扩散、T3/T4 线弹性体积项生成确定性逐单元 COO 和
节点载荷，也能应用规范 CSR。T3/T4 并行入口使用显式线程数、静态连续分块和
固定载荷归并；原串行入口保持兼容。任何非零状态都使全部输出缓冲区无效，调用方
不得使用部分结果。布局和兼容规则见 `docs/specifications/NATIVE_P1_ABI.md`。

## 构建与测试

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
