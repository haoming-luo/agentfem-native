# C++20 Native 内核

这是独立实现、只依赖标准库的生产性能内核。它通过手写 CPython Stable ABI
适配器进入验证 wheel，但不替代可读 NumPy 数学参考。公开边界是真正的 C ABI，
因此绑定层保持轻薄且可替换。

ABI 1.6 候选在既有 1.5 基础上新增显式 IC0/LDLT 研究入口；默认求解器不变。
既有路径为静态各向异性 P1 扩散、T3/T4 线弹性体积项生成确定性逐单元 COO 和
节点载荷，也允许已准备的串行 T3/T4 省略重复 COO 行列索引；同时能应用规范
CSR，并按原 COO 贡献顺序回填其数值。T3/T4 并行入口使用显式线程数、静态连续
分块和固定载荷归并；原串行非空缓冲区行为保持兼容。任何非零状态都使全部输出
缓冲区无效，调用方不得使用部分结果。布局和兼容规则见
`docs/specifications/NATIVE_P1_ABI.md`。

## 构建与测试

原生 Windows、macOS、Linux 使用相同构建命令：

```text
cmake -S native_spikes/cpp20 -B work/cpp20-build -DCMAKE_BUILD_TYPE=Release
cmake --build work/cpp20-build --config Release
ctest --test-dir work/cpp20-build -C Release --output-on-failure
```

Clang/GCC 开发机可启用地址与未定义行为检查：

```text
cmake -S native_spikes/cpp20 -B work/cpp20-sanitize -DAFN_ENABLE_SANITIZERS=ON
cmake --build work/cpp20-sanitize
ctest --test-dir work/cpp20-sanitize --output-on-failure
```

每个 ABI 增量必须完成原生三平台、内存安全、安装态、版本元数据与可读参考的
完整输出对照。开发 wheel 验收不意味着对外发布二进制包。
