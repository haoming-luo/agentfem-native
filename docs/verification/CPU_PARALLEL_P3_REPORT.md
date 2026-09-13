# P3 确定性 CPU 并行装配候选报告

**日期：** 2026-09-13

**成熟度：** 本地实现候选；Tier-1 平台验收待执行

**范围：** C++ ABI 1.3 的 T3/T4 体积装配

## 交付内容

- 保留 ABI 1.0–1.2 的所有串行函数和默认单线程行为；
- 新增 T3/T4 显式 `thread_count` 并行入口，不读取环境变量或硬件默认并发数；
- 单元按输入顺序静态连续分块，各线程只写独占 COO 区间；
- 节点载荷使用线程私有缓冲区，并按线程编号固定归并；
- 线程数和额外工作区进入 `ExecutionPlan`、计划摘要、能力清单与执行收据；
- C ABI 捕获线程/临时资源失败并返回稳定状态码，异常不越过 ABI；
- Python Stable ABI 包装在执行 C++ 时释放 GIL。

本增量不包含线程池、自动线程选择、CSR 图复用、SIMD、并行 SpMV、MPI 或 GPU，
也不把 Gate 2 或 P3 整体标记为完成。

## 数值与确定性证据

本地科学套件共 172 项测试通过。新增证据验证：

1. T3/T4 的并行 rows、columns、data 与串行输出逐位相同；
2. 固定输入和线程数重复执行时，四个输出数组均逐位相同；
3. 不同线程数的载荷仅允许固定归约分组导致的舍入级差异；
4. 零线程、非整数和布尔线程数在进入生产计算前失败；
5. C++ 头文件、直接 ABI 契约、Python 包装、Agent 计划与证据字段一致；
6. AddressSanitizer 与 UndefinedBehaviorSanitizer 的直接 C++ 契约通过；
7. 本地 `cp311-abi3` wheel 构建并安装到隔离目录后，172 项测试再次通过；
8. 独立性扫描通过，未引入第三方有限元实现或新依赖。

## macOS arm64 本地性能记录

环境为 Python 3.12.14、NumPy 2.3.5、10 逻辑 CPU、AgentFEM Native 0.7.0a1。
每项先预热，随后取七次热进程墙钟时间的中位数。

| 单元 | 规模 | 串行 | 2 线程 | 4 线程 | 8 线程 | 最佳加速 |
|---|---:|---:|---:|---:|---:|---:|
| T3 | 131,072 cells / 132,098 DOF | 6.812 ms | 3.775 ms | 2.697 ms | 2.808 ms | 2.53× |
| T4 | 10,368 cells / 6,591 DOF | 7.628 ms | 4.064 ms | 2.159 ms | 1.906 ms | 4.00× |

T3 最大并行载荷差为 `1.70e-21`，T4 为 `6.51e-19`；所有记录的 COO 完全相同，
固定线程数重复性全部通过。原始机器可读记录见
[`benchmarks/results/2026-09-13-darwin-arm64-cpu-parallel.json`](../../benchmarks/results/2026-09-13-darwin-arm64-cpu-parallel.json)，
重跑入口为 [`benchmarks/cpu_parallel_assembly.py`](../../benchmarks/cpu_parallel_assembly.py)。

## 声明边界与下一步

- 本记录只说明当前 macOS arm64 机器上的方向，不推断 Windows、Linux 或其他
  CPU 的性能；
- 2.53×/4.00× 尚未达到长期“八核至少 6×”目标，后续需通过复用稀疏图、减少
  中间 COO、线程池/批次粒度和 SIMD 继续优化，而不是夸大当前结果；
- ABI 1.3 必须通过 Windows x86_64、Linux x86_64、macOS x86_64/arm64 的 wheel、
  C/C++ 契约、sanitizer、Python Stable ABI 与科学套件，才成为正式平台证据；
- CI 成功也只关闭本次并行增量的平台验收，不自动完成 Kernel Contract 0.2、
  Gate 2 或 P3 的其他剩余项。
