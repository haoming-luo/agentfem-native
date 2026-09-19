# Kernel Contract 0.2：AgentFEM 线性力学边界

状态：Gate 2 限定范围内已验证；兼容 0.1 热传导请求。

## 产品目标

Kernel Contract 是 AgentFEM 与自主求解内核之间唯一稳定的计算边界。它让
AgentFEM 在执行前回答“能不能算、准备怎样算、预计消耗多少”，在执行后回答
“实际怎样算、数值结果是什么、证据绑定到哪一个请求”。它不是通用弱形式
语言，也不承载 AgentFEM 的 GUI、智能体状态或当前 AF-IR 内部对象。

0.2 的最小闭环覆盖：

- 0.1 已有的二维 P1 三角形稳态扩散；
- 二维 T3 小应变线弹性，平面应力或平面应变；
- 三维 T4 小应变线弹性；
- 命名节点集、边/面集和单元集；
- 常量体力、位移约束、边界力和分区材料；
- 原生稀疏、NumPy 有界对照和可选 SciPy 提供者；
- 执行前资源计划，以及执行后的结果、运行环境和摘要证据。

## 兼容性

`contract` 固定为 `agentfem.native-kernel-request`。实现同时接受
`contract_version = 0.1.0` 和 `0.2.0`：

- 0.1 只接受原有二维稳态热传导，返回 0.1 结果形状；
- 0.2 接受热传导与线性静力学，返回带计划和证据的 0.2 结果；
- 不支持的版本、物理、维数、提供者和装配路径必须在装配前以稳定错误码和
  JSON 路径拒绝；
- 版本化 Schema 随 Python 包安装，不依赖网络。

## 0.2 线性力学请求

`study` 使用 `analysis = linear_static`、`physics = solid_mechanics`，维数是
2 或 3。网格采用零基索引：二维单元为三个节点、边界实体为两个节点；三维
单元为四个节点、边界实体为三个节点。

`physics.material` 至少给出 `young_modulus` 和 `poisson_ratio`。二维可给出
`model = plane_stress | plane_strain` 和正厚度 `thickness`。`body_force` 是
与维数一致的有限常向量。`materials` 用 `cell_set` 和同形的 `material`
覆盖默认材料。

`dirichlet` 每项包含 `node_set`、可空的 `component` 和 `value`。给定分量时
值为标量；分量为空时值必须是完整位移向量。二维分量为 `x/y`，三维为
`x/y/z`。`traction` 每项包含 `boundary_set` 和完整向量 `value`。

`procedure.kind` 固定为 `linear_elasticity`。`linear_algebra` 可为 `native`、
`native_sparse`、`numpy`、`scipy` 或 `auto`。`assembly` 二维可为 `auto`、
`reference`、`vectorized`、`native`；三维不接受 `vectorized`。`thread_count`
必须为正整数。

## 规划、执行与证据

`plan_kernel_request` 只构造拥有的网格和问题对象并生成计划，不装配、不求解。
计划给出问题类型、能力成熟度、单元/自由度规模、COO 条目数、CSR 非零上界、
峰值内存保守上界、实际选择的提供者/装配路径、线程工作区、结构摘要和警告。

`run_kernel_request` 必须复用同一计划并通过 `execute_plan` 执行，禁止维护第二套
求解分派。0.2 成功结果还包含：

- 节点位移、单元应变和单元应力；
- 自由自由度残差、合力、反力、平衡范数和应变能；
- 可选力学 VTK 文件的相对路径与 SHA-256；
- 请求规范 JSON 的摘要、执行计划摘要和执行证据摘要；
- 操作系统、架构、Python、NumPy、原生 ABI、装配路径和线性代数收敛信息。

摘要证明“这些字节和这次执行相互绑定”，不证明模型代表真实工程。T3/T4 的
`verified` 来自 G2-01 至 G2-12 科学、契约和三平台证据闭环，而不是单次成功
执行；范围外能力不得继承这个成熟度。

## 明确不做

0.2 不包含通用弱形式 DSL、非线性全局迭代、动力学 JSON 契约、MPI、GPU、
高阶单元、接触或 AgentFEM 私有对象适配。这些能力只有在 Gate 2 通过且有独立
规格、证据和资源边界后才能进入生产路径。
