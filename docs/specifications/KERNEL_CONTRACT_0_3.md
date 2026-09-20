# Kernel Contract 0.3：AgentFEM 线性动力边界

**状态：** 实现候选  
**兼容性：** 保留 0.1 热传导与 0.2 线性静力语义

## 请求模型

线性动力请求必须声明：

```text
study.analysis = linear_transient
study.physics = solid_mechanics
study.dimension = 2
procedure.kind = linear_dynamics
```

网格和小应变弹性材料沿用 0.2 的 T3 表达；`physics.density` 必须为正有限数。
`procedure` 还包含 `integrator`、`time_step`、`steps`、`mass` 和可选
`load_scale`。当前积分方法为：

- `central_difference`：只接受 `mass=lumped`；
- `newmark_average_acceleration`：接受 `mass=consistent` 或 `lumped`。

`load_scale` 可以是有限常数，或包含严格递增 `times` 与等长有限 `values` 的
分段线性曲线。曲线必须覆盖从初始/重启时间到本执行段末的完整绝对时间区间，
Native 不做静默外推。

## 状态与重启

`initial_state.displacement` 和 `initial_state.velocity` 接受 `[node][2]` 数组，
省略时为零。固定自由度上的初值必须为零。请求携带 `restart` 时不得再携带非空
初始状态；重启对象包含方法、步号、时间、位移、速度、加速度和确定性摘要。

成功结果返回完整接受状态历史、能量/外力功/平衡误差、约束反力和最后状态
检查点。检查点摘要不绑定请求本身；请求、计划和执行另由证据摘要分别绑定。

## 资源与控制

`execution.budget` 可声明 `maximum_dofs`、`maximum_peak_bytes` 和
`maximum_steps`。规划阶段返回保守历史内存上界；执行前再次验证预算。实时取消
和进度回调由调用方传入 `ExecutionContext`，不序列化进 JSON。

预算超限、取消、显式时间步不安全、载荷曲线覆盖不足、检查点不一致及不支持的
范围必须以稳定错误码和 JSON 路径失败，不产生伪成功结果。

## 明确排除

0.3 只覆盖无阻尼 T3 线性瞬态和零位移固定约束。不支持非零/时变位移约束、
阻尼、T4 动力、非线性、自适应减步、MPI、GPU 或时间序列 VTK。接口存在不等于
科学成熟度晋级；只有 Gate 3 线性验收矩阵和三平台证据全部关闭后才能标记为
`verified`。
