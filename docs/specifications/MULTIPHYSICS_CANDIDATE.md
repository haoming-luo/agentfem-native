# 二维/三维热—力、有限变形与断裂候选

日期：2026-10-26。目标：`implemented` / `experimental`，不继承 Gate 2/3 的验证等级。

## 六小时执行顺序与验收

1. 共用 T3/T4 仿射几何、标量/向量预备 CSR 图；二维/三维稳态与瞬态热。
2. 小应变热弹性，温度场 → 热应变载荷 → 位移/应力/反力。
3. 有限变形材料解析切线、总拉格朗日装配、Newton、线搜索和减步回滚。
4. AT2 位移—损伤交替求解、节点界限约束与不可逆演化。
5. AgentFEM 显式候选入口、可运行示例、轻量回归与性能证据。
6. 统一审核、安装态和平台验收、路线图与产品汇报。

每项先完成恒等式、解析响应和失败测试再进入下一层。时间盒不等同于全部工程
能力承诺。接触、自接触、壳梁、弧长屈曲跟踪、混合近不可压缩、动态裂纹、塑性
断裂和大变形断裂耦合仍需独立规格。

## 热与线性耦合

仿射 d 维单纯形有 m=d+1 个节点，正体积 V，梯度 G[a,J]。
K_ab=V G_a k G_b；C_ab=rho*c*V*(1+delta_ab)/(m*(m+1))。
后向 Euler： (C/dt+K) T[n+1]=C T[n]/dt+q[n+1]。
热流 q_phys=-k grad(T)。边界输入为向内热通量，故正值进入右端。
Robin 为 h(T-Tambient)，左端添加面质量乘 h，右端添加 h*Tambient 面积分。
每步记录 C(Tnew-Told)/dt+K*Tnew-q 的约束反力以及全局热量余额。
导热张量仅允许相对最大分量 `64*eps` 的舍入级非对称，并取对称均值；不得以
NumPy 默认绝对容差让微小量纲下的非对称张量通过正定准入。

热弹性 sigma=D*(B*u-eps_th)，f_th=integral B^T D eps_th。
三维 eps_th=alpha*(T-Tref)*[1,1,1,0,0,0]。
二维平面应力有效热应变为 alpha*dT*[1,1,0]；平面应变为
(1+nu)*alpha*dT*[1,1,0]，恢复 sigma_zz 时须使用真实三维热应变。
这是温度驱动力学的顺序单向耦合，不声称热弹性阻尼或机械耗散热反馈。

## 有限变形

F_iJ=delta_iJ+sum_a u_ai G_aJ，二维使用 F33=1 的平面应变嵌入。
沿用已记录的可压缩 Neo-Hookean W(F)，J>0；A_iJkL=dP_iJ/dF_kL：

`A_iJkL=mu*delta_ik*delta_JL+lambda*FinvT_iJ*FinvT_kL+(mu-lambda*ln J)*FinvT_iL*FinvT_kJ`。

r_ai=V*P_iJ*G_aJ-f_ai，K_ai,bk=V*G_aJ*A_iJkL*G_bL。
可选均匀逐单元热伸长 s=1+alpha*(T-Tref)>0，Fe=F/s，使用
W_ref(F,T)=W(Fe)，P=P_e/s，A=A_e/s²；明确能量以参考体积计。
Newton 用真实自由残差收敛；试探 J<=0 拒绝；线搜索缩步；失败增量二分。
已接受位移不被拒绝试探修改。一般切线未必正定，不能假设 CG 一定适用；
使用自主重启 GMRES 或显式可选通用提供者。当前不跟踪极限点后的平衡分支。
取消可在 Krylov/Newton/交替迭代安全检查点发生，错误只附带已提交状态；不是
已有持久化检查点协议。程序内迭代计数不冒充物理步数预算。

固定拓扑、固定约束的 Newton 在首次线性求解时准备一次 `CSRConstraintPlan`，
后续只变换当前切线的数值与右端；不得复用旧切线或旧数值预条件器。缓存局限于
一次 `solve` 调用，图或约束不一致必须拒绝。Python 诊断开关
`reuse_constraint_plan=False` 仅用于等价与完整求解计时，不改变物理输入/容差。

## 拓扑与刚体准入

候选单纯形算子拒绝未被任何单元使用的节点。按单元节点邻接连通分量检查零空间：
稳态热要求每个分量至少一个固定温度或正 Robin 面；力学要求每个分量的受约束
刚体迹达到二维 3 / 三维 6 的秩。刚体基先去掉该分量的原点并按几何尺度归一化，
因此不会仅因模型平移或长度单位改变而改变判定。该检查是必要条件，不替代铰接
机构、零能模式或病态条件数检测；不得把无载荷零残差视为唯一解证明。

## T4 线性动力候选

三维一致质量 `M_ai,bj=rho*V*(1+delta_ab)*delta_ij/20`；集中质量为行和。
复用已存在的中心差分/Newmark 程序，但 T4 空间能力只标记 implemented，当前
仅 Python API；质量/模态时间阶/零约束/重启证据不等价于三维波传播完整验证。

## AT2 准静态脆性断裂

采用各向同性无张压分裂模型：
E(u,d)=integral [((1-d)^2+kappa)*psi0(eps(u))+Gc/(2ell)*d²
+Gc*ell/2*|grad d|²] dV - f^T u。
每单元 B*u 常量，损伤使用 P1：积分 (1-d)^2 必须使用一致单纯形质量，
不能用单元平均损伤平方代替。固定 u 时
A_d=(2psi0+Gc/ell)*M0+Gc*ell*K0，b_d=2psi0*V/m。
在 d_previous<=d<=1 的箱约束内求解凸二次子问题，并检查 KKT；不能只截断
无约束解。交替位移和损伤直至场变化与平衡残差都满足容差，再接受整步。
不使用历史最大能量替代变分约束，因此有同一离散势能的下降检验。
本模型会在压缩下损伤，仅为隔离候选；不用于裂面闭合、压缩断裂或混合模态
工程结论。预裂纹可由初始 d=1 节点给出；加载卸载不允许愈合。

## 数学资料与独立性

后续轻量制造解：单位方形/立方体 `T=sum_i x_i(1-x_i)`，`k=1`，体热源
`q=2*d`，外边界固定解析温度。热弹性平面应变/三维使用 `u_i=x_i²`，
`T=Tref+sum_i x_i`，常数 alpha。均匀材料体力每分量为
`-(4*mu+2*lambda)+(2*mu+3*lambda)*alpha`。在 4/8 划分上，检查体积加权
单元重心采样误差：温度/位移二阶、热流一阶；这里是声明的采样范数，不是精确
积分 L2 误差，也不代替更广的非结构网格验收。真值由方程代入得到，不依赖其他
有限元软件。

- Sierra/SM 总拉格朗日公开理论说明：
  https://www.sandia.gov/files/sierra/SM_Users_5_20/main/elements/sections.html
- Gerasimov 与 De Lorenzis，不可逆相场约束研究：
  https://arxiv.org/abs/1811.05334
- Illinois 2025 相场断裂课程的变分能量与不可逆约束：
  https://pamies.cee.illinois.edu/assets/pdf/Short_Course_USNCCM_2025.pdf
- COMSOL 热弹性公开理论：
  https://doc.comsol.com/6.0/doc/com.comsol.help.mems/mems_ug_sme.08.13.html

只使用理论、方程和项目既有独立实现。搜索引擎曾返回一个其他 FEM 项目的源码
片段，未打开或作为实现输入；新实现从上述方程独立推导，不采用第三方 FEM 代码。
