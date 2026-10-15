# Buck Neural Event ODE 全局设计与训练规范

原设计日期：2026-08-25
配置同步：2026-09-13

本文从 `2026-08-25-buck-neural-event-ode-design.md` 派生，统一定义模型、数据、训练目标、配置与验收规范；原设计保留。

文档范围：MATLAB/Simulink 数据链、MAT→NPZ v3、Python DCM 模型、训练与评估。
适用目录：`topologies/non_isolated_unidirectional/buck/`

配置依据：MATLAB 电路与数据生成参数以 `topologies/non_isolated_unidirectional/buck/configs/hybrid_event_matlab_v3/project.json` 及实际 `buck_ideal_hybrid.slx` 为准。Python 入口默认选择 `input_voltage_relative.json`；中心差分与联合电压导数监督通过 `central_derivative_rollout.json` 显式选择。本文描述全局方法与实现契约，具体运行状态、日志和验收结果由各运行产物记录。

## 1. 目标

输出仅支持 DCM 的神经事件模型。模型必须用同一组参数拟合固定 DCM 负载下不同输入电压和不同初始状态的完整开关波形，并在每个 PWM 关断区间自行学习续流二极管自然截止事件。所有正式数据的每个记录周期都必须在下一 PWM 上升沿前发生一次自然 `OFF_DIODE -> IDLE` 事件并形成正 IDLE 区间。

正式模型只采用可学习神经事件函数：

$$
g_\phi(z_i,z_v,s)=0.
$$

本版固定负载为：

$$
R=80\Omega.
$$

该值高于理想临界负载 $R_{\mathrm{crit}}\approx40\Omega$，可提供明确的自然截止事件和正 IDLE 区间，同时不过度接近 BCM 边界。$R$ 只作为电路配置和数据 metadata 保存，不作为神经网络输入。

本版固定 $L,C,R,D,f_s$，只研究输入电压和初始状态变化，不声称泛化到任意元件参数或任意硬件寄生参数。

地面真值来自本拓扑自包含的 `matlab/` 模型与脚本；原始 MAT 和处理后 NPZ 分别保存在 Buck 的 `data/raw/hybrid_event_matlab_v3/` 和 `data/processed/hybrid_event_matlab_v3/`。

## 2. 训练配置与运行方式

以下路径均相对 `topologies/non_isolated_unidirectional/buck/`。配置位于 `configs/hybrid_event_matlab_v3/`：

| 配置 | 网络坐标 | 模式预训练差分 | rollout 电压导数损失权重 | 用途 |
| --- | --- | --- | ---: | --- |
| `project.json` | 固定参考坐标 | 前向差分 | 0 | MATLAB 数据契约与固定归一化基线 |
| `input_voltage_relative.json` | 输入电压相对坐标，保留 Vin 条件 | 前向差分 | 0 | Python CLI 默认；实验 A |
| `central_derivative.json` | 同上 | 中心差分 | 0 | 实验 B |
| `central_derivative_rollout.json` | 同上 | 中心差分 | 0.01 | 联合导数监督；实验 C |

后三种配置的 `config_status` 为 `experimental`，`run_profile` 为 `formal`；正式课程长度不等同于已经完成效果验收。

从 Buck 目录启动配置 C 的示例（每个配置、seed 和 IDLE profile 使用独立输出目录）：

```powershell
python -u train.py --config configs/hybrid_event_matlab_v3/central_derivative_rollout.json --device cuda --seed 20260804 --idle-profile soft_idle_loss --out-dir artifacts/runs/central_derivative_rollout/seed_20260804/soft_idle_loss
```

恢复已有运行时使用原输出目录，并追加对应的 `--resume <run-dir>/checkpoint.pt`；仅在原进程退出后启动恢复。同一输出目录通过 `training.lock` 防止并发写入。A/B/C 的完整 checkpoint 不能跨配置恢复；从头训练的配置实验也不能直接拿旧 run 的最终权重冒充新训练结果。

相关说明见 [导数监督实验](../../../topologies/non_isolated_unidirectional/buck/docs/derivative-supervision.md)、[输入电压相对归一化](../../../topologies/non_isolated_unidirectional/buck/docs/normalization-experiment.md) 和 [GPU 执行与 batch 覆盖](../../../topologies/non_isolated_unidirectional/buck/docs/gpu-performance.md)。

## 3. 电路与时间规范

### 3.1 固定参数

首版保留当前电路参数：

| 参数 | 数值 |
| --- | ---: |
| 电感 $L$ | $500\,\mu\mathrm H$ |
| 电容 $C$ | $100\,\mu\mathrm F$ |
| 负载 $R$ | $80\Omega$ |
| 占空比 $D$ | $0.5$ |
| 开关频率 $f_s$ | $20\,\mathrm{kHz}$ |
| 周期 $T_s$ | $50\,\mu\mathrm s$ |
| 每周期采样区间 | 100 |
| 归一化 RK4 步长 $h$ | 0.01 |
| 半步验证步长 | 0.005 |

理想 Buck 的临界负载近似为：

$$
R_{\mathrm{crit}}
=
\frac{2L}{(1-D)T_s}
\approx40\Omega.
$$

### 3.2 归一化

采用周期归一化时间：

$$
\tau=\frac{t}{T_s}.
$$

状态与工况归一化为：

$$
\hat i_L=\frac{i_L}{1\,\mathrm A},
\qquad
\hat v_C=\frac{v_C}{32\,\mathrm V},
$$

$$
\hat v_S=\frac{v_S}{32\,\mathrm V}.
$$

负载固定为 $R=80\Omega$。

2026-09-10 更新：上述固定参考坐标仍用于积分器、监督损失与评估。网络内部采用输入电压相对坐标：

$$
s=\hat v_S>0,\qquad z_i=\hat i_L/s,\qquad z_v=\hat v_C/s=v_C/v_S.
$$

ON/OFF 神经网络学习 $dz/d\tau=F_\theta(z_i,z_v,s)$，包装层返回 $d\hat x/d\tau=sF_\theta$。事件网络使用 $g_\phi(z_i,z_v,s)$，保留绝对输入电压条件，不强制整体的齐次缩放。IDLE 保持两维神经网络输入 $(z_i,z_v)$，输出同样乘 $s$；因此该模式仍带有相对导数不依赖绝对电压的假设。

每条轨迹的输入电压必须恒定且严格为正；不适用于时变 Vin。电流、电压导数不硬编码电路方程；默认 soft IDLE 仍由神经网络输出两种导数。所有损失仍以固定参考坐标计算，不能改用相对坐标 NRMSE 与旧表直接比较。时间、事件映射、网络层宽和激活、训练数据保持不变；中心差分、电压导数损失及运行时 batch 以第 2、11 节为准。

默认 Python 配置为 `input_voltage_relative.json`，固定归一化基线保留在 `project.json`。不同归一化配置从头训练；完整 checkpoint 检查配置/结构签名，纯权重使用独立版本键拒绝旧固定归一化及旧强制缩放权重。详细方案见 `topologies/non_isolated_unidirectional/buck/docs/normalization-experiment.md`。


## 4. 混合状态机

### 4.1 连续状态和离散模式

连续状态为：

$$
x=(i_L,v_C).
$$

离散模式为：

| 编码 | 模式 | 含义 |
| ---: | --- | --- |
| 0 | `ON` | MOSFET 导通 |
| 1 | `OFF_DIODE` | MOSFET 关断，二极管续流 |
| 2 | `IDLE` | 二极管截止，电感电流为零 |

### 4.2 已知 PWM 事件

PWM 上升沿和下降沿是已知外生事件：

- 周期开始：切换到 `ON`；
- $\tau\bmod1=D$：切换到 `OFF_DIODE`；
- PWM 事件严格对齐 $h=0.01$ 网格；
- 连续状态在 PWM 事件处不跳变。

### 4.3 神经状态事件

仅在 `OFF_DIODE` 模式监测：

$$
g_\phi(z_i,z_v,s).
$$

符号约定：

- $g_\phi>0$：尚未发生自然截止；
- $g_\phi=0$：发生 `OFF_DIODE -> IDLE` 事件；
- 沿 OFF 反事实动力学越过事件面后 $g_\phi<0$。

每个 PWM 周期最多允许一次该事件。进入 `IDLE` 后，本周期不再调用事件网络。

### 4.4 恒等事件映射与 IDLE 对照配置

自然截止事件只改变离散模式，不改变连续状态。该映射对应参考论文 Algorithm 1 的瞬时更新函数 $h$。事件前状态记为 $x^-$，事件后状态记为 $x^+$，两种配置统一采用恒等映射：

$$
x^+=h_{\mathrm{id}}(x^-)=x^-,
$$

即：

$$
i_L^+=i_L^-,
\qquad
v_C^+=v_C^-.
$$

理想事件应由神经事件函数定位到 $i_L\approx0$，因此训练直接惩罚事件处电流：

$$
L_{\mathrm{event-current}}
=
\left|\hat i_L(\hat t^*)\right|^2.
$$

保留两个正式配置：

- `soft_idle_loss`：IDLE 网络自由学习 $d\hat i_L/d\tau$ 和 $d\hat v_C/d\tau$，用软损失约束电流；
- `hard_idle_constraint`：使用相同事件恒等映射，但在 IDLE 中强制 $d\hat i_L/d\tau=0$，作为结构化对照。

两种配置都不向 IDLE 网络输入 $v_S$，因为理想非同步 Buck 在该模式与输入源断开。

## 5. 神经网络结构

### 5.1 ON/OFF 向量场

`f_on` 和 `f_off` 是互不共享参数的 MLP：
$$
(z_i,z_v,s)
\longrightarrow
\left(
\frac{d\hat i_L}{d\tau},
\frac{d\hat v_C}{d\tau}
\right).
$$

上式右侧为包装层还原后的导数；原始 MLP 输出为相对坐标导数，还原时乘 $s$。

每个网络结构为：

```text
3 -> 64 -> 64 -> 2
```

隐藏层使用 `Tanh`，输出层线性。

### 5.2 IDLE 向量场

两种配置使用相同的原始 IDLE MLP，以隔离结构约束的影响：

```text
2 -> 64 -> 64 -> 2
```

输入为 $(z_i,z_v)$，还原到固定参考坐标后的输出为：

$$
(u_i,u_v)=s f_{\mathrm{IDLE},\theta}(z_i,z_v).
$$

`soft_idle_loss` 返回：

$$
\left(
\frac{d\hat i_L}{d\tau},
\frac{d\hat v_C}{d\tau}
\right)
=(u_i,u_v).
$$

`hard_idle_constraint` 返回：
$$
\left(
\frac{d\hat i_L}{d\tau},
\frac{d\hat v_C}{d\tau}
\right)
=(0,u_v).
$$

因此两种配置具有相同主干、输入和参数规模，事件映射也相同；唯一前向结构差异是是否将 IDLE 电流导数替换为零。固定负载的影响由训练数据隐式包含。

### 5.3 神经事件函数

事件网络结构为：

```text
3 -> 64 -> 64 -> 1
```

输入为：

$$
(z_i,z_v,s).
$$

隐藏层使用平滑的 `Tanh`，输出层线性。输出是无量纲事件得分，不解释为电流。

所有线性层使用 Xavier normal 初始化，偏置初始化为零。事件网络必须先通过有监督预训练门槛，才允许接入 Hybrid RK4 联合训练，因此不使用随机大方差“碰根”初始化。

## 6. 批量 Hybrid RK4

### 6.1 选择

主训练和正式推理使用本拓扑自包含的批量 Hybrid RK4。它使用标准 PyTorch autograd。

### 6.2 固定网格步进

每个固定步执行：

1. 根据 PWM 网格事件和逐样本模式选择向量场；
2. 对 `OFF_DIODE` 样本执行完整 RK4 候选步；
3. 计算步前、步后事件值；
4. 检测正到非正穿越；
5. 对穿越样本定位步内事件；
6. 保持连续状态不变并将离散模式切换到 `IDLE`；
7. 用所选 soft/hard IDLE 动力学积分剩余子步；
8. 未穿越样本保留完整 OFF 候选状态。

穿越掩码为：

```python
crossed = (
    (mode == OFF_DIODE)
    & (g0 > 0)
    & (g1 <= 0)
)
```

### 6.3 两次可微割线细化

完整步给出初始区间 $\alpha\in[0,1]$。先计算：

$$
\alpha_0
=
\operatorname{clip}
\left(
\frac{g_0}{g_0-g_1},0,1
\right).
$$

随后执行两次细化。每次使用长度为 $\alpha h$ 的 OFF RK4 子步重新计算候选事件状态和事件值，并用张量 `where` 更新正负括区间。正负括区间保证割线分母非零，所有候选 $\alpha$ 被限制在当前括区间内。

细化后的事件状态通过分数 RK4 子步获得，不使用状态线性插值：

$$
x_e
=
\operatorname{RK4}
\left(
f_{\mathrm{OFF}},x_k,\alpha h
\right).
$$

恒等映射下事件状态仍为 $x_e$。事件后的剩余子步为：

$$
x_{k+1}
=
\operatorname{RK4}
\left(
f_{\mathrm{IDLE}},x_e,h(1-\alpha)
\right).
$$

$\alpha$、事件状态和后续轨迹均保留 autograd 计算图。括区间选择和事件数量仍是分段可微的离散决策；这是事件方法的固有限制。

### 6.4 异常保护

- 若 OFF 起点 $g_\phi\le0$，记录 `invalid_initial_guard`，按 $\alpha=0$ 安全进入 IDLE，并施加强事件符号损失；
- 若本周期无正负穿越，则保持 `OFF_DIODE` 到下一 PWM 上升沿以保证数值有限，同时记录 DCM 漏事件；该结果不得被视为受支持工况；
- 每周期首个自然事件后禁用事件检测，防止重复触发；
- 非有限状态、非有限事件值或割线分母异常必须中止该批；当前默认在 rollout 返回前集中检查轨迹、被替换的候选状态和事件值，事件细化仍即时检查。调试可使用 `check_finite_every_step=True` 恢复逐步检查；
- 使用 $h=0.005$ 对正式权重做半步一致性评估。

## 7. Simulink 真值与事件标注

### 7.1 模型升级

单一活动模型为 `buck_ideal_hybrid.slx`。真值电路使用 Simscape Electrical 物理网络，包含 `Solver Configuration` 和 `Electrical Reference`，并允许续流二极管自然截止进入 IDLE。旧 Specialized Power Systems 模型及其 `powergui`、`Series RLC Branch`、器件 snubber 参数不再定义当前真值电路。

模型必须显式记录：

- `iL`；
- `vC`；
- `gate`；
- 续流二极管串联支路电流 `iD`。

Diode 支路使用 Simscape `Current Sensor` 测量阳极到阴极的电流，经 `PS-Simulink Converter` 输出并记录为 `iD`；该方向在续流导通时定义为正。电感电流和输出电压也分别经传感器及转换器输出为 `iL`、`vC`。`diode_on` 不依赖库内部不可移植状态，而由原始 `iD` 按第 8.4 节的迟滞规则确定。

当前 Diode 使用分段线性模型，固定 `Ron=1e-4 ohm`、`Vf=1e-4 V`、`Goff=1e-8 S`、`CJ=0 pF`，并关闭 reverse recovery；MOSFET 使用 Ideal Switching 模型，固定 `Rds=1e-4 ohm`、`Goff=1e-8 S`、`Vth=0.5 V`，保护二极管模式为 external。`Goff` 是 Simscape 器件的关断态电导，不等同于旧 Specialized Power Systems 的并联 RC snubber。电感采用 `r=0`、`g=1e-9 S`，输出电容采用 `r=1e-6 ohm`、`g=0`，用于当前连续物理网络的数值定义。模型中的实际设置和 `project.json` 审计副本必须同步，生成器将 JSON 中的器件参数写入轨迹 metadata；若两者不一致，不得将生成结果作为正式数据。

### 7.2 事件真值

事件真值由已知 `gate` 边界和原始 `iD` 决定；`diode_on` 只是由同一 `iD` 迟滞生成的审计标签：

- 只在 `gate=0` 的 OFF 区间搜索；
- 二极管从导通变为截止且早于下一 PWM 上升沿，标记自然事件；
- 使用变步长原始日志确定事件时刻，不先量化到 $0.01T_s$ 网格；
- 在事件时刻插值得到 $(i_L,v_C)$；
- $i_L$ 仅校验事件物理一致性，不作为模型推理 guard。

求解器保持 `ode23t`，`RelTol=1e-8`、`AbsTol=auto`、`AutoScaleAbsTol=on`，最大步长 $5\times10^{-8}\,\mathrm s=0.001T_s$，使用非自适应零交叉算法。事件时刻标签必须来自原始变步长日志，固定网格只用于轨迹输出。

### 7.3 DCM 周期验收

正式数据不再划分 CCM、BCM、DCM 类别。每个记录周期都必须满足 DCM 条件：

$$
t^*\le t_{end}-\varepsilon_{DCM}.
$$

所有 split 统一使用：

$$
\varepsilon_{DCM}=0.02T_s,
$$

确保固定 $R=80\Omega$ 的每个事件后都存在可学习的正 IDLE 区间，并排除与 PWM 边界重合的 BCM 周期。

每个周期必须恰好包含一次自然 `OFF_DIODE -> IDLE` 事件。`natural_event_mask` 在正式数据中必须全为真；保留该字段是为了审计和明确拒绝漏标，而不是支持无事件工况。

### 7.4 数据拒收与失败分类

候选轨迹仅在 DCM 物理验收失败时拒收并补采。任一记录周期满足以下任一条件时，整条轨迹均按 `BuckNeuralODE:DCMViolation` 拒收：

- 未恰好包含一次自然 `OFF_DIODE -> IDLE` 事件；
- 唯一事件不满足 $t_{off}<t^*\le t_{next\_on}-\varepsilon_{DCM}$；该条件同时排除 OFF 区间外事件、BCM 边界事件和不足 $\varepsilon_{DCM}$ 的 IDLE；
- 事件处电流、IDLE 区间电流或全轨迹反向电流不满足第 8.4 节规定的容差。

以下情况不是可由补采掩盖的工况拒收，而是模型、生成器或日志链错误；出现任一情况必须立即终止整个 split：

- 任一必需原始日志的时间或数据包含非有限值；
- 原始 `gate` 与配置定义的 PWM 边沿或固定网格不对齐；
- 必需信号或 schema 缺失，或任一变步长日志未覆盖完整记录区间；
- 由原始 `iD` 和 `gate` 派生的 `diode_on`、事件时刻或 `interval_mode` 与迟滞状态机、事件括区间或对应 OFF 区间不一致。

这里“补采”仅指放弃当前 LHS 候选并继续使用下一个候选，不是对同一条 CCM 或 BCM 波形重新插值或固定网格重采样。产生 CCM 或 BCM 的候选轨迹不得写入正式数据。

## 8. 工况与数据集

### 8.1 固定 DCM 工作点

所有数据固定：

$$
R=80\Omega.
$$

只对输入电压和初始状态采样。电阻不形成数据维度、不参与网络输入，也不设置负载插值或外推 split。Simulink 导出和 NPZ metadata 逐轨迹记录 $R=80\Omega$。

### 8.2 数据划分

| split | 输入电压 | 每个电压的初值数 | 数量 | 周期 |
| --- | --- | ---: | ---: | ---: |
| `train` | 12,16,20,24,28,32 V | 12 | 72 | 20 |
| `validation` | 14,22,30 V | 6 | 18 | 20 |
| `test_seen` | 训练电压 | 6 | 36 | 200 |
| `test_interpolation` | 13,15,17,18,19,21,23,25,26,27,29,31 V | 3 | 36 | 200 |
| `test_voltage_extrapolation` | 10,34,36 V | 6 | 18 | 200 |
| `test_extrapolation_high` | 50,55,60,65,70,75,80 V | 3 | 21 | 200 |

所有 split 都固定 $R=80\Omega$，只保存逐周期 DCM 合格轨迹，并使用互不相同的初值和固定随机种子。高电压 split 保留为强电压外推诊断，不作为核心通过门槛。

### 8.3 初始状态采样

沿用原 CCM 数据生成器的二维 Latin hypercube 采样方式，并按当前活动配置固定负载和初值因子范围。对每个输入电压定义简单参考值：

$$
i_{L,\mathrm{ref}}=\frac{Dv_S}{R},
\qquad
v_{C,\mathrm{ref}}=Dv_S.
$$

它们只用于生成初值，不声明为 DCM 精确稳态。初值为：

$$
i_{L0}=k_i i_{L,\mathrm{ref}},
\qquad
k_i\in[0.75,1.35],
$$

$$
v_{C0}=k_v v_{C,\mathrm{ref}},
\qquad
k_v\in[1.22,1.30].
$$

每个输入电压继续使用二维 Latin hypercube 对 $(k_i,k_v)$ 取样。记录阶段保留这些初值产生的真实暂态；若任一记录周期不是 DCM，则拒收该候选并取下一个候选。

### 8.4 MATLAB 数据生成契约

本节记录 `initialize_buck_model.m`、`generate_dataset.m` 和 `export_trajectory.m` 的现行实施契约。MATLAB 数据生成参数以当前活动 `project.json` 和实际模型为准；字段、阈值、插值方法或拒收规则发生变化时，必须同步更新本设计后再继续下游训练。

#### 8.4.1 模型块与参数

活动模型为 `buck_ideal_hybrid.slx`。现有块和变量绑定固定如下：

| Block Path（模型根目录下） | 用途 | 必须设置 |
| --- | --- | --- |
| `Voltage Source` | 输入电源 | `dc_voltage=vs` |
| `Inductor` | 电感 | `l=L`、`r=0`、`g=1e-9`、`i_L_specify=on`、`i_L=iL0`、初值优先级 High |
| `Capacitor1` | 输出电容 | `c=C`、`r=1e-6`、`g=0`、`vc_specify=on`、`vc=vC0`、初值优先级 High |
| `Resistor` | 负载 | `R=R` |
| `Pulse\nGenerator1` | PWM | Time based、`Period=Ts`、`PulseWidth=50`、相位从 ON 开始 |
| `Diode1` | 续流二极管 | Piecewise linear、`Ron=1e-4`、`Vf=1e-4`、`Goff=1e-8`、`CJ=0 pF`、reverse recovery off |
| `MOSFET\n(Ideal,\nSwitching)` | 主开关 | Ideal switching、`Rds=1e-4`、`Goff=1e-8`、`Vth=0.5`、protection diode external |
| `Current Sensor1` | 二极管支路电流 | 阳极到阴极为正，经 `PS-Simulink Converter2` 输出并记录为 `iD` |
| `Current Sensor` | 电感电流 | 经 `PS-Simulink Converter1` 输出并记录为 `iL` |
| `Voltage Sensor` | 输出电压 | 经 `PS-Simulink Converter` 输出并记录为 `vC` |
| `Solver Configuration` | Simscape 物理网络求解 | 与同一电气网络连接 |
| `Electrical Reference` | 电气参考地 | 与同一电气网络连接 |

`project.json` 的 `matlab.device_library` 必须为 `Simscape Electrical`，`matlab.diode` 和 `matlab.mosfet` 必须保存与上表一致的模型类型、导通参数、关断电导和保护二极管设置。JSON 是 metadata 的审计来源，不会自动覆盖 `.slx` 中的块参数；生成正式数据前必须核对两处一致。若当前物理网络出现奇异或初始状态未生效，生成器必须立即终止；在模型经过单独审查前不得继续生成正式数据。

#### 8.4.2 求解器与日志

| 项目 | 固定值 |
| --- | --- |
| Solver | `ode23t` |
| `RelTol` | `1e-8` |
| `AbsTol` | `auto` |
| `AutoScaleAbsTol` | `on` |
| `MaxStep` | `5e-8` s |
| Zero-crossing algorithm | `Nonadaptive` |
| Simscape 物理网络 | `Solver Configuration` + `Electrical Reference` |
| Signal logging variable | `logsout` |
| Logging decimation | 1 |

`project.json` 当前保留 `matlab.powergui_mode="Continuous"` 作为旧 Specialized Power Systems 配置遗留字段；它不控制当前 Simscape Electrical 物理网络，也不进入轨迹 metadata 或训练语义。除非后续同时更新活动配置及相关审计，否则本文按现有配置保留并记录该字段。`logsout` 必须唯一包含 `iL`、`vC`、`gate`、`iD` 四个 `timeseries`。连续信号保留求解器原始变步长时间戳；不得先重采样再提取事件。每条记录仿真从 $t=0$ 的 PWM 上升沿开始，停止时间严格为 $C T_s$，其中 $C$ 是 split 配置的记录周期数。

#### 8.4.3 固定随机种子和轨迹生成

沿用原 CCM 生成器：每个 split 只初始化一次 MATLAB RNG，然后按配置中的电压顺序依次生成轨迹。

| split | `data_seed` |
| --- | ---: |
| `train` | 20260803 |
| `validation` | 20260813 |
| `test_seen` | 20260823 |
| `test_interpolation` | 20260902 |
| `test_voltage_extrapolation` | 20260912 |
| `test_extrapolation_high` | 20260920 |

进入电压循环前调用：

```matlab
rng(data_seed, "twister");
```

对于每个输入电压，使用原生成器的 `latin_hypercube_2d` 生成 $(k_i,k_v)$ 候选，范围固定为：

$$
k_i\in[0.75,1.35],
\qquad
k_v\in[1.22,1.30].
$$

候选数量为该电压目标轨迹数的 10 倍，与原 CCM 生成器一致。候选被 DCM 验收拒绝后继续取下一项；候选耗尽仍不足目标数量时，以 `BuckNeuralODE:InsufficientDCMTrajectories` 失败。

轨迹 ID 沿用简单的 split 内全局顺序：

```text
{split}_{accepted_index:06d}
```

例如 `train_000001`。metadata 保存实际的 `current_factor` 和 `voltage_factor`。

#### 8.4.4 二极管自然截止事件提取

权威事件信号是方向校正后的原始 `iD`。固定阈值为：

| 名称 | 数值 |
| --- | ---: |
| `diode_on_threshold_a` | $2\times10^{-5}$ A |
| `diode_off_threshold_a` | $1\times10^{-5}$ A |
| `event_current_abs_tol_a` | $1\times10^{-4}$ A |
| `idle_current_abs_tol_a` | $1\times10^{-4}$ A |
| `negative_current_abs_tol_a` | $1\times10^{-4}$ A |
| `idle_diode_current_abs_tol_a` | $1\times10^{-5}$ A |
| `event_guard_time_s` | $1\times10^{-7}$ s |
| `dcm_margin_tau` | 0.02 |

每个周期按以下顺序处理：

1. 用已知 $D=0.5$ 和 $T_s$ 定义 ON、OFF 边界，同时核对原始 `gate` 的上升沿、下降沿；边沿时间误差不得超过 $10^{-12}$ s。
2. 只在 $[t_{off},t_{next\_on})$ 搜索。`iD` 首次超过 `diode_on_threshold_a` 后，将二极管状态置为导通；未出现该条件则拒收。
3. 对 $|i_D|\le$ `diode_off_threshold_a` 的原始样本只在事件状态机中视为零，不修改保存的电流值。
4. 导通后第一次从正电流进入上述零带时形成事件括区间。事件时间在括区间两端对 $i_D=0$ 做线性插值并限制在括区间内；若线性根不在括区间，则使用零带端点。
5. 在原始 `iL`、`vC` 上按事件时间线性插值得到 `event_state`。必须满足 $|i_L(t^*)|\le10^{-4}$ A。
6. 从 $t^*+$ `event_guard_time_s` 到下一 PWM 上升沿，必须同时满足 $|i_L|\le10^{-4}$ A、$|i_D|\le10^{-5}$ A，且 `iD` 不得再次超过导通阈值；否则分别判为 IDLE 电流违规或重复事件。
7. 必须满足 $t^*\le t_{next\_on}-0.02T_s$。每个周期恰好一个合格事件，否则整条轨迹拒收。
8. 全轨迹还必须满足 $i_L\ge-10^{-4}$ A，防止数值解进入非物理反向电流。

`diode_on` 是由上述迟滞状态机生成的诊断标签，不使用 $i_L=0$ 构造事件时间。事件括区间的两个时间和两个原始 `iD` 值必须写入 MAT 文件用于审计。

#### 8.4.5 固定网格和模式标签

对于 $C$ 个记录周期：

$$
T=100C+1,
\qquad
time\_s=(0,dt,\ldots,CT_s)^\mathsf T.
$$

重采样规则固定为：

- `iL`、`vC`、`iD`：在 `time_s` 上线性插值；
- `gate`：在每个固定区间中点以前值保持采样，阈值 0.5；
- `diode_on`：将原始迟滞状态以前值保持采样到固定区间中点；
- `pwm_gate`、`diode_on`、`interval_mode`：长度均为 $T-1$，对应固定区间中点；
- 原始日志必须覆盖 $[0,CT_s]$，覆盖容差为 $100\,\mathrm{eps}(\max(1,CT_s))$。

模式编码为 `uint8`：

| 编码 | 模式 |
| ---: | --- |
| 0 | `ON` |
| 1 | `OFF_DIODE` |
| 2 | `IDLE` |

区间中点处 `gate=1` 时标记 `ON`；`gate=0` 且中点早于本周期事件时标记 `OFF_DIODE`；中点大于或等于事件时标记 `IDLE`。事件恰落在区间中点时，该区间标记为 `IDLE`。状态端点不承担离散模式归属。

#### 8.4.6 原始 MAT v3 schema

每条被接受轨迹保存为一个 MATLAB `-v7` 文件，必须包含：

| 字段 | MATLAB 类型与形状 | 含义 |
| --- | --- | --- |
| `time_s` | `double (T,1)` | 固定状态时间 |
| `inductor_current_a` | `double (T,1)` | 固定网格 $i_L$ |
| `capacitor_voltage_v` | `double (T,1)` | 固定网格 $v_C$ |
| `diode_current_a` | `double (T,1)` | 固定网格原始符号 $i_D$ |
| `input_voltage_v` | `double scalar` | $v_S$ |
| `load_resistance_ohm` | `double scalar` | 必须为 80 |
| `initial_state` | `double (1,2)` | 实际固定网格首状态 |
| `pwm_gate` | `uint8 (T-1,1)` | 区间 PWM |
| `diode_on` | `uint8 (T-1,1)` | 迟滞导通标签 |
| `interval_mode` | `uint8 (T-1,1)` | 三模式标签 |
| `natural_event_mask` | `logical (C,1)` | 必须全真 |
| `event_time_s` | `double (C,1)` | 绝对事件时间 |
| `event_time_tau` | `double (C,1)` | 周期内事件时间，范围 $(0.5,0.98]$ |
| `idle_duration_tau` | `double (C,1)` | $1-\mathrm{event\_time\_tau}$ |
| `event_state` | `double (C,2)` | 事件处 $(i_L,v_C)$ |
| `event_current_a` | `double (C,1)` | `event_state(:,1)` 的审计副本 |
| `event_bracket_time_s` | `double (C,2)` | 事件括区间时间 |
| `event_bracket_diode_current_a` | `double (C,2)` | 括区间原始 $i_D$ |
| `metadata` | `struct scalar` | 参数、种子、版本和审计信息 |

`metadata` 至少包含：`trajectory_id`、`split`、`schema_version`、`random_seed`、`matlab_release`、`input_voltage_v`、`load_resistance_ohm`、`initial_current_a`、`initial_voltage_v`、`current_factor`、`voltage_factor`、`num_cycles`、`L`、`C`、`R`、`fs`、`D`、`dt`、`samples_per_cycle`、求解器设置、`device_library`、`diode_model_type`、`diode_Ron_ohm`、`diode_Vf_v`、`diode_Goff_siemens`、`diode_junction_capacitance_pf`、`diode_reverse_recovery`、`mosfet_model_type`、`mosfet_Rds_on_ohm`、`mosfet_Goff_siemens`、`mosfet_threshold_voltage_v`、`mosfet_body_diode_mode`、全部事件阈值和全部验收余量。

固定网格 MAT 不保存完整变步长日志，以控制 200 周期数据体积；只保存事件括区间。事件提取和全部原始日志验收必须在 `export_trajectory.m` 返回前完成。

#### 8.4.7 输出与拒收

沿用原 CCM 生成器：单条合格轨迹直接保存为 `{trajectory_id}.mat`，同名文件直接覆盖。

候选轨迹未满足“每周期恰好一个 DCM 自然事件”、事件电流、IDLE 电流或反向电流要求时，`export_trajectory.m` 统一抛出 `BuckNeuralODE:DCMViolation`。`generate_dataset.m` 打印一行拒收原因并继续使用下一个 LHS 候选，不另建拒收日志。

`SingularTopology`、`NonFiniteState`、`PWMAlignmentMismatch`、`InsufficientTimeRange`、schema/日志缺失及任何未知异常都表示模型或生成器错误，必须立即终止整个 split，不允许靠补采掩盖。

split 完成后打印实际保存数量，并检查它等于配置目标数量。MAT 文件的 `source_sha256` 由 `mat_to_npz.py` 在转换时计算。

## 9. 数据 schema v3

单一活动 schema 命名为 `hybrid_event_matlab_v3`。配置和数据路径统一升级：

```text
configs/hybrid_event_matlab_v3/project.json
data/raw/hybrid_event_matlab_v3/
data/processed/hybrid_event_matlab_v3/
```

每个 NPZ split 至少包含：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `states` | `(N,T,2)` | $i_L,v_C$ |
| `input_voltage` | `(N,)` | $v_S$ |
| `load_resistance` | `(N,)` | 固定 $R=80\Omega$，只用于审计 |
| `time_s` | `(T,)` | 固定输出时间 |
| `pwm_gate` | `(N,T-1)` | PWM 区间值 |
| `diode_on` | `(N,T-1)` | 由原始 $i_D$ 迟滞生成的导通标签 |
| `interval_mode` | `(N,T-1)` | ON/OFF_DIODE/IDLE |
| `natural_event_mask` | `(N,C)` | DCM 自然事件审计，正式数据必须全真 |
| `event_time_tau` | `(N,C)` | 周期内归一化事件时间，正式数据不得为 NaN |
| `idle_duration_tau` | `(N,C)` | 事件到下一 PWM 上升沿的归一化 IDLE 时长 |
| `event_state` | `(N,C,2)` | 事件状态，正式数据不得为 NaN |
| `event_current_a` | `(N,C)` | 真值事件电感电流审计 |
| `diode_current_a` | `(N,T)` | 固定网格二极管电流诊断 |
| `initial_state` | `(N,2)` | 初值 |
| `trajectory_ids` | `(N,)` | 唯一 ID |
| `metadata` | `(N,)` | JSON metadata |
| `source_sha256` | `(N,)` | MAT 文件校验哈希 |

`interval_mode` 不再是全 split 共用的一维数组，因为不同轨迹的 DCM 事件时间不同。

转换器必须校验：

- MAT 文件数量和配置一致；
- 轨迹 ID、输入电压、初值和随机种子一致；
- 模式只能按 `ON -> OFF_DIODE -> IDLE -> ON` 合法转移；
- 每个周期恰好一个自然事件，事件时间、IDLE 时长、`diode_on` 和 `interval_mode` 一致；
- 所有记录周期满足对应 split 的 DCM 最小 IDLE 时长与 MATLAB 数值容差；
- 原始 metadata 的 schema 与配置一致；
- v2 数据和 checkpoint 被明确拒绝，而不是静默兼容。

## 10. 事件网络预训练

### 10.1 正样本和事件面

对于每个 DCM 周期：

- 事件前、排除事件邻域的 OFF_DIODE 状态要求 $g_\phi\ge m$；
- 真值事件状态要求 $g_\phi=0$。

采用当前配置 $m=0.01$ 的无量纲 guard margin，预训练和联合损失均从配置读取。

### 10.2 负侧反事实样本

真实系统事件后立即切换到 IDLE，因此不直接把整段 IDLE 当负样本。先完成 `f_off` 预训练，然后从真值事件状态继续使用冻结的 OFF 网络作反事实延伸：

$$
x_{cf}(\Delta\tau)
=
\operatorname{RK4}
\left(
f_{\mathrm{OFF}},x_e,\Delta\tau
\right),
$$

$$
\Delta\tau\in\{0.0025,0.005,0.01\}.
$$

要求：

$$
g_\phi(x_{cf})\le-m.
$$

生成的反事实状态和 `f_off` 输出均 `detach`，只更新事件网络。这些状态仅用于 guard 预训练，不写入真实轨迹，也不作为 Buck 波形真值。

### 10.3 预训练损失

$$
L_{\mathrm{event-pre}}
=
L_{\mathrm{pre}}
+L_{\mathrm{surface}}
+L_{\mathrm{cf}},
$$

其中各项为等权均值；当前实现从训练 surface/pre 样本池分别均匀随机抽样，validation 使用完整固定样本集进行确定性评估。

事件网络接入联合 rollout 前，必须在 validation 真值轨迹上满足：

- DCM 自然事件检出率不低于 99%；
- 事件时间 MAE 不高于 $0.01T_s$；
- OFF 起点 guard 非正比例为 0；
- 每个 OFF 区间恰好一个零点；
- 预测事件严格早于下一 PWM 上升沿。

正式运行未通过门槛时训练入口保存断点并停止在 `event_pretrain`，不进入联合 rollout。Smoke 可显式绕过失败门槛以检查执行链，不能作为事件验收通过的证据。

## 11. 训练流程

### 11.1 模式动力学预训练

B/C 配置通过 `training.derivative_method="central"` 从真值状态构造中心差分监督：

$$
d_k=\frac{\hat x_{k+1}-\hat x_{k-1}}{\tau_{k+1}-\tau_{k-1}}.
$$

标签先在固定参考归一化的 `float64` 数据上计算，再随 batch 转为 `float32`。不代入电路方程，也不更改 NPZ schema。中心点左右两个区间必须同模式，排除首尾点、PWM 边界及自然事件邻域；当前 mask 固定用于每周期 100 个区间、占空比 0.5，不能直接宣称适用于其他网格或占空比。A 配置保留前向差分；B/C 使用中心差分。

模式预训练规则：

- 只使用同一模式内部相邻点；
- 丢弃 PWM 边界、自然事件两侧及事件邻域 $\pm2$ 个样本；
- ON、OFF_DIODE、IDLE 使用模式平衡采样；
- IDLE 原始网络监督 $d\hat v_C/d\tau$；
- `soft_idle_loss` 还在真值 IDLE 状态上用 $L_{\mathrm{idle-tangent}}$ 监督 $d\hat i_L/d\tau\approx0$；
- `hard_idle_constraint` 在前向计算中将 $d\hat i_L/d\tau$ 替换为零，原始 $u_i$ 头不参与该配置的损失。

### 11.2 课程阶段

| 阶段 | 窗口 | 最大 epoch | 默认 batch | 向量场 LR | 事件网络 LR |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mode_dynamics` | 局部 | 200 | 256 | $10^{-3}$ | 冻结 |
| `event_pretrain` | 局部/单 OFF 段 | 600 | 512 | 冻结 | $5\times10^{-4}$ |
| `one_cycle` | 1 周期 | 200 | 64 | $2\times10^{-4}$ | $10^{-4}$ |
| `five_cycle` | 5 周期 | 130 | 16 | $10^{-4}$ | $5\times10^{-5}$ |
| `ten_cycle` | 10 周期 | 80 | 8 | $10^{-4}$ | $5\times10^{-5}$ |
| `twenty_cycle` | 20 周期 | 60 | 4 | $5\times10^{-5}$ | $2.5\times10^{-5}$ |

模式预训练按 `batch_size // 3` 为每个模式分配样本，因此配置 256 对应每模式 85 个样本，共 255 个。Rollout 训练可按设备容量显式覆盖各阶段 batch；表中记录 JSON 默认值。

Rollout validation batch 始终使用 JSON 的 64/16/8/4。训练 batch 的优先级为：逐阶段 `--stage-batch-size` > 全局 `--rollout-batch-size` > checkpoint 的 `training_batch_sizes` > 旧阶段报告的 `training_batch_size` > JSON 默认。覆盖值不改变配置哈希，但会改变每 epoch 的优化器更新次数；必须随 checkpoint 保存，不能在恢复时悄悄切换。

优化器采用 AdamW，weight decay $10^{-6}$，梯度范数裁剪为 5。模式动力学及 rollout 使用 ReduceLROnPlateau（patience 10、factor 0.5），早停 patience 30，相对改善阈值 0.5%。事件预训练使用固定 LR $5\times10^{-4}$，不使用平台调度，独立早停 patience 600；默认每 10 epoch 及最后一 epoch 检查事件门槛，通过后提前结束。

验收计划使用 3 个固定模型种子 `20260804/20260805/20260806`；入口每次只运行一个 seed，默认取第一个。Smoke 模式每阶段 1 epoch，限制窗口数量，但必须包含低、中、高输入电压以及不同初值产生的异步 DCM 事件。

### 11.3 联合损失

归一化状态损失：

$$
L_{\mathrm{state}}
=
\operatorname{mean}
\left(
\|\hat x_{pred}-\hat x_{true}\|_2^2
\right).
$$

对成功匹配的真值自然事件：

$$
L_{\mathrm{time}}
=
\operatorname{mean}
\left(
\frac{\hat t^*-t^*}{T_s}
\right)^2.
$$

事件 guard 损失沿用预训练的 surface/pre/counterfactual 约束；联合阶段每批从真值轨迹抽取 guard 样本，防止事件网络被波形损失带偏。

对一个批次内成功预测事件的 $N_c$ 个周期，令 $K_c$ 为第 $c$ 个周期真值 IDLE 区间在预测轨迹上的固定网格索引。事件处电流损失为：

$$
L_{\mathrm{event-current}}
=
\frac{1}{N_c}
\sum_{c=1}^{N_c}
\left|\hat i_L(\hat t_c^*)\right|^2.
$$

IDLE 电流状态损失为：

$$
L_{\mathrm{idle-state}}
=
\frac{1}{N_c}
\sum_{c=1}^{N_c}
\frac{1}{|K_c|}
\sum_{k\in K_c}
\left|\hat i_{L,c,k}\right|^2.
$$

IDLE 切向损失为：

$$
L_{\mathrm{idle-tangent}}
=
\frac{1}{N_c}
\sum_{c=1}^{N_c}
\frac{1}{|K_c|}
\sum_{k\in K_c}
\left|
\frac{d\hat i_{L,c,k}}{d\tau}
\right|^2.
$$

IDLE 末端损失为：

$$
L_{\mathrm{idle-end}}
=
\frac{1}{N_c}
\sum_{c=1}^{N_c}
\left|
\hat i_L(\tau_{\mathrm{on},c}^{-})
\right|^2.
$$

所有 IDLE 损失都先在每个周期内部平均，再对周期平均，避免较长 IDLE 区间获得更高样本权重。模式动力学预训练在真值 IDLE 状态上计算切向损失；当前联合实现中，事件电流取预测事件电流；`idle_state`、`idle_tangent`、`idle_end` 在预测状态上计算，但区间 mask 来自真值 `interval_mode`，不使用预测 IDLE 区间划分。上述 $K_c$ 应按该真值 IDLE mask 解释；它与独立评估中的预测事件/IDLE 指标需区分。

Soft/Hard 两种 IDLE profile 使用相同损失定义。联合训练的全局目标为：

$$
\begin{aligned}
L
={}&L_{\mathrm{state}}
+1.0L_{\mathrm{time}}
+0.1L_{\mathrm{guard}}\\
&+1.0L_{\mathrm{event-current}}
+1.0L_{\mathrm{idle-state}}\\
&+0.1L_{\mathrm{idle-tangent}}
+1.0L_{\mathrm{idle-end}}\\
&+10^{-4}L_{\mathrm{ripple}}
+\lambda_v L_{\mathrm{voltage-derivative}}.
\end{aligned}
$$

其中 $\lambda_v$ 对应 `training.loss_weights.voltage_derivative`：A/B 配置为 0（缺省值），C 配置为 0.01。电压导数损失的完整定义及代码对应见第 11.4 节。

逐项报告包含 `voltage_derivative`；调度、早停和最佳权重选择使用各自配置的总 validation loss。不同损失定义的 A/B/C total 不能直接横比，应使用相同波形、事件及导数诊断指标，且公平对照须统一 seed、batch、窗口与训练预算。

在 `hard_idle_constraint` 中，实际 IDLE 电流导数被替换为零，因此 $L_{\mathrm{idle-tangent}}=0$；其余电流损失继续约束恒等事件映射下的事件定位残余。`ripple` 只在 `twenty_cycle` 阶段启用，继续采用线性端点去趋势后的 $v_C$ 峰峰值相对误差平方。

若预测在 PWM 边界前没有事件，记录一次 DCM 漏触发，并由真值 guard/surface 损失提供可微学习信号；若出现多个根、OFF 起点立即触发或事件落入 BCM 容差，记录事件结构错误，并由 sign、事件电流和后续波形损失共同纠正。离散错误计数本身只作为指标，不伪装为连续梯度。

### 11.4 电压导数监督：公式与代码对应

`voltage_derivative` 约束真值状态处的局部电压变化率，与积分轨迹的状态损失共同训练向量场。它不引入额外网络，不代入电容电路方程，也不对预测 rollout 波形做数值差分。

#### 11.4.1 真值标签与归一化

令轨迹编号为 $b$、网格索引为 $k$。采用第 3.2 节的固定参考坐标：

$$
\hat v_{C,b,k}=\frac{v_{C,b,k}}{V_{\mathrm{ref}}},
\qquad V_{\mathrm{ref}}=32\,\mathrm V,
\qquad \tau_k=\frac{t_k}{T_s}.
$$

中心差分电压导数标签为：

$$
q_{b,k}
=\frac{\hat v_{C,b,k+1}-\hat v_{C,b,k-1}}
{\tau_{k+1}-\tau_{k-1}}
=\frac{\hat v_{C,b,k+1}-\hat v_{C,b,k-1}}{2h},
\qquad h=0.01.
$$

因此分母为 0.02；标签对应 $d\hat v_C/d\tau$，与物理电压导数的关系为：

$$
\frac{d\hat v_C}{d\tau}
=\frac{T_s}{V_{\mathrm{ref}}}\frac{dv_C}{dt}.
$$

`training/data.py::central_derivative_labels` 在完整轨迹上使用 `float64` 计算并缓存两个状态分量的中心差分。`collate` 将对应窗口的标签写入 `derivative_target`，有效性写入 `derivative_valid`，随 batch 转换到训练设备与精度。电压损失只读取第二个分量（索引 1）：

```python
tau = t / (t[100] - t[0])
dx[:, 1:-1] = (
    (x[:, 2:] - x[:, :-2])
    / (tau[2:] - tau[:-2])[None, :, None]
)
```

有效中心点必须满足：左右区间同模式；周期相位索引 $p=k\bmod100$ 满足 $2<p<98$；$|p-50|>2$；距该周期自然事件的浮点网格位置大于 2。完整轨迹首尾点无效。该 mask 的定义依赖每周期 100 个区间及占空比 0.5，用于排除跨模式差分及切换邻域。

#### 11.4.2 网络预测与模式平均

在真值状态 $\hat x_{b,k}^{\mathrm{true}}$ 上，使用真值模式 $m$ 的向量场预测电压导数：

$$
p_{b,k}
=\left[f_{\theta,m}
\left(\hat x_{b,k}^{\mathrm{true}},s_b\right)\right]_v,
\qquad s_b=\frac{V_{\mathrm{in},b}}{32\,\mathrm V}.
$$

对于输入电压相对归一化，$z=\hat x/s$，ON/OFF 的包装层输出为 $sF_{\theta,m}(z,s)$，IDLE 为 $sF_{\theta,\mathrm{IDLE}}(z)$。因此 `model.vector_field(... )[:, 1]` 已还原为 $d\hat v_C/d\tau$，可以直接与 $q_{b,k}$ 比较，无需再乘除输入电压。

令 $S_m$ 为 batch 内真值模式为 $m$ 且导数标签有效的 $(b,k)$ 集合；只取窗口末点之前的位置，与 `interval_mode` 对齐。令 $\mathcal M=\{m:|S_m|>0\}$，则：

$$
L_{\mathrm{voltage-derivative}}
=\frac{1}{|\mathcal M|}
\sum_{m\in\mathcal M}
\frac{1}{|S_m|}
\sum_{(b,k)\in S_m}
\left(p_{b,k}-q_{b,k}\right)^2.
$$

先对每个模式的全部有效样本求 MSE，再对有样本的模式等权平均；无有效模式时返回零。此处不先按轨迹或周期平均。对应 `training/losses.py::voltage_derivative_loss`：

```python
x = batch["target"][:-1]
vs = batch["vs"].expand(x.shape[0], -1, -1)
values = []
for mode in (MODE_ON, MODE_OFF_DIODE, MODE_IDLE):
    mask = batch["derivative_valid"][:-1] & (batch["interval_mode"] == mode)
    if mask.any():
        prediction = model.vector_field(mode, x[mask], vs[mask])[:, 1]
        truth = batch["derivative_target"][:-1, :, 1][mask]
        values.append((prediction - truth).square().mean())
loss = torch.stack(values).mean() if values else x.new_zeros(())
```

#### 11.4.3 总损失、梯度与启用范围

`training/losses.py::rollout_losses` 按配 置权重启用该项：

```python
if weights.get("voltage_derivative", 0) > 0:
    terms["voltage_derivative"] = voltage_derivative_loss(model, batch)
terms["total"] = sum(
    weights.get(name, 0.0) * value
    for name, value in terms.items() if name != "total"
)
```

对应配置 C：

$$
L_{\mathrm{total}}
=L_{\mathrm{原有联合损失}}
+0.01L_{\mathrm{voltage-derivative}}.
$$

`training/loops.py::run_rollout_stage` 对 `terms["total"]` 执行 `backward()`。导数项向 ON/OFF/IDLE 的电压输出及各自隐藏层传播梯度；隐藏层参数同时参与电流输出，因此也可能间接影响电流动态。该项不调用事件网络或事件定位过程，不直接向事件网络参数提供梯度。Hard IDLE 仅将电流导数归零，电压导数监督仍有效。

权重大于零时，导数监督在 `one_cycle`、`five_cycle`、`ten_cycle`、`twenty_cycle` 的训练及 validation 中均启用。它不受仅在 `twenty_cycle` 启用的 ripple 开关控制；权重缺省或为零时不构造 rollout 导数标签，也不添加该损失项。

`training.derivative_method` 决定模式动力学预训练使用前向还是中心差分；`training.loss_weights.voltage_derivative` 决定联合 rollout 是否增加电压导数损失。两者是独立配置项：B/C 的模式预训练均使用中心差分，只有 C 在联合训练继续使用此电压监督。Rollout 电压监督的标签固定使用中心差分。

调度、早停与最佳权重选择使用含该项的 validation 总损失，逐项报告保留未乘权重的 `voltage_derivative`。A/B/C 的比较应统一数据、seed、batch 和训练预算，并使用共同的波形及事件指标；不能把不同目标函数的总损失数值直接当作优劣结论。

### 11.5 运行设备

正式训练、联合 rollout、半步检查和评估支持 Mac MPS 和 NVIDIA CUDA，均使用 `float32`。新归一化配置默认 auto（CUDA 优先，其次 MPS），无 GPU 时停止；CPU 仅显式指定。可用 --device cuda:N 选择单个 NVIDIA GPU。CUDA 关闭 TF32，不启用 AMP。配置换行符统一后计算指纹，设备覆盖使用命令行，不修改配置。
原始与处理后数据继续保存为 `float64`，进入训练批次时转换为 `float32`。

## 12. Checkpoint 与复现

Checkpoint 必须新增：

- schema version；
- 配置文件 SHA-256；
- 模型结构签名；
- 当前训练阶段；
- 模型、优化器和调度器状态；
- Python、NumPy、Torch CPU/MPS/所选 CUDA GPU RNG 状态；
- 已完成阶段和阶段门槛结果；
- 归一化常数；
- 固定负载 $R=80\Omega$；
- 恒等事件映射标识；
- `soft_idle_loss` 或 `hard_idle_constraint` 配置名；
- 四项电流损失权重；
- Hybrid RK4 步长和割线细化次数；
- 每个 rollout 阶段的实际 `training_batch_sizes`，包括尚未开始的阶段。

训练中的当前阶段每完成 2 个 epoch 原子覆盖一次 `checkpoint.pt`；阶段因正常
结束或早停退出时，无论 epoch 是否为 2 的倍数都再保存一次。断点必须记录
当前阶段已完成的 epoch、best/stall、优化器和调度器状态，使 `--resume` 从
下一 epoch 继续。一次中断最多损失尚未完成的 2 个 epoch。

旧两模式 checkpoint 与新模型结构不兼容，必须给出明确错误。正式训练不得静默加载旧权重。

`checkpoint.pt` 保存用于恢复的最新训练状态；`best_validation/<stage>.pt` 独立保存各 rollout 阶段实际观察到的最低有限 validation loss 权重，每次严格下降即更新，不受早停的 0.5% 阈值限制。训练结束先保存 `last.pt`，再从最后训练阶段的最佳记录导出 `best.pt` 用于最终评估；无最佳记录时回退到最后权重，并在 `best_selection.json` 中记录原因。`best.pt`、`last.pt` 和最佳选择 bundle 不是完整恢复 checkpoint；`--resume` 应使用 `checkpoint.pt`。

## 13. 评估指标与验收

### 13.1 波形指标

| 场景 | 验收门槛 |
| --- | ---: |
| validation 20 周期归一化 RMSE | $\le1\%$ |
| test_seen 200 周期平均 RMSE | $\le2\%$ |
| test_interpolation 200 周期平均 RMSE | $\le3\%$ |
| 电压外推 200 周期平均 RMSE | $\le5\%$ |
| seen/interpolation 的 $v_C$ 纹波平均相对误差 | $\le10\%$ |
| seen/interpolation 的 $v_C$ 纹波 P95 相对误差 | $\le20\%$ |

所有指标必须按输入电压和初值扰动区间分别报告，不能只报告混合均值。`test_extrapolation_high` 单独报告，不作为核心硬门槛。所有结果都必须核对 $R=80\Omega$；任何其他负载或被检测为 CCM/BCM 的真值数据都属于数据生成错误，不进入模型指标。

### 13.2 事件指标

| 指标 | 验收门槛 |
| --- | ---: |
| DCM 事件漏触发率 | $\le1\%$ |
| 每周期恰好一次事件比例 | $\ge99\%$ |
| 预测事件落入 BCM/周期边界比例 | 0 |
| 事件时间 MAE | $\le0.005T_s$ |
| 事件时间 P95 | $\le0.01T_s$ |
| 事件处 $|i_L|$ 平均值 | $\le10\,\mathrm{mA}$ |
| 事件处 $|i_L|$ P95 | $\le20\,\mathrm{mA}$ |
| IDLE 全区间 $|i_L|$ 平均值 | $\le10\,\mathrm{mA}$ |
| IDLE 全区间 $|i_L|$ P95 | $\le20\,\mathrm{mA}$ |
| 下一 PWM 上升沿前 $|i_L|$ 平均值 | $\le10\,\mathrm{mA}$ |
| 下一 PWM 上升沿前 $|i_L|$ P95 | $\le20\,\mathrm{mA}$ |
| 每个 OFF 区间额外伪根 | 0 |
| OFF 起点 guard 非正比例 | 0 |

事件指标使用 Simulink 二极管状态真值；触发电流只是物理一致性审计。

### 13.3 数值收敛

同一正式权重分别以 $h=0.01$ 和 $h=0.005$ 运行：

- 事件时间平均差不超过 $0.002T_s$；
- 归一化波形 RMSE 差不超过 0.1 个百分点；
- 周期事件存在性不得改变超过 0.5%；
- 非有限状态数量必须为 0。

三个模型种子都必须满足事件安全门槛；至少两个种子满足全部波形门槛。报告三种子的均值、标准差和最差种子。

### 13.4 Soft/Hard 对照

`soft_idle_loss` 和 `hard_idle_constraint` 必须使用完全相同的数据 split、模型种子、ON/OFF/事件网络结构、优化阶段、损失权重和 Hybrid RK4 设置。两者都采用恒等事件映射，不设置 `soft_no_physics`。

对照报告至少包含：

- 波形和事件指标；
- 四项电流损失及对应物理量；
- soft 配置的 IDLE $d\hat i_L/d\tau$ RMS 和 P95；
- hard 配置实际 IDLE $d\hat i_L/d\tau$ 必须严格为零；
- 200 周期稳定性、训练时间和推理时间。

本实验不预设 soft 必须优于 hard；两者分别代表软物理约束和硬结构约束。

## 14. 文件修改矩阵

### 14.1 重命名

| 当前路径 | 新路径 |
| --- | --- |
| `configs/known_pwm_matlab_v2/` | `configs/hybrid_event_matlab_v3/` |
| `matlab/scripts/buck_ideal_ccm.slx` | `matlab/scripts/buck_ideal_hybrid.slx` |

这是 v2→v3 的历史迁移记录，不是当前待执行的重命名；当前在同一 v3 schema 下保留第 2 节列出的训练配置。

### 14.2 修改

| 文件 | 修改内容 |
| --- | --- |
| `README.md` | 固定 $R=80\Omega$ 的 DCM-only 目标、v3 数据链、命令和验收说明 |
| `configs/hybrid_event_matlab_v3/project.json` | 固定 $R=80\Omega$、schema、网络、Hybrid RK4、训练和验收配置 |
| `matlab/scripts/initialize_buck_model.m` | 定义固定 DCM 电路参数、模型路径和周期归一化输出网格 |
| `matlab/scripts/generate_dataset.m` | 沿用原生成模式：split 固定种子、二维初值 LHS、DCM 拒收补采 |
| `matlab/scripts/export_trajectory.m` | 原始 `iD` 事件提取、容差验收、固定网格和 MAT v3 保存 |
| `scripts/mat_to_npz.py` | v3 schema 转换、一致性校验和 MAT source SHA-256 |
| `training/model.py` | DCM 三向量场、恒等事件映射、soft/hard IDLE 配置和事件 MLP |
| `training/data.py` | 固定负载审计、模式/事件加载、初值样本和窗口 |
| `training/losses.py` | guard、surface、event time、四项电流和 ripple 损失 |
| `training/loops.py` | mode/event 预训练门槛、soft/hard 联合课程训练 |
| `training/evaluate.py` | 按输入电压/初值的波形、事件、IDLE 电流、soft/hard 对照和半步指标 |
| `training/checkpoint.py` | schema/config/模型签名和完整 RNG 状态 |
| `train.py` | 新模型入口、阶段编排、移除 backprop 路线开关 |
| `analysis/*.py` | 按输入电压和初值绘图，支持 DCM 事件与 IDLE 标记 |
| `docs/CHANGE_LOG.md` | v2 到 v3 的不兼容升级记录 |
| `docs/ROUTE.md` | Hybrid RK4 与神经事件唯一路线 |
| `docs/requirements-alignment.md` | 用本设计替换旧 CCM 历史目标或明确归档状态 |
| 工作区 `README.md`、`docs/topology_catalog.md` | Buck 能力更新为 DCM-only Neural Event ODE |

### 14.3 新增

| 文件 | 职责 |
| --- | --- |
| `configs/hybrid_event_matlab_v3/experiments/soft_idle_loss.json` | 默认 soft IDLE 配置和 $1.0/1.0/0.1/1.0$ 电流损失权重 |
| `configs/hybrid_event_matlab_v3/experiments/hard_idle_constraint.json` | 恒等事件映射、IDLE 电流导数硬归零对照配置 |
| `configs/hybrid_event_matlab_v3/input_voltage_relative.json` | 输入电压相对坐标默认配置 A |
| `configs/hybrid_event_matlab_v3/central_derivative.json` | 中心差分预训练对照 B |
| `configs/hybrid_event_matlab_v3/central_derivative_rollout.json` | 中心差分与 0.01 电压导数损失配置 C |
| `training/batching.py` | 逐阶段训练 batch 解析与恢复优先级 |
| `training/best_weights.py` | 各 rollout 阶段最佳 validation 权重选择 |
| `training/run_lock.py` | 同输出目录并发训练保护 |
| `tests/test_derivative_supervision.py` | 中心差分、边界 mask 与导数损失验证 |
| `tests/test_stage_batches.py`、`tests/test_best_weights.py` | batch 恢复与最佳权重选择验证 |
| `training/hybrid_rk4.py` | 批量 RK4、两次割线、逐样本模式和事件记录 |
| `tests/test_hybrid_rk4.py` | 解析 DCM 事件、异步批事件、漏事件保护和梯度测试 |
| `tests/test_event_network.py` | 网络结构、输入、符号和预训练门槛测试 |
| `tests/test_idle_constraint.py` | 恒等事件映射、soft/hard 导数和 IDLE 电流损失 |
| `tests/test_hybrid_data_schema.py` | MAT/NPZ v3 字段、形状、合法转移测试 |
| `tests/test_training_smoke.py` | 三预训练/rollout 最小闭环测试 |
| `tests/test_half_step_convergence.py` | $h=0.01/0.005$ 数值一致性 |

### 14.4 替换过时测试

旧 `test_regression_equivalence.py` 的两模式权重等价目标不再成立。它应替换为“v2 checkpoint/data 被清晰拒绝”的不兼容迁移测试，而不是删除迁移审计能力。

## 15. 测试策略

电压导数监督的专项检查包括：中心差分数值与固定参考归一化一致；PWM/自然事件邻域 mask 正确；损失只读取电压分量并按有效模式等权平均；无有效样本时返回零；真值状态输入、向量场梯度及权重开关正确；B/C 预训练差分选择与 rollout 导数监督开关互相独立。

### 15.1 Hybrid RK4 单元测试

使用解析线性系统：

$$
\frac{di}{d\tau}=-a,
\qquad
g_\phi(i)=wi+b.
$$

验证：

- 解析事件时间；
- 两次割线误差；
- $\partial t^*/\partial w$、$\partial t^*/\partial b$ 与有限差分一致；
- 一个批次内不同 DCM 事件时间互不干扰；
- 漏事件样本在 PWM 边界被识别为错误而不产生非有限状态；
- 事件前后连续状态严格相同；
- 恒等映射保留事件状态梯度；
- soft 配置剩余子步使用完整 IDLE 导数；
- hard 配置剩余子步将 IDLE 电流导数严格替换为零；
- 事件后不重复触发。

### 15.2 数据测试

使用小型合成 NPZ fixture 验证：

- 每轨迹模式数组；
- 每周期恰好一个 DCM 自然事件；
- 事件时间和 IDLE 时长；
- 无事件、BCM 边界事件和多事件轨迹被拒绝；
- 非法模式转移拒绝；
- 固定负载 $R=80\Omega$ 和 metadata 一致；
- schema 不包含模型电导输入；
- v2 schema 明确拒绝。

### 15.3 训练测试

Smoke 数据必须同时包含：

- 低、中、高输入电压下的 DCM 周期；
- 同批不同事件时间；
- 每周期严格一个自然事件；
- 明确的 OFF_DIODE 和正 IDLE 区间。

另用故障注入 fixture 验证无事件、多事件和 BCM 边界样本会被数据转换或训练门槛拒绝。对四项电流损失分别验证公式、逐周期平均、mask 和梯度；验证 soft/hard 两个配置的事件网络预训练损失下降、联合 rollout 可反传、checkpoint 可恢复、evaluate-only 可运行。配置测试必须明确拒绝不存在的 `soft_no_physics`。

## 16. 运行产物

每个正式 run 至少输出：

- `checkpoint.pt`；
- `best.pt`、`last.pt`、`best_selection.json`；
- `best_validation/<stage>.pt`（已记录 validation 改善的 rollout 阶段）；
- `runtime_environment.json`；
- `metrics.json`；
- `metrics_by_voltage.json`；
- `idle_soft_vs_hard.json`；
- `events.csv`；
- 波形/事件对比图；
- 按输入电压和初值的漏触发、事件时间、事件电流和 IDLE 时长图；
- $h=0.01/0.005$ 收敛报告；
- 运行配置副本。

所有产物位于 Buck 自有 `artifacts/`，不提交 Git。

## 17. 风险与控制

| 风险 | 控制措施 |
| --- | --- |
| 事件网络无根或漏事件 | 有监督预训练门槛、所有周期都有 DCM surface/counterfactual 样本、PWM 边界故障标记 |
| 数据混入 CCM/BCM | 每周期自然事件和最小 IDLE 时长硬验收，转换器拒绝 |
| 固定负载配置漂移 | 配置、MAT、NPZ 和 checkpoint 四处校验 $R=80\Omega$，网络不输入 $R$ 或 $G$ |
| 神经事件提前触发 | 恒等映射不掩盖误差，使用 $L_{\mathrm{event-current}}$、事件时间和事件电流门槛 |
| soft IDLE 中电流漂移 | $L_{\mathrm{idle-state}}$、$L_{\mathrm{idle-tangent}}$、$L_{\mathrm{idle-end}}$ 和 200 周期验收；hard 配置作对照 |
| 多根/重复事件 | 每周期一次状态机、负侧符号训练、半步检查 |
| 初值扰动造成非 DCM 周期 | 逐周期最小 IDLE 时长验收，拒收后按同一输入电压补采 |
| 批内事件时间不同 | 逐样本掩码和分数子步 |
| Simscape 关断态泄漏污染事件 | 固定并审计 Diode/MOSFET 的 `Goff`，同时执行事件电流与 IDLE 电流容差验收 |
| 长 rollout 累积误差 | 1/5/10/20 周期课程和 200 周期独立评估 |
| 旧 checkpoint 静默误用 | schema、配置哈希、模型签名强校验 |

## 18. 实施顺序

实施必须按测试驱动顺序进行：

1. 写 v3 路径、schema 和旧版本拒绝测试；
2. 升级 Simulink 日志与固定 $R=80\Omega$ 的 DCM 数据生成；
3. 完成 MAT 到 NPZ v3 转换和数据单元测试；
4. 实现三向量场、事件网络、恒等事件映射和 soft/hard IDLE 配置；
5. 实现批量 Hybrid RK4 与解析事件/梯度测试；
6. 实现模式动力学预训练；
7. 实现反事实负样本和事件网络预训练门槛；
8. 实现联合课程训练和 checkpoint；
9. 实现分输入电压/初值、事件、IDLE 时长和半步评估；
10. 运行 smoke、三种子正式训练与最终指标验收。

数据生成、模式预训练、事件门槛、联合课程与独立评估按上述依赖顺序执行。MATLAB 参数以 `project.json` 和实际模型为准；硬门槛、schema 或模型语义发生变化时，必须同步更新本设计并提交独立 Git 变更。各配置的三种子与 Soft/Hard 对照按第 13 节统一验收，结果保存在对应运行产物中。

跨后端恢复模型和优化器允许继续训练，但 RNG 序列及浮点运算不保证相同。设备、软件版本和数值精度记录在 `runtime_environment.json` 中，用于复现与跨后端结果解释。
