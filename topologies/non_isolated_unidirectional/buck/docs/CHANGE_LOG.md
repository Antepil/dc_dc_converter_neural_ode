# 2026-09-10 输入电压相对归一化

- 网络内部采用 `z_i=(iL/1 A)/(Vin/32 V)`、`z_v=vC/Vin`，ON/OFF 与事件网络保留 `Vin/32 V` 输入。
- IDLE 保持两维相对状态输入；各模式神经导数乘 `Vin/32 V` 还原为积分器坐标。
- 损失、导数监督与评估保留固定参考尺度；无新增电路方程或物理损失。
- Python 默认配置和输出目录切换到独立归一化版本；`project.json` 与 Git 基线 `a55a077` 保留旧方案。
- 权重标记区分固定参考、旧强制缩放试验及当前保留电压条件版本。
- 本版尚无正式训练结果。坐标与权重检查通过；检查过程曾误执行一次旧合成数据短训，未加载正式数据、未保存训练产物。

> [!NOTE]
> 下方 v1/v2 内容是历史记录。当前可执行路径和命令以 [../README.md](../README.md) 为准。

# known_pwm_matlab_v2 → hybrid_event_matlab_v3

- 修复事件预训练学习率被随机 validation 反复衰减至近零的问题：事件阶段改用
  固定学习率、全量确定性 validation、向量化周期 gate，并在 gate 通过后立即
  结束；旧 event checkpoint 恢复时自动重置学习率和不可比较的 best/stall；
  事件得分 `guard_margin` 从 `0.1` 调整为 `0.01`，降低近事件反事实约束造成的
  零面偏移。该配置变更要求重新训练，不复用旧配置 checkpoint。

- 正式训练、评估和半步检查统一使用 MPS/float32，不再静默回退 CPU；
- 当前阶段每 2 个 epoch 原子更新 checkpoint，并可恢复 optimizer、scheduler、
  best/stall、MPS RNG 后从下一 epoch 继续；
- 删除 MATLAB 单元/接受测试及其过时实施计划，DCM 数据合法性由生成筛选和
  Python v3 数据入口审计。

日期：2026-09-03

- 训练目标从严格 CCM 两模式升级为固定 `R=80 Ω` 的 DCM-only 三模式模型；
- 连续状态仍为 `(iL, vC)`，新增 `OFF_DIODE → IDLE` 神经事件面；
- 事件映射为恒等映射，不再把事件电流投影到零；
- `torchdiffeq`/adjoint/按绝对时间 PWM 调度被固定网格 Hybrid RK4 取代；
- NPZ schema 切换为 `hybrid_event_matlab_v3`，旧数据和 checkpoint 明确拒绝；
- 新增 mode/event 预训练、validation gate、1/5/10/20-cycle 课程、四项电流损失、
  半步评估和 soft/hard IDLE 对照；
- MATLAB 初始电压因子以活动配置为准，从历史 `[0.8,1.2]` 更新为 `[1.22,1.30]`。

Python 训练、评估和分析脚本不保留 V2 运行分支；历史实现由 Git 保存。

# known_pwm_matlab_v1 → v2 变更说明

日期：2026-08-15　对应需求对齐记录：`../requirements-alignment.md`

## 变更动机

需求对齐（grilling）确认了三处数据侧/模型侧修正：

1. vS 必须进入网络输入（(iL, vC, vS)）
2. CCM 门槛过低（v1 全局 0.16 A，train 最低 min(iL)=0.1632 A，裕度仅 ~1.5%）
3. 模型配置需要表达两个独立网络 f_ON / f_OFF

## v2 数据配置改动

### 1. CCM 门槛改为按 vS 动态推导

原方案 A（margin_fraction 1.2–1.3 × 稳态理论最小电流）经模拟验证在数学上不可行：

- 稳态理论最小电流 iL_min_ss = D·vS/R − vS·D·Ts/(2L) 本身就是稳态纹波下沿
- 要求整条轨迹 min(iL) ≥ 1.2 × iL_min_ss 意味着高于物理下沿，接受率仅 3–8%
  （vS=12 V 时门槛 0.36 A，而稳态最小值 0.30 A），20 条/电压无法凑齐

v2 采用修正后的规则：

- threshold(vS) = max(floor_absolute_a, margin_fraction × iL_min_ss(vS))
- margin_fraction = 1.0（要求任何瞬态都不低于稳态理论纹波下沿）
- floor_absolute_a = 0.2 A（绝对安全下限）
- margin_fraction 必须 ≤ 1.0，配置中带说明

实测接受率（20 周期、LHS 初始因子 [0.75,1.35]×[0.8,1.2]、N=20000 批量模拟）：

| vS (V) | 门槛 (A) | 接受率 | 期望接受数/200 次尝试 |
| ------ | -------- | ------ | ---------------------- |
| 12     | 0.300    | 18.3%  | ~37（≥20 充足）        |
| 16     | 0.400    | 18.3%  | ~37                    |
| 20     | 0.500    | 18.3%  | ~37                    |
| 24     | 0.600    | 18.3%  | ~37                    |
| 28     | 0.700    | 18.3%  | ~37                    |
| 32     | 0.800    | 18.3%  | ~37                    |

接受率在所有 vS 下相同（系统线性、按 vS 缩放后归一化动力学一致）。train 每电压需 20 条、
attempt 上限 200（10×trajectories_per_voltage），期望接受 ~37 条，充足。
initial_current_factor 保持 [0.75, 1.35] 不变，无需提高下限。

### 2. metadata 增加 CCM 审计字段

- `theoretical_min_current_a`：该 vS 的稳态理论最小电流
- `ccm_minimum_current_a`：该轨迹实际使用的动态门槛（按 vS）
- `ccm_passed` / `ccm_margin_a`：校验结果与裕度（export_trajectory 写入）

### 3. model 配置改为两个独立网络

- `architecture: "two_mode_mlp"`，on/off 各一套 (input_dim=3, 64×2, output_dim=2)
- `input_spec: ["iL_norm", "vC_norm", "vS_norm"]`——vS 进网络输入
- ρ 为增广状态（`phase_rho: true`），导数固定不学习（`rho_derivative_fixed: true`），不作为网络输入
- normalization 增加 `input_voltage_ref_v: 32.0`

## 其他确认

- Simulink 模型求解器：ode23t 变步长，MaxStep=5e-8 s（= dt/10，满足 max step ≤ dt/2），
  RelTol=10*128*eps——开关瞬间附近 vC 重采样精度已确认无虞，无需改动模型。
- 占空比锁死 D=0.5 的约束保持（事件落在 100 采样/周期网格边界，event_root_finding=false 的前提）。

## 待办（模型侧，不在本数据配置范围）

- 训练管线读取 v2 配置：按 (iL, vC, vS) 归一化输入、两个网络分别前向
- local_mode 有限差分斜率监督需过滤跨事件点对
- 分段 adjoint（torchdiffeq 逐段 odeint_adjoint，固定 step_size=dt）

---

# v2 → v2.1 变更说明（test_extrapolation_high 强外推测试）

日期：2026-08-18

## 变更动机

现有 test_extrapolation（10/34/36 V）外推幅度有限（vS_norm ≤ 1.125）。
新增 50–80 V 强外推划分，评估 vS_norm 超出 [0.375, 1.0] 训练区间
（最高 80 V 时 vS_norm=2.5）的模型行为。

## 数据侧改动

- project.json 新增 split `test_extrapolation_high`：
  voltages [50, 55, 60, 65, 70, 75, 80]、每电压 10 条、200 周期、seed 20260920。
  config_version → 2.1.0。
- 新增 `scripts/mat_to_npz.py`：.mat → npz 转换脚本（此前为手工转换，仓库无脚本）。
  已用 `--check` 对旧 test_extrapolation 回归校验，与既有 npz 逐数组完全一致。
- CCM 门槛按 vS 动态推导不变：50 V→1.25 A … 80 V→2.0 A，接受率与低电压一致（~18%）。

## 训练侧改动（配合评估）

- data.py SPLIT_FILES 增加 test_extrapolation_high
- evaluate.py：评分逻辑抽成 _rollout_metrics（含非有限值 NaN 保护）；
  新增 evaluate_by_voltage / print_by_voltage / save_voltage_plots
- train_buck.py 新增 `--evaluate-only`：加载 best.pt（或 --resume checkpoint）
  直接评估，含逐电压分解与波形图，结果写 metrics_eval.json

## 实测（best.pt，ten_cycle 31/60 时的最优权重，2026-08-18）

- 1 周期 rollout：50 V→RMSE 0.15%、80 V→1.04%（对比训练域 0.04%）
- 10 周期 rollout：50 V→0.18%、80 V→均值 1.80%/最大 3.36%
- vC 纹波相对误差保持 <2.3%（各电压、1/10 周期）
- 20 周期稳态对比（80 V）：iL 均值偏高 3.9%（3.712 vs 3.572 A），
  vC 均值偏高 0.6%，iL 纹波 −3.6%，vC 纹波 −16%
