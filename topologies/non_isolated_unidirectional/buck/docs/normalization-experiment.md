# 输入电压相对归一化实验

本实验保留 Neural Event ODE：ON、OFF、IDLE 的电流与电压导数均由原尺寸 Tanh MLP 输出，截止事件仍由神经事件网络定位。train.py 默认采用 input_voltage_relative.json；project.json 保留为固定参考基线；不引入电感、电容微分方程或新的物理损失。

## 坐标和实验假设

积分器坐标仍为 x=(iL/1 A, vC/32 V)，时间仍为 tau=t/50 μs。令 s=Vin/32 V，网络内部输入 z=x/s，因此 z_i=iL/(s·1 A)，z_v=vC/Vin。ON/OFF 和事件网络的第三输入保留 s，维度和参数数量不变。IDLE 网络输入同样使用 z，但不增加输入维度。

神经网络输出 dz/dtau，包装层以 dx/dtau=s·NN(z,s) 还原；IDLE 同理。事件得分为 g(z,s)。事件映射仍为恒等，不投影电流。

ON/OFF 与事件网络保留绝对电压条件，不强制整体满足严格幅值缩放。IDLE 维持原两维输入，其相对坐标下的导数仍假定不依赖绝对 Vin；这是保留参数规模的限制，若以后要表达该模式的非齐次效应需单独扩展设计。所有导数仍来自神经网络。仅用于每条轨迹 Vin 恒定且严格正值的当前数据。若 Vin 随时间变化，坐标导数还需要额外项，本实现不用于该情况。

损失、导数监督、事件电流和 IDLE 惩罚、评估反归一化均保留原固定参考坐标，因此不会因把高压误差除以更大的 Vin 而人为降低报告指标。归一化常数不通过测试轨迹拟合，也不使用未来输出统计量。输入和输出范围检查只用于诊断，不作为调参数据。

## 数据范围核对

| 坐标 | 训练集电压状态范围 | 50–80 V 测试电压状态范围 |
|---|---:|---:|
| 原 vC/32 V | 0.229480–0.650828 | 0.959116–1.616079 |
| 新 vC/Vin | 0.610949–0.653106 | 0.613834–0.649617 |

对应归一化电流最大值从训练 0.856605 / 高压 2.179595，变成训练 0.886498 / 高压 0.882368。边缘范围重合不保证联合分布一致或泛化成功。

## 控制变量与验收

- 基线：configs/hybrid_event_matlab_v3/project.json。
- 实验：configs/hybrid_event_matlab_v3/input_voltage_relative.json。
- 相同数据、随机种子、网络层宽、优化器、学习率、损失权重、训练阶段和积分设置；仅改变网络输入/导数的坐标处理。
- 从头训练，不能复用旧权重；新权重含坐标版本标记，旧权重和新模型严格加载时互相拒绝。完整 checkpoint 另有配置和结构签名。
- 先以 seed 20260804 完成训练；若结果有收益，再执行另两个既定随机种子。
- 用原固定 1 A、32 V 口径评估 1、10、20、50、200 周期，逐电压报告 NRMSE、电流/电压绝对 RMSE、均值偏差、逐周期纹波及事件指标。
- 必须通过已有域内和近外推验收，并检验高压误差是否下降。高压尚非原配置核心门槛，不把程序完成等同于外推成功。
- 旧 best.pt 实际为最终阶段权重；本轮不同时改变选择规则，以免混入另一实验因素。

## 运行

从工作区根目录执行，默认输出位于 artifacts/hybrid_event_matlab_v3/input_voltage_relative_conditioned_v2/soft_idle_loss，也可独立指定：

```bash
python3 topologies/non_isolated_unidirectional/buck/train.py \
  --config topologies/non_isolated_unidirectional/buck/configs/hybrid_event_matlab_v3/input_voltage_relative.json \
  --seed 20260804 \
  --out-dir topologies/non_isolated_unidirectional/buck/artifacts/input_voltage_relative_20260910/seed_20260804
```

恢复时沿用上述配置和输出目录，增加 --resume 指向该实验的 checkpoint.pt。

当前仅做代码与设计更新，不启动正式数据训练。检查过程中曾因路径失误执行一次旧合成数据短训，未保存模型；该测试已移除。新增的静态前向检查覆盖输入电压条件、导数还原、梯度和权重坐标隔离，不执行优化器更新。

Git 基线：a55a077。旧版强制缩放试验的版本标记与本版不同，不能载入本版模型；已有 smoke 文件不是本版训练结果。

设备适配更新：本配置 runtime.device 为 auto，CUDA 优先、其次 MPS，无 GPU 不自动降级。跨设备对比需记录后端，数值结果不保证逐位一致。为隔离归一化收益，基线与新方案应在相同后端对比；固定基线配置可用 --device cuda 在 NVIDIA 上运行。
