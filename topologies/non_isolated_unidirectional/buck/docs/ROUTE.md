> 2026-09-10 归一化更新：Python 默认配置改为 `configs/hybrid_event_matlab_v3/input_voltage_relative.json`，网络使用相对状态并保留 ON/OFF/事件的输入电压条件。下文 `project.json` 为数据生成及固定归一化基线配置；进行本轮训练时应使用新配置。详见 [归一化设计](normalization-experiment.md)。

# Buck DCM 训练路线

当前唯一训练路线是固定网格 Hybrid RK4 + autograd。模型显式传入逐样本
`ON`、`OFF_DIODE`、`IDLE` 模式；PWM 边沿在网格上切换，OFF 段自然事件由
神经事件面和两次可微割线细化定位，事件映射为恒等映射。

课程阶段依次为：

1. `mode_dynamics`：三个模式的同模式有限差分，按模式平衡采样；
2. `event_pretrain`：surface、事件前 guard 和 counterfactual 约束；该阶段使用
   `guard_margin=0.01`、固定学习率、全量确定性 validation，并每 10 个 epoch
   检查一次 gate；
3. `one_cycle`、`five_cycle`、`ten_cycle`、`twenty_cycle`：Hybrid rollout 联合训练。

`event_pretrain` 的 validation gate 未通过时，正式训练会停止，不进入联合
rollout；gate 一旦通过则立即结束预训练并保存。恢复旧 plateau-scheduler
checkpoint 时会保留 Adam 动量，但把事件学习率恢复到配置值，并重新建立
best/stall 基线。`--smoke` 仅为调试允许继续并打印警告。

训练入口：

```bash
python3 topologies/non_isolated_unidirectional/buck/train.py \
  --config topologies/non_isolated_unidirectional/buck/configs/hybrid_event_matlab_v3/project.json
```

`soft_idle_loss` 和 `hard_idle_constraint` 共享全部数据、网络和课程设置，
只在 IDLE 电流导数是否硬归零上形成结构对照。旧 CCM/V2 的 `odeint`、adjoint
路线和两模式 checkpoint 不属于当前实现。
