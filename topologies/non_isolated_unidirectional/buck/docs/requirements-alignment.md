# Buck 需求对齐（DCM v3）

本文替换旧 CCM/V2 对齐记录。旧内容仍可从 Git 历史查看，不作为当前训练
需求、配置或验收依据。

## 当前唯一活动配置

`configs/hybrid_event_matlab_v3/project.json`

- 只接受固定 `R=80 Ω` 的 DCM 数据；每周期恰好一个
  `OFF_DIODE → IDLE` 自然事件。
- 连续状态为 `(iL_norm, vC_norm)`，外部输入为 `vS_norm`；负载不输入网络。
- `f_on`/`f_off` 为 `3→64→64→2`，`f_idle` 为 `2→64→64→2`，事件网络为
  `3→64→64→1`。
- PWM 使用 100 个固定区间/周期，`D=0.5`；自然事件由神经事件面定位，事件
  映射为恒等映射，不做电流投影。
- 训练路线是 `mode_dynamics → event_pretrain → 1/5/10/20-cycle` Hybrid RK4
  rollout；旧 `odeint`、adjoint 和 CCM 两模式路线已删除。

## 已落地代码

| 文件 | 职责 |
| --- | --- |
| `training/data.py` | NPZ v3 schema、固定负载/模式审计、归一化、模式平衡点对和窗口 |
| `training/model.py` | 三模式向量场、事件网络、soft/hard IDLE 和恒等映射 |
| `training/hybrid_rk4.py` | 批量固定网格 RK4、两次可微割线和逐样本事件状态机 |
| `training/losses.py` | state、event time、guard、四项 IDLE/事件电流和 ripple 损失 |
| `training/loops.py` | 三个预训练/门槛和联合课程训练 |
| `training/checkpoint.py` | schema、配置哈希、模型签名、优化状态和 RNG 状态 |
| `training/evaluate.py` | 波形、事件、IDLE、输入电压、初值分位和半步指标 |
| `train.py` | 阶段编排、gate 停止、resume、evaluate-only |

## 验收入口

```bash
python3 topologies/non_isolated_unidirectional/buck/train.py --smoke
```

正式训练前应确认 MATLAB/Simulink 生成链和 `scripts/mat_to_npz.py` 输出的
所有 split 均通过 DCM v3 schema。`event_pretrain` validation gate 未通过时，
正式入口停止在该阶段；`--smoke` 仅允许打印警告以便测试后续代码连通性。
