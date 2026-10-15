# 数据导数监督对照

## 已启动的完整重训：2026-09-13

用户确认后，已从头启动 C 方案，输出目录 `artifacts/runs/central_derivative_rollout_20260913`。seed=20260804，与历史 CUDA run 一致。训练 batch：one_cycle=64、five_cycle=64、ten_cycle=64、twenty_cycle=16。使用完整阶段预算；不是 8 次更新的短微调。启动记录、配置快照、stdout/stderr 均保存在新目录，原模型保留。

在 Buck 目录查看日志：

```powershell
Get-Content artifacts/runs/central_derivative_rollout_20260913/train.log -Tail 20 -Wait
```

只有当前训练进程已退出时才执行恢复；不要同时启动第二个进程：

```powershell
python -u train.py --config configs/hybrid_event_matlab_v3/central_derivative_rollout.json --device cuda --seed 20260804 --idle-profile soft_idle_loss --stage-batch-size one_cycle=64 --stage-batch-size five_cycle=64 --stage-batch-size ten_cycle=64 --stage-batch-size twenty_cycle=16 --out-dir artifacts/runs/central_derivative_rollout_20260913 --resume artifacts/runs/central_derivative_rollout_20260913/checkpoint.pt
```

新增监督仍为真值状态上的数据电压导数损失，在全部 rollout 阶段启用，初始权重 0.01；不把预测状态与不对应的真值导数强行配对。当前只启动 C 组完整训练，下面 A/B 为可选的严格消融方案，未自动启动。

本实验不加入解析电路方程，不改变网络结构。原始数据和既有 CUDA run 保留。

## 三组完整训练配置

| 组别 | 配置 | 模式预训练标签 | Rollout 电压导数权重 |
|---|---|---|---:|
| A | input_voltage_relative.json | 原前向差分 | 0 |
| B | central_derivative.json | 同模式中心差分 | 0 |
| C | central_derivative_rollout.json | 同模式中心差分 | 0.01 |

0.01 是初始实验权重，不是已验证的最优值。三组使用同一 seed、阶段预算、batch 设置和数据划分，分别从头训练，以隔离标签和辅助监督的作用。原历史 run 的 batch 计划可能不同，不直接充当严格匹配的 A 组。

在 Buck 目录使用可用的 CUDA Python 执行；三组顺序运行，避免争用 GPU：

```powershell
python train.py --config configs/hybrid_event_matlab_v3/input_voltage_relative.json --device cuda --seed 20260913 --out-dir artifacts/runs/derivative_A
python train.py --config configs/hybrid_event_matlab_v3/central_derivative.json --device cuda --seed 20260913 --out-dir artifacts/runs/derivative_B
python train.py --config configs/hybrid_event_matlab_v3/central_derivative_rollout.json --device cuda --seed 20260913 --out-dir artifacts/runs/derivative_C
```

恢复 C 组必须使用 C 组 checkpoint 和原配置：

```powershell
python train.py --config configs/hybrid_event_matlab_v3/central_derivative_rollout.json --device cuda --seed 20260913 --out-dir artifacts/runs/derivative_C --resume artifacts/runs/derivative_C/checkpoint.pt
```

不要用旧 run 的 checkpoint 搭配新配置进行 resume；配置指纹不同且旧 checkpoint 已完成课程，不能代表新目标下的续训。正式三组训练尚未由本次短对照脚本启动。

## 实现

`training/data.py` 在 float64 归一化状态上计算中心差分，以原数据时间间隔和一个开关周期转换为归一化时间。当前数据契约固定每周期 100 点、占空比 0.5。标签与中心状态对齐，排除 PWM 边沿及自然事件附近两格、首尾和跨模式样本。派生结果只在内存缓存；原 NPZ 不变。

预训练 central 选项监督两个状态的导数。Rollout 的辅助项仅监督电压导数，使用真值状态作为网络输入，以免把偏离真值的预测状态错误配对到真值导数。分别求 ON、OFF_DIODE、IDLE 内 MSE，再平均三个模式。没有物理方程标签，没有事件网络的直接辅助梯度。缺失有效样本的模式不参与平均。

启用时，验证报告增加 `voltage_derivative`，总损失包含其加权值，scheduler 和 best checkpoint 也使用这个总损失。因此不能直接比较 A/B/C 的 total 大小；必须比较相同定义的波形、导数及事件指标。默认权重为零时不构造 rollout 导数标签，保持原行为。

## 快速辅助项对照

`scripts/compare_derivative_supervision.py` 从现有 `artifacts/runs/cuda/best.pt` 相同权重出发，分别运行权重 0 和 0.01；相同 8 次更新、batch=6、窗口=5 周期、AdamW LR=1e-6，重新初始化优化器。每个 batch 包含全部六个训练电压。评估全部 18 条验证轨迹的 20 周期自由 rollout，不使用测试集训练或选模。

输出为 `artifacts/derivative_pilot_20260913`，包含来源 SHA256、指标和两组最终纯权重。脚本拒绝覆盖已有目录。纯权重不是可精确续训 checkpoint。这只比较 rollout 辅助项，不能替代从头训练的三组消融，也没有证明高压外推改善。

2026-09-13 实测（18 条验证轨迹、20 周期）：

| 模型 | iL RMSE mA | vC RMSE mV | 归一化电压导数 MSE |
|---|---:|---:|---:|
| 原 best.pt | 0.09470 | 2.23889 | 7.72318e-8 |
| 普通微调 8 次 | 0.21631 | 5.69089 | 7.78460e-8 |
| 中心差分辅助微调 8 次 | 0.21732 | 5.70442 | 7.77338e-8 |

辅助项相对普通微调的导数 MSE 只略低约 0.14%，波形没有改善；两组都逊于原模型。单次短实验不具有统计显著性结论。微调重新初始化了优化器，使用 5 周期窗口并未启用最终 20 周期阶段的 ripple 项，因此退化不能单独归因于中心差分。保留原最佳模型，不提升任何试验权重为 best。下一步正式消融应使用上面的完整课程，不能将这次结果表述为中心差分预训练已经验证失败或成功。

实现验证：`python -m unittest discover -s tests`，37 项测试通过，包括二次曲线导数精度/时间单位、事件屏蔽、向量场梯度、central 预训练和 rollout 验证指标接入。
