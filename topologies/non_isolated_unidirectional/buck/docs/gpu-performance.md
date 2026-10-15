# GPU 执行优化（2026-09-12）

针对小型 MLP、长时间积分和小 batch 的执行开销进行优化：

- 已知单一模式时直接运行对应 MLP；混合模式继续使用稠密选择，避免新增动态索引。
- PWM ON 段以及整个 batch 已进入 IDLE 后，跳过事件索引查询。
- 完整轨迹、事件分数和被事件更新覆盖的试探状态在返回前统一检查有限性；割线细化与事件后状态仍即时检查。调试可传入 `hybrid_rk4(..., check_finite_every_step=True)`。
- 训练日志在设备上累加已 detach 的 loss，每个 epoch 输出时转为 Python 数值。

模型结构、配置、float32 精度、RK4 步长、事件细化次数与默认 batch size 保持原设置。原 checkpoint 可用于续训；已经运行的 Python 进程需重新启动并从 checkpoint 恢复才能使用新代码。标量模式调用不再为未使用的网络构建零梯度图，独立调用该接口的自定义优化器代码应区分 None 梯度与零梯度；现有训练流程在联合 rollout 中仍使用混合模式路径。

## 复现测速

从仓库根目录执行：

```powershell
python topologies/non_isolated_unidirectional/buck/scripts/benchmark_rollout.py --device cuda --cycles 1 --batches 4 16 64 --repeats 3
python topologies/non_isolated_unidirectional/buck/scripts/benchmark_rollout.py --device cuda --cycles 20 --batches 4 16 --repeats 3
```

Mac 使用 `--device mps`，并保持 cycles、batch 和 repeats 一致。脚本使用固定种子、固定参考归一化配置和合成输入，预热后测量 forward + backward，并在计时边界同步设备。它不包含数据读取、完整训练损失、optimizer update、validation 或 checkpoint I/O，不能直接代表完整 epoch 时间。

可传 `--reference-dir <目录>`，目录中放优化前的 `model.py` 与 `hybrid_rk4.py`；脚本会逐项比较轨迹、事件诊断和参数梯度，再报告各自耗时。比较基线应来自修改前的实际工作区，避免丢失未提交的归一化改动。

## 本机初步测量

RTX 3060 Laptop GPU（6 GB），PyTorch 2.5.1+cu121，float32，3 次测量中位数。合成输入的事件行为不代表真实训练分布；其他进程和笔记本功耗状态可能影响测量。

| 周期数 | Batch | 修改前（秒） | 修改后（秒） | 加速比 |
| --- | --- | --- | --- | --- |
| 1 | 4 | 1.595 | 0.516 | 3.09× |
| 1 | 16 | 1.418 | 0.522 | 2.72× |
| 1 | 64 | 1.698 | 0.493 | 3.45× |
| 20 | 4 | 36.482 | 25.358 | 1.44× |
| 20 | 16 | 37.908 | 25.141 | 1.51× |

1 周期、batch 4 的测量在补齐被事件覆盖的试探状态检查后重跑，其余测量在该检查补齐前启动；最终代码的 CPU/CUDA 输出及梯度均与原始快照匹配。更大的 batch 在这个小网络测试中提高吞吐量，但会改变每个 epoch 的更新次数和训练行为，因此没有自动修改训练 batch 配置。

28 项回归测试全部通过，覆盖模式输出/输入与参数梯度、不同归一化与 IDLE profile、异步事件、事件缺失、无效初始 guard、非有限值拒绝、三阶段训练与 checkpoint。CUDA 输出及梯度还与修改前快照直接对比；未完成 Mac 实测或完整重训练精度评估。

## 实际训练批次调优

2026-09-12 第二次排查发现两个训练进程同时使用 `artifacts/runs/cuda`，且 NVIDIA 报告 86°C、`SW Thermal Slowdown: Active`。已停止重复进程，保留 epoch 16 checkpoint 备份后进行独占 GPU 测速。新的训练入口用操作系统文件锁阻止同一输出目录的并发运行；进程终止后锁自动释放，磁盘上留下 `training.lock` 不表示仍被占用。

在当前输入电压相对归一化模型、实际数据、five_cycle checkpoint 上，测量 forward、完整训练损失、backward、梯度裁剪和 AdamW update（预热后两次平均）：

| Training batch | 每步秒数 | 每 epoch 更新数 | 估计训练部分秒数 | PyTorch 峰值分配 MB |
| --- | --- | --- | --- | --- |
| 16 | 6.79 | 72 | 489 | 57 |
| 64 | 7.89 | 18 | 142 | 166 |
| 128 | 8.38 | 9 | 75 | 314 |

这些估计不含 validation、数据整理和 checkpoint I/O，也会受笔记本温度影响。当时五周期选择 batch 64；每 epoch 仍覆盖全部 1,152 个训练窗口，validation 保持原有 batch 16。由于更新次数改变，这是一项训练超参数调整，不能仅根据 epoch 秒数推断达到同等精度的总时间；需要继续对比验证误差和实际优化器更新数。

恢复后的首个完整 epoch（17/130）实测 **137.9 秒**：训练 105.0 秒、validation 32.8 秒、18 次更新，validation loss 为 `8.196e-7`。之前日志稳定在约 401–405 秒，按 epoch 耗时约快 2.9 倍；这个差异包含批次调整、消除重复进程与运行温度变化，不能归因于单项优化。该 validation loss 没有显示首个 epoch 的明显退化，但还不能证明长期收敛等价。

## 当前续训方案：按阶段设置 batch

已停止统一 batch 64 的进程，并备份当前 checkpoint（ten_cycle，第 8 个 epoch）。此前 five_cycle 已完成；后续按下表继续，不重置已完成轮数、优化器、学习率调度器或早停状态：

| 阶段 | 训练 batch | 训练窗口数 | 更新次数/epoch | validation batch | epoch 上限 |
| --- | --- | --- | --- | --- | --- |
| ten_cycle | 16 | 792 | 50（末批 8 个窗口） | 8 | 80 |
| twenty_cycle | 4 | 72 | 18 | 4 | 60 |

原始配置文件不修改，因此现有 checkpoint 的配置指纹仍匹配。这次续训属于中途调整 batch 的混合训练实验，不能标记为从头使用固定 batch 的对照实验。

从仓库根目录执行以下命令，从 ten_cycle 第 9 个 epoch 继续。不要再加旧的 `--rollout-batch-size 64`：

```powershell
python -X utf8 -u topologies/non_isolated_unidirectional/buck/train.py `
  --device cuda `
  --stage-batch-size ten_cycle=16 `
  --stage-batch-size twenty_cycle=4 `
  --out-dir topologies/non_isolated_unidirectional/buck/artifacts/runs/cuda `
  --resume topologies/non_isolated_unidirectional/buck/artifacts/runs/cuda/checkpoint.pt
```

`--stage-batch-size STAGE=N` 可重复使用，仅调整指定 rollout 阶段的训练 batch；事件预训练与 validation batch 不变。优先级为：本次指定阶段参数 > 本次全局 `--rollout-batch-size` > checkpoint 的 `training_batch_sizes` > 旧 checkpoint 的阶段报告 > 配置默认值。重复指定同一阶段、未知阶段和非正 batch 会报错。

新 checkpoint 同时保存当前与未来阶段的完整 `training_batch_sizes` 计划。因此首次按以上命令保存后，即使之后不传覆盖参数，twenty_cycle=4 也能恢复；旧 checkpoint 尚未保存该计划时必须使用以上参数，避免恢复报告中的 ten_cycle=64。启动日志会显示未完成阶段的 batch 计划。每两轮及阶段结束时保存，尚未保存的轮次会重算。

暂停实际训练后可用 `scripts/benchmark_training_batch.py --checkpoint artifacts/runs/cuda/checkpoint.pt` 从 Buck 拓扑目录复现真实数据测速；该脚本只在内存中更新模型，不写回实验 checkpoint。新增 batch override 和跨进程锁测试后，共 30 项测试通过。
