# Buck Neural Event ODE

分类：非隔离单向变换器 → 基本单电感型 → Buck。

本目录是自包含的 Buck Neural Event ODE 项目。MATLAB 真值模型采用 Simscape Electrical 物理网络，DCM v3 数据生成代码按 `project.json` 记录求解器、器件和事件筛选参数；Python MAT→NPZ、Neural Event ODE 训练和 Hybrid RK4 评估入口已按设计文档实现。

## 固定物理与数值约束

- 只接受 DCM，每个记录周期必须恰好一次 `OFF_DIODE -> IDLE` 自然事件。
- 固定负载为 $R=80\,\Omega$，负载不作为神经网络输入。
- 连续状态为 $(i_L,v_C)$，外部工况为 $v_S$。
- 占空比 $D=0.5$，开关频率 $20\,\mathrm{kHz}$。
- 每周期 100 个固定输出区间。
- MATLAB 求解器为 `ode23t`，`RelTol=1e-8`、`MaxStep=5e-8` s。
- 真值电路使用 Simscape Electrical 的分段线性 Diode 和 Ideal Switching MOSFET；二者的关断态均由 `Goff=1e-8 S` 数值正则化，不使用旧 Specialized Power Systems 的 snubber 参数。
- 二极管自然截止事件只由原始支路电流 `iD` 标注；`iL` 只用于物理一致性审计。

## MATLAB 数据链

```text
matlab/scripts/buck_ideal_hybrid.slx
    -> generate_dataset.m
    -> extract_hybrid_trajectory.m
    -> export_trajectory.m
    -> data/raw/hybrid_event_matlab_v3/<split>/*.mat
```

MATLAB 数据生成及固定归一化基线配置：

```text
configs/hybrid_event_matlab_v3/project.json
```

当前 Python 默认训练配置为 `configs/hybrid_event_matlab_v3/input_voltage_relative.json`：网络使用 `vC/Vin` 与相对电流，ON/OFF/事件保留 `Vin/32 V` 条件。导数由神经网络输出后还原到原积分尺度，损失和评估口径不变。默认产物写入独立的 `artifacts/hybrid_event_matlab_v3/input_voltage_relative_conditioned_v2/<idle-profile>`。旧权重评估需要显式传入 `--config configs/hybrid_event_matlab_v3/project.json`（路径相对本拓扑），新方式必须从头训练。详见 [归一化设计](docs/normalization-experiment.md)。

旧 `known_pwm_matlab_v2` 配置和 `buck_ideal_ccm.slx` 已删除，不提供兼容分支。

## Simscape 器件配置

`buck_ideal_hybrid.slx` 是实际仿真的器件参数来源，`project.json` 中的 `matlab.device_library`、`matlab.diode` 和 `matlab.mosfet` 是同一配置的文本化审计副本。修改器件参数时必须同步更新两处，数据生成器会把 JSON 中的审计值写入每条 MAT 的 `metadata`。

| 器件 | Simscape 设置 | `project.json` 字段 |
| --- | --- | --- |
| Diode | Piecewise linear | `model_type=piecewise_linear` |
| Diode | `Ron=1e-4 ohm` | `Ron_ohm=1e-4` |
| Diode | `Vf=1e-4 V` | `Vf_v=1e-4` |
| Diode | `Goff=1e-8 S` | `Goff_siemens=1e-8` |
| Diode | `CJ=0 pF`、reverse recovery off | `junction_capacitance_pf=0`、`reverse_recovery=off` |
| MOSFET | Ideal switching | `model_type=ideal_switching` |
| MOSFET | `Rds=1e-4 ohm` | `Rds_on_ohm=1e-4` |
| MOSFET | `Goff=1e-8 S` | `Goff_siemens=1e-8` |
| MOSFET | `Vth=0.5 V` | `threshold_voltage_v=0.5` |
| MOSFET | external protection diode | `body_diode_mode=external` |

生成正式数据前，至少核对 JSON 可以解析且器件字段存在：

```bash
python3 -m json.tool \
  topologies/non_isolated_unidirectional/buck/configs/hybrid_event_matlab_v3/project.json \
  >/dev/null
```

## 生成数据

从 MATLAB 切换到脚本目录：

```matlab
cd('topologies/non_isolated_unidirectional/buck/matlab/scripts');
split_name = "train";
generate_dataset;
```

可用 split：

- `train`
- `validation`
- `test_seen`
- `test_interpolation`
- `test_voltage_extrapolation`
- `test_extrapolation_high`

生成器对每个 split 只设置一次 Twister RNG，按配置电压顺序使用二维 LHS 候选。同名轨迹文件会直接覆盖，可重复生成。仅 `BuckNeuralODE:DCMViolation` 会触发候选补采；奇异拓扑、日志缺失、PWM 错位或未知异常会立即终止。MAT metadata 中的 `device_library`、Diode 和 MOSFET 字段必须与生成该数据的 `.slx` 设置一致。

## 数据转换与训练入口

拓扑自包含训练入口为 `topologies/non_isolated_unidirectional/buck/train.py`。下面的 Python 命令从包含 `dc_dc_converter_neural_ode/` 的上一级目录执行；训练入口只接受 DCM v3 数据和 checkpoint。

```bash
# Convert the train split
python3 dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/scripts/mat_to_npz.py --split train
```

```bash
# Full training
python3 dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/train.py --out-dir dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/artifacts/runs/full

# Smoke test
python3 dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/train.py --smoke --out-dir dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/artifacts/smoke

# Hard-IDLE 对照
python3 dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/train.py --idle-profile hard_idle_constraint --out-dir dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/artifacts/runs/hard_idle

# Resume training
python3 dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/train.py --resume dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/artifacts/checkpoint.pt --out-dir dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/artifacts/resume

# Resume a legacy event checkpoint whose plateau scheduler reduced LR too far;
# the override applies only to event_pretrain, not to later rollout stages.
python3 dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/train.py --resume dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/artifacts/checkpoint.pt --event-pretrain-epochs 700 --out-dir dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/artifacts/resume

# Evaluate existing weights
python3 dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/train.py --evaluate-only --resume dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/artifacts/best.pt --out-dir dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/artifacts/eval
```

训练每完成 2 个 epoch 原子覆盖一次 `checkpoint.pt`，阶段结束或早停时也会
保存。`--resume` 会恢复当前阶段的模型、优化器、scheduler、best/stall 和
随机数状态，并从下一 epoch 继续。事件预训练使用固定学习率；恢复旧的
plateau-scheduler checkpoint 时会自动将学习率恢复到配置值。若旧 checkpoint
已经到达原事件阶段上限，可用 `--event-pretrain-epochs` 单独扩展该阶段，后续
rollout 仍采用各自的配置轮数。

Python DCM 专项测试：

```bash
cd dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck
python3 -m unittest discover -s tests -v
```

当前归一化配置默认使用 `auto`/`float32`：优先 CUDA，其次 MPS，不自动回退 CPU。固定归一化基线 project.json 仍保留原 mps 设置，可通过 --device cuda 覆盖。仅在明确调试时使用 --device cpu。

MATLAB 测试不作为训练前置步骤。生成脚本负责按配置筛选 DCM 轨迹，Python
数据入口在构造模型前审计 NPZ v3 schema、固定负载、模式顺序、事件和数值
有限性。

需求来源：

- `docs/superpowers/specs/2026-08-25-buck-neural-event-ode-design.md`

## Mac 与 Windows NVIDIA 训练

同一套 train.py、Neural Event ODE、Hybrid RK4 和评估代码适配 MPS/CUDA。设备选择位于 train.py 的 `_device`；`--device` 可选 auto、mps、cuda、cuda:N、cpu。cuda:N 是单 GPU 选择，不是多 GPU 分布式训练。CUDA 保持 float32，关闭 TF32，不启用 AMP。

Windows 建议使用 Python 3.11 的独立环境。从 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 选择 Windows / Pip / 与驱动和 GPU 兼容的 CUDA 版本并执行其安装命令，然后在工作区根目录执行：

```powershell
python -m pip install -r topologies/non_isolated_unidirectional/buck/requirements.txt
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
python topologies/non_isolated_unidirectional/buck/train.py --device cuda --out-dir topologies/non_isolated_unidirectional/buck/artifacts/relative_cuda/seed_20260804
```

Mac 使用相同命令，把 `--device cuda` 改成 `--device mps`。使用当前默认归一化配置时可省略 --device，自动选 GPU。Windows 和 Mac 都可以使用这些正斜杠路径；以上命令均从仓库根目录运行。

训练数据未提交 Git。迁移电脑时另行复制 `data/processed/hybrid_event_matlab_v3/*.npz`，或用 `--data-dir` 指定该目录；仅训练需要 train.npz 和 validation.npz，完整评估还需其他测试 split。训练无需在 Windows 上重新运行 MATLAB。

续训在相同命令后增加 `--resume <checkpoint.pt路径>`，保持配置文件不变，仅通过 --device 切换后端。checkpoint 保存 CPU 格式的模型/优化器状态，加载后由 PyTorch 按参数设备恢复优化器张量；新增所选 CUDA GPU 的 RNG 状态。跨 MPS/CUDA 的 RNG 无法互相转换，缺少目标后端状态时保留启动种子并提示，因此跨后端不保证逐步一致。旧固定归一化权重仍须使用 project.json，不能迁移到新归一化模型。

配置 SHA-256 统一 LF/CRLF 换行符，保留原 LF 文件的指纹。`runtime_environment.json` 记录实际后端、GPU 名称、PyTorch/CUDA 版本和精度。CUDA rollout 前后向测速、执行优化与复现命令见 [GPU 性能说明](docs/gpu-performance.md)；完整训练与 MPS 性能仍需分别实测。

当前 CUDA 实验后续采用 **ten_cycle batch=16、twenty_cycle batch=4**，validation batch 保持配置值。原始配置指纹不变，分阶段覆盖计划会写入新 checkpoint。从仓库根目录续训：

```powershell
python -X utf8 -u topologies/non_isolated_unidirectional/buck/train.py `
  --device cuda `
  --stage-batch-size ten_cycle=16 `
  --stage-batch-size twenty_cycle=4 `
  --out-dir topologies/non_isolated_unidirectional/buck/artifacts/runs/cuda `
  --resume topologies/non_isolated_unidirectional/buck/artifacts/runs/cuda/checkpoint.pt
```

本命令替代之前统一 `--rollout-batch-size 64` 的续训命令。已有进程占用同一目录时，训练锁会拒绝第二个写入进程。参数优先级与 checkpoint 兼容行为见 [GPU 性能说明](docs/gpu-performance.md)。

## 先诊断再延长训练

可在长训练停止后，用独立目录执行短时 batch 对照：

```powershell
python -X utf8 -u topologies/non_isolated_unidirectional/buck/scripts/diagnose_and_compare.py `
  --checkpoint topologies/non_isolated_unidirectional/buck/artifacts/runs/cuda/checkpoint.pt `
  --out-dir topologies/non_isolated_unidirectional/buck/artifacts/my_batch_pilot `
  --seconds 180 --batches 16 32 64
```

输出目录必须尚不存在。脚本先复制 checkpoint，再对训练/验证子集进行 1、5、10、20 周期诊断；每组从相同权重、优化器状态和采样序列开始，以同步后的实际训练秒数为预算。验证和保存时间另计，预算最多超出一次更新；验证使用固定的电压/初始电流分层子集。短对照固定 checkpoint 中的学习率，不运行按 epoch 的 scheduler。结果、波形、每组最好与最后权重均独立保存，不写回原实验。试验权重不含完整续训状态，不能直接当作 `--resume` checkpoint。

正式 rollout 训练现在每个 epoch 检查并保存该阶段真正观察到的最佳验证权重到 `best_validation/<stage>.pt`（选择信息与权重的原子 bundle，不是续训 checkpoint）。完成训练后 `last.pt` 保存最后权重，`best.pt` 使用最后训练阶段的最佳验证权重进行评估，`best_selection.json` 记录来源；旧运行没有最佳权重记录时明确回退到最后权重，无法追回未保存的历史最佳。不同阶段的损失不互相比较。阶段报告还记录实际 `optimizer_updates` 和最近一轮训练/验证时间。
