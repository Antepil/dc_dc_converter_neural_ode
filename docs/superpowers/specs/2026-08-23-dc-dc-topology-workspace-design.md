# DC–DC 变换器 Neural ODE 总工作区设计

日期：2026-08-23

状态：已确认，待实施计划

## 1. 目标

在当前仓库中新建独立总工作区 `dc_dc_converter_neural_ode/`。今后的 Buck、Boost、Ćuk、隔离型和双向 DC–DC 拓扑工作均在该目录内开展。

首期完整复制现有 Buck 项目，并适配新目录，使新副本可以独立完成：

- MATLAB/Simulink 数据生成；
- MAT 到 NPZ 转换；
- Neural Event ODE 训练；
- checkpoint 断点续训；
- 仅评估和波形分析。

现有 `training/`、`neural_ode_v2/`、根目录文档、数据和训练产物保持原样：不移动、不重命名、不删除、不作为本轮重构对象。当前工作树中用户已有的未提交修改也不得被恢复或覆盖。

## 2. 设计原则

1. **拓扑自包含**：每个拓扑拥有自己的训练入口、模型、数据代码、配置、仿真、数据和产物。
2. **先有真实复用，再提取公共层**：首期不创建 `framework/`、插件契约、注册表或统一 CLI。接入第二个真实拓扑后，再比较实现并提取已经验证的公共代码。
3. **旧项目只读保留**：新项目通过复制建立，之后新旧副本互不自动同步。
4. **行为保持**：目录适配不得改变 Buck 的数学模型、数据格式、网络参数名、积分设置、训练课程、损失或评估定义。
5. **不创建空支持**：未实现的拓扑只记录规划路径，不创建空目录或占位实现。

## 3. 拓扑分类

规划使用四个稳定大类，目录名采用英文 `snake_case`：

```text
topologies/
├── non_isolated_unidirectional/
│   ├── buck/
│   ├── boost/
│   ├── buck_boost/
│   └── cuk/
├── multiphase/
│   └── two_phase_interleaved_buck/
├── isolated_unidirectional/
│   ├── forward/
│   ├── flyback/
│   ├── push_pull/
│   ├── half_bridge/
│   └── full_bridge/
└── bidirectional/
    ├── non_isolated_buck_boost/
    └── dab/
```

“基本单电感型”“电容传能高阶型”“单端型”“桥式型”等细分类写入对应拓扑的 `README.md`，不继续增加目录深度。

首期只实际创建：

```text
topologies/non_isolated_unidirectional/buck/
```

二极管全桥整流级不属于首期 DC–DC 拓扑工作区内容，不创建 `components/rectifiers/`。确有公共整流组件需求时再单独设计。

## 4. 新工作区结构

```text
dc_dc_converter_neural_ode/
├── README.md
├── AGENTS.md
├── .gitignore
├── docs/
│   ├── architecture.md
│   ├── topology_catalog.md
│   └── superpowers/specs/
└── topologies/
    └── non_isolated_unidirectional/
        └── buck/
            ├── README.md
            ├── train.py
            ├── requirements.txt
            ├── configs/
            │   └── known_pwm_matlab_v2/project.json
            ├── training/
            │   ├── __init__.py
            │   ├── data.py
            │   ├── model.py
            │   ├── loops.py
            │   ├── losses.py
            │   ├── evaluate.py
            │   └── checkpoint.py
            ├── analysis/
            │   ├── __init__.py
            │   ├── _bootstrap.py
            │   ├── plot_extra_hi.py
            │   ├── plot_ripple.py
            │   ├── plot_ripple_vc.py
            │   └── plot_waveforms.py
            ├── matlab/scripts/
            │   ├── buck_ideal_ccm.slx
            │   ├── generate_dataset.m
            │   ├── export_trajectory.m
            │   ├── initialize_buck_model.m
            │   └── trajectory_file_path.m
            ├── scripts/
            │   └── mat_to_npz.py
            ├── docs/
            │   ├── ROUTE.md
            │   ├── requirements-alignment.md
            │   └── CHANGE_LOG.md
            ├── data/
            ├── artifacts/
            └── tests/
```

`training/train.py` 在新副本中改名为 `buck/training/loops.py`，避免与拓扑级入口 `buck/train.py` 重名。其训练循环逻辑不变。

## 5. 完整复制范围

| 现有内容 | 新位置 |
| --- | --- |
| `training/train_buck.py` | `buck/train.py` |
| `training/data.py`、`model.py`、`train.py`、`losses.py`、`evaluate.py`、`checkpoint.py` | `buck/training/`，其中原 `train.py` 改名为 `loops.py` |
| `training/plot_*.py` | `buck/analysis/` |
| `training/requirements.txt` | `buck/requirements.txt` |
| `training/artifacts/` | `buck/artifacts/` |
| `train.log` | `buck/artifacts/train.log` |
| `neural_ode_v2/configs/` | `buck/configs/` |
| `neural_ode_v2/matlab/` | `buck/matlab/` |
| `neural_ode_v2/scripts/` | `buck/scripts/` |
| `neural_ode_v2/data/` | `buck/data/`，raw、processed、CSV、archive 和 manifest 全部复制 |
| `ROUTE.md` | `buck/docs/ROUTE.md` |
| `requirements-alignment.md` | `buck/docs/requirements-alignment.md` |
| `neural_ode_v2/CHANGE_LOG.md` | `buck/docs/CHANGE_LOG.md` |

复制的是完整有效工作内容。`.DS_Store`、`__pycache__/`、`*.pyc`、`*.slxc` 和 `slprj/` 属于可再生缓存，不进入新工作区。

新工作区中的 `data/`、`artifacts/` 和训练日志由该工作区自己的 `.gitignore` 排除，但文件必须实际复制并通过数量、大小及关键文件校验。现有有效内容约增加 225 MB 磁盘占用。

## 6. 新架构适配

### 6.1 Python 入口与导入

`buck/train.py` 是 Buck 的独立主入口。它使用包限定导入，例如从 `training.data`、`training.model` 和 `training.loops` 导入，不依赖旧根目录 `training/`，也不通过修改 `PYTHONPATH` 寻找旧代码。

新默认路径均由 `Path(__file__).resolve()` 相对 Buck 根目录计算：

- 配置：`buck/configs/known_pwm_matlab_v2/project.json`；
- 数据：`buck/data/processed/known_pwm_matlab_v2/`；
- 产物：`buck/artifacts/`。

从仓库根目录、总工作区目录或 Buck 目录启动时，路径解析结果必须一致。`--config`、`--data-dir`、`--out-dir`、`--resume`、`--backprop`、`--evaluate-only` 等现有 CLI 参数和语义保持不变。

### 6.2 分析脚本

`analysis/_bootstrap.py` 通过自身位置确定 Buck 根目录，将该目录加入模块搜索路径，并集中提供配置、数据、checkpoint、指标和波形输出的本地默认路径。各分析脚本先加载该 bootstrap，再从本地 `training` 包导入模型与数据代码。因此，从任意当前工作目录直接执行 `python3 <buck>/analysis/plot_waveforms.py` 都能工作，且不会引用旧 `training/` 或旧数据目录。

### 6.3 MATLAB 与数据转换

新 MATLAB 脚本必须仅解析新 Buck 项目中的配置、模型和数据输出位置。当前 `generate_dataset.m` 通过 `script_dir` 向上两级定位 Buck 根目录；复制到 `buck/matlab/scripts/` 后继续得到正确路径。

`buck/scripts/mat_to_npz.py` 从自身位置向上一级定位 Buck 根目录，并默认使用新项目中的：

- `configs/known_pwm_matlab_v2/project.json`；
- `data/raw/known_pwm_matlab_v2/`；
- `data/processed/known_pwm_matlab_v2/`。

所有路径必须显式落在新工作区，不允许缺失时静默回退到旧项目。

### 6.4 checkpoint 与数值兼容

复制现有 `checkpoint.pt`、`best.pt` 和评估产物。新代码保持 `TwoModeVectorField` 的模块属性名和 `state_dict` 键不变，因此旧权重可以直接加载。

以下内容不得在本轮更改：

- 两个独立 MLP 的结构与初始化；
- 状态和输入归一化；
- D=0.5、100 点/周期、RK4 和步长 0.01；
- standard/adjoint 路线语义；
- local_mode 与 1/5/10/20 周期课程；
- MSE、ripple loss、优化器、调度器、早停和梯度裁剪；
- NPZ schema 与评估指标。

## 7. 文档

`dc_dc_converter_neural_ode/README.md` 说明总工作区用途、已实现拓扑和规划分类。

`topology_catalog.md` 记录用户确认的完整分类，但明确区分“规划”与“已实现”。

Buck 的 `README.md` 提供以下新路径命令：

```bash
python3 dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/train.py
python3 dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/train.py --smoke
python3 dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/train.py \
  --resume dc_dc_converter_neural_ode/topologies/non_isolated_unidirectional/buck/artifacts/checkpoint.pt
```

新工作区 `AGENTS.md` 记录拓扑自包含、旧项目不可修改、未接入拓扑不建空目录等约束。

## 8. 错误处理

- 缺少配置、必要 NPZ、权重或 checkpoint 时，错误信息必须打印新工作区中的完整目标路径。
- 未知 split、checkpoint 路线不匹配等现有拒绝逻辑保持不变。
- 不允许自动搜索或使用旧目录中的同名文件。
- 完整复制失败时必须停止，不得留下被误认为可用的半成品 Buck 项目。

## 9. 验收标准

1. **旧项目未变化**：实施前后，所有现有路径的内容与 Git 状态保持不变；只新增 `dc_dc_converter_neural_ode/`。
2. **复制完整**：raw、processed、CSV、archive、manifest、artifacts 和有效源文件均存在于新 Buck 项目；缓存文件除外。
3. **路径独立**：新工作区代码中不存在对旧 `training/`、`Buck_neural_ode_v2/` 或 `neural_ode_v2/` 的运行时依赖。
4. **入口可用**：新 Buck 入口和分析脚本的 `--help` 可运行，模块可正常导入。
5. **数据等价**：同一 NPZ 在新旧代码下得到相同 split、张量、归一化结果和窗口索引。
6. **模型等价**：加载复制的 `best.pt`，固定输入 rollout 的输出在严格容差内与旧代码一致。
7. **续训可用**：使用复制的 checkpoint 再制作测试副本，完成短程 resume 冒烟；不得修改原始新旧 checkpoint。
8. **评估可用**：`--evaluate-only` 能从新目录加载复制权重和数据并写入新 `artifacts/`。
9. **文档准确**：README 命令、目录树和实际文件一致；未实现拓扑不得表述为已支持。

## 10. 非目标

本轮不包括：

- 修改现有旧项目；
- 实现 Boost、Ćuk、交错 Buck、隔离型或双向拓扑；
- 设计通用多模式向量场；
- 跨拓扑共享网络权重；
- 创建统一训练框架、插件注册表或总 CLI；
- 建模整流级与 DC–DC 级联；
- 重训、调参或改变当前验收指标。

第二个拓扑真正落地后，再基于至少两个实现评估是否提取 `shared/` 公共模块。
