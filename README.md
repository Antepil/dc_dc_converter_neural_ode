# DC–DC Converter Neural ODE

本目录是 DC–DC 变换器 Neural ODE 的统一工作区。每个拓扑是一个自包含项目，拥有独立的数据生成、配置、训练、评估和产物目录。

## 已实现

- Buck DCM v3 Neural Event ODE：`topologies/non_isolated_unidirectional/buck/`

其他拓扑均处于规划状态，见 [`docs/topology_catalog.md`](docs/topology_catalog.md)。

## Buck 快速开始

从仓库根目录运行：

```bash
python3 topologies/non_isolated_unidirectional/buck/train.py --smoke \
  --out-dir topologies/non_isolated_unidirectional/buck/artifacts/smoke
```

现有仓库根目录中的旧 Buck 项目保留不动；后续开发以本工作区为准。
