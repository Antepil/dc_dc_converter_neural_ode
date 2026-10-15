#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_extrapolation_high 的 DCM v3 多电压波形汇总。"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

if __package__:
    from ..training.data import load_data, normalize
    from ..training.hybrid_rk4 import hybrid_rk4
    from ..training.model import NeuralEventHybridODE
    from ._bootstrap import DEFAULT_CONFIG, DEFAULT_DATA_DIR, DEFAULT_WAVEFORM_DIR, DEFAULT_WEIGHTS
else:
    from _bootstrap import DEFAULT_CONFIG, DEFAULT_DATA_DIR, DEFAULT_WAVEFORM_DIR, DEFAULT_WEIGHTS
    from dc_dc_converter_neural_ode.topologies.non_isolated_unidirectional.buck.training.data import load_data, normalize
    from dc_dc_converter_neural_ode.topologies.non_isolated_unidirectional.buck.training.hybrid_rk4 import hybrid_rk4
    from dc_dc_converter_neural_ode.topologies.non_isolated_unidirectional.buck.training.model import NeuralEventHybridODE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--out-dir", default=DEFAULT_WAVEFORM_DIR)
    parser.add_argument("--length", type=int, default=200)
    args = parser.parse_args()
    cfg = json.loads(open(args.config, encoding="utf-8").read())
    refs = cfg["normalization"]
    data = normalize(load_data(args.data_dir), refs).get("test_extrapolation_high")
    if data is None:
        raise SystemExit("数据目录缺少 test_extrapolation_high.npz")
    model = NeuralEventHybridODE(cfg["model"], cfg["experiments"]["default"])
    payload = torch.load(args.weights, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model_state" in payload:
        if payload.get("schema_version") != "hybrid_event_matlab_v3":
            raise ValueError("权重不是 hybrid_event_matlab_v3")
        payload = payload["model_state"]
    model.load_state_dict(payload)
    model = model.double().eval()
    length = min(args.length, (data["states"].shape[1] - 1) // 100)
    os.makedirs(args.out_dir, exist_ok=True)
    voltages = torch.unique(data["input_voltage"])
    fig, axes = plt.subplots(2, len(voltages), figsize=(3.2 * len(voltages), 5), sharex=True)
    if len(voltages) == 1:
        axes = axes[:, None]
    with torch.no_grad():
        for column, voltage in enumerate(voltages):
            index = (data["input_voltage"] == voltage).nonzero(as_tuple=True)[0][0]
            prediction = hybrid_rk4(model, data["states_norm"][index:index + 1, 0], data["vs_norm"][index:index + 1], length).states[:, 0]
            target = data["states_norm"][index, :length * 100 + 1]
            start = max(0, length - 2) * 100
            time = torch.arange(start, length * 100 + 1, dtype=torch.float64) / 100.0
            scale = torch.tensor([refs["current_ref_a"], refs["voltage_ref_v"]], dtype=torch.float64)
            for row, label in enumerate(("iL (A)", "vC (V)")):
                axes[row, column].plot(time, target[start:, row] * scale[row], "k-")
                axes[row, column].plot(time, prediction[start:, row] * scale[row], "r--")
                axes[row, column].set_ylabel(label)
                axes[row, column].grid(True, alpha=0.3)
                if row == 0:
                    axes[row, column].set_title(f"vS={float(voltage):.0f} V")
    axes[-1, 0].set_xlabel("tau (cycles)")
    fig.suptitle("DCM voltage extrapolation · last two cycles")
    fig.tight_layout()
    path = os.path.join(args.out_dir, "extra_hi_waveforms_all.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"已保存: {path}")


if __name__ == "__main__":
    main()
