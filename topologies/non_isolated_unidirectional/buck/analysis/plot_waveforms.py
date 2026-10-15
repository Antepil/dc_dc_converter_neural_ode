#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DCM v3 轨迹波形辅助脚本，供 ripple 图脚本复用。"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__:
    from ..training.checkpoint import load_checkpoint
    from ..training.data import load_data, normalize
    from ..training.hybrid_rk4 import hybrid_rk4
    from ..training.model import NeuralEventHybridODE
    from ._bootstrap import DEFAULT_CKPT, DEFAULT_CONFIG, DEFAULT_DATA_DIR, DEFAULT_WAVEFORM_DIR
else:
    from _bootstrap import DEFAULT_CKPT, DEFAULT_CONFIG, DEFAULT_DATA_DIR, DEFAULT_WAVEFORM_DIR
    from dc_dc_converter_neural_ode.topologies.non_isolated_unidirectional.buck.training.checkpoint import load_checkpoint
    from dc_dc_converter_neural_ode.topologies.non_isolated_unidirectional.buck.training.data import load_data, normalize
    from dc_dc_converter_neural_ode.topologies.non_isolated_unidirectional.buck.training.hybrid_rk4 import hybrid_rk4
    from dc_dc_converter_neural_ode.topologies.non_isolated_unidirectional.buck.training.model import NeuralEventHybridODE


SPLIT_TITLES = {
    "train": "Train",
    "validation": "Validation",
    "test_seen": "Test seen",
    "test_interpolation": "Test interpolation",
    "test_voltage_extrapolation": "Test voltage extrapolation",
    "test_extrapolation_high": "Test extrapolation high",
}


def load_model(cfg, ckpt_path):
    model = NeuralEventHybridODE(cfg["model"], cfg["experiments"]["default"])
    checkpoint = load_checkpoint(ckpt_path)
    model.load_state_dict(checkpoint["model_state"])
    return model.double().eval()


def rollout(model, states, vs, length):
    """从每条轨迹初值执行 DCM Hybrid rollout，返回 ``(N,T,2)``。"""
    with torch.no_grad():
        result = hybrid_rk4(model, states[:, 0].double(), vs.double(), length)
    return result.states.permute(1, 0, 2).cpu()


def per_cycle_metrics(pred, target, length):
    single = pred.dim() == 2
    if single:
        pred, target = pred.unsqueeze(0), target.unsqueeze(0)
    rmse = np.zeros((pred.shape[0], length))
    ripple = np.zeros((pred.shape[0], length))
    for cycle in range(length):
        segment = slice(cycle * 100 + 1, cycle * 100 + 101)
        rmse[:, cycle] = (torch.sqrt((pred[:, segment] - target[:, segment]).square().mean(dim=(1, 2))) * 100.0).numpy()
        segment = slice(cycle * 100, cycle * 100 + 101)
        line = torch.linspace(0.0, 1.0, 101, dtype=pred.dtype)
        pred_v = pred[:, segment, 1]
        target_v = target[:, segment, 1]
        pred_d = pred_v - (pred_v[:, :1] + line[None] * (pred_v[:, -1:] - pred_v[:, :1]))
        target_d = target_v - (target_v[:, :1] + line[None] * (target_v[:, -1:] - target_v[:, :1]))
        pred_ptp = pred_d.amax(1) - pred_d.amin(1)
        target_ptp = target_d.amax(1) - target_d.amin(1)
        ripple[:, cycle] = ((pred_ptp - target_ptp).abs() / (target_ptp + 1e-6) * 100.0).numpy()
    return (rmse[0], ripple[0]) if single else (rmse, ripple)


def denorm(states, refs):
    ref = torch.tensor([refs["current_ref_a"], refs["voltage_ref_v"]], dtype=states.dtype)
    return states * ref


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--out-dir", default=DEFAULT_WAVEFORM_DIR)
    parser.add_argument("--length", type=int, default=None)
    args = parser.parse_args()
    cfg = json.loads(open(args.config, encoding="utf-8").read())
    refs = cfg["normalization"]
    splits = normalize(load_data(args.data_dir), refs)
    model = load_model(cfg, args.ckpt)
    os.makedirs(args.out_dir, exist_ok=True)
    for name, data in splits.items():
        length = args.length or (data["states"].shape[1] - 1) // 100
        target = data["states_norm"][:, :length * 100 + 1]
        prediction = rollout(model, target, data["vs_norm"], length)
        _, ripple = per_cycle_metrics(prediction, target, length)
        index = 0
        physical_target = denorm(target[index], refs)
        physical_prediction = denorm(prediction[index], refs)
        start = max(0, length - 2) * 100
        t = np.arange(start, length * 100 + 1) / 100.0
        fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
        for channel, label in enumerate(("iL (A)", "vC (V)")):
            axes[channel].plot(t, physical_target[start:, channel], "k-", label="data")
            axes[channel].plot(t, physical_prediction[start:, channel], "r--", label="model")
            axes[channel].set_ylabel(label)
            axes[channel].grid(True, alpha=0.3)
            axes[channel].legend(loc="upper right")
        axes[-1].set_xlabel("tau (cycles)")
        axes[0].set_title(f"{SPLIT_TITLES.get(name, name)} · {length}-cycle rollout · ripple {ripple[index, -1]:.3f}%")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"steady_{name}.png"), dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
