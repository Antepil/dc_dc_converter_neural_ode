#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vC（输出电压）纹波聚焦图：5 splits × (原始 vC / 去趋势纹波 mV)，2 周期。

输出：buck/artifacts/waveforms/vc_ripple_overview.png
每列标题标注峰峰测量：data / model / 相对误差。
"""

import argparse
import json
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__:
    from ._bootstrap import DEFAULT_WAVEFORM_DIR
    from .plot_waveforms import (
        DEFAULT_CKPT,
        DEFAULT_CONFIG,
        DEFAULT_DATA_DIR,
        SPLIT_TITLES,
        denorm,
        load_model,
        per_cycle_metrics,
        rollout,
    )
    from ..training.data import load_data, normalize
else:
    from _bootstrap import DEFAULT_WAVEFORM_DIR
    from dc_dc_converter_neural_ode.topologies.non_isolated_unidirectional.buck.analysis.plot_waveforms import (
        DEFAULT_CKPT,
        DEFAULT_CONFIG,
        DEFAULT_DATA_DIR,
        SPLIT_TITLES,
        denorm,
        load_model,
        per_cycle_metrics,
        rollout,
    )
    from dc_dc_converter_neural_ode.topologies.non_isolated_unidirectional.buck.training.data import load_data, normalize

OUT = os.path.join(DEFAULT_WAVEFORM_DIR, "vc_ripple_overview.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    cfg = json.load(open(DEFAULT_CONFIG, encoding="utf-8"))
    refs = cfg["normalization"]
    splits = normalize(load_data(DEFAULT_DATA_DIR), refs)
    model = load_model(cfg, DEFAULT_CKPT)

    names = ["train", "validation", "test_seen",
             "test_interpolation", "test_voltage_extrapolation"]
    fig, axes = plt.subplots(2, 5, figsize=(20, 7.5), sharex=True)

    for col, name in enumerate(names):
        data = splits[name]
        n_cycles = (data["states_norm"].shape[1] - 1) // 100
        L = n_cycles                     # 满长 rollout（train/val=20，test_*=200）
        target = data["states_norm"][:, :100 * L + 1]
        pred = rollout(model, data["states_norm"], data["vs_norm"], L)
        _, ripple = per_cycle_metrics(pred, target, L)      # (N, L)

        # 该 split 第一条轨迹（物理单位）
        i = 0
        pd = denorm(pred[i], refs)[:, 1]
        td = denorm(target[i], refs)[:, 1]
        st = slice(100 * (L - 2), 100 * L + 1)
        x = np.arange(201) / 100.0                          # 2 周期含端点

        # 行 0：原始 vC（绝对值视角，可见锯齿纹波叠加在 DC 电平上）
        ax = axes[0, col]
        ax.plot(x, td[st].numpy(), "k-", lw=1.1, label="data")
        ax.plot(x, pd[st].numpy(), "r--", lw=1.1, label="model")
        if col == 0:
            ax.set_ylabel("vC (V)")
            ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # 行 1：逐周期线性端点去趋势纹波（mV），并测峰峰
        ax = axes[1, col]
        lin = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
        d_t, d_p = [], []
        for k in range(2):
            seg = slice(st.start + k * 100, st.start + k * 100 + 101)
            d_t.append(td[seg] - (td[seg][0] + lin * (td[seg][-1] - td[seg][0])))
            d_p.append(pd[seg] - (pd[seg][0] + lin * (pd[seg][-1] - pd[seg][0])))
        d_t = torch.cat(d_t)                                # 202 点
        d_p = torch.cat(d_p)
        x2 = np.arange(202) / 100.0
        ax.plot(x2, d_t.numpy() * 1e3, "k-", lw=1.1, label="data")
        ax.plot(x2, d_p.numpy() * 1e3, "r--", lw=1.1, label="model")
        if col == 0:
            ax.set_ylabel("vC ripple (mV)")
            ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel("tau (cycles)")
        ax.grid(True, alpha=0.3)

        # 峰峰测量（2 周期平均，去趋势域）
        pp_d = float(max(d_t[:101].max() - d_t[:101].min(),
                         d_t[101:].max() - d_t[101:].min()))
        pp_p = float(max(d_p[:101].max() - d_p[:101].min(),
                         d_p[101:].max() - d_p[101:].min()))
        err = abs(pp_p - pp_d) / pp_d * 100.0
        axes[0, col].set_title(
            f"{SPLIT_TITLES[name]}\n"
            f"Vpp: data {pp_d * 1e3:.1f} mV | model {pp_p * 1e3:.1f} mV | "
            f"err {err:.1f}%", fontsize=9)

    fig.suptitle("vC (output voltage) ripple: 2 steady-state cycles, "
                 "20-cycle rollout from t=0 (data = black, model = red dashed)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print("saved:", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
