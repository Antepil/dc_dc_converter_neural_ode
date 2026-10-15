#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DCM v3 Hybrid rollout 评估与波形输出。"""

from __future__ import annotations

import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .hybrid_rk4 import hybrid_rk4, rk4_step
from .losses import ripple_loss
from .model import MODE_IDLE, MODE_OFF_DIODE


SPLIT_SHORT = {
    "train": "train",
    "validation": "val",
    "test_seen": "seen",
    "test_interpolation": "interp",
    "test_voltage_extrapolation": "voltage-extra",
    "test_extrapolation_high": "extra-hi",
}


def _off_root_diagnostics(model, rollout, vs, length):
    """在每个预测 OFF 起点上延长 OFF 动力学，统计完整关断段的根。"""
    batch_size = vs.shape[0]
    root_count = torch.zeros(batch_size, length, dtype=torch.long, device=vs.device)
    invalid_guard = torch.zeros(batch_size, length, dtype=torch.bool, device=vs.device)
    first_root_time = torch.zeros(batch_size, length, dtype=vs.dtype, device=vs.device)
    off_mode = torch.full((batch_size,), MODE_OFF_DIODE, dtype=torch.long, device=vs.device)
    for cycle in range(length):
        x = rollout.states[cycle * 100 + 50]
        previous = model.event_value(x, vs)
        invalid_guard[:, cycle] = previous <= 0.0
        for interval in range(50):
            next_x = rk4_step(model, off_mode, x, vs, 0.01)
            current = model.event_value(next_x, vs)
            crossed = (previous > 0.0) & (current <= 0.0)
            first = crossed & (root_count[:, cycle] == 0)
            alpha = previous / (previous - current).clamp_min(1e-12)
            root_time = 0.5 + interval * 0.01 + alpha * 0.01
            first_root_time[:, cycle] = torch.where(first, root_time, first_root_time[:, cycle])
            root_count[:, cycle] += crossed.long()
            previous, x = current, next_x
    return root_count, invalid_guard, first_root_time


def _rollout_split(model, data, length, device, refs, batch_size=32):
    states, vs = data["states_norm"], data["vs_norm"]
    dtype = next(model.parameters()).dtype
    outputs = []
    for start in range(0, states.shape[0], batch_size):
        stop = min(start + batch_size, states.shape[0])
        y0 = states[start:stop, 0].to(device=device, dtype=dtype)
        vbatch = vs[start:stop].to(device=device, dtype=dtype)
        rollout = hybrid_rk4(model, y0, vbatch, length)
        root_count, invalid_guard, first_root_time = _off_root_diagnostics(
            model, rollout, vbatch, length
        )
        target = states[start:stop, : length * 100 + 1].transpose(0, 1).to(device=device, dtype=dtype)
        event_target_time = data["event_time_tau"][start:stop, :length].to(device=device, dtype=dtype)
        event_target_current = data["event_current_norm"][start:stop, :length].to(device=device, dtype=dtype)
        rmse = torch.sqrt((rollout.states - target).square().mean(dim=(0, 2)))
        ripple = torch.stack([
            ripple_loss(rollout.states[:, i:i + 1, 1], target[:, i:i + 1, 1]) for i in range(stop - start)
        ])
        event_time = (rollout.event_time_tau - event_target_time).abs()
        event_current = (rollout.event_current_norm - event_target_current).abs()
        idle_mask = rollout.interval_mode == MODE_IDLE
        idle_values = rollout.states[:-1, :, 0][idle_mask].abs()
        idle_vs = vbatch.expand(rollout.states.shape[0] - 1, -1, -1).reshape(-1, 1)
        idle_tangent = model.idle_vector_field(rollout.states[:-1].reshape(-1, 2), idle_vs).reshape_as(rollout.states[:-1])[..., 0]
        idle_tangent = idle_tangent[idle_mask].abs()
        idle_end_values = rollout.states[100::100, :, 0].abs().reshape(-1)
        outputs.append({
            "start": start,
            "stop": stop,
            "rmse": rmse.cpu(),
            "ripple": ripple.cpu(),
            "missed": rollout.missed_event.cpu(),
            "event_mask": rollout.event_mask.cpu(),
            "root_count": root_count.cpu(),
            "invalid_guard": invalid_guard.cpu(),
            "first_root_time": first_root_time.cpu(),
            "event_time": event_time.cpu(),
            "event_time_pred": rollout.event_time_tau.cpu(),
            "event_time_target": event_target_time.cpu(),
            "event_current_pred": rollout.event_current_norm.abs().cpu(),
            "event_current": event_current.cpu(),
            "idle_current": idle_values.cpu(),
            "idle_tangent": idle_tangent.cpu(),
            "idle_end_current": idle_end_values.cpu(),
        })
    return outputs


def _append_event_rows(rows, split_name, data, outputs, length, refs):
    for output in outputs:
        for local, trajectory in enumerate(range(output["start"], output["stop"])):
            trajectory_id = str(data["trajectory_ids"][trajectory])
            voltage = float(data["input_voltage"][trajectory])
            for cycle in range(length):
                rows.append({
                    "split": split_name,
                    "trajectory_id": trajectory_id,
                    "input_voltage_v": voltage,
                    "cycle": cycle,
                    "predicted_event": int(output["root_count"][local, cycle] > 0),
                    "root_count": int(output["root_count"][local, cycle]),
                    "predicted_event_time_tau": float(output["first_root_time"][local, cycle]),
                    "target_event_time_tau": float(output["event_time_target"][local, cycle]),
                    "absolute_event_time_error_tau": float(output["event_time"][local, cycle]),
                    "predicted_event_current_a": float(output["event_current_pred"][local, cycle] * refs["current_ref_a"]),
                    "invalid_off_start_guard": int(output["invalid_guard"][local, cycle]),
                })


def _summarize(outputs, refs):
    rmse = torch.cat([item["rmse"] for item in outputs])
    ripple = torch.cat([item["ripple"] for item in outputs])
    root_count = torch.cat([item["root_count"].reshape(-1) for item in outputs])
    invalid_guard = torch.cat([item["invalid_guard"].reshape(-1) for item in outputs])
    event_time = torch.cat([item["event_time"].reshape(-1) for item in outputs])
    event_time_pred = torch.cat([item["event_time_pred"].reshape(-1) for item in outputs])
    event_current = torch.cat([item["event_current"].reshape(-1) for item in outputs])
    idle_current = torch.cat([item["idle_current"] for item in outputs])
    idle_tangent = torch.cat([item["idle_tangent"] for item in outputs])
    idle_end_current = torch.cat([item["idle_end_current"] for item in outputs])
    valid_event = (root_count == 1) & ~invalid_guard
    return {
        "rmse_mean_pct": float(rmse.mean() * 100.0),
        "rmse_p95_pct": float(torch.quantile(rmse, 0.95) * 100.0),
        "rmse_max_pct": float(rmse.max() * 100.0),
        "ripple_mean_relative_error": float(torch.sqrt(ripple).mean()),
        "ripple_p95_relative_error": float(torch.sqrt(ripple).quantile(0.95)),
        "event_miss_rate": float((root_count == 0).float().mean()),
        "event_existence_rate": float((root_count > 0).float().mean()),
        "exactly_one_event_rate": float((root_count == 1).float().mean()),
        "event_at_boundary_rate": float(((event_time_pred <= 0.5 + 1e-7) | (event_time_pred >= 1.0 - 1e-7))[valid_event].float().mean()) if valid_event.any() else 0.0,
        "extra_roots_per_off": float(torch.clamp(root_count - 1, min=0).float().mean()),
        "nonpositive_off_start_guard_rate": float(invalid_guard.float().mean()),
        "event_time_mae_tau": float(event_time[valid_event].mean()) if valid_event.any() else float("nan"),
        "event_time_p95_tau": float(event_time[valid_event].quantile(0.95)) if valid_event.any() else float("nan"),
        "idle_duration_mae_tau": float(event_time[valid_event].mean()) if valid_event.any() else float("nan"),
        "event_current_mean_a": float(event_current[valid_event].mean() * refs["current_ref_a"]) if valid_event.any() else float("nan"),
        "event_current_p95_a": float(event_current[valid_event].quantile(0.95) * refs["current_ref_a"]) if valid_event.any() else float("nan"),
        "idle_current_mean_a": float(idle_current.mean() * refs["current_ref_a"]) if idle_current.numel() else float("nan"),
        "idle_current_p95_a": float(idle_current.quantile(0.95) * refs["current_ref_a"]) if idle_current.numel() else float("nan"),
        "idle_current_derivative_rms": float(torch.sqrt(idle_tangent.square().mean())) if idle_tangent.numel() else float("nan"),
        "idle_current_derivative_p95": float(idle_tangent.quantile(0.95)) if idle_tangent.numel() else float("nan"),
        "idle_end_current_mean_a": float(idle_end_current.mean() * refs["current_ref_a"]),
        "idle_end_current_p95_a": float(idle_end_current.quantile(0.95) * refs["current_ref_a"]),
    }


def evaluate_all(model, splits, rollout_cycles, device, refs=None, event_rows=None):
    refs = refs or {"current_ref_a": 1.0, "input_voltage_ref_v": 32.0}
    model_d = model.to(device=device, dtype=torch.float32)
    model_d.eval()
    report = {}
    with torch.no_grad():
        for name, data in splits.items():
            report[name] = {}
            available = (data["states"].shape[1] - 1) // 100
            lengths = [length for length in rollout_cycles if length <= available]
            for length in lengths:
                outputs = _rollout_split(model_d, data, length, device, refs)
                report[name][str(length)] = _summarize(outputs, refs)
                if event_rows is not None and length == max(lengths):
                    _append_event_rows(event_rows, name, data, outputs, length, refs)
    return report


def save_event_rows(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_event_plot(path, rows):
    """按输入电压展示事件时间误差和事件电流。"""
    if not rows:
        return
    voltage = torch.tensor([row["input_voltage_v"] for row in rows])
    time_error = torch.tensor([row["absolute_event_time_error_tau"] for row in rows])
    current = torch.tensor([row["predicted_event_current_a"] for row in rows])
    exists = torch.tensor([row["predicted_event"] for row in rows], dtype=torch.bool)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(voltage[exists], time_error[exists], s=6, alpha=0.45)
    axes[0].set_ylabel("|event time error| (tau)")
    axes[1].scatter(voltage[exists], current[exists], s=6, alpha=0.45)
    axes[1].set_ylabel("|event current| (A)")
    for axis in axes:
        axis.set_xlabel("input voltage (V)")
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_by_voltage(model, data, rollout_cycles, device, refs=None):
    refs = refs or {"input_voltage_ref_v": 32.0, "current_ref_a": 1.0}
    result = {}
    with torch.no_grad():
        for voltage in torch.unique(data["input_voltage"]):
            indices = (data["input_voltage"] == voltage).nonzero(as_tuple=True)[0]
            subset = {key: value[indices] if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == data["states"].shape[0] else value for key, value in data.items()}
            rows = {}
            for length in rollout_cycles:
                if length <= (subset["states"].shape[1] - 1) // 100:
                    rows[str(length)] = _summarize(_rollout_split(model, subset, length, device, refs), refs)
            result[float(voltage) * refs["input_voltage_ref_v"]] = rows
    return result


def evaluate_half_step(model, data, rollout_cycles, device, refs=None):
    """比较 h=.01 与 h=.005 的下采样轨迹和事件一致性。"""
    refs = refs or {"current_ref_a": 1.0}
    model_d = model.to(device=device, dtype=torch.float32).eval()
    report = {}
    with torch.no_grad():
        for length in rollout_cycles:
            if length > (data["states"].shape[1] - 1) // 100:
                continue
            y0 = data["states_norm"][:, 0].to(device=device, dtype=torch.float32)
            vs = data["vs_norm"].to(device=device, dtype=torch.float32)
            coarse = hybrid_rk4(model_d, y0, vs, length, step_size_tau=0.01, samples_per_cycle=100)
            fine = hybrid_rk4(model_d, y0, vs, length, step_size_tau=0.005, samples_per_cycle=200)
            fine_down = fine.states[::2]
            report[str(length)] = {
                "rmse_percentage_point_delta": float(torch.sqrt((coarse.states - fine_down).square().mean()) * 100.0),
                "event_time_mean_delta_tau": float((coarse.event_time_tau - fine.event_time_tau).abs().mean()),
                "event_existence_change_rate": float((coarse.event_mask != fine.event_mask).float().mean()),
            }
    return report


def evaluate_by_initial_bin(model, data, rollout_cycles, device, refs=None, n_bins=3):
    """按初始电感电流分位区间报告 rollout 指标。"""
    refs = refs or {"current_ref_a": 1.0, "input_voltage_ref_v": 32.0}
    initial_current = data["initial_state"][:, 0]
    edges = torch.quantile(initial_current, torch.linspace(0.0, 1.0, n_bins + 1, dtype=initial_current.dtype))
    result = {}
    for index in range(n_bins):
        if index == n_bins - 1:
            selected = (initial_current >= edges[index]) & (initial_current <= edges[index + 1])
        else:
            selected = (initial_current >= edges[index]) & (initial_current < edges[index + 1])
        trajectory = selected.nonzero(as_tuple=True)[0]
        if trajectory.numel() == 0:
            continue
        subset = {key: value[trajectory] if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == data["states"].shape[0] else value for key, value in data.items()}
        result[f"bin_{index + 1}"] = evaluate_all(model, {"subset": subset}, rollout_cycles, device, refs)["subset"]
    return result


def assess_acceptance(report, half_step, cfg):
    """把 project.json 的硬门槛映射到已完成的正式评估。"""
    limits = cfg["evaluation"]["acceptance"]
    checks = {}

    def add(name, value, limit, relation="max"):
        if value is None:
            checks[name] = {"status": "not_evaluated", "limit": limit}
            return
        passed = math.isfinite(value) and (value <= limit if relation == "max" else value >= limit)
        checks[name] = {"status": "passed" if passed else "failed", "value": value, "limit": limit}

    waveform = (
        ("validation", "20", "validation_normalized_rmse_max"),
        ("test_seen", "200", "test_seen_rmse_max"),
        ("test_interpolation", "200", "test_interpolation_rmse_max"),
        ("test_voltage_extrapolation", "200", "test_voltage_extrapolation_rmse_max"),
    )
    for split, length, key in waveform:
        row = report.get(split, {}).get(length)
        add(key, row["rmse_mean_pct"] / 100.0 if row else None, limits[key])

    core_rows = []
    for split in ("validation", "test_seen", "test_interpolation", "test_voltage_extrapolation"):
        rows = report.get(split, {})
        if rows:
            core_rows.append(rows[max(rows, key=int)])
    max_metrics = {
        "event_miss_rate_max": "event_miss_rate",
        "event_at_boundary_rate_max": "event_at_boundary_rate",
        "event_time_mae_tau_max": "event_time_mae_tau",
        "event_time_p95_tau_max": "event_time_p95_tau",
        "event_current_mean_a_max": "event_current_mean_a",
        "event_current_p95_a_max": "event_current_p95_a",
        "idle_current_mean_a_max": "idle_current_mean_a",
        "idle_current_p95_a_max": "idle_current_p95_a",
        "idle_end_current_mean_a_max": "idle_end_current_mean_a",
        "idle_end_current_p95_a_max": "idle_end_current_p95_a",
        "extra_roots_per_off_max": "extra_roots_per_off",
        "nonpositive_off_start_guard_rate_max": "nonpositive_off_start_guard_rate",
    }
    for key, metric in max_metrics.items():
        values = [row[metric] for row in core_rows]
        add(key, max(values) if values else None, limits[key])
    exactly = [row["exactly_one_event_rate"] for row in core_rows]
    add("exactly_one_event_rate_min", min(exactly) if exactly else None, limits["exactly_one_event_rate_min"], "min")

    for key, metric in (
        ("ripple_mean_relative_error_max", "ripple_mean_relative_error"),
        ("ripple_p95_relative_error_max", "ripple_p95_relative_error"),
    ):
        values = [report[split]["200"][metric] for split in ("test_seen", "test_interpolation") if "200" in report.get(split, {})]
        add(key, max(values) if values else None, limits[key])

    half_rows = list(half_step.values())
    for key, metric in (
        ("half_step_event_time_mean_delta_tau_max", "event_time_mean_delta_tau"),
        ("half_step_rmse_percentage_point_delta_max", "rmse_percentage_point_delta"),
        ("half_step_event_existence_change_rate_max", "event_existence_change_rate"),
    ):
        add(key, max((row[metric] for row in half_rows), default=None), limits[key])
    add("nonfinite_state_count_max", 0.0, limits["nonfinite_state_count_max"])
    statuses = [item["status"] for item in checks.values()]
    return {
        "complete": "not_evaluated" not in statuses,
        "passed": bool(statuses) and all(status == "passed" for status in statuses),
        "checks": checks,
    }


def print_report(result):
    print("\n===== DCM Hybrid 评估（归一化 RMSE % / 事件安全）=====")
    for name, rows in result.items():
        for length, metrics in rows.items():
            print(f"{SPLIT_SHORT.get(name, name):>14} L={length:>3} "
                  f"RMSE={metrics['rmse_mean_pct']:.4f}% "
                  f"miss={metrics['event_miss_rate']:.3f} "
                  f"event_mae={metrics['event_time_mae_tau']:.4f}")


def print_by_voltage(result):
    print("\n===== 按输入电压评估 =====")
    for voltage, rows in sorted(result.items()):
        for length, metrics in rows.items():
            print(f"vS={voltage:.1f} V L={length}: RMSE={metrics['rmse_mean_pct']:.4f}%")


def _plot_one(model, data, out_path, title, length=20):
    length = min(length, (data["states"].shape[1] - 1) // 100)
    parameter = next(model.parameters())
    model_d = model.eval()
    with torch.no_grad():
        rollout = hybrid_rk4(
            model_d,
            data["states_norm"][:1, 0].to(device=parameter.device, dtype=parameter.dtype),
            data["vs_norm"][:1].to(device=parameter.device, dtype=parameter.dtype),
            length,
        )
    target = data["states_norm"][0, :length * 100 + 1]
    prediction = rollout.states[:, 0].cpu()
    t = torch.arange(length * 100 + 1, dtype=torch.float64) / 100.0
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for channel, label in enumerate(("iL (norm)", "vC (norm)")):
        axes[channel].plot(t, target[:, channel], "k-", label="data")
        axes[channel].plot(t, prediction[:, channel], "r--", label="model")
        axes[channel].set_ylabel(label)
        axes[channel].grid(True, alpha=0.3)
        axes[channel].legend(loc="upper right")
    axes[-1].set_xlabel("tau (cycles)")
    axes[0].set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_plot(model, splits, out_path, plot_cycles=20):
    name = next(iter(splits))
    _plot_one(model, splits[name], out_path, f"DCM Hybrid waveform: {name}", plot_cycles)


def save_voltage_plots(model, data, out_dir, prefix="extra_hi"):
    os.makedirs(out_dir, exist_ok=True)
    for voltage in torch.unique(data["input_voltage"]):
        index = (data["input_voltage"] == voltage).nonzero(as_tuple=True)[0][0]
        path = os.path.join(out_dir, f"{prefix}_{float(voltage):.0f}v.png")
        subset = {key: value[index:index + 1] if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == data["states"].shape[0] else value for key, value in data.items()}
        _plot_one(model, subset, path, f"DCM Hybrid vS={float(voltage):.0f} V")
