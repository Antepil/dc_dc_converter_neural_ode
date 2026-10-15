#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DCM v3 数据契约：加载、归一化和训练样本组织。

MATLAB/Simulink 导出的 NPZ 已经完成拓扑和 DCM 合法性检查。本模块只做
训练入口必须的 schema 审计，并把固定网格上的轨迹整理成两类监督：
同模式有限差分点对和周期对齐 rollout 窗口。
"""

from __future__ import annotations

import json
import os
from typing import Iterable

import numpy as np
import torch

from .model import MODE_IDLE, MODE_OFF_DIODE, MODE_ON


SAMPLES_PER_CYCLE = 100
FIXED_LOAD_OHM = 80.0
SPLIT_FILES = {
    "train": "train.npz",
    "validation": "validation.npz",
    "test_seen": "test_seen.npz",
    "test_interpolation": "test_interpolation.npz",
    "test_voltage_extrapolation": "test_voltage_extrapolation.npz",
    "test_extrapolation_high": "test_extrapolation_high.npz",
}
REQUIRED_FIELDS = (
    "states", "input_voltage", "load_resistance", "time_s", "pwm_gate",
    "diode_on", "interval_mode", "natural_event_mask", "event_time_tau",
    "idle_duration_tau", "event_state", "event_current_a", "diode_current_a",
    "initial_state", "trajectory_ids", "metadata", "source_sha256",
)


def _as_tensor(array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array)).to(dtype=dtype)


def _parse_metadata(values: np.ndarray) -> list[dict]:
    return [json.loads(str(value)) for value in values.tolist()]


def _audit_split(name: str, data: dict) -> None:
    """审计 v3 固定形状和模式编码；失败应在模型构造前暴露。"""
    schemas = {item.get("schema_version") for item in data["metadata"]}
    if schemas != {"hybrid_event_matlab_v3"}:
        raise ValueError(f"{name}: metadata schema_version 必须是 hybrid_event_matlab_v3")
    states = data["states"]
    n_traj, n_points, state_dim = states.shape
    for key in ("states", "input_voltage", "load_resistance", "time_s", "event_time_tau", "idle_duration_tau", "event_state", "event_current_a", "diode_current_a", "initial_state"):
        if not np.isfinite(data[key]).all():
            raise ValueError(f"{name}: {key} 含非有限值")
    n_intervals = n_points - 1
    if state_dim != 2 or n_intervals != data["interval_mode"].shape[1]:
        raise ValueError(f"{name}: states/interval_mode shape 不符合 v3")
    if n_intervals % SAMPLES_PER_CYCLE != 0:
        raise ValueError(f"{name}: 时间网格不是 {SAMPLES_PER_CYCLE} 点/周期")
    n_cycles = n_intervals // SAMPLES_PER_CYCLE
    if data["input_voltage"].shape != (n_traj,):
        raise ValueError(f"{name}: input_voltage 必须是 (N,)")
    if data["time_s"].shape != (n_points,) or data["diode_current_a"].shape != (n_traj, n_points):
        raise ValueError(f"{name}: time_s/diode_current_a shape 错误")
    if data["initial_state"].shape != (n_traj, 2):
        raise ValueError(f"{name}: initial_state shape 错误")
    if data["load_resistance"].shape != (n_traj,) or not np.allclose(
        data["load_resistance"], FIXED_LOAD_OHM
    ):
        raise ValueError(f"{name}: DCM 训练只支持 R={FIXED_LOAD_OHM:g} ohm")
    if data["pwm_gate"].shape != (n_traj, n_intervals):
        raise ValueError(f"{name}: pwm_gate shape 错误")
    if data["diode_on"].shape != (n_traj, n_intervals):
        raise ValueError(f"{name}: diode_on shape 错误")
    if data["natural_event_mask"].shape != (n_traj, n_cycles):
        raise ValueError(f"{name}: natural_event_mask shape 错误")
    for key in ("event_time_tau", "idle_duration_tau"):
        if data[key].shape != (n_traj, n_cycles):
            raise ValueError(f"{name}: {key} shape 错误")
    if data["event_state"].shape != (n_traj, n_cycles, 2):
        raise ValueError(f"{name}: event_state shape 错误")
    if data["event_current_a"].shape != (n_traj, n_cycles):
        raise ValueError(f"{name}: event_current_a shape 错误")
    if data["interval_mode"].min() < MODE_ON or data["interval_mode"].max() > MODE_IDLE:
        raise ValueError(f"{name}: interval_mode 含未知模式编码")
    if not np.all(data["natural_event_mask"]):
        raise ValueError(f"{name}: DCM v3 训练要求每周期恰好一个自然事件")
    for trajectory in range(n_traj):
        for cycle in range(n_cycles):
            segment = data["interval_mode"][trajectory, cycle * SAMPLES_PER_CYCLE:(cycle + 1) * SAMPLES_PER_CYCLE]
            if not np.all(segment[:50] == MODE_ON) or segment[50] != MODE_OFF_DIODE or not np.all(np.isin(segment[51:], (MODE_OFF_DIODE, MODE_IDLE))) or not np.any(segment == MODE_IDLE):
                raise ValueError(f"{name}: trajectory {trajectory} cycle {cycle} 模式顺序不是 ON→OFF_DIODE→IDLE")


def load_data(data_dir: str) -> dict[str, dict]:
    """加载目录中存在的 v3 split；缺少 train/validation 由入口处理。"""
    splits: dict[str, dict] = {}
    for name, filename in SPLIT_FILES.items():
        path = os.path.join(data_dir, filename)
        if not os.path.isfile(path):
            continue
        with np.load(path, allow_pickle=True) as source:
            missing = [field for field in REQUIRED_FIELDS if field not in source.files]
            if missing:
                raise ValueError(f"{name}: 缺少 v3 字段 {missing}")
            raw = {
                "states": np.asarray(source["states"], dtype=np.float64),
                "input_voltage": np.asarray(source["input_voltage"], dtype=np.float64),
                "load_resistance": np.asarray(source["load_resistance"], dtype=np.float64),
                "time_s": np.asarray(source["time_s"], dtype=np.float64),
                "pwm_gate": np.asarray(source["pwm_gate"], dtype=np.int64),
                "diode_on": np.asarray(source["diode_on"], dtype=np.int64),
                "interval_mode": np.asarray(source["interval_mode"], dtype=np.int64),
                "natural_event_mask": np.asarray(source["natural_event_mask"], dtype=bool),
                "event_time_tau": np.asarray(source["event_time_tau"], dtype=np.float64),
                "idle_duration_tau": np.asarray(source["idle_duration_tau"], dtype=np.float64),
                "event_state": np.asarray(source["event_state"], dtype=np.float64),
                "event_current_a": np.asarray(source["event_current_a"], dtype=np.float64),
                "diode_current_a": np.asarray(source["diode_current_a"], dtype=np.float64),
                "initial_state": np.asarray(source["initial_state"], dtype=np.float64),
                "trajectory_ids": np.asarray(source["trajectory_ids"]),
                "metadata": _parse_metadata(source["metadata"]),
                "source_sha256": np.asarray(source["source_sha256"]),
            }
        _audit_split(name, raw)
        splits[name] = {
            "states": _as_tensor(raw["states"], torch.float64),
            "input_voltage": _as_tensor(raw["input_voltage"], torch.float64),
            "load_resistance": _as_tensor(raw["load_resistance"], torch.float64),
            "time_s": _as_tensor(raw["time_s"], torch.float64),
            "pwm_gate": _as_tensor(raw["pwm_gate"], torch.long),
            "diode_on": _as_tensor(raw["diode_on"], torch.long),
            "interval_mode": _as_tensor(raw["interval_mode"], torch.long),
            "natural_event_mask": torch.from_numpy(raw["natural_event_mask"]),
            "event_time_tau": _as_tensor(raw["event_time_tau"], torch.float64),
            "idle_duration_tau": _as_tensor(raw["idle_duration_tau"], torch.float64),
            "event_state": _as_tensor(raw["event_state"], torch.float64),
            "event_current_a": _as_tensor(raw["event_current_a"], torch.float64),
            "diode_current_a": _as_tensor(raw["diode_current_a"], torch.float64),
            "initial_state": _as_tensor(raw["initial_state"], torch.float64),
            "trajectory_ids": raw["trajectory_ids"],
            "metadata": raw["metadata"],
            "source_sha256": raw["source_sha256"],
        }
    return splits


def normalize(splits: dict[str, dict], refs: dict) -> dict[str, dict]:
    """构造积分、损失和评估的固定参考坐标。

    网络内部的 Vin 相对坐标由 model.py 转换；此处保持统一评分尺度。
    """
    state_ref = torch.tensor(
        [refs["current_ref_a"], refs["voltage_ref_v"]], dtype=torch.float64
    )
    for data in splits.values():
        data["states_norm"] = data["states"] / state_ref
        data["vs_norm"] = data["input_voltage"][:, None] / refs["input_voltage_ref_v"]
        data["event_state_norm"] = data["event_state"] / state_ref
        data["event_current_norm"] = data["event_current_a"] / refs["current_ref_a"]
        data["diode_current_norm"] = data["diode_current_a"] / refs["current_ref_a"]
        data["initial_state_norm"] = data["initial_state"] / state_ref
    return splits


def make_windows(
    data: dict, window_cycles: int, samples_per_cycle: int = SAMPLES_PER_CYCLE
) -> list[tuple[int, int]]:
    """返回 ``(trajectory_index, start_cycle)``，窗口严格对齐 PWM 周期。"""
    _, n_points, _ = data["states_norm"].shape
    n_cycles = (n_points - 1) // samples_per_cycle
    return [
        (trajectory, cycle)
        for trajectory in range(data["states_norm"].shape[0])
        for cycle in range(n_cycles - window_cycles + 1)
    ] if window_cycles <= n_cycles else []


def collate(
    data: dict,
    windows: Iterable[tuple[int, int]],
    window_cycles: int,
    device: torch.device | str,
    dtype: torch.dtype,
    samples_per_cycle: int = SAMPLES_PER_CYCLE,
    derivative_supervision: bool = False,
) -> dict[str, torch.Tensor]:
    """把周期窗口整理成 rollout 所需的时间优先批次。"""
    windows = list(windows)
    trajectory = torch.tensor([item[0] for item in windows], dtype=torch.long)
    cycle = torch.tensor([item[1] for item in windows], dtype=torch.long)
    starts = cycle * samples_per_cycle
    offsets = torch.arange(window_cycles * samples_per_cycle + 1, dtype=torch.long)
    state_index = starts[:, None] + offsets[None, :]
    target = data["states_norm"][trajectory[:, None], state_index]
    target = target.transpose(0, 1).to(device=device, dtype=dtype)
    interval_offsets = torch.arange(window_cycles * samples_per_cycle, dtype=torch.long)
    event_offsets = torch.arange(window_cycles, dtype=torch.long)
    extra = {}
    if derivative_supervision:
        dx, valid = central_derivative_labels(data)
        extra = {
            "derivative_target": dx[trajectory[:, None], state_index].transpose(0, 1).to(device=device, dtype=dtype),
            "derivative_valid": valid[trajectory[:, None], state_index].transpose(0, 1).to(device=device),
        }
    return {
        **extra,
        "y0": target[0],
        "target": target,
        "vs": data["vs_norm"][trajectory].to(device=device, dtype=dtype),
        "interval_mode": data["interval_mode"][trajectory[:, None], starts[:, None] + interval_offsets[None, :]]
        .transpose(0, 1).to(device=device),
        "event_time_tau": data["event_time_tau"][trajectory[:, None], cycle[:, None] + event_offsets[None, :]]
        .to(device=device, dtype=dtype),
        "event_state": data["event_state_norm"][trajectory[:, None], cycle[:, None] + event_offsets[None, :]]
        .to(device=device, dtype=dtype),
        "event_current": data["event_current_norm"][trajectory[:, None], cycle[:, None] + event_offsets[None, :]]
        .to(device=device, dtype=dtype),
        "event_mask": data["natural_event_mask"][trajectory[:, None], cycle[:, None] + event_offsets[None, :]]
        .to(device=device),
    }


def central_derivative_labels(data: dict):
    """Float64 centered d(x/ref)/d(t/Ts), masked inside each mode; no circuit equations."""
    if "_central_derivative_labels" in data:
        return data["_central_derivative_labels"]
    x, modes, t = data["states_norm"], data["interval_mode"], data["time_s"]
    if x.dtype != torch.float64:
        raise ValueError("Derivative labels require float64 normalized source states")
    dx = torch.zeros_like(x)
    valid = torch.zeros(x.shape[:2], dtype=torch.bool)
    k = torch.arange(1, x.shape[1] - 1)
    tau = t / (t[100] - t[0])
    dx[:, 1:-1] = (x[:, 2:] - x[:, :-2]) / (tau[2:] - tau[:-2])[None, :, None]
    phase = k % 100
    event = (k // 100)[None, :] * 100 + data["event_time_tau"][:, k // 100] * 100
    valid[:, 1:-1] = ((modes[:, k-1] == modes[:, k]) & (phase > 2) & (phase < 98)
                      & ((phase - 50).abs() > 2) & ((k[None, :] - event).abs() > 2))
    data["_central_derivative_labels"] = (dx, valid)
    return dx, valid


def build_mode_pairs(
    data: dict,
    *,
    step_size_tau: float = 0.01,
    samples_per_cycle: int = SAMPLES_PER_CYCLE,
    boundary_exclusion_intervals: int = 2,
    event_exclusion_intervals: int = 2,
    method: str = "forward",
) -> dict[int, dict[str, torch.Tensor]]:
    """构造三个模式内部的有限差分点对并排除切换邻域。"""
    states, modes, vs = data["states_norm"], data["interval_mode"], data["vs_norm"]
    if method == "central":
        dx, valid = central_derivative_labels(data)
        out = {}
        for mode in (MODE_ON, MODE_OFF_DIODE, MODE_IDLE):
            trajectory, index = (valid[:, :-1] & (modes == mode)).nonzero(as_tuple=True)
            out[mode] = {"x": states[trajectory, index], "dx": dx[trajectory, index], "vs": vs[trajectory]}
        return out
    if method != "forward":
        raise ValueError(f"Unknown derivative method: {method}")
    n_traj, n_points, _ = states.shape
    keep: dict[int, list[tuple[int, int]]] = {MODE_ON: [], MODE_OFF_DIODE: [], MODE_IDLE: []}
    for trajectory in range(n_traj):
        row = modes[trajectory]
        for k in range(n_points - 2):
            mode = int(row[k])
            if mode not in keep or int(row[k + 1]) != mode:
                continue
            phase = k % samples_per_cycle
            if phase < boundary_exclusion_intervals or phase >= samples_per_cycle - boundary_exclusion_intervals:
                continue
            if abs(phase - round(samples_per_cycle * 0.5)) <= boundary_exclusion_intervals:
                continue
            cycle = k // samples_per_cycle
            event_idx = cycle * samples_per_cycle + round(float(data["event_time_tau"][trajectory, cycle]) * samples_per_cycle)
            if abs(k - event_idx) <= event_exclusion_intervals:
                continue
            keep[mode].append((trajectory, k))
    out: dict[int, dict[str, torch.Tensor]] = {}
    for mode, pairs in keep.items():
        if not pairs:
            out[mode] = {"x": states.new_empty((0, 2)), "dx": states.new_empty((0, 2)), "vs": vs.new_empty((0, 1))}
            continue
        trajectory = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        index = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        x0 = states[trajectory, index]
        x1 = states[trajectory, index + 1]
        out[mode] = {"x": x0, "dx": (x1 - x0) / step_size_tau, "vs": vs[trajectory]}
    return out


def build_event_samples(
    data: dict,
    *,
    samples_per_cycle: int = SAMPLES_PER_CYCLE,
    event_exclusion_intervals: int = 2,
) -> dict[str, torch.Tensor]:
    """提取 event pretrain 的表面点和事件前 OFF 点。"""
    states, modes = data["states_norm"], data["interval_mode"]
    surface_x, surface_vs, pre_x, pre_vs = [], [], [], []
    n_traj, n_cycles = data["event_time_tau"].shape
    for trajectory in range(n_traj):
        for cycle in range(n_cycles):
            event_idx = cycle * samples_per_cycle + round(float(data["event_time_tau"][trajectory, cycle]) * samples_per_cycle)
            surface_x.append(data["event_state_norm"][trajectory, cycle])
            surface_vs.append(data["vs_norm"][trajectory])
            off_start = cycle * samples_per_cycle + round(samples_per_cycle * 0.5)
            for k in range(off_start, min(event_idx - event_exclusion_intervals, off_start + samples_per_cycle)):
                if int(modes[trajectory, k]) == MODE_OFF_DIODE:
                    pre_x.append(states[trajectory, k])
                    pre_vs.append(data["vs_norm"][trajectory])
    return {
        "surface_x": torch.stack(surface_x),
        "surface_vs": torch.stack(surface_vs),
        "pre_x": torch.stack(pre_x),
        "pre_vs": torch.stack(pre_vs),
    }
