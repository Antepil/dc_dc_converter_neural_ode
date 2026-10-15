#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 Buck hybrid_event_matlab_v3 MAT 轨迹转换为训练用 NPZ。"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import scipy.io
from scipy.io.matlab import MatlabOpaque


SCHEMA_VERSION = "hybrid_event_matlab_v3"
MAT_FIELDS = {
    "time_s",
    "inductor_current_a",
    "capacitor_voltage_v",
    "diode_current_a",
    "input_voltage_v",
    "load_resistance_ohm",
    "initial_state",
    "pwm_gate",
    "diode_on",
    "interval_mode",
    "natural_event_mask",
    "event_time_s",
    "event_time_tau",
    "idle_duration_tau",
    "event_state",
    "event_current_a",
    "event_bracket_time_s",
    "event_bracket_diode_current_a",
    "metadata",
}


def _scalar(value, name):
    value = np.asarray(value)
    if value.size != 1:
        raise ValueError(f"{name} 应为标量，实际形状 {value.shape}")
    return value.reshape(-1)[0].item()


def _array(data, name, shape, dtype):
    value = np.asarray(data[name], dtype=dtype)
    if value.size != int(np.prod(shape)):
        raise ValueError(
            f"{name} 元素数 {value.size} 与期望形状 {shape} 不一致"
        )
    value = value.reshape(shape)
    if np.issubdtype(value.dtype, np.floating) and not np.all(np.isfinite(value)):
        raise ValueError(f"{name} 含 NaN 或 Inf")
    return value


def _opaque_string(value, expected, name):
    """SciPy 无法解码 MAT v7 中的 MATLAB string；验证类型后取配置值。"""
    if value.size != 1:
        raise ValueError(f"metadata.{name} 的 MATLAB string 不是标量")
    if value["s1"].item() != b"MCOS" or value["s2"].item() != b"string":
        raise ValueError(f"metadata.{name} 不是 MATLAB string")
    if expected is None:
        raise ValueError(f"metadata.{name} 无法从配置解析")
    return expected


def _metadata_value(value, expected, name):
    if isinstance(value, MatlabOpaque):
        return _opaque_string(value, expected, name)
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"metadata.{name} 不是标量")
        return _metadata_value(value.reshape(-1)[0], expected, name)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, (bool, int, float, str)):
        return value
    raise ValueError(f"metadata.{name} 类型不支持: {type(value).__name__}")


def _expected_metadata_strings(config, split_name, trajectory_id):
    matlab = config["matlab"]
    return {
        "trajectory_id": trajectory_id,
        "split": split_name,
        "schema_version": SCHEMA_VERSION,
        "solver": matlab["solver"],
        "solver_absolute_tolerance": matlab["absolute_tolerance"],
        "zero_crossing_algorithm": matlab["zero_crossing_algorithm"],
        "device_library": matlab["device_library"],
        "diode_model_type": matlab["diode"]["model_type"],
        "diode_reverse_recovery": matlab["diode"]["reverse_recovery"],
        "mosfet_model_type": matlab["mosfet"]["model_type"],
        "mosfet_body_diode_mode": matlab["mosfet"]["body_diode_mode"],
    }


def _metadata_json(raw_metadata, expected_strings):
    if not isinstance(raw_metadata, dict):
        raise ValueError("metadata 必须是标量 struct")
    metadata = {
        name: _metadata_value(value, expected_strings.get(name), name)
        for name, value in raw_metadata.items()
    }
    for name, expected in expected_strings.items():
        if name not in metadata:
            raise ValueError(f"metadata 缺少 {name}")
        if metadata[name] != expected:
            raise ValueError(
                f"metadata.{name}={metadata[name]!r}，期望 {expected!r}"
            )
    return json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def load_trajectory(mat_path, config, split_name, split_cfg,
                    trajectory_id, expected_voltage):
    """读取一条 MAT，并核对 v3 的形状、事件和模式。"""
    with open(mat_path, "rb") as stream:
        blob = stream.read()
    data = scipy.io.loadmat(mat_path, simplify_cells=True)
    missing = sorted(MAT_FIELDS.difference(data))
    if missing:
        raise ValueError(f"{trajectory_id} 缺少 MAT v3 字段: {missing}")

    dataset = config["dataset"]
    circuit = config["circuit"]
    event_cfg = dataset["event_detection"]
    cycles = split_cfg["cycles"]
    samples = dataset["samples_per_cycle"]
    intervals = cycles * samples
    points = intervals + 1
    ts = 1.0 / circuit["switching_frequency_hz"]
    dt = ts / samples

    input_voltage = float(_scalar(data["input_voltage_v"], "input_voltage_v"))
    load_resistance = float(_scalar(
        data["load_resistance_ohm"], "load_resistance_ohm"
    ))
    if input_voltage != float(expected_voltage):
        raise ValueError(
            f"{trajectory_id} 输入电压 {input_voltage} != {expected_voltage}"
        )
    if load_resistance != float(circuit["load_ohm"]):
        raise ValueError(
            f"{trajectory_id} 负载 {load_resistance} != {circuit['load_ohm']}"
        )

    time_s = _array(data, "time_s", (points,), np.float64)
    expected_time = np.arange(points, dtype=np.float64) * dt
    tolerance = 100 * np.finfo(np.float64).eps * max(1.0, expected_time[-1])
    if not np.allclose(time_s, expected_time, rtol=0, atol=tolerance):
        raise ValueError(f"{trajectory_id} time_s 不符合固定网格")

    states = np.column_stack((
        _array(data, "inductor_current_a", (points,), np.float64),
        _array(data, "capacitor_voltage_v", (points,), np.float64),
    ))
    diode_current = _array(data, "diode_current_a", (points,), np.float64)
    initial_state = _array(data, "initial_state", (2,), np.float64)
    if not np.array_equal(initial_state, states[0]):
        raise ValueError(f"{trajectory_id} initial_state 与首个状态点不一致")

    pwm_gate = _array(data, "pwm_gate", (intervals,), np.uint8)
    diode_on = _array(data, "diode_on", (intervals,), np.uint8)
    interval_mode = _array(data, "interval_mode", (intervals,), np.uint8)
    expected_pwm = (
        np.tile(np.arange(samples), cycles)
        < round(circuit["duty_cycle"] * samples)
    ).astype(np.uint8)
    if not np.array_equal(pwm_gate, expected_pwm):
        raise ValueError(f"{trajectory_id} pwm_gate 不符合固定 PWM 网格")

    natural_event_mask = _array(
        data, "natural_event_mask", (cycles,), np.bool_
    )
    event_time_tau = _array(data, "event_time_tau", (cycles,), np.float64)
    idle_duration_tau = _array(
        data, "idle_duration_tau", (cycles,), np.float64
    )
    if not np.all(natural_event_mask):
        raise ValueError(f"{trajectory_id} natural_event_mask 并非全真")
    lower = circuit["duty_cycle"]
    upper = 1.0 - event_cfg["dcm_margin_tau"]
    if np.any(event_time_tau <= lower) or np.any(event_time_tau > upper + 1e-12):
        raise ValueError(f"{trajectory_id} event_time_tau 超出 DCM 范围")
    if not np.allclose(
        idle_duration_tau, 1.0 - event_time_tau, rtol=0, atol=1e-12
    ):
        raise ValueError(f"{trajectory_id} idle_duration_tau 与事件时间不一致")

    event_time_s = _array(data, "event_time_s", (cycles,), np.float64)
    if not np.allclose(
        event_time_s,
        (np.arange(cycles) + event_time_tau) * ts,
        rtol=0,
        atol=tolerance,
    ):
        raise ValueError(f"{trajectory_id} event_time_s 与 event_time_tau 不一致")
    event_state = _array(data, "event_state", (cycles, 2), np.float64)
    event_current = _array(data, "event_current_a", (cycles,), np.float64)
    if not np.array_equal(event_current, event_state[:, 0]):
        raise ValueError(f"{trajectory_id} event_current_a 与 event_state 不一致")

    midpoint_tau = (np.arange(samples, dtype=np.float64) + 0.5) / samples
    expected_mode = np.concatenate([
        np.where(
            midpoint_tau < circuit["duty_cycle"],
            0,
            np.where(midpoint_tau < tau, 1, 2),
        )
        for tau in event_time_tau
    ]).astype(np.uint8)
    if not np.array_equal(interval_mode, expected_mode):
        raise ValueError(f"{trajectory_id} interval_mode 与事件时间不一致")
    if not np.array_equal(diode_on, (interval_mode == 1).astype(np.uint8)):
        raise ValueError(f"{trajectory_id} diode_on 与 interval_mode 不一致")

    bracket_time = _array(
        data, "event_bracket_time_s", (cycles, 2), np.float64
    )
    bracket_current = _array(
        data, "event_bracket_diode_current_a", (cycles, 2), np.float64
    )
    if np.any(event_time_s < bracket_time[:, 0]) or np.any(
        event_time_s > bracket_time[:, 1]
    ):
        raise ValueError(f"{trajectory_id} 事件时间不在括区间内")
    if np.any(bracket_current[:, 0] <= 0) or np.any(
        np.abs(bracket_current[:, 1]) > event_cfg["diode_off_threshold_a"]
    ):
        raise ValueError(f"{trajectory_id} 事件括区间电流不符合阈值")

    expected_strings = _expected_metadata_strings(
        config, split_name, trajectory_id
    )
    metadata = _metadata_json(data["metadata"], expected_strings)
    return {
        "states": states,
        "input_voltage": input_voltage,
        "load_resistance": load_resistance,
        "time_s": time_s,
        "pwm_gate": pwm_gate,
        "diode_on": diode_on,
        "interval_mode": interval_mode,
        "natural_event_mask": natural_event_mask,
        "event_time_tau": event_time_tau,
        "idle_duration_tau": idle_duration_tau,
        "event_state": event_state,
        "event_current_a": event_current,
        "diode_current_a": diode_current,
        "initial_state": initial_state,
        "trajectory_id": trajectory_id,
        "metadata": metadata,
        "source_sha256": hashlib.sha256(blob).hexdigest(),
    }


def convert_split(split_name, config, raw_root, out_dir, check_only=False):
    if config["dataset"].get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"配置 schema 必须是 {SCHEMA_VERSION}")
    split_cfg = config["dataset"]["splits"][split_name]
    voltages = split_cfg["voltages_v"]
    per_voltage = split_cfg["trajectories_per_voltage"]
    expected_count = len(voltages) * per_voltage
    raw_dir = os.path.join(raw_root, split_name)
    if not os.path.isdir(raw_dir):
        raise ValueError(f"找不到原始数据目录: {raw_dir}")

    expected_names = [
        f"{split_name}_{index:06d}.mat"
        for index in range(1, expected_count + 1)
    ]
    mat_files = sorted(
        name for name in os.listdir(raw_dir)
        if name.endswith(".mat") and name.startswith(split_name + "_")
    )
    if mat_files != expected_names:
        raise ValueError(
            f"{split_name} 文件数或连续编号不符："
            f"实际 {len(mat_files)}，期望 {expected_count}"
        )

    expected_voltages = [
        voltage for voltage in voltages for _ in range(per_voltage)
    ]
    trajectories = []
    for filename, voltage in zip(mat_files, expected_voltages):
        trajectory_id = filename[:-4]
        trajectories.append(load_trajectory(
            os.path.join(raw_dir, filename),
            config,
            split_name,
            split_cfg,
            trajectory_id,
            voltage,
        ))

    shared_time = trajectories[0]["time_s"]
    if any(not np.array_equal(item["time_s"], shared_time)
           for item in trajectories[1:]):
        raise ValueError(f"{split_name} 各轨迹 time_s 不一致")

    arrays = {
        "states": np.stack([item["states"] for item in trajectories]),
        "input_voltage": np.asarray(
            [item["input_voltage"] for item in trajectories], dtype=np.float64
        ),
        "load_resistance": np.asarray(
            [item["load_resistance"] for item in trajectories], dtype=np.float64
        ),
        "time_s": shared_time,
        "pwm_gate": np.stack([item["pwm_gate"] for item in trajectories]),
        "diode_on": np.stack([item["diode_on"] for item in trajectories]),
        "interval_mode": np.stack([
            item["interval_mode"] for item in trajectories
        ]),
        "natural_event_mask": np.stack([
            item["natural_event_mask"] for item in trajectories
        ]),
        "event_time_tau": np.stack([
            item["event_time_tau"] for item in trajectories
        ]),
        "idle_duration_tau": np.stack([
            item["idle_duration_tau"] for item in trajectories
        ]),
        "event_state": np.stack([
            item["event_state"] for item in trajectories
        ]),
        "event_current_a": np.stack([
            item["event_current_a"] for item in trajectories
        ]),
        "diode_current_a": np.stack([
            item["diode_current_a"] for item in trajectories
        ]),
        "initial_state": np.stack([
            item["initial_state"] for item in trajectories
        ]),
        "trajectory_ids": np.asarray([
            item["trajectory_id"] for item in trajectories
        ]),
        "metadata": np.asarray([item["metadata"] for item in trajectories]),
        "source_sha256": np.asarray([
            item["source_sha256"] for item in trajectories
        ], dtype="<U64"),
    }

    out_path = os.path.join(out_dir, f"{split_name}.npz")
    if not check_only:
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(out_path, **arrays)
        size_mb = os.path.getsize(out_path) / 1e6
        print(
            f"已写出 {out_path}（{len(trajectories)} 条轨迹，{size_mb:.1f} MB）"
        )
    return arrays, out_path


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Buck MAT v3 → NPZ v3 转换")
    parser.add_argument("--split", required=True, help="split 名")
    parser.add_argument(
        "--config",
        default=os.path.join(
            project_root, "configs", SCHEMA_VERSION, "project.json"
        ),
    )
    parser.add_argument(
        "--raw-dir",
        default=os.path.join(project_root, "data", "raw", SCHEMA_VERSION),
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(project_root, "data", "processed", SCHEMA_VERSION),
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    try:
        with open(args.config, encoding="utf-8") as stream:
            config = json.load(stream)
        if args.split not in config["dataset"]["splits"]:
            raise ValueError(f"project.json 中不存在 split: {args.split}")
        arrays, out_path = convert_split(
            args.split,
            config,
            args.raw_dir,
            args.out_dir,
            check_only=args.check,
        )
        if args.check:
            if not os.path.isfile(out_path):
                raise ValueError(f"--check 需要已有 NPZ: {out_path}")
            with np.load(out_path, allow_pickle=False) as old:
                if set(old.files) != set(arrays):
                    raise ValueError("NPZ 字段集合不一致")
                mismatches = [
                    name for name, value in arrays.items()
                    if value.shape != old[name].shape
                    or not np.array_equal(value, old[name])
                ]
            if mismatches:
                raise ValueError(f"NPZ 数组不一致: {mismatches}")
            print("回归校验通过：与现有 NPZ 完全一致")
    except (KeyError, OSError, ValueError) as error:
        sys.exit(f"转换失败: {error}")


if __name__ == "__main__":
    main()
