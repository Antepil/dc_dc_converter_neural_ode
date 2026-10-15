from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from training.model import MODE_IDLE, MODE_OFF_DIODE, MODE_ON


BUCK_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = BUCK_ROOT / "configs" / "hybrid_event_matlab_v3" / "project.json"


def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


class LinearEventModel(nn.Module):
    """ON/IDLE 静止、OFF 恒定下降、线性事件面的解析测试模型。"""

    def __init__(self, off_rate=1.0, root=0.25, idle_current_rate=0.0):
        super().__init__()
        self.off_rate = nn.Parameter(torch.tensor(float(off_rate)))
        self.event_weight = nn.Parameter(torch.tensor(1.0))
        self.event_bias = nn.Parameter(torch.tensor(-float(root)))
        self.idle_current_rate = float(idle_current_rate)

    def vector_field(self, mode, x, vs):
        mode = torch.as_tensor(mode, device=x.device, dtype=torch.long)
        if mode.ndim == 0:
            mode = mode.expand(x.shape[0])
        zero = torch.zeros_like(x)
        off = torch.stack((-self.off_rate.expand(x.shape[0]), torch.zeros_like(x[:, 1])), dim=1)
        idle = torch.stack((torch.full_like(x[:, 0], self.idle_current_rate), torch.zeros_like(x[:, 1])), dim=1)
        return torch.where((mode == MODE_ON)[:, None], zero, torch.where((mode == MODE_OFF_DIODE)[:, None], off, idle))

    def event_value(self, x, vs):
        return self.event_weight * x[:, 0] + self.event_bias

    def apply_event(self, x):
        return x


def synthetic_split(event_times=(0.70, 0.75, 0.80)):
    n_traj = len(event_times)
    states = torch.zeros(n_traj, 101, 2, dtype=torch.float64)
    modes = torch.zeros(n_traj, 100, dtype=torch.long)
    voltages = torch.tensor([12.0, 24.0, 32.0][:n_traj], dtype=torch.float64)
    for trajectory, event_time in enumerate(event_times):
        event_index = round(event_time * 100)
        current_at_off = 0.35 + 0.05 * trajectory
        states[trajectory, :51, 0] = torch.linspace(0.10, current_at_off, 51)
        states[trajectory, :51, 1] = 0.25 + 0.01 * trajectory
        states[trajectory, 50:event_index + 1, 0] = torch.linspace(current_at_off, 0.0, event_index - 49)
        states[trajectory, 50:, 1] = torch.linspace(0.25 + 0.01 * trajectory, 0.24 + 0.01 * trajectory, 51)
        modes[trajectory, 50:event_index] = MODE_OFF_DIODE
        modes[trajectory, event_index:] = MODE_IDLE
    event_time = torch.tensor(event_times, dtype=torch.float64)[:, None]
    event_state = torch.stack(
        [states[index, round(value * 100)] for index, value in enumerate(event_times)]
    )[:, None]
    data = {
        "states": states.clone(),
        "states_norm": states,
        "input_voltage": voltages,
        "vs_norm": (voltages / 32.0)[:, None],
        "load_resistance": torch.full((n_traj,), 80.0, dtype=torch.float64),
        "time_s": torch.arange(101, dtype=torch.float64) * 5e-7,
        "pwm_gate": torch.cat((torch.ones(n_traj, 50), torch.zeros(n_traj, 50)), dim=1).long(),
        "diode_on": (modes == MODE_OFF_DIODE).long(),
        "interval_mode": modes,
        "natural_event_mask": torch.ones(n_traj, 1, dtype=torch.bool),
        "event_time_tau": event_time,
        "idle_duration_tau": 1.0 - event_time,
        "event_state": event_state.clone(),
        "event_state_norm": event_state,
        "event_current_a": event_state[..., 0].clone(),
        "event_current_norm": event_state[..., 0].clone(),
        "diode_current_a": torch.zeros(n_traj, 101, dtype=torch.float64),
        "diode_current_norm": torch.zeros(n_traj, 101, dtype=torch.float64),
        "initial_state": states[:, 0].clone(),
        "initial_state_norm": states[:, 0].clone(),
        "trajectory_ids": np.asarray([f"synthetic_{index}" for index in range(n_traj)]),
        "metadata": [{"schema_version": "hybrid_event_matlab_v3"} for _ in range(n_traj)],
        "source_sha256": np.asarray(["0" * 64] * n_traj),
    }
    return data


def write_npz(path: Path, *, load_ohm=80.0, invalid_transition=False):
    data = synthetic_split((0.75,))
    modes = data["interval_mode"].numpy().astype(np.uint8)
    if invalid_transition:
        modes[0, 70] = MODE_ON
    metadata = np.asarray([json.dumps({"schema_version": "hybrid_event_matlab_v3"})])
    np.savez(
        path,
        states=data["states"].numpy(),
        input_voltage=data["input_voltage"].numpy(),
        load_resistance=np.asarray([load_ohm], dtype=np.float64),
        time_s=data["time_s"].numpy(),
        pwm_gate=data["pwm_gate"].numpy().astype(np.uint8),
        diode_on=data["diode_on"].numpy().astype(np.uint8),
        interval_mode=modes,
        natural_event_mask=np.ones((1, 1), dtype=bool),
        event_time_tau=data["event_time_tau"].numpy(),
        idle_duration_tau=data["idle_duration_tau"].numpy(),
        event_state=data["event_state"].numpy(),
        event_current_a=data["event_current_a"].numpy(),
        diode_current_a=data["diode_current_a"].numpy(),
        initial_state=data["initial_state"].numpy(),
        trajectory_ids=np.asarray(["synthetic_0"]),
        metadata=metadata,
        source_sha256=np.asarray(["0" * 64]),
    )
