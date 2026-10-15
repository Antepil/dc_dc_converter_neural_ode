#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Buck DCM 的批量 Hybrid RK4 积分器。

该模块实现设计文档第 6 节规定的固定网格混合积分：PWM 边沿在网格上
切换模式，``OFF_DIODE`` 中由神经事件面检测正到非正穿越，并用两次可微
割线细化事件位置。事件后采用恒等状态映射，再用 soft/hard IDLE 向量场
完成剩余子步。

输入数据的 schema、负载、周期数和 DCM 合法性由上游数据入口检查；本模块
不重复验证那些配置假设，只在运行过程中保护非有限状态和非有限事件值。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch

from .model import MODE_IDLE, MODE_OFF_DIODE, MODE_ON


@dataclass
class HybridRollout:
    """一次批量 rollout 的状态和事件诊断。

    ``states`` 使用时间优先布局 ``(T, B, 2)``，``interval_mode`` 使用
    ``(T-1, B)``；事件字段使用批次优先布局：
    ``event_time_tau``/``event_mask``/``missed_event``/``invalid_initial_guard``
    为 ``(B, C)``，``event_state`` 为 ``(B, C, 2)``。
    """

    states: torch.Tensor
    interval_mode: torch.Tensor
    event_time_tau: torch.Tensor
    event_state: torch.Tensor
    event_mask: torch.Tensor
    missed_event: torch.Tensor
    invalid_initial_guard: torch.Tensor

    @property
    def event_current_norm(self) -> torch.Tensor:
        """事件处归一化电感电流，形状 ``(B, C)``。"""
        return self.event_state[..., 0]

def rk4_step(
    model,
    mode: Union[int, torch.Tensor],
    x: torch.Tensor,
    vs: torch.Tensor,
    dt: Union[float, torch.Tensor],
) -> torch.Tensor:
    """对一个批次执行一次可微 RK4 步。

    ``dt`` 可以是标量，也可以是每个样本一个步长 ``(B,)``；后者用于事件
    后的不同剩余子步。``mode`` 由调用方显式传入，不在这里进行 PWM 推断。
    """
    step = torch.as_tensor(dt, device=x.device, dtype=x.dtype)
    if step.ndim == 0:
        step = step.expand(x.shape[0])
    step = step.reshape(-1, 1)

    k1 = model.vector_field(mode, x, vs)
    k2 = model.vector_field(mode, x + 0.5 * step * k1, vs)
    k3 = model.vector_field(mode, x + 0.5 * step * k2, vs)
    k4 = model.vector_field(mode, x + step * k3, vs)
    return x + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def _check_finite(name: str, value: torch.Tensor) -> None:
    """对积分中的数值结果做有限性保护。"""
    if not torch.isfinite(value).all():
        bad = (~torch.isfinite(value)).any(dim=-1).nonzero(as_tuple=False).flatten()
        raise FloatingPointError(f"Hybrid RK4 {name} 出现非有限值，batch={bad.tolist()}")


def _secant_event(
    model,
    x0: torch.Tensor,
    vs: torch.Tensor,
    g0: torch.Tensor,
    g1: torch.Tensor,
    step_size: float,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """在一个已括住根的 OFF 步内执行可微割线细化。

    ``g0`` 必须为正，``g1`` 必须为非正。每次细化都重新执行长度为
    ``alpha * step_size`` 的 OFF RK4 子步，事件状态不使用线性插值。
    """
    lower = torch.zeros_like(g0)
    upper = torch.ones_like(g0)
    g_lower = g0
    g_upper = g1
    denominator = g_lower - g_upper
    alpha = torch.clamp(g_lower / denominator, 0.0, 1.0)

    off_mode = MODE_OFF_DIODE

    def evaluate(candidate_alpha):
        candidate_x = rk4_step(
            model, off_mode, x0, vs, candidate_alpha * step_size
        )
        candidate_g = model.event_value(candidate_x, vs)
        _check_finite("割线候选事件值", candidate_g.unsqueeze(1))
        return candidate_x, candidate_g

    # 先评估初始割线位置，再执行配置指定的两次细化。
    _, g_candidate = evaluate(alpha)
    positive = g_candidate > 0.0
    lower = torch.where(positive, alpha, lower)
    g_lower = torch.where(positive, g_candidate, g_lower)
    upper = torch.where(positive, upper, alpha)
    g_upper = torch.where(positive, g_upper, g_candidate)

    for _ in range(iterations):
        denominator = g_lower - g_upper
        secant = lower + g_lower * (upper - lower) / denominator
        alpha = torch.minimum(torch.maximum(secant, lower), upper)
        _, g_candidate = evaluate(alpha)
        positive = g_candidate > 0.0
        lower = torch.where(positive, alpha, lower)
        g_lower = torch.where(positive, g_candidate, g_lower)
        upper = torch.where(positive, upper, alpha)
        g_upper = torch.where(positive, g_upper, g_candidate)

    x_event = rk4_step(model, off_mode, x0, vs, alpha * step_size)
    _check_finite("事件状态", x_event)
    return alpha, x_event


def _full_batch_from_subset(
    base: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """把子批次结果放回完整批次，同时保留 autograd 图。"""
    return base.index_copy(0, indices, values)


def _mask_from_indices(
    indices: torch.Tensor, batch_size: int, device: torch.device
) -> torch.Tensor:
    """将子批次索引转换为完整批次的布尔掩码。"""
    return torch.zeros(batch_size, dtype=torch.bool, device=device).index_fill(
        0, indices, True
    )


def hybrid_rk4(
    model,
    y0: torch.Tensor,
    vs: torch.Tensor,
    n_cycles: int,
    *,
    duty: float = 0.5,
    step_size_tau: float = 0.01,
    samples_per_cycle: int = 100,
    secant_refinement_iterations: int = 2,
    check_finite_every_step: bool = False,
) -> HybridRollout:
    """执行固定周期数的批量 DCM Hybrid RK4 rollout。

    参数 ``y0`` 为 ``(B,2)`` 初始状态，``vs`` 为 ``(B,1)`` 或 ``(B,)``
    归一化输入电压。返回的状态包含初始点，因此形状为
    ``(100*n_cycles+1, B, 2)``。

    默认在返回前统一检查完整轨迹，避免每一步 CPU/GPU 同步；事件细化
    仍即时检查。调试时可用 ``check_finite_every_step=True`` 恢复逐步检查。
    """
    batch_size = y0.shape[0]
    vs = vs.to(device=y0.device, dtype=y0.dtype)
    if vs.ndim == 1:
        vs = vs.unsqueeze(1)
    if vs.shape[0] == 1:
        vs = vs.expand(batch_size, -1)

    states = [y0]
    interval_modes = []
    cycle_event_times = []
    cycle_event_states = []
    cycle_event_masks = []
    cycle_missed = []
    cycle_invalid_guards = []
    event_values_to_check = []
    replaced_states_to_check = []

    on_steps = round(duty * samples_per_cycle)
    off_mode = torch.full(
        (batch_size,), MODE_OFF_DIODE, dtype=torch.long, device=y0.device
    )
    idle_mode = torch.full(
        (batch_size,), MODE_IDLE, dtype=torch.long, device=y0.device
    )

    x = y0
    for cycle in range(n_cycles):
        mode = torch.full(
            (batch_size,), MODE_ON, dtype=torch.long, device=y0.device
        )
        event_found = torch.zeros(batch_size, dtype=torch.bool, device=y0.device)
        invalid_guard = torch.zeros(batch_size, dtype=torch.bool, device=y0.device)
        event_time = torch.zeros_like(x[:, 0])
        event_state = torch.zeros_like(x)
        all_idle = False

        for interval in range(samples_per_cycle):
            if interval == on_steps:
                mode = off_mode
            step_mode = mode

            tau_start = interval * step_size_tau
            # PWM ON 段的模式由时间网格确定，无需从 GPU 读回索引。
            field_mode = MODE_ON if interval < on_steps else (MODE_IDLE if all_idle else step_mode)
            x_full = rk4_step(model, field_mode, x, vs, step_size_tau)
            if check_finite_every_step:
                _check_finite("状态", x_full)

            if interval < on_steps or all_idle:
                x = x_full
                interval_modes.append(step_mode)
                states.append(x)
                continue

            off_indices = (mode == MODE_OFF_DIODE).nonzero(
                as_tuple=False
            ).flatten()
            if off_indices.numel() == 0:
                x = x_full
                interval_modes.append(step_mode)
                states.append(x)
                continue

            x_off = x.index_select(0, off_indices)
            vs_off = vs.index_select(0, off_indices)
            g0 = model.event_value(x_off, vs_off)
            g1 = model.event_value(
                x_full.index_select(0, off_indices), vs_off
            )
            if check_finite_every_step:
                _check_finite("事件值", torch.stack([g0, g1], dim=1))
            else:
                event_values_to_check.extend((g0.detach(), g1.detach()))

            invalid_local = g0 <= 0.0
            crossed_local = (g0 > 0.0) & (g1 <= 0.0)
            crossed_indices = off_indices[crossed_local]
            invalid_indices = off_indices[invalid_local]
            trigger_indices = off_indices[invalid_local | crossed_local]

            if crossed_indices.numel() > 0:
                cross_positions = crossed_local.nonzero(as_tuple=False).flatten()
                alpha_cross, x_event_cross = _secant_event(
                    model,
                    x_off.index_select(0, cross_positions),
                    vs_off.index_select(0, cross_positions),
                    g0.index_select(0, cross_positions),
                    g1.index_select(0, cross_positions),
                    step_size_tau,
                    secant_refinement_iterations,
                )
            else:
                alpha_cross = x_off.new_empty((0,))
                x_event_cross = x_off.new_empty((0, 2))

            # invalid_initial_guard 采用 alpha=0；不把 iL 投影到零。
            alpha_invalid = x_off.new_zeros((invalid_indices.numel(),))
            x_event_invalid = x.index_select(0, invalid_indices)

            candidate_time = x[:, 0].new_zeros((batch_size,))
            candidate_state = x.new_zeros((batch_size, 2))
            candidate_alpha = x[:, 0].new_zeros((batch_size,))
            if crossed_indices.numel() > 0:
                candidate_time = _full_batch_from_subset(
                    candidate_time,
                    crossed_indices,
                    tau_start + alpha_cross * step_size_tau,
                )
                candidate_state = _full_batch_from_subset(
                    candidate_state, crossed_indices, x_event_cross
                )
                candidate_alpha = _full_batch_from_subset(
                    candidate_alpha, crossed_indices, alpha_cross
                )
            if invalid_indices.numel() > 0:
                candidate_time = _full_batch_from_subset(
                    candidate_time, invalid_indices,
                    x[:, 0].new_zeros((invalid_indices.numel(),)) + tau_start,
                )
                candidate_state = _full_batch_from_subset(
                    candidate_state, invalid_indices, x_event_invalid
                )
                candidate_alpha = _full_batch_from_subset(
                    candidate_alpha, invalid_indices, alpha_invalid
                )

            if trigger_indices.numel() > 0:
                if not check_finite_every_step:
                    # 事件更新会替换 x_full 的部分行；这些试探状态也必须检查。
                    replaced_states_to_check.append(x_full.detach())
                trigger_mask = _mask_from_indices(
                    trigger_indices, batch_size, x.device
                )
                invalid_mask = _mask_from_indices(
                    invalid_indices, batch_size, x.device
                )
                trigger_state = candidate_state.index_select(0, trigger_indices)
                trigger_vs = vs.index_select(0, trigger_indices)
                trigger_alpha = candidate_alpha.index_select(0, trigger_indices)
                x_idle = rk4_step(
                    model,
                    MODE_IDLE,
                    model.apply_event(trigger_state),
                    trigger_vs,
                    (1.0 - trigger_alpha) * step_size_tau,
                )
                _check_finite("事件后状态", x_idle)
                x_next = _full_batch_from_subset(x_full, trigger_indices, x_idle)
                x = x_next
                event_time = torch.where(
                    trigger_mask,
                    candidate_time,
                    event_time,
                )
                event_state = torch.where(
                    trigger_mask.unsqueeze(1),
                    candidate_state,
                    event_state,
                )
                event_found = event_found | trigger_mask
                invalid_guard = invalid_guard | invalid_mask
                mode = torch.where(event_found, idle_mode, mode)
                # 索引长度已在事件分支中获得；无需另一次 GPU reduction。
                all_idle = trigger_indices.numel() == off_indices.numel()
            else:
                x = x_full

            interval_modes.append(step_mode)
            states.append(x)

        cycle_event_times.append(event_time)
        cycle_event_states.append(event_state)
        cycle_event_masks.append(event_found)
        cycle_missed.append(~event_found)
        cycle_invalid_guards.append(invalid_guard)

    stacked_states = torch.stack(states, dim=0)
    _check_finite("完整轨迹", stacked_states.reshape(-1, 2))
    if replaced_states_to_check:
        _check_finite("事件前试探状态", torch.cat(replaced_states_to_check))
    if event_values_to_check:
        _check_finite("轨迹事件值", torch.cat(event_values_to_check).reshape(-1, 1))
    return HybridRollout(
        states=stacked_states,
        interval_mode=torch.stack(interval_modes, dim=0),
        event_time_tau=torch.stack(cycle_event_times, dim=1),
        event_state=torch.stack(cycle_event_states, dim=1),
        event_mask=torch.stack(cycle_event_masks, dim=1),
        missed_event=torch.stack(cycle_missed, dim=1),
        invalid_initial_guard=torch.stack(cycle_invalid_guards, dim=1),
    )
