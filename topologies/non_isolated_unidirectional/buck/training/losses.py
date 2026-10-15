#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DCM 训练损失：模式导数、事件面和 Hybrid rollout 诊断。"""

from __future__ import annotations

import torch

from .hybrid_rk4 import rk4_step
from .model import MODE_IDLE, MODE_OFF_DIODE, MODE_ON


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).mean()


def masked_mse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    mask = mask.expand_as(pred)
    values = (pred - target).square()
    return values[mask].mean() if mask.any() else values.new_zeros(())


def _cycle_masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """先按周期平均，再对周期和 batch 平均，避免长 IDLE 段占更高权重。"""
    cycles = values.shape[0] // 100
    values = values[: cycles * 100].reshape(cycles, 100, -1)
    mask = mask[: cycles * 100].reshape(cycles, 100, -1)
    counts = mask.sum(dim=1)
    per_cycle = (values * mask).sum(dim=1) / counts.clamp_min(1)
    valid = counts > 0
    return per_cycle[valid].mean() if valid.any() else values.new_zeros(())


def event_pretrain_loss(
    g_surface: torch.Tensor,
    g_pre: torch.Tensor,
    g_counterfactual: torch.Tensor,
    guard_margin: float,
) -> dict[str, torch.Tensor]:
    """事件表面 g=0、事件前 g>=margin、反事实后 g<=-margin。"""
    surface = g_surface.square().mean()
    pre = torch.relu(guard_margin - g_pre).square().mean()
    counterfactual = torch.relu(g_counterfactual + guard_margin).square().mean()
    return {
        "surface": surface,
        "guard": pre,
        "counterfactual": counterfactual,
        "total": surface + pre + counterfactual,
    }


def ripple_loss(pred_vc: torch.Tensor, tgt_vc: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """线性端点去趋势后的相对峰峰值误差平方。"""
    n_points = pred_vc.shape[0]
    line = torch.linspace(0.0, 1.0, n_points, device=pred_vc.device, dtype=pred_vc.dtype)[:, None]
    pred_detrended = pred_vc - (pred_vc[0] + line * (pred_vc[-1] - pred_vc[0]))
    target_detrended = tgt_vc - (tgt_vc[0] + line * (tgt_vc[-1] - tgt_vc[0]))
    pred_ptp = pred_detrended.amax(0) - pred_detrended.amin(0)
    target_ptp = target_detrended.amax(0) - target_detrended.amin(0)
    return ((pred_ptp - target_ptp) / (target_ptp + eps)).square().mean()


def voltage_derivative_loss(model, batch):
    """Mode-balanced teacher-state voltage derivative MSE in normalized coordinates."""
    x = batch["target"][:-1]
    vs = batch["vs"].expand(x.shape[0], -1, -1)
    values = []
    for mode in (MODE_ON, MODE_OFF_DIODE, MODE_IDLE):
        mask = batch["derivative_valid"][:-1] & (batch["interval_mode"] == mode)
        if mask.any():
            prediction = model.vector_field(mode, x[mask], vs[mask])[:, 1]
            truth = batch["derivative_target"][:-1, :, 1][mask]
            values.append((prediction - truth).square().mean())
    return torch.stack(values).mean() if values else x.new_zeros(())


def rollout_losses(
    model,
    rollout,
    batch: dict,
    weights: dict,
    *,
    use_ripple: bool = False,
    guard_margin: float = 0.1,
    counterfactual_offsets=(0.0025, 0.005, 0.01),
) -> dict[str, torch.Tensor]:
    """计算设计文档中的 state/event/IDLE/ripple 项，返回逐项和 total。"""
    pred, target = rollout.states, batch["target"]
    terms: dict[str, torch.Tensor] = {"state": mse_loss(pred, target)}
    event_mask = batch["event_mask"]
    terms["event_time"] = masked_mse(
        rollout.event_time_tau, batch["event_time_tau"], event_mask
    )
    event_current = rollout.event_current_norm
    terms["event_current"] = event_current[event_mask].square().mean() if event_mask.any() else event_current.new_zeros(())
    guard_indices = torch.arange(50, target.shape[0] - 1, 100, device=target.device)
    guard_x = target.index_select(0, guard_indices).reshape(-1, 2)
    guard_vs = batch["vs"].expand(guard_indices.numel(), -1, -1).reshape(-1, 1)
    surface_x = batch["event_state"].reshape(-1, 2)
    cycles = batch["event_state"].shape[1]
    surface_vs = batch["vs"].repeat_interleave(cycles, dim=0)
    offsets = torch.as_tensor(counterfactual_offsets, device=pred.device, dtype=pred.dtype)
    counterfactual_x = rk4_step(
        model,
        MODE_OFF_DIODE,
        surface_x.repeat_interleave(offsets.numel(), dim=0),
        surface_vs.repeat_interleave(offsets.numel(), dim=0),
        offsets.repeat(surface_x.shape[0]),
    )
    guard_terms = event_pretrain_loss(
        model.event_value(surface_x, surface_vs),
        model.event_value(guard_x, guard_vs),
        model.event_value(counterfactual_x, surface_vs.repeat_interleave(offsets.numel(), dim=0)),
        guard_margin,
    )
    terms["guard"] = guard_terms["total"]

    idle_mask = batch["interval_mode"] == MODE_IDLE
    idle_current = pred[1:, :, 0]
    terms["idle_state"] = _cycle_masked_mean(idle_current.square(), idle_mask)
    idle_vs = batch["vs"].expand(pred.shape[0] - 1, -1, -1).reshape(-1, 1)
    pred_dx = model.idle_vector_field(pred[:-1].reshape(-1, 2), idle_vs).reshape_as(pred[:-1])
    terms["idle_tangent"] = _cycle_masked_mean(pred_dx[..., 0].square(), idle_mask)
    next_mode = torch.roll(batch["interval_mode"], shifts=-1, dims=0)
    next_mode[-1] = MODE_ON
    next_mode[99::100] = MODE_ON
    idle_end = idle_mask & (next_mode == MODE_ON)
    terms["idle_end"] = _cycle_masked_mean(pred[1:, :, 0].square(), idle_end)
    terms["ripple"] = ripple_loss(pred[..., 1], target[..., 1]) if use_ripple else pred.new_zeros(())
    if weights.get("voltage_derivative", 0) > 0:
        terms["voltage_derivative"] = voltage_derivative_loss(model, batch)
    terms["total"] = sum(weights.get(name, 0.0) * value for name, value in terms.items() if name != "total")
    return terms
