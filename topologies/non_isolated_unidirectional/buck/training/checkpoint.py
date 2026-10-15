#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DCM v3 断点：权重、优化状态、配置签名和随机数状态。"""

from __future__ import annotations

import os

import torch


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_cpu(item) for item in value]
    return value


def save_checkpoint(
    path: str,
    *,
    model,
    optimizer=None,
    scheduler=None,
    stage: str,
    epoch: int,
    best: float,
    stall: int,
    stage_report: dict | None,
    completed_stages: list[str] | None,
    config_sha256: str,
    normalization: dict,
    loss_weights: dict,
    hybrid_settings: dict,
    idle_profile: str,
    rng_state: dict,
    optimizer_state=None,
    scheduler_state=None,
    training_batch_sizes=None,
):
    """原子保存完整训练状态；V2 checkpoint 不会被此入口接受。"""
    payload = {
        "schema_version": "hybrid_event_matlab_v3",
        "model_signature": model.architecture_signature(),
        "model_state": _to_cpu(model.state_dict()),
        "optimizer_state": _to_cpu(optimizer_state if optimizer_state is not None else (optimizer.state_dict() if optimizer else None)),
        "scheduler_state": _to_cpu(scheduler_state if scheduler_state is not None else (scheduler.state_dict() if scheduler else None)),
        "stage": stage,
        "epoch": epoch,
        "best": best,
        "stall": stall,
        "stage_report": dict(stage_report or {}),
        "completed_stages": list(completed_stages or []),
        "training_batch_sizes": dict(training_batch_sizes or {}),
        "config_sha256": config_sha256,
        "normalization": dict(normalization),
        "load_resistance_ohm": 80.0,
        "loss_weights": dict(loss_weights),
        "hybrid_settings": dict(hybrid_settings),
        "idle_profile": idle_profile,
        "rng_state": _to_cpu(rng_state),
    }
    temporary = path + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: str) -> dict:
    """加载并确认是当前 DCM v3 checkpoint。"""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "hybrid_event_matlab_v3":
        raise ValueError("checkpoint 不是 hybrid_event_matlab_v3，拒绝混用 V2 权重")
    return payload
