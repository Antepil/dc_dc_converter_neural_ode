#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DCM v3 课程训练循环。"""

from __future__ import annotations

import json
import time

import torch
import torch.nn as nn

from .data import (
    build_event_samples,
    build_mode_pairs,
    collate,
    make_windows,
)
from .hybrid_rk4 import hybrid_rk4, rk4_step
from .losses import event_pretrain_loss, rollout_losses
from .model import MODE_IDLE, MODE_OFF_DIODE, MODE_ON


def _scheduler(optimizer, cfg):
    settings = cfg["training"]["learning_rate_scheduler"]
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=settings["factor"],
        patience=settings["patience_epochs"],
    )


def _clip(model, cfg):
    nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip_norm"])


def _early_stop(best, stall, value, cfg):
    improvement = cfg["training"]["minimum_relative_improvement_fraction"]
    if value < best * (1.0 - improvement):
        return value, 0
    return best, stall + 1


def _set_requires_grad(model, names: set[str] | None):
    for name, parameter in model.named_parameters():
        parameter.requires_grad = names is None or any(name.startswith(prefix) for prefix in names)


def _restore_training_state(optimizer, scheduler, resume):
    if resume is None:
        return 0, float("inf"), 0
    optimizer.load_state_dict(resume["optimizer_state"])
    if scheduler is not None and resume.get("scheduler_state") is not None:
        scheduler.load_state_dict(resume["scheduler_state"])
    return resume["epoch"], resume["best"], resume["stall"]


def _optimizer_updates(optimizer) -> int:
    """返回 Adam 已执行的真实 update 数，避免用 epoch 推测。"""
    updates = [
        int(state["step"])
        for state in optimizer.state.values()
        if "step" in state
    ]
    return max(updates, default=0)


def _limit_windows(data, windows, maximum):
    if maximum is None or len(windows) <= maximum:
        return windows
    anchors = []
    voltage_order = torch.argsort(data["input_voltage"])
    event_order = torch.argsort(data["event_time_tau"][:, 0])
    for trajectory in torch.cat((voltage_order[[0, len(voltage_order) // 2, -1]], event_order)).tolist():
        candidate = (trajectory, 0)
        if candidate in windows and candidate not in anchors:
            anchors.append(candidate)
    anchors.extend(window for window in windows if window not in anchors)
    return anchors[:maximum]


def run_mode_dynamics(
    model, train_data, val_data, stage, cfg, device, dtype, *,
    max_batches=None, max_epochs=None, resume=None,
    checkpoint_every=2, checkpoint_callback=None,
):
    """模式内有限差分预训练，三个模式按 batch 等量采样。"""
    method = cfg["training"].get("derivative_method", "forward")
    train_pairs = build_mode_pairs(train_data, method=method)
    val_pairs = build_mode_pairs(val_data, method=method)
    _set_requires_grad(model, {"f_on", "f_off", "f_idle"})
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=stage["vector_field_learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = _scheduler(optimizer, cfg)
    batch_size = stage["batch_size"]
    per_mode = max(1, batch_size // 3)
    tensors = {
        mode: {key: value.to(device=device, dtype=dtype) for key, value in values.items()}
        for mode, values in train_pairs.items()
    }
    val_tensors = {
        mode: {key: value.to(device=device, dtype=dtype) for key, value in values.items()}
        for mode, values in val_pairs.items()
    }
    n_steps = max(len(values["x"]) // per_mode for values in tensors.values())
    n_steps = min(n_steps, max_batches) if max_batches else n_steps
    start_epoch, best, stall = _restore_training_state(optimizer, scheduler, resume)
    epochs = max_epochs or stage["max_epochs"]
    last_epoch = start_epoch
    for epoch in range(start_epoch, epochs):
        model.train()
        total = torch.zeros((), device=device, dtype=dtype)
        for _ in range(max(1, n_steps)):
            mode_losses = []
            for mode in (MODE_ON, MODE_OFF_DIODE, MODE_IDLE):
                sample = tensors[mode]
                index = torch.randint(len(sample["x"]), (per_mode,), device=device)
                pred = model.vector_field(mode, sample["x"].index_select(0, index), sample["vs"].index_select(0, index))
                mode_losses.append((pred - sample["dx"].index_select(0, index)).square().mean())
            loss = sum(mode_losses) / len(mode_losses)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            _clip(model, cfg)
            optimizer.step()
            total += loss.detach()
        model.eval()
        with torch.no_grad():
            values = []
            for mode in (MODE_ON, MODE_OFF_DIODE, MODE_IDLE):
                sample = val_tensors[mode]
                pred = model.vector_field(mode, sample["x"], sample["vs"])
                values.append((pred - sample["dx"]).square().mean())
            val_loss = float(torch.stack(values).mean())
        scheduler.step(val_loss)
        best, stall = _early_stop(best, stall, val_loss, cfg)
        last_epoch = epoch + 1
        print(f"  [mode_dynamics] epoch {last_epoch}/{epochs} | train {float(total) / max(n_steps, 1):.3e} | val {val_loss:.3e}")
        if checkpoint_callback and last_epoch % checkpoint_every == 0:
            checkpoint_callback(
                last_epoch, best, stall,
                {"val_mode_derivative_mse": best, "epochs": last_epoch},
                optimizer.state_dict(), scheduler.state_dict(),
            )
        if stall >= cfg["training"]["early_stopping_patience_epochs"]:
            break
    return {"val_mode_derivative_mse": best, "epochs": last_epoch, "_stall": stall,
            "_optimizer_state": optimizer.state_dict(),
            "_scheduler_state": scheduler.state_dict()}


def run_event_pretrain(
    model, train_data, val_data, stage, cfg, device, dtype, *,
    max_batches=None, max_epochs=None, resume=None,
    checkpoint_every=2, checkpoint_callback=None,
):
    """冻结向量场，仅训练神经事件面及其 guard/counterfactual 约束。"""
    train_samples = build_event_samples(train_data)
    val_samples = build_event_samples(val_data)
    _set_requires_grad(model, {"event_network"})
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=stage["event_network_learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    # 事件损失包含随机抽样的 hinge 项，ReduceLROnPlateau 会把学习率过早压到
    # 接近零。该阶段保持固定学习率；后续 rollout 阶段仍使用全局 scheduler。
    scheduler = None
    margin = cfg["training"]["event_pretraining"]["guard_margin"]
    offsets = torch.tensor(
        cfg["training"]["event_pretraining"]["counterfactual_offsets_tau"],
        device=device,
        dtype=dtype,
    )
    batch_size = stage["batch_size"]
    start_epoch, best, stall = _restore_training_state(optimizer, scheduler, resume)
    for group in optimizer.param_groups:
        group["lr"] = stage["event_network_learning_rate"]
    # 兼容曾使用 plateau scheduler 的 event checkpoint。旧 best/stall 来自随机
    # validation，不能与现在的全量确定性 validation 直接比较。
    if resume is not None and resume.get("scheduler_state") is not None:
        best, stall = float("inf"), 0
    epochs = max_epochs or stage["max_epochs"]
    early_stopping_patience = stage.get(
        "early_stopping_patience_epochs",
        cfg["training"]["early_stopping_patience_epochs"],
    )
    gate_interval = stage.get("validation_gate_interval_epochs", 10)

    def tensors(samples):
        return {key: value.to(device=device, dtype=dtype) for key, value in samples.items()}

    train_t, val_t = tensors(train_samples), tensors(val_samples)
    steps = max(1, len(train_t["surface_x"]) // batch_size)
    steps = min(steps, max_batches) if max_batches else steps

    def loss_for(sample, n=None):
        if n is None:
            # validation 必须覆盖全部样本，保证 scheduler/early-stop 指标可复现。
            surface_x = sample["surface_x"]
            surface_vs = sample["surface_vs"]
            pre_x = sample["pre_x"]
            pre_vs = sample["pre_vs"]
        else:
            surface_index = torch.randint(len(sample["surface_x"]), (n,), device=device)
            pre_index = torch.randint(len(sample["pre_x"]), (n,), device=device)
            surface_x = sample["surface_x"].index_select(0, surface_index)
            surface_vs = sample["surface_vs"].index_select(0, surface_index)
            pre_x = sample["pre_x"].index_select(0, pre_index)
            pre_vs = sample["pre_vs"].index_select(0, pre_index)
        n_surface = len(surface_x)
        g_surface = model.event_value(surface_x, surface_vs)
        g_pre = model.event_value(pre_x, pre_vs)
        off_mode = MODE_OFF_DIODE
        x_cf = rk4_step(
            model,
            off_mode,
            surface_x.repeat_interleave(len(offsets), dim=0),
            surface_vs.repeat_interleave(len(offsets), dim=0),
            offsets.repeat(n_surface),
        )
        g_cf = model.event_value(x_cf, surface_vs.repeat_interleave(len(offsets), dim=0))
        return event_pretrain_loss(g_surface, g_pre, g_cf, margin)

    last_epoch = start_epoch
    last_gate = None
    val_terms = {}
    for epoch in range(start_epoch, epochs):
        model.train()
        train_totals = {name: torch.zeros((), device=device, dtype=dtype)
                        for name in ("surface", "guard", "counterfactual", "total")}
        for _ in range(max(1, steps)):
            terms = loss_for(train_t, batch_size)
            optimizer.zero_grad(set_to_none=True)
            terms["total"].backward()
            _clip(model, cfg)
            optimizer.step()
            for name in train_totals:
                train_totals[name] += terms[name].detach()
        model.eval()
        with torch.no_grad():
            deterministic_terms = loss_for(val_t)
            val_terms = {name: float(value) for name, value in deterministic_terms.items()}
        val_loss = val_terms["total"]
        best, stall = _early_stop(best, stall, val_loss, cfg)
        last_epoch = epoch + 1
        completed_updates = _optimizer_updates(optimizer)
        total_updates = completed_updates + (epochs - last_epoch) * steps
        learning_rate = optimizer.param_groups[0]["lr"]

        print(
            f"  [event_pretrain] epoch {last_epoch}/{epochs} | "
            f"updates {completed_updates}/{total_updates} | "
            f"train {float(train_totals['total']) / max(steps, 1):.3e} | "
            f"val {val_loss:.3e} | "
            f"surface {val_terms['surface']:.3e} | "
            f"guard {val_terms['guard']:.3e} | "
            f"cf {val_terms['counterfactual']:.3e} | "
            f"lr {learning_rate:.2e}"
        )

        should_check_gate = last_epoch % gate_interval == 0 or last_epoch == epochs
        if should_check_gate:
            last_gate = event_pretrain_gate(model, val_data, cfg, device, dtype)
            print(
                "  [event_pretrain gate] "
                + json.dumps(last_gate, ensure_ascii=False)
            )

        if checkpoint_callback and last_epoch % checkpoint_every == 0:
            current_report = {
                "val_event_pretrain_loss": best,
                "last_val_loss_terms": val_terms,
                "epochs": last_epoch,
            }
            if last_gate is not None:
                current_report["validation_gate"] = last_gate
            checkpoint_callback(
                last_epoch, best, stall,
                current_report,
                optimizer.state_dict(), None,
            )
        if last_gate is not None and last_gate["passed"]:
            break
        if stall >= early_stopping_patience:
            break
    _set_requires_grad(model, None)
    return {"val_event_pretrain_loss": best, "last_val_loss_terms": val_terms,
            "validation_gate": last_gate, "epochs": last_epoch, "_stall": stall,
            "_optimizer_state": optimizer.state_dict(),
            "_scheduler_state": None}


def event_pretrain_gate(model, data, cfg, device, dtype):
    """在进入联合 rollout 前检查每个 OFF 段的事件安全性。"""
    samples_per_cycle = 100
    step_size = 1.0 / samples_per_cycle
    on_steps = round(cfg["circuit"]["duty_cycle"] * samples_per_cycle)
    off_steps = samples_per_cycle - on_steps
    n_trajectories, n_cycles = data["event_time_tau"].shape
    model.eval()
    with torch.no_grad():
        off_start_indices = (
            torch.arange(n_cycles, dtype=torch.long) * samples_per_cycle + on_steps
        )
        x = data["states_norm"][:, off_start_indices].reshape(-1, 2).to(
            device=device, dtype=dtype
        )
        vs = data["vs_norm"][:, None, :].expand(-1, n_cycles, -1).reshape(-1, 1).to(
            device=device, dtype=dtype
        )
        target = data["event_time_tau"].reshape(-1).to(device=device, dtype=dtype)
        mode = MODE_OFF_DIODE
        previous = model.event_value(x, vs)
        guard_bad = previous <= 0.0
        root_count = torch.zeros_like(previous, dtype=torch.long)
        first_root = torch.zeros_like(previous)

        for interval in range(off_steps):
            next_x = rk4_step(model, mode, x, vs, step_size)
            current = model.event_value(next_x, vs)
            crossed = (previous > 0.0) & (current <= 0.0)
            denominator = torch.where(
                crossed, previous - current, torch.ones_like(previous)
            )
            alpha = previous / denominator
            root = on_steps * step_size + interval * step_size + alpha * step_size
            first_root = torch.where(crossed & (root_count == 0), root, first_root)
            root_count = root_count + crossed.long()
            previous, x = current, next_x

        detections = root_count > 0
        exactly_one = root_count == 1
        before_pwm = detections & (first_root < 1.0)
        errors = (first_root[exactly_one] - target[exactly_one]).abs()
    gate_cfg = cfg["training"]["event_pretraining"]["validation_gate"]
    report = {
        "event_detection_rate": float(detections.float().mean()),
        "event_time_mae_tau": float(errors.mean()) if errors.numel() else float("inf"),
        "nonpositive_off_start_guard_rate": float(guard_bad.float().mean()),
        "exactly_one_root_rate": float(exactly_one.float().mean()),
        "event_before_next_pwm_rate": float(before_pwm.float().mean()),
    }
    report["passed"] = (
        report["event_detection_rate"] >= gate_cfg["event_detection_rate_min"]
        and report["event_time_mae_tau"] <= gate_cfg["event_time_mae_tau_max"]
        and report["nonpositive_off_start_guard_rate"] <= gate_cfg["nonpositive_off_start_guard_rate_max"]
        and report["exactly_one_root_rate"] >= gate_cfg["exactly_one_root_rate_min"]
        and report["event_before_next_pwm_rate"] >= gate_cfg["event_before_next_pwm_rate_min"]
    )
    return report


def _rollout_validation(model, data, windows, stage, cfg, device, dtype, use_ripple):
    if not windows:
        return {"total": float("inf")}
    values = {}
    with torch.no_grad():
        for start in range(0, len(windows), stage["batch_size"]):
            batch = collate(data, windows[start:start + stage["batch_size"]], stage["window_cycles"], device, dtype,
                            derivative_supervision=cfg["training"]["loss_weights"].get("voltage_derivative", 0) > 0)
            result = hybrid_rk4(
                model,
                batch["y0"],
                batch["vs"],
                stage["window_cycles"],
                duty=cfg["circuit"]["duty_cycle"],
            )
            terms = rollout_losses(
                model, result, batch, cfg["training"]["loss_weights"],
                use_ripple=use_ripple,
                guard_margin=cfg["training"]["event_pretraining"]["guard_margin"],
                counterfactual_offsets=cfg["training"]["event_pretraining"]["counterfactual_offsets_tau"],
            )
            for name, value in terms.items():
                values.setdefault(name, []).append(value)
    return {name: float(torch.stack(items).mean()) for name, items in values.items()}


def run_rollout_stage(
    model, train_data, val_data, stage, cfg, device, dtype, *,
    max_windows=None, max_epochs=None, resume=None,
    checkpoint_every=2, checkpoint_callback=None,
    best_callback=None,
):
    """固定网格 Hybrid RK4 的周期课程阶段。"""
    train_windows = make_windows(train_data, stage["window_cycles"])
    val_windows = make_windows(val_data, stage["window_cycles"])
    if max_windows:
        train_windows = _limit_windows(train_data, train_windows, max_windows)
    if not train_windows:
        return {"val_rollout_loss": float("inf"), "epochs": 0, "_stall": 0,
                "_optimizer_state": None, "_scheduler_state": None}
    _set_requires_grad(model, None)
    groups = [{"params": list(model.f_on.parameters()) + list(model.f_off.parameters()) + list(model.f_idle.parameters()), "lr": stage["vector_field_learning_rate"]}]
    if stage["event_network_learning_rate"] is not None:
        groups.append({"params": model.event_network.parameters(), "lr": stage["event_network_learning_rate"]})
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg["training"]["weight_decay"])
    scheduler = _scheduler(optimizer, cfg)
    use_ripple = stage["name"] == cfg["training"]["ripple_loss"]["enabled_stage"]
    start_epoch, best, stall = _restore_training_state(optimizer, scheduler, resume)
    epochs = max_epochs or stage["max_epochs"]
    last_epoch = start_epoch
    val_terms = dict(resume.get("last_val_loss_terms", {})) if resume else {}
    training_batch_size = stage.get("training_batch_size", stage["batch_size"])
    epoch_timing = {}
    print(f"  [{stage['name']}] train_windows={len(train_windows)} | "
          f"train_batch={training_batch_size} | validation_batch={stage['batch_size']}")
    for epoch in range(start_epoch, epochs):
        model.train()
        order = torch.randperm(len(train_windows)).tolist()
        total, batches = torch.zeros((), device=device, dtype=dtype), 0
        started = time.time()
        for start in range(0, len(order), training_batch_size):
            selected = [train_windows[index] for index in order[start:start + training_batch_size]]
            batch = collate(train_data, selected, stage["window_cycles"], device, dtype,
                            derivative_supervision=cfg["training"]["loss_weights"].get("voltage_derivative", 0) > 0)
            result = hybrid_rk4(model, batch["y0"], batch["vs"], stage["window_cycles"], duty=cfg["circuit"]["duty_cycle"])
            terms = rollout_losses(
                model, result, batch, cfg["training"]["loss_weights"],
                use_ripple=use_ripple,
                guard_margin=cfg["training"]["event_pretraining"]["guard_margin"],
                counterfactual_offsets=cfg["training"]["event_pretraining"]["counterfactual_offsets_tau"],
            )
            optimizer.zero_grad(set_to_none=True)
            terms["total"].backward()
            _clip(model, cfg)
            optimizer.step()
            total += terms["total"].detach()
            batches += 1
        train_average = float(total) / max(batches, 1)
        training_seconds = time.time() - started
        validation_started = time.time()
        val_terms = _rollout_validation(model, val_data, val_windows, stage, cfg, device, dtype, use_ripple)
        validation_seconds = time.time() - validation_started
        epoch_timing = {"training": training_seconds, "validation": validation_seconds}
        val_loss = val_terms["total"]
        scheduler.step(val_loss)
        best, stall = _early_stop(best, stall, val_loss, cfg)
        last_epoch = epoch + 1
        if best_callback is not None:
            best_callback(model, last_epoch, val_loss)
        print(f"  [{stage['name']}] epoch {last_epoch}/{epochs} | train {train_average:.3e} | val {val_loss:.3e} | {time.time() - started:.1f}s | train {training_seconds:.1f}s | validation {validation_seconds:.1f}s | batches {batches}")
        if checkpoint_callback and last_epoch % checkpoint_every == 0:
            checkpoint_callback(
                last_epoch, best, stall,
                {"val_rollout_loss": best, "last_val_loss_terms": val_terms, "epochs": last_epoch,
                 "training_batch_size": training_batch_size,
                 "optimizer_updates": _optimizer_updates(optimizer), "last_epoch_timing_seconds": epoch_timing},
                optimizer.state_dict(), scheduler.state_dict(),
            )
        if stall >= cfg["training"]["early_stopping_patience_epochs"]:
            break
    return {"val_rollout_loss": best, "last_val_loss_terms": val_terms, "epochs": last_epoch, "_stall": stall,
            "training_batch_size": training_batch_size,
            "optimizer_updates": _optimizer_updates(optimizer), "last_epoch_timing_seconds": epoch_timing,
            "_optimizer_state": optimizer.state_dict(),
            "_scheduler_state": scheduler.state_dict()}
