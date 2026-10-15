#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Buck DCM Neural Event ODE 训练入口。

本文件负责组装配置、数据、模型、训练阶段、checkpoint 和评估；
具体的数据处理、网络结构、积分器和损失实现位于 ``training/`` 中。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

# 同时兼容 ``python -m ...train`` 的包调用和 ``python train.py`` 的脚本调用。
if __package__:
    from .training.checkpoint import load_checkpoint, save_checkpoint
    from .training.data import load_data, normalize
    from .training.evaluate import (
        evaluate_all,
        evaluate_by_voltage,
        evaluate_by_initial_bin,
        evaluate_half_step,
        assess_acceptance,
        print_by_voltage,
        print_report,
        save_plot,
        save_event_rows,
        save_event_plot,
        save_voltage_plots,
    )
    from .training.loops import event_pretrain_gate, run_event_pretrain, run_mode_dynamics, run_rollout_stage
    from .training.model import NeuralEventHybridODE
    from .training.run_lock import exclusive_run
    from .training.batching import parse_stage_batch, resolve_training_batch_sizes
    from .training.best_weights import BestValidationWeights
else:
    from training.checkpoint import load_checkpoint, save_checkpoint
    from training.data import load_data, normalize
    from training.evaluate import (
        evaluate_all,
        evaluate_by_voltage,
        evaluate_by_initial_bin,
        evaluate_half_step,
        assess_acceptance,
        print_by_voltage,
        print_report,
        save_plot,
        save_event_rows,
        save_event_plot,
        save_voltage_plots,
    )
    from training.loops import event_pretrain_gate, run_event_pretrain, run_mode_dynamics, run_rollout_stage
    from training.model import NeuralEventHybridODE
    from training.run_lock import exclusive_run
    from training.batching import parse_stage_batch, resolve_training_batch_sizes
    from training.best_weights import BestValidationWeights


BUCK_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = str(BUCK_ROOT / "configs" / "hybrid_event_matlab_v3" / "input_voltage_relative.json")
DEFAULT_DATA_DIR = str(BUCK_ROOT / "data" / "processed" / "hybrid_event_matlab_v3")
DEFAULT_OUT_DIR = str(BUCK_ROOT / "artifacts" / "hybrid_event_matlab_v3")


def _config_sha256(*config_paths: str) -> str:
    """计算本次训练所有配置文件的联合 SHA-256 指纹。

    checkpoint 会保存该指纹；恢复训练时只有当当前配置完全一致才允许
    继续，以避免无意中混用不同实验的权重。

    Args:
        config_paths: 按顺序参与指纹计算的配置文件路径。

    Returns:
        十六进制 SHA-256 字符串。
    """
    digest = hashlib.sha256()
    for path in config_paths:
        digest.update(Path(path).read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def _tupleize(value):
    """递归地把 list 恢复为 tuple，以满足 Python RNG 状态的输入格式。"""
    return tuple(_tupleize(item) for item in value) if isinstance(value, list) else value


def _device(requested: str) -> str:
    """优先 CUDA、其次 MPS；CPU 仅显式选择，不静默降级。"""
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
        raise RuntimeError("未检测到 CUDA/MPS GPU；检查 PyTorch 安装，或显式指定 --device cpu")
    if requested == "cpu":
        return requested
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("当前 Python 无法使用 MPS；请检查 Mac Metal/PyTorch 环境")
        return requested
    if requested == "cuda" or (requested.startswith("cuda:") and requested[5:].isdigit()):
        index = int(requested.split(":")[1]) if ":" in requested else 0
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用；请安装支持 NVIDIA GPU 的 PyTorch 并检查驱动")
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA 设备 {index} 不存在；可用设备数 {torch.cuda.device_count()}")
        return f"cuda:{index}"
    raise ValueError("设备必须为 auto、mps、cuda、cuda:N 或 cpu")


def _save_stage(
    path, model, stage_name, epoch, stage_report, completed, cfg, config_hash,
    *, best, stall, optimizer_state=None, scheduler_state=None,
):
    """保存一个可精确续训的阶段 checkpoint。

    除模型、优化器和 scheduler 外，还保存 Python、NumPy、PyTorch 的随机数
    状态、已完成阶段和配置指纹。因此 ``--resume`` 不只是重新加载权重，
    而是尽量从同一训练状态继续。

    Args:
        path: checkpoint 输出路径。
        model: 当前 Neural Event ODE 模型。
        stage_name: 当前课程阶段名称。
        epoch: 当前阶段已完成的 epoch。
        stage_report: 各阶段累计的训练报告。
        completed: 已完成阶段名称列表。
        cfg: 合并实验覆盖后的项目配置。
        config_hash: 由 ``_config_sha256`` 生成的配置指纹。
        best: 当前阶段的最佳验证指标。
        stall: 未取得有效改善的连续 epoch 数。
        optimizer_state: 可选的优化器状态。
        scheduler_state: 可选的学习率调度器状态。
    """
    # RNG 状态必须和模型状态一起保存，才能在续训时复现采样顺序。
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if next(model.parameters()).device.type == "mps":
        rng_state["torch_mps"] = torch.mps.get_rng_state()
    if next(model.parameters()).device.type == "cuda":
        rng_state["torch_cuda"] = torch.cuda.get_rng_state(next(model.parameters()).device)
    save_checkpoint(
        path,
        model=model,
        stage=stage_name,
        epoch=epoch,
        best=best,
        stall=stall,
        stage_report=stage_report,
        completed_stages=completed,
        training_batch_sizes=cfg["runtime"].get("training_batch_sizes"),
        config_sha256=config_hash,
        normalization=cfg["normalization"],
        loss_weights=cfg["training"]["loss_weights"],
        hybrid_settings={
            "duty": cfg["circuit"]["duty_cycle"],
            "step_size_tau": 0.01,
            "samples_per_cycle": 100,
            "secant_refinement_iterations": 2,
        },
        idle_profile=model.idle_profile,
        rng_state=rng_state,
        optimizer_state=optimizer_state,
        scheduler_state=scheduler_state,
    )


def _best_metric(report):
    """从不同类型的阶段报告中提取统一的“最佳指标”。

    模式动力学、事件预训练和 rollout 使用不同的验证指标键，该函数为
    checkpoint 保存逻辑屏蔽这种差异。
    """
    return next(
        report[key]
        for key in ("val_rollout_loss", "val_mode_derivative_mse", "val_event_pretrain_loss")
        if key in report
    )


def _write_json(path, value):
    """以 UTF-8 和易读缩进写入 JSON，并保留中文字符。"""
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_run_artifacts(out_dir, profile, metrics, by_voltage, event_rows, cfg, experiment):
    """写出一次训练或评估的结构化产物。

    产物包含总体指标、分电压指标、半步长收敛性、本次实际配置、事件
    CSV/图表，以及 soft/hard IDLE 两种 profile 的汇总对比。

    Args:
        out_dir: 本次运行的产物目录。
        profile: 当前 IDLE 约束 profile 名称。
        metrics: 总体训练和评估结果。
        by_voltage: 按输入电压分组的评估结果。
        event_rows: 逐事件诊断记录。
        cfg: 本次运行的完整项目配置。
        experiment: 当前 profile 的实验覆盖配置。
    """
    # 先保存单次运行的可机器读结果和事件诊断。
    _write_json(os.path.join(out_dir, "metrics.json"), metrics)
    _write_json(os.path.join(out_dir, "metrics_by_voltage.json"), by_voltage)
    _write_json(os.path.join(out_dir, "half_step_convergence.json"), metrics["half_step"])
    _write_json(os.path.join(out_dir, "run_config.json"), {"project": cfg, "experiment": experiment})
    save_event_rows(os.path.join(out_dir, "events.csv"), event_rows)
    save_event_plot(os.path.join(out_dir, "events_by_voltage.png"), event_rows)

    # 再把当前 profile 合并到 soft/hard IDLE 对照报告中。
    run_dir = Path(out_dir)
    comparison_root = run_dir.parent if run_dir.name == profile else run_dir
    comparison_path = comparison_root / "idle_soft_vs_hard.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8")) if comparison_path.is_file() else {"profiles": {}}
    profile_report = dict(comparison["profiles"].get(profile, {}))
    profile_report.update({
        "evaluation": metrics["evaluation"],
        "half_step": metrics["half_step"],
        "timing_seconds": metrics["timing_seconds"],
    })
    if "stages" in metrics:
        profile_report["stages"] = metrics["stages"]
    comparison["profiles"][profile] = profile_report
    comparison["complete"] = all(name in comparison["profiles"] for name in ("soft_idle_loss", "hard_idle_constraint"))
    _write_json(comparison_path, comparison)
    _write_json(run_dir / "idle_soft_vs_hard.json", comparison)


def main():
    """解析命令行并编排 Buck DCM 的训练、续训或纯评估流程。

    主流程依次为：加载配置与实验 profile、固定随机种子、加载数据、
    构建模型、可选恢复 checkpoint、执行分阶段课程训练，最后评估并写出
    产物。使用 ``--evaluate-only`` 时会跳过训练阶段。
    """
    # 1. 命令行参数：命令行值优先于 project.json 中的默认值。
    parser = argparse.ArgumentParser(description="Buck DCM Neural Event ODE 训练")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None, help="auto | mps | cuda | cuda:N | cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--idle-profile", choices=["soft_idle_loss", "hard_idle_constraint"], default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-window-cycles", type=int, default=20)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--rollout-batch-size", type=int, default=None,
                        help="Override rollout training batch size; validation batching stays unchanged")
    parser.add_argument("--stage-batch-size", type=parse_stage_batch, action="append", default=[],
                        metavar="STAGE=N", help="Per-stage training batch; repeatable, overrides global/saved batch")
    parser.add_argument(
        "--event-pretrain-epochs",
        type=int,
        default=None,
        help="仅覆盖 event_pretrain 的目标 epoch；用于修复后续训而不改变 rollout 轮数",
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.rollout_batch_size is not None and args.rollout_batch_size < 1:
        parser.error("--rollout-batch-size must be positive")

    # 2. 配置组装：先读主配置，再用选中 profile 的 loss_weights 覆盖它。
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    try:
        resolve_training_batch_sizes(cfg["training"]["stages"], global_batch=args.rollout_batch_size,
                                     stage_batches=args.stage_batch_size)
    except ValueError as exc:
        parser.error(str(exc))
    profile = args.idle_profile or cfg["experiments"]["default"]
    if profile not in cfg["model"]["supported_idle_profiles"]:
        raise ValueError(f"不支持的 IDLE profile: {profile}")
    experiment_path = Path(args.config).parent / cfg["experiments"]["config_directory"] / f"{profile}.json"
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    cfg["training"]["loss_weights"].update(experiment["loss_weights"])
    config_hash = _config_sha256(args.config, str(experiment_path))
    network_normalization = cfg["model"].get("network_normalization", "fixed_reference")
    output_root = Path(DEFAULT_OUT_DIR)
    if network_normalization != "fixed_reference":
        output_root = output_root / "input_voltage_relative_conditioned_v2"
    args.out_dir = args.out_dir or str(output_root / profile)
    with exclusive_run(args.out_dir):
        _run(args, cfg, profile, experiment, config_hash)


def _run(args, cfg, profile, experiment, config_hash):
    network_normalization = cfg["model"].get("network_normalization", "fixed_reference")
    device = _device(args.device or cfg["runtime"]["device"])
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
        # 保持 float32 数值口径，不自动启用 TF32/AMP。
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    dtype = getattr(torch, cfg["runtime"]["training_dtype"])
    seed = args.seed if args.seed is not None else cfg["training"]["model_seeds"][0]
    # 统一固定所有会影响训练采样和初始化的随机数源。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "mps":
        torch.mps.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    os.makedirs(args.out_dir, exist_ok=True)
    _write_json(os.path.join(args.out_dir, "runtime_environment.json"), {
        "device": device, "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda, "dtype": str(dtype),
        "gpu_name": torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else device,
    })

    # 3. 数据入口会先审计 DCM v3 schema，然后按配置生成无量纲字段。
    splits = normalize(load_data(args.data_dir), cfg["normalization"])
    if network_normalization == "input_voltage_relative":
        for split_name, data in splits.items():
            if not torch.all(data["input_voltage"] > 0):
                raise ValueError(f"{split_name}: 输入电压相对归一化要求 Vin > 0")
    if "train" not in splits or "validation" not in splits:
        raise FileNotFoundError("data-dir 必须包含 train.npz 和 validation.npz")
    if args.smoke:
        splits = {name: value for name, value in splits.items() if name in ("train", "validation")}
    print(f"config={args.config}\ndata={args.data_dir}\ndevice={device}\nprofile={profile}")
    print("splits:", {name: tuple(data["states"].shape) for name, data in splits.items()})

    # 4. 构建三模式 Neural Event ODE，并根据需要恢复权重与训练状态。
    model = NeuralEventHybridODE(cfg["model"], idle_profile=profile).to(device=device, dtype=dtype)
    checkpoint = None
    raw_weights = None
    if args.resume:
        if args.evaluate_only:
            candidate = torch.load(args.resume, map_location="cpu", weights_only=False)
            if isinstance(candidate, dict) and "model_state" in candidate and candidate.get("schema_version") != "hybrid_event_matlab_v3":
                raise ValueError("checkpoint 不是 hybrid_event_matlab_v3，拒绝混用 V2 权重")
            checkpoint = candidate if isinstance(candidate, dict) and candidate.get("schema_version") == "hybrid_event_matlab_v3" else None
            raw_weights = candidate.get("model_state", candidate) if isinstance(candidate, dict) else candidate
        else:
            checkpoint = load_checkpoint(args.resume)
    if checkpoint:
        # 续训必须同时匹配 schema、配置、IDLE profile 和模型结构。
        if checkpoint["config_sha256"] != config_hash:
            raise ValueError("checkpoint 与当前 project.json 不匹配")
        if checkpoint["idle_profile"] != profile:
            raise ValueError("checkpoint 的 IDLE profile 与当前运行不匹配")
        if checkpoint["model_signature"] != model.architecture_signature():
            raise ValueError("checkpoint 的模型结构签名与当前配置不匹配")
        model.load_state_dict(checkpoint["model_state"])
        # 恢复 RNG 状态，使后续采样顺序尽可能与中断前一致。
        rng = checkpoint.get("rng_state", {})
        if "python" in rng:
            random.setstate(_tupleize(rng["python"]))
        if "numpy" in rng:
            state = rng["numpy"]
            np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), state[2], state[3], state[4]))
        if "torch_cpu" in rng:
            torch.set_rng_state(rng["torch_cpu"])
        if device == "mps" and "torch_mps" in rng:
            torch.mps.set_rng_state(rng["torch_mps"])
        if device.startswith("cuda") and "torch_cuda" in rng:
            torch.cuda.set_rng_state(rng["torch_cuda"], torch.device(device))
        backend_rng = "torch_cuda" if device.startswith("cuda") else "torch_" + device
        if backend_rng not in rng:
            print("当前设备没有对应的已保存 RNG 状态，使用当前种子；跨后端续训不保证逐步一致")
        completed = list(checkpoint["completed_stages"])
        stage_report = dict(checkpoint["stage_report"])
    else:
        completed, stage_report = [], {}

    cfg["runtime"]["training_batch_sizes"] = resolve_training_batch_sizes(
        cfg["training"]["stages"], checkpoint, args.rollout_batch_size, args.stage_batch_size)
    if not args.evaluate_only:
        remaining_batches = {name: size for name, size in cfg["runtime"]["training_batch_sizes"].items()
                             if name not in completed}
        print("rollout training batches:", remaining_batches)

    # 5A. 纯评估路径：加载权重后直接计算波形、事件、分组和收敛性指标。
    if args.evaluate_only:
        weights = checkpoint["model_state"] if checkpoint else (raw_weights if raw_weights is not None else torch.load(
            os.path.join(args.out_dir, "best.pt"), map_location="cpu", weights_only=False
        ))
        model.load_state_dict(weights)
        rollout_cycles = [1, 10] if args.smoke else cfg["evaluation"]["rollout_cycles"]
        evaluation_started = time.perf_counter()
        event_rows = []
        result = evaluate_all(model, splits, rollout_cycles, device, cfg["normalization"], event_rows)
        print_report(result)
        by_voltage = {
            name: evaluate_by_voltage(model, data, rollout_cycles, device, cfg["normalization"])
            for name, data in splits.items()
        }
        half_step = evaluate_half_step(model, splits["validation"], rollout_cycles[:2], device, cfg["normalization"])
        initial_bins = {
            name: evaluate_by_initial_bin(model, data, rollout_cycles, device, cfg["normalization"])
            for name, data in splits.items()
        }
        acceptance = assess_acceptance(result, half_step, cfg)
        metrics = {"evaluation": result, "by_voltage": by_voltage, "half_step": half_step, "by_initial_state_bin": initial_bins, "acceptance": acceptance, "timing_seconds": {"evaluation": time.perf_counter() - evaluation_started}}
        _write_json(os.path.join(args.out_dir, "metrics_eval.json"), metrics)
        _write_run_artifacts(args.out_dir, profile, metrics, by_voltage, event_rows, cfg, experiment)
        return

    # 5B. 训练路径：依配置顺序执行模式动力学、事件预训练和 rollout 课程。
    training_started = time.perf_counter()
    checkpoint_path = os.path.join(args.out_dir, "checkpoint.pt")
    checkpoint_every = cfg["training"]["checkpoint_interval_epochs"]
    last_trained_stage = checkpoint["stage"] if checkpoint else None
    for stage in cfg["training"]["stages"]:
        stage = dict(stage)
        name = stage["name"]
        if stage["window_cycles"]:
            stage["training_batch_size"] = cfg["runtime"]["training_batch_sizes"][name]
        if name in completed:
            print(f"[{name}] 已完成，跳过")
            continue
        if stage["window_cycles"] and stage["window_cycles"] > args.max_window_cycles:
            print(f"[{name}] 超过 --max-window-cycles，跳过")
            continue
        # 只恢复 checkpoint 中正在进行且尚未完成的当前阶段。
        last_trained_stage = name
        resume = None
        if checkpoint and checkpoint["stage"] == name and name not in completed:
            resume = {
                "epoch": checkpoint["epoch"],
                "best": checkpoint["best"],
                "stall": checkpoint["stall"],
                "optimizer_state": checkpoint["optimizer_state"],
                "scheduler_state": checkpoint["scheduler_state"],
                "last_val_loss_terms": stage_report.get(name, {}).get("last_val_loss_terms", {}),
            }

        def save_progress(epoch, best, stall, current_report, optimizer_state, scheduler_state):
            """将训练循环回调传回的中间状态写入当前 checkpoint。"""
            reports = dict(stage_report)
            reports[name] = current_report
            _save_stage(
                checkpoint_path, model, name, epoch, reports, completed, cfg, config_hash,
                best=best, stall=stall,
                optimizer_state=optimizer_state, scheduler_state=scheduler_state,
            )

        # smoke/命令行可缩短阶段；正式运行则由每个 stage 的 max_epochs 决定。
        epochs = args.epochs or (cfg["training"]["smoke"]["epochs_per_stage"] if args.smoke else None)
        if name == "event_pretrain" and args.event_pretrain_epochs is not None:
            epochs = args.event_pretrain_epochs
        # 三类阶段使用不同的数据组织和优化目标。
        if name == "mode_dynamics":
            report = run_mode_dynamics(
                model, splits["train"], splits["validation"], stage, cfg, device, dtype,
                max_batches=2 if args.smoke else None, max_epochs=epochs,
                resume=resume, checkpoint_every=checkpoint_every,
                checkpoint_callback=save_progress,
            )
        elif name == "event_pretrain":
            report = run_event_pretrain(
                model, splits["train"], splits["validation"], stage, cfg, device, dtype,
                max_batches=2 if args.smoke else None, max_epochs=epochs,
                resume=resume, checkpoint_every=checkpoint_every,
                checkpoint_callback=save_progress,
            )
            gate = report.get("validation_gate")
            if gate is None:
                gate = event_pretrain_gate(model, splits["validation"], cfg, device, dtype)
                print(
                    "  [event_pretrain gate] "
                    + json.dumps(gate, ensure_ascii=False)
                )
            report["validation_gate"] = gate
            # 正式训练必须通过事件安全门槛后才能进入联合 rollout。
            if not gate["passed"] and not args.smoke:
                optimizer_state = report.pop("_optimizer_state", None)
                scheduler_state = report.pop("_scheduler_state", None)
                stall = report.pop("_stall")
                stage_report[name] = report
                _save_stage(
                    checkpoint_path, model, name, report["epochs"], stage_report,
                    completed, cfg, config_hash,
                    best=_best_metric(report), stall=stall,
                    optimizer_state=optimizer_state, scheduler_state=scheduler_state,
                )
                raise RuntimeError("event_pretrain validation gate 未通过，停止进入联合 rollout")
            if not gate["passed"]:
                print("  警告：smoke 模式允许 event_pretrain gate 未通过")
        else:
            best_selection = BestValidationWeights(args.out_dir, name, config_hash)
            report = run_rollout_stage(
                model, splits["train"], splits["validation"], stage, cfg, device, dtype,
                max_windows=args.max_windows or (32 if args.smoke else None), max_epochs=epochs,
                resume=resume, checkpoint_every=checkpoint_every,
                checkpoint_callback=save_progress,
                best_callback=best_selection.consider,
            )
        # 报告中以下划线开头的字段为 checkpoint 内部状态，不写入对外指标。
        optimizer_state = report.pop("_optimizer_state", None)
        scheduler_state = report.pop("_scheduler_state", None)
        stall = report.pop("_stall")
        stage_report[name] = report
        completed.append(name)
        _save_stage(
            checkpoint_path, model, name, report["epochs"], stage_report,
            completed, cfg, config_hash,
            best=_best_metric(report), stall=stall,
            optimizer_state=optimizer_state, scheduler_state=scheduler_state,
        )

    # 6. 所有允许执行的阶段结束后，保存纯权重并运行统一评估。
    last_path = os.path.join(args.out_dir, "last.pt")
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, last_path)
    selection = BestValidationWeights(args.out_dir, last_trained_stage, config_hash)
    if selection.path.is_file():
        selected = selection.load()
        model.load_state_dict(selected["weights"])
        _write_json(os.path.join(args.out_dir, "best_selection.json"), {
            "stage": selected["stage"], "epoch": selected["epoch"],
            "validation_loss": selected["metric"], "source": str(selection.path),
        })
    else:
        _write_json(os.path.join(args.out_dir, "best_selection.json"), {
            "stage": last_trained_stage, "source": last_path,
            "reason": "No recorded validation-best weights; using final weights",
        })
    best_path = os.path.join(args.out_dir, "best.pt")
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, best_path)
    rollout_cycles = [1, 10] if args.smoke else cfg["evaluation"]["rollout_cycles"]
    training_seconds = time.perf_counter() - training_started
    evaluation_started = time.perf_counter()
    event_rows = []
    result = evaluate_all(model, splits, rollout_cycles, device, cfg["normalization"], event_rows)
    print_report(result)
    by_voltage = {
        name: evaluate_by_voltage(model, data, rollout_cycles, device, cfg["normalization"])
        for name, data in splits.items()
    }
    if "test_voltage_extrapolation" in by_voltage:
        print_by_voltage(by_voltage["test_voltage_extrapolation"])
        save_voltage_plots(model, splits["test_voltage_extrapolation"], os.path.join(args.out_dir, "waveforms"), "voltage_extra")
    half_step = evaluate_half_step(model, splits["validation"], rollout_cycles[:2], device, cfg["normalization"])
    initial_bins = {
        name: evaluate_by_initial_bin(model, data, rollout_cycles, device, cfg["normalization"])
        for name, data in splits.items()
    }
    acceptance = assess_acceptance(result, half_step, cfg)
    metrics = {"config": args.config, "data_dir": args.data_dir, "seed": seed, "idle_profile": profile, "stages": stage_report, "evaluation": result, "by_voltage": by_voltage, "half_step": half_step, "by_initial_state_bin": initial_bins, "acceptance": acceptance, "timing_seconds": {"training": training_seconds, "evaluation": time.perf_counter() - evaluation_started}}
    _write_run_artifacts(args.out_dir, profile, metrics, by_voltage, event_rows, cfg, experiment)
    save_plot(model, splits, os.path.join(args.out_dir, "steady_state_compare.png"))
    print(f"已保存权重: {best_path}")


if __name__ == "__main__":
    main()
