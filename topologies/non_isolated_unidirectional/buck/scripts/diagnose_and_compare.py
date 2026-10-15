"""Small checkpoint diagnostic and equal-training-time batch pilot on CUDA.

Writes only to a new output directory; never changes the source experiment.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train import DEFAULT_CONFIG, DEFAULT_DATA_DIR, _config_sha256
from training.checkpoint import load_checkpoint
from training.data import load_data, normalize, make_windows, collate
from training.model import NeuralEventHybridODE, MODE_OFF_DIODE, MODE_IDLE
from training.hybrid_rk4 import hybrid_rk4, rk4_step
from training.losses import rollout_losses


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def selected_indices(data, per_voltage):
    chosen = []
    for voltage in torch.unique(data["input_voltage"], sorted=True):
        indices = (data["input_voltage"] == voltage).nonzero().flatten()
        ordered = indices[torch.argsort(data["initial_state_norm"][indices, 0])]
        positions = torch.linspace(0, len(ordered) - 1, min(per_voltage, len(ordered))).long()
        chosen.extend(ordered[positions].tolist())
    return chosen


def diagnosis(model, data, indices, output, label, cfg):
    model.eval()
    batch = collate(data, [(i, 0) for i in indices], 20, "cuda", torch.float32)
    with torch.no_grad():
        result = hybrid_rk4(model, batch["y0"], batch["vs"], 20)
        # Count roots on all OFF starts in parallel, including after first roots.
        x = result.states[50::100].reshape(-1, 2)
        vs = batch["vs"].expand(20, -1, -1).reshape(-1, 1)
        g0 = model.event_value(x, vs)
        roots = torch.zeros_like(g0, dtype=torch.long)
        for _ in range(50):
            x = rk4_step(model, MODE_OFF_DIODE, x, vs, 0.01)
            g1 = model.event_value(x, vs)
            roots += ((g0 > 0) & (g1 <= 0)).long()
            g0 = g1
        if not torch.isfinite(x).all() or not torch.isfinite(g0).all():
            raise FloatingPointError("Nonfinite extended OFF diagnostic")
    roots = roots.reshape(20, len(indices)).T.cpu()
    states, target = result.states.detach().cpu(), batch["target"].cpu()
    modes = result.interval_mode.cpu()
    event_mask = result.event_mask.cpu()
    guards = result.invalid_initial_guard.cpu()
    event_error = (result.event_time_tau - batch["event_time_tau"]).abs().cpu()
    refs = cfg["normalization"]
    scale = torch.tensor([refs["current_ref_a"], refs["voltage_ref_v"]])
    report = {"indices": indices, "voltages": data["input_voltage"][indices].tolist(),
              "trajectory_ids": [str(data["trajectory_ids"][i]) for i in indices], "horizons": {}}
    for cycles in (1, 5, 10, 20):
        pred, truth = states[:cycles * 100 + 1], target[:cycles * 100 + 1]
        error = pred - truth
        physical_rmse = ((error * scale).square().mean(dim=(0, 1))).sqrt()
        idle = pred[:-1, :, 0][modes[:cycles * 100] == MODE_IDLE].abs() * scale[0]
        valid = event_mask[:, :cycles] & ~guards[:, :cycles] & (roots[:, :cycles] == 1)
        event = event_error[:, :cycles][valid]
        report["horizons"][str(cycles)] = {
            "normalized_rmse_pct": float(error.square().mean(dim=(0, 2)).sqrt().mean() * 100),
            "current_rmse_a": float(physical_rmse[0]), "voltage_rmse_v": float(physical_rmse[1]),
            "minimum_current_a": float(pred[..., 0].min() * scale[0]),
            "event_time_mae_us": float(event.mean() * refs["time_ref_s"] * 1e6) if event.numel() else None,
            "event_miss_rate": float((roots[:, :cycles] == 0).float().mean()),
            "extra_roots_mean": float((roots[:, :cycles] - 1).clamp_min(0).float().mean()),
            "invalid_guard_rate": float(guards[:, :cycles].float().mean()),
            "idle_current_mean_a": float(idle.mean()) if idle.numel() else None,
            "idle_current_max_a": float(idle.max()) if idle.numel() else None,
        }
    fig, axes = plt.subplots(len(indices), 2, figsize=(12, 2.4 * len(indices)), squeeze=False)
    t = torch.arange(len(states)).numpy() / 100
    for row, index in enumerate(indices):
        for channel, unit in enumerate(("iL (A)", "vC (V)")):
            ax = axes[row, channel]
            ax.plot(t, target[:, row, channel].numpy() * float(scale[channel]), label="reference", lw=1)
            ax.plot(t, states[:, row, channel].numpy() * float(scale[channel]), label="prediction", lw=0.8, alpha=0.8)
            ax.set_title(f"{float(data['input_voltage'][index]):g} V, trajectory {index}")
            ax.set_ylabel(unit)
            ax.set_xlabel("switching cycles")
            ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(output / f"{label}.png", dpi=130)
    plt.close(fig)
    write_json(output / f"{label}.json", report)
    print(json.dumps({"diagnosis": label, "horizons": report["horizons"]}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--batches", nargs="+", type=int, default=[16, 32, 64])
    args = parser.parse_args()
    if args.seconds <= 0 or any(b <= 0 for b in args.batches):
        parser.error("Positive time and batch sizes required")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    source_bytes = args.checkpoint.read_bytes()
    snapshot = args.out_dir / "source_checkpoint.pt"
    snapshot.write_bytes(source_bytes)
    checkpoint = load_checkpoint(str(snapshot))
    cfg = json.loads(Path(DEFAULT_CONFIG).read_text(encoding="utf-8"))
    profile_path = Path(DEFAULT_CONFIG).parent / cfg["experiments"]["config_directory"] / (checkpoint["idle_profile"] + ".json")
    assert checkpoint["config_sha256"] == _config_sha256(DEFAULT_CONFIG, str(profile_path))
    cfg["training"]["loss_weights"].update(json.loads(profile_path.read_text(encoding="utf-8"))["loss_weights"])
    stage = next(s for s in cfg["training"]["stages"] if s["name"] == checkpoint["stage"])
    if not stage["window_cycles"]:
        raise ValueError("Requires rollout-stage checkpoint")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    splits = normalize(load_data(DEFAULT_DATA_DIR), cfg["normalization"])
    train_indices = selected_indices(splits["train"], 1)
    val_indices = selected_indices(splits["validation"], 2)

    def new_model():
        model = NeuralEventHybridODE(cfg["model"], checkpoint["idle_profile"]).cuda()
        model.load_state_dict(checkpoint["model_state"])
        return model

    summary = {"checkpoint_sha256": hashlib.sha256(source_bytes).hexdigest(),
               "stage": checkpoint["stage"], "epoch": checkpoint["epoch"],
               "budget_training_seconds": args.seconds, "validation_indices": val_indices,
               "limitations": "Single seed, short pilot, small fixed validation subset; not final acceptance. Training budget excludes validation/save time, may overshoot by one update. No epoch-based LR scheduler in the pilot; checkpoint LRs held constant.",
               "arms": []}
    model = new_model()
    summary["baseline_train"] = diagnosis(model, splits["train"], train_indices, args.out_dir, "baseline_train", cfg)
    summary["baseline_validation"] = diagnosis(model, splits["validation"], val_indices, args.out_dir, "baseline_validation", cfg)
    write_json(args.out_dir / "summary.json", summary)
    del model
    val_batch = collate(splits["validation"], [(i, 0) for i in val_indices], stage["window_cycles"], "cuda", torch.float32)
    windows = make_windows(splits["train"], stage["window_cycles"])

    def terms_for(model, batch):
        result = hybrid_rk4(model, batch["y0"], batch["vs"], stage["window_cycles"])
        return rollout_losses(model, result, batch, cfg["training"]["loss_weights"],
                              use_ripple=stage["name"] == "twenty_cycle",
                              guard_margin=cfg["training"]["event_pretraining"]["guard_margin"],
                              counterfactual_offsets=cfg["training"]["event_pretraining"]["counterfactual_offsets_tau"])

    def validate(model):
        model.eval()
        with torch.no_grad():
            terms = {k: float(v) for k, v in terms_for(model, val_batch).items()}
        if not all(math.isfinite(v) for v in terms.values()):
            raise FloatingPointError("Nonfinite validation loss")
        return terms

    for size in args.batches:
        arm_dir = args.out_dir / f"batch_{size}"
        arm_dir.mkdir()
        model = new_model()
        optimizer = torch.optim.AdamW([
            {"params": list(model.f_on.parameters()) + list(model.f_off.parameters()) + list(model.f_idle.parameters())},
            {"params": model.event_network.parameters()},
        ])
        optimizer.load_state_dict(deepcopy(checkpoint["optimizer_state"]))
        generator = torch.Generator().manual_seed(20260912)
        order = torch.randperm(len(windows), generator=generator).tolist()
        cursor = 0
        initial = validate(model)
        best = initial["total"]
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, arm_dir / "best_weights.pt")
        arm = {"batch": size, "initial_validation": initial, "checks": [], "best_update": 0,
               "best_validation_loss": best, "learning_rates": [g["lr"] for g in optimizer.param_groups]}
        elapsed = 0.0
        updates = examples = 0
        next_check = 60.0
        wall_started = time.perf_counter()
        print(f"batch={size} starting, initial_validation={best:.6e}", flush=True)
        while elapsed < args.seconds:
            torch.cuda.synchronize()
            started = time.perf_counter()
            if cursor >= len(order):
                order = torch.randperm(len(windows), generator=generator).tolist()
                cursor = 0
            selected = order[cursor:cursor + size]
            cursor += len(selected)
            batch = collate(splits["train"], [windows[i] for i in selected], stage["window_cycles"], "cuda", torch.float32)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            terms = terms_for(model, batch)
            terms["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            elapsed += time.perf_counter() - started
            updates += 1
            examples += len(selected)
            del terms, batch
            if elapsed >= next_check or elapsed >= args.seconds:
                values = validate(model)
                record = {"updates": updates, "examples": examples, "training_seconds": elapsed, "validation": values}
                arm["checks"].append(record)
                if values["total"] < best:
                    best = values["total"]
                    arm["best_update"] = updates
                    arm["best_validation_loss"] = best
                    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, arm_dir / "best_weights.pt")
                write_json(arm_dir / "progress.json", arm)
                print(json.dumps({"batch": size, **record}), flush=True)
                next_check += 60.0
        arm.update(training_seconds=elapsed, wall_seconds=time.perf_counter() - wall_started,
                   updates=updates, examples=examples)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, arm_dir / "last_weights.pt")
        model.load_state_dict(torch.load(arm_dir / "best_weights.pt", map_location="cuda", weights_only=True))
        arm["best_diagnostics"] = diagnosis(model, splits["validation"], val_indices, arm_dir, "best_validation", cfg)
        write_json(arm_dir / "progress.json", arm)
        summary["arms"].append(arm)
        write_json(args.out_dir / "summary.json", summary)
        del model, optimizer
    print("FINISHED: " + str(args.out_dir), flush=True)


if __name__ == "__main__":
    main()
