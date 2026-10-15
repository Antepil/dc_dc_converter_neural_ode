"""Benchmark real checkpoint/data batches without changing the saved experiment."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train import DEFAULT_CONFIG, DEFAULT_DATA_DIR
from training.checkpoint import load_checkpoint
from training.data import load_data, normalize, make_windows, collate
from training.model import NeuralEventHybridODE
from training.hybrid_rk4 import hybrid_rk4
from training.losses import rollout_losses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batches", nargs="+", type=int, default=[16, 64, 128])
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    checkpoint = load_checkpoint(args.checkpoint)
    experiment = json.loads((Path(args.config).parent / cfg["experiments"]["config_directory"] /
                             (checkpoint["idle_profile"] + ".json")).read_text(encoding="utf-8"))
    cfg["training"]["loss_weights"].update(experiment["loss_weights"])
    stage = next(s for s in cfg["training"]["stages"] if s["name"] == checkpoint["stage"])
    if not stage["window_cycles"]:
        raise ValueError("Benchmark requires a rollout-stage checkpoint")
    data = normalize(load_data(args.data_dir), cfg["normalization"])["train"]
    windows = make_windows(data, stage["window_cycles"])
    torch.manual_seed(0)
    order = torch.randperm(len(windows)).tolist()
    print(json.dumps({"stage": checkpoint["stage"], "epoch": checkpoint["epoch"],
                      "windows": len(windows), "gpu": torch.cuda.get_device_name()}), flush=True)
    for size in args.batches:
        model = NeuralEventHybridODE(cfg["model"], checkpoint["idle_profile"]).cuda()
        model.load_state_dict(checkpoint["model_state"])
        optimizer = torch.optim.AdamW([
            {"params": list(model.f_on.parameters()) + list(model.f_off.parameters()) + list(model.f_idle.parameters())},
            {"params": model.event_network.parameters()},
        ])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        batch = collate(data, [windows[i] for i in order[:size]], stage["window_cycles"], "cuda", torch.float32)

        def step():
            optimizer.zero_grad(set_to_none=True)
            result = hybrid_rk4(model, batch["y0"], batch["vs"], stage["window_cycles"], duty=cfg["circuit"]["duty_cycle"])
            terms = rollout_losses(model, result, batch, cfg["training"]["loss_weights"],
                                   use_ripple=stage["name"] == cfg["training"]["ripple_loss"]["enabled_stage"],
                                   guard_margin=cfg["training"]["event_pretraining"]["guard_margin"],
                                   counterfactual_offsets=cfg["training"]["event_pretraining"]["counterfactual_offsets_tau"])
            terms["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip_norm"])
            optimizer.step()
            return float(terms["total"].detach())

        step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            loss = step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            if not math.isfinite(loss):
                raise FloatingPointError("Nonfinite benchmark loss")
        median = statistics.median(times)
        print(json.dumps({"batch": size, "step_seconds": median, "loss": loss,
                          "estimated_training_seconds": math.ceil(len(windows) / size) * median,
                          "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20}), flush=True)
        del model, optimizer, batch


if __name__ == "__main__":
    main()
