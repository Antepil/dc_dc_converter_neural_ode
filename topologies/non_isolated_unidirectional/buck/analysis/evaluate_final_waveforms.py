"""Export continuous final-weight predictions for all saved data splits on CUDA."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.data import load_data, normalize
from training.model import NeuralEventHybridODE
from training.hybrid_rk4 import hybrid_rk4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise ValueError("Output directory must be new or empty")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    saved_config = json.loads((args.run_dir / "run_config.json").read_text(encoding="utf-8"))
    cfg = saved_config.get("project", saved_config.get("config", saved_config))
    metrics = json.loads((args.run_dir / "metrics.json").read_text(encoding="utf-8"))
    weights_path = args.run_dir / "best.pt"
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = NeuralEventHybridODE(cfg["model"], metrics["idle_profile"]).cuda().eval()
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    data_path = Path(metrics["data_dir"])
    if not data_path.is_dir():
        data_path = ROOT / "data/processed/hybrid_event_matlab_v3"
    splits = normalize(load_data(str(data_path)), cfg["normalization"])
    metadata = {"weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
                "selection": json.loads((args.run_dir / "best_selection.json").read_text(encoding="utf-8")),
                "normalization": cfg["normalization"], "dtype": "float32", "device": torch.cuda.get_device_name(),
                "splits": {}}
    lengths = sorted({(d["states_norm"].shape[1] - 1) // 100 for d in splits.values()})
    for length in lengths:
        names = [name for name, d in splits.items() if (d["states_norm"].shape[1] - 1) // 100 == length]
        y0 = torch.cat([splits[name]["states_norm"][:, 0] for name in names]).cuda().float()
        vs = torch.cat([splits[name]["vs_norm"] for name in names]).cuda().float()
        predicted = []
        started = time.perf_counter()
        with torch.no_grad():
            for start in range(0, length, 20):
                result = hybrid_rk4(model, y0, vs, min(20, length - start), duty=cfg["circuit"]["duty_cycle"])
                predicted.append(result.states[0 if start == 0 else 1:].cpu())
                y0 = result.states[-1]
                print(f"horizon={length}: {min(start + 20, length)}/{length} cycles, batch={len(vs)}, {time.perf_counter()-started:.1f}s", flush=True)
                del result
        prediction = torch.cat(predicted).numpy()
        cursor = 0
        for name in names:
            d = splits[name]
            n = len(d["states_norm"])
            pred = prediction[:, cursor:cursor+n]
            target = d["states_norm"].transpose(0, 1).numpy()
            error = pred.astype(np.float64) - target
            nrmse = np.sqrt(np.mean(error**2, axis=(0, 2))) * 100
            channels = np.sqrt(np.mean(error**2, axis=(0, 1))) * np.array([cfg["normalization"]["current_ref_a"], cfg["normalization"]["voltage_ref_v"]])
            np.savez_compressed(args.out_dir / f"{name}_predictions.npz", prediction=pred, target=target,
                                input_voltage_v=d["input_voltage"].numpy(), trajectory_ids=d["trajectory_ids"])
            previous = metrics["evaluation"][name][str(length)]["rmse_mean_pct"]
            metadata["splits"][name] = {"cycles": length, "trajectories": n, "normalized_rmse_mean_pct": float(nrmse.mean()),
                                        "current_rmse_ma": float(channels[0]*1000), "voltage_rmse_mv": float(channels[1]*1000),
                                        "previous_normalized_rmse_mean_pct": previous}
            print(name, json.dumps(metadata["splits"][name]), flush=True)
            cursor += n
        (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
