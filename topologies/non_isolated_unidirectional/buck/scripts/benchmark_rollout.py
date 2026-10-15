"""Measure synchronized rollout forward/backward latency and batch throughput.

Optional --reference-dir contains pre-change model.py and hybrid_rk4.py.
This uses synthetic inputs, not a full training epoch or accuracy benchmark.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
import types

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.model import NeuralEventHybridODE
from training.hybrid_rk4 import hybrid_rk4


def reference_modules(directory):
    package = types.ModuleType("benchmark_reference")
    package.__path__ = [str(directory)]
    sys.modules[package.__name__] = package
    loaded = {}
    for name in ("model", "hybrid_rk4"):
        spec = importlib.util.spec_from_file_location(
            f"{package.__name__}.{name}", directory / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded["model"].NeuralEventHybridODE, loaded["hybrid_rk4"].hybrid_rk4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--batches", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reference-dir", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = json.loads((ROOT / "configs/hybrid_event_matlab_v3/project.json").read_text(encoding="utf-8"))
    variants = [("optimized", NeuralEventHybridODE, hybrid_rk4)]
    if args.reference_dir:
        variants.insert(0, ("baseline", *reference_modules(args.reference_dir)))

    def sync():
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        elif args.device == "mps":
            torch.mps.synchronize()

    print(json.dumps({"device": args.device, "torch": torch.__version__,
                      "cycles": args.cycles, "repeats": args.repeats}), flush=True)
    for batch in args.batches:
        reference_result = reference_grads = None
        for name, model_class, rollout in variants:
            torch.manual_seed(7)
            model = model_class(cfg["model"]).to(args.device)
            x = torch.tensor([[0.5, 0.25]], device=args.device).repeat(batch, 1)
            vs = torch.ones(batch, 1, device=args.device)

            def step():
                model.zero_grad(set_to_none=True)
                result = rollout(model, x, vs, args.cycles)
                (result.states.square().mean() + result.event_time_tau.square().mean()).backward()
                return result

            result = step()  # warmup and numerical comparison outside timing
            actual = {key: value.detach().cpu() for key, value in vars(result).items()}
            grads = {key: (torch.zeros_like(p) if p.grad is None else p.grad).detach().cpu()
                     for key, p in model.named_parameters()}
            del result
            if name == "baseline":
                reference_result, reference_grads = actual, grads
            elif reference_result is not None:
                for key in actual:
                    torch.testing.assert_close(actual[key], reference_result[key], atol=2e-6, rtol=2e-5)
                for key in grads:
                    torch.testing.assert_close(grads[key], reference_grads[key], atol=2e-6, rtol=2e-5)
            sync()
            if args.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(args.repeats):
                sync()
                start = time.perf_counter()
                result = step()
                sync()
                times.append(time.perf_counter() - start)
                del result
            median = statistics.median(times)
            print(json.dumps({"variant": name, "batch": batch,
                              "median_seconds": median,
                              "windows_per_second": batch / median,
                              "samples_seconds": times,
                              "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20
                              if args.device.startswith("cuda") else None,
                              "reference_parity": reference_result is not None and name != "baseline"}), flush=True)


if __name__ == "__main__":
    main()
