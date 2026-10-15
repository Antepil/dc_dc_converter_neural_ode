"""Bounded CUDA auxiliary-loss pilot from identical weights, batches and optimizer settings."""
import json
import hashlib
import sys
import time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.data import load_data, normalize, collate
from training.model import NeuralEventHybridODE
from training.hybrid_rk4 import hybrid_rk4
from training.losses import rollout_losses, voltage_derivative_loss


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    source = ROOT / 'artifacts/runs/cuda/best.pt'
    cfg = json.loads((ROOT / 'artifacts/runs/cuda/run_config.json').read_text(encoding='utf8'))['project']
    data = normalize(load_data(str(ROOT / 'data/processed/hybrid_event_matlab_v3')), cfg['normalization'])
    out = ROOT / 'artifacts/derivative_pilot_20260913'
    out.mkdir(exist_ok=False)
    initial = torch.load(source, map_location='cuda', weights_only=True)
    def batch(split, windows, cycles):
        return collate(data[split], windows, cycles, 'cuda', torch.float32, derivative_supervision=True)
    validation = batch('validation', [(i, 0) for i in range(18)], 20)
    def evaluate(model):
        with torch.no_grad():
            result = hybrid_rk4(model, validation['y0'], validation['vs'], 20)
            error = result.states-validation['target']
            rmse = error.square().mean((0, 1)).sqrt()
            return {'iL_rmse_a': float(rmse[0]*cfg['normalization']['current_ref_a']),
                    'vC_rmse_v': float(rmse[1]*cfg['normalization']['voltage_ref_v']),
                    'voltage_derivative_mse': float(voltage_derivative_loss(model, validation)),
                    'event_time_mae_tau': float((result.event_time_tau-validation['event_time_tau']).abs().mean())}
    report = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'scope': '8 five-cycle updates per arm; all 18 validation trajectories, 20-cycle free rollout; no test data',
              'arms': {}}
    # Exactly the same eight batches, each covering all six training voltages.
    windows = [[(12*v+(step % 12), step % 16) for v in range(6)] for step in range(8)]
    for name, weight in [('control', 0.0), ('central_rollout', 0.01)]:
        torch.manual_seed(20260913)
        model = NeuralEventHybridODE(cfg['model'], idle_profile='soft_idle_loss').cuda()
        model.load_state_dict(initial)
        if 'baseline' not in report:
            report['baseline'] = evaluate(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=cfg['training']['weight_decay'])
        weights = dict(cfg['training']['loss_weights'], voltage_derivative=weight)
        start = time.time()
        for selected in windows:
            b = batch('train', selected, 5)
            result = hybrid_rk4(model, b['y0'], b['vs'], 5)
            terms = rollout_losses(model, result, b, weights,
                                   guard_margin=cfg['training']['event_pretraining']['guard_margin'],
                                   counterfactual_offsets=cfg['training']['event_pretraining']['counterfactual_offsets_tau'])
            optimizer.zero_grad(set_to_none=True)
            terms['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['gradient_clip_norm'])
            optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.time()-start
        report['arms'][name] = dict(evaluate(model), training_seconds=elapsed, weight=weight, updates=8)
        torch.save(model.state_dict(), out / f'{name}_weights.pt')
        (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
        print(name, report['arms'][name], flush=True)


if __name__ == '__main__':
    main()
