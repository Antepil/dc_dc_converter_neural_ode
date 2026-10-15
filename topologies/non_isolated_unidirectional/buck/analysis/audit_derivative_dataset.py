"""Read-only dataset audit; difference agreement is not derivative ground truth."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import whosmat


def audit(root):
    result = {}
    seen = {}
    duplicates = []
    with np.load(root / 'data/processed/hybrid_event_matlab_v3/train.npz') as train:
        train_relative = train['states'] / train['input_voltage'].reshape(-1, 1, 1)
        lower = train_relative.min(axis=(0, 1))
        upper = train_relative.max(axis=(0, 1))
    for path in sorted((root / 'data/processed/hybrid_event_matlab_v3').glob('*.npz')):
        with np.load(path, allow_pickle=False) as d:
            x = d['states']
            v = d['input_voltage'].reshape(-1)
            m = d['interval_mode']
            t = d['time_s']
            dt = np.diff(t)
            k = np.arange(x.shape[1] - 2)
            phase = k % 100
            event_idx = (k // 100)[None, :] * 100 + np.rint(d['event_time_tau'][:, k // 100] * 100)
            keep = (m[:, :-1] == m[:, 1:]) & (phase >= 2) & (phase < 98) & (np.abs(phase - 50) > 2) & (np.abs(k[None, :] - event_idx) > 2)
            forward = np.diff(x[:, :, 1], axis=1) / dt
            relative_x = x / v[:, None, None]
            entry = {
                'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'shape': list(x.shape), 'dtype': str(x.dtype),
                'fields': list(d.files), 'finite_states': bool(np.isfinite(x).all()),
                'voltages_v': np.unique(v).tolist(), 'load_ohm': np.unique(d['load_resistance']).tolist(),
                'dt_s_min_max': [float(dt.min()), float(dt.max())],
                'initial_iL_over_Vin_range': [float(relative_x[:, 0, 0].min()), float(relative_x[:, 0, 0].max())],
                'initial_vC_over_Vin_range': [float(relative_x[:, 0, 1].min()), float(relative_x[:, 0, 1].max())],
                'vC_over_Vin_range': [float(relative_x[:, :, 1].min()), float(relative_x[:, :, 1].max())],
                'relative_state_outside_train_bounding_box_fraction': float(np.mean(np.any((relative_x < lower) | (relative_x > upper), axis=-1))),
                'cycle_net_voltage_rise_fraction': float(np.mean(np.diff(x[:, ::100, 1], axis=1) > 0)),
                'metadata_example': json.loads(str(d['metadata'][0])), 'modes': {},
            }
            for j, name in enumerate(['ON', 'OFF_DIODE', 'IDLE']):
                mask = keep & (m[:, :-1] == j)
                slopes = forward[:, :-1][mask]
                # Interior centered estimates: all four surrounding intervals must have same mode.
                idx = np.arange(2, x.shape[1] - 2)
                center_mask = keep[:, idx] & (m[:, idx] == j)
                for offset in [-2, -1, 1]:
                    center_mask &= m[:, idx + offset] == j
                vc = x[:, :, 1]
                c1 = (vc[:, idx+1] - vc[:, idx-1]) / (t[idx+1] - t[idx-1])
                c2 = (vc[:, idx+2] - vc[:, idx-2]) / (t[idx+2] - t[idx-2])
                a, b = c1[center_mask], c2[center_mask]
                f = forward[:, idx][center_mask]
                rms = lambda z: float(np.sqrt(np.mean(z*z)))
                entry['modes'][name] = {
                    'interval_count': int((m == j).sum()), 'usable_forward_pairs': int(mask.sum()),
                    'positive_dvC_fraction': float(np.mean(slopes > 0)),
                    'dvC_V_per_s_quantiles_0_05_50_95_100': np.quantile(slopes, [0, .05, .5, .95, 1]).tolist(),
                    'center_comparison_samples': int(center_mask.sum()),
                    'center_h_vs_2h_rms_over_slope_rms_pct': rms(a-b)/max(rms(a), 1e-12)*100,
                    'forward_vs_center_rms_over_slope_rms_pct': rms(f-a)/max(rms(a), 1e-12)*100,
                    'relative_state_min': relative_x[:, :-2][mask].min(axis=0).tolist(),
                    'relative_state_max': relative_x[:, :-2][mask].max(axis=0).tolist(),
                }
            # Exact state duplication check, independent of MAT metadata/hash differences.
            for i, state in enumerate(x):
                digest = hashlib.sha256(state.tobytes()).hexdigest()
                label = f'{path.stem}:{d["trajectory_ids"][i]}'
                if digest in seen:
                    duplicates.append([seen[digest], label])
                seen[digest] = label
            raw = root / 'data/raw/hybrid_event_matlab_v3' / path.stem
            mats = sorted(raw.glob('*.mat'))
            entry['raw_mat_count'] = len(mats)
            entry['raw_mat_example_fields'] = [a[0] for a in whosmat(mats[0])] if mats else []
            result[path.stem] = entry
    return {'splits': result, 'exact_duplicate_state_trajectories': duplicates,
            'notes': ['Centered difference agreement measures internal consistency, not error against an independent derivative truth.',
                      'Relative state axes are iL/Vin and vC/Vin; bounding ranges do not measure joint support.']}


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    report = audit(root)
    out = root / 'artifacts/dataset_derivative_audit_20260913'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    for name, e in report['splits'].items():
        print(name, e['shape'], e['voltages_v'], 'initial vC/Vin', e['initial_vC_over_Vin_range'])
        for mode, s in e['modes'].items():
            print(mode, 'pairs', s['usable_forward_pairs'], 'positive%', round(s['positive_dvC_fraction']*100, 2),
                  'center consistency%', round(s['center_h_vs_2h_rms_over_slope_rms_pct'], 5),
                  'forward-center%', round(s['forward_vs_center_rms_over_slope_rms_pct'], 4))
    print('Exact duplicates:', report['exact_duplicate_state_trajectories'])
    print('Report: artifacts/dataset_derivative_audit_20260913/audit.json')
