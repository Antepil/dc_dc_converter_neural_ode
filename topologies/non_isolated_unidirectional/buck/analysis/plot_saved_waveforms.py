#!/usr/bin/env python3
"""Plot saved review predictions without rerunning or changing the trained model."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'buck-waveforms-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPLITS = {
    'train': ('Training', '训练集'),
    'validation': ('Validation', '验证集'),
    'test_seen': ('Seen-voltage test', '同电压测试'),
    'test_interpolation': ('Interpolation test', '电压插值'),
    'test_voltage_extrapolation': ('Near-extrapolation test', '近域外推'),
    'test_extrapolation_high': ('High-voltage extrapolation', '高压外推'),
}
COLORS = ('#222b36', '#dc493a')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titleweight': 'semibold', 'savefig.facecolor': 'white'})


def style_axis(ax):
    ax.grid(alpha=.18)
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_xlabel('Switching cycles (1 cycle = 50 us)')


def draw_pair(ax, t, target, prediction, lw=1.35):
    ax.plot(t, target, color=COLORS[0], lw=lw, label='Simscape truth')
    ax.plot(t, prediction, color=COLORS[1], lw=lw, ls='--', label='Neural ODE')
    style_axis(ax)


def plot_trajectory(task):
    record, pred, target, out = task
    cycles = record['cycles']
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 7.2))
    physical_scale = np.array([record['current_ref_a'], record['voltage_ref_v']])
    pred, target = pred * physical_scale, target * physical_scale
    for channel, ylabel in enumerate(('Inductor current iL (A)', 'Capacitor voltage vC (V)')):
        for col, (a, b) in enumerate(((0, min(5, cycles) * 100), (0, cycles * 100), ((cycles - 2) * 100, cycles * 100))):
            ax = axes[channel, col]
            if col == 1:
                # Full-horizon envelopes remain legible even when 200 switching cycles overlap.
                time = np.arange(cycles) + .5
                for arr, color, ls, label in ((target, COLORS[0], '-', 'Truth cycle mean'), (pred, COLORS[1], '--', 'Model cycle mean')):
                    windows = np.lib.stride_tricks.sliding_window_view(arr[:, channel], 101)[::100]
                    ax.fill_between(time, windows.min(1), windows.max(1), color=color, alpha=.12)
                    ax.plot(time, windows.mean(1), color=color, ls=ls, lw=1.25, label=label)
                style_axis(ax)
            else:
                draw_pair(ax, np.arange(a, b + 1) / 100, target[a:b+1, channel], pred[a:b+1, channel])
            ax.set_ylabel(ylabel)
            if channel == 0:
                ax.set_title(('First 5 cycles: waveform', f'Full {cycles} cycles: mean + min/max', 'Last 2 cycles: waveform')[col])
    axes[0, 0].legend(loc='best', fontsize=8)
    axes[0, 1].legend(loc='best', fontsize=8)
    fig.suptitle(f"{SPLITS[record['split']][0]} | Vs = {record['voltage_v']:g} V | {record['trajectory_id']}\n"
                 f"NRMSE = {record['nrmse_pct']:.4f}%   |   iL RMSE = {record['current_rmse_ma']:.3f} mA   |   vC RMSE = {record['voltage_rmse_mv']:.3f} mV", fontsize=14, y=.99)
    fig.text(.5, .012, 'Truth is black; prediction is red dashed. The middle panels show cycle means and min/max envelopes. No truth resets during rollout.', ha='center', fontsize=9, color='#536273')
    fig.tight_layout(rect=(0, .035, 1, .97))
    path = Path(out) / record['image']
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def overview(rows, out, split=None):
    selected = [(r, p, y) for r, p, y in rows if split is None or r['split'] == split]
    fig, axes = plt.subplots(len(selected), 2, figsize=(13, 2.45 * len(selected)), squeeze=False)
    for row, (record, pred, target) in enumerate(selected):
        cycles = record['cycles']; a = (cycles - 2) * 100
        for channel, (label, scale) in enumerate((('iL (A)', record['current_ref_a']), ('vC (V)', record['voltage_ref_v']))):
            ax = axes[row, channel]
            draw_pair(ax, np.arange(a, cycles * 100 + 1) / 100, target[a:, channel] * scale, pred[a:, channel] * scale)
            ax.set_ylabel(label)
            ax.set_title(f"{SPLITS[record['split']][0]} | {record['voltage_v']:g} V | NRMSE {record['nrmse_pct']:.4f}%", fontsize=10)
            ax.set_xlabel('Switching cycles')
    axes[0, 0].legend(fontsize=8)
    title = 'Worst trajectory in each dataset: final 2 cycles' if split is None else f'{SPLITS[split][0]}: worst trajectory at each voltage, final 2 cycles'
    fig.suptitle(title, fontsize=14, y=1 - .12 / fig.get_figheight())
    fig.tight_layout(rect=(0, 0, 1, 1 - .42 / fig.get_figheight()))
    filename = 'all_splits_overview.png' if split is None else f'{split}_voltages.png'
    fig.savefig(Path(out) / filename, dpi=140)
    plt.close(fig)
    return filename


def make_gallery(records, out, source, refs):
    payload = json.dumps(records, ensure_ascii=False).replace('</', '<\\/')
    labels = json.dumps({k: v[1] for k, v in SPLITS.items()}, ensure_ascii=False)
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Buck Neural ODE 波形图册</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f5f7;color:#202e3a;font:15px system-ui,-apple-system,sans-serif}main{max-width:1560px;margin:auto;padding:28px}h1{font-size:27px;margin:0 0 10px}p{line-height:1.65;color:#526272}.controls{display:flex;gap:16px;flex-wrap:wrap;padding:18px;background:white;border-radius:10px;margin:20px 0}label{display:flex;gap:6px;flex-direction:column;font-size:13px;color:#526272}select,button{padding:9px 12px;border:1px solid #d7dee4;border-radius:6px;background:white;color:#202e3a;font-size:14px}button{cursor:pointer}button:hover{background:#edf3f7}#info{margin:15px 0;padding:12px 16px;background:#e8eef3;border-radius:8px}img{display:block;max-width:100%;height:auto;background:white;border-radius:8px}details{margin-top:28px}summary{cursor:pointer;font-weight:600}a{color:#236590}#open{margin-left:12px}footer{margin-top:24px;font-size:13px;color:#667783}
</style><main><h1>Buck Neural ODE 波形图册</h1>
<p>201 条轨迹 · 6 个数据集 · 37 个“数据集—电压”组合。黑色为 Simscape 真值，红色虚线为模型预测。<br>每张图显示前 5 周期波形、全程周期均值与峰谷包络、末尾 2 周期波形；全程包络的阴影不是置信区间。所有预测只从初始真值出发。</p>
<div class="controls"><label>数据集<select id="split"></select></label><label>输入电压<select id="voltage"></select></label><label>轨迹 / 初始条件<select id="trajectory"></select></label><label>&nbsp;<button id="worst">当前电压最差轨迹</button></label><label>&nbsp;<button id="previous">上一条</button></label><label>&nbsp;<button id="next">下一条</button></label></div>
<div id="info"></div><a id="download">下载当前 PNG</a><a id="open" target="_blank">单独打开原图</a><img id="waveform" alt="当前轨迹的电流和电压波形" style="margin-top:12px">
<details open><summary>当前数据集：各电压最差轨迹总览（末尾 2 周期）</summary><p>按完整轨迹 NRMSE 选取每个电压的最差样本。单条曲线不代表整个数据集的平均误差。</p><img id="voltageOverview"></details>
<details><summary>六个数据集的最差轨迹总览</summary><img src="all_splits_overview.png" loading="lazy"></details>
<footer>固定工况：R=80 Ω，D=0.5，20 kHz，每周期 100 个区间。NRMSE 按 iL/1 A、vC/32 V 计算；它不是相对每条信号幅值的百分误差。<br>来源：SOURCE_PLACEHOLDER。直接使用已保存预测，未重新训练或推理。<br><a href="manifest.json">完整图册索引及指标 JSON</a></footer></main>
<script>const records=DATA_PLACEHOLDER;const labels=LABELS_PLACEHOLDER;
const split=document.querySelector('#split'),voltage=document.querySelector('#voltage'),trajectory=document.querySelector('#trajectory');
function option(value,text){const o=document.createElement('option');o.value=value;o.textContent=text;return o}
for(const [key,label] of Object.entries(labels))split.append(option(key,label));split.value='test_seen';
function currentGroup(){return records.filter(r=>r.split===split.value&&r.voltage_v===Number(voltage.value))}
function loadVoltage(){voltage.replaceChildren();for(const v of [...new Set(records.filter(r=>r.split===split.value).map(r=>r.voltage_v))].sort((a,b)=>a-b))voltage.append(option(v,v+' V'));loadTrajectory()}
function loadTrajectory(){trajectory.replaceChildren();for(const r of currentGroup())trajectory.append(option(r.trajectory_id,r.trajectory_id+' · '+r.nrmse_pct.toFixed(4)+'%'));selectWorst()}
function selectWorst(){const r=currentGroup().reduce((a,b)=>a.nrmse_pct>b.nrmse_pct?a:b);trajectory.value=r.trajectory_id;show()}
function show(){const r=currentGroup().find(r=>r.trajectory_id===trajectory.value);document.querySelector('#waveform').src=r.image;document.querySelector('#voltageOverview').src=r.split+'_voltages.png';document.querySelector('#download').href=r.image;document.querySelector('#download').download=r.image.split('/').pop();document.querySelector('#open').href=r.image;document.querySelector('#info').textContent=labels[r.split]+' | '+r.voltage_v+' V | '+r.cycles+' 周期 | 初值 iL='+r.initial_current_a.toFixed(5)+' A，vC='+r.initial_voltage_v.toFixed(5)+' V | NRMSE='+r.nrmse_pct.toFixed(4)+'% | 电流 RMSE='+r.current_rmse_ma.toFixed(3)+' mA | 电压 RMSE='+r.voltage_rmse_mv.toFixed(3)+' mV'}
split.onchange=loadVoltage;voltage.onchange=loadTrajectory;trajectory.onchange=show;document.querySelector('#worst').onclick=selectWorst;document.querySelector('#previous').onclick=()=>{trajectory.selectedIndex=(trajectory.selectedIndex+trajectory.options.length-1)%trajectory.options.length;show()};document.querySelector('#next').onclick=()=>{trajectory.selectedIndex=(trajectory.selectedIndex+1)%trajectory.options.length;show()};loadVoltage();
</script></html>'''
    html = html.replace('DATA_PLACEHOLDER', payload).replace('LABELS_PLACEHOLDER', labels).replace('SOURCE_PLACEHOLDER', source.name)
    (out / 'index.html').write_text(html, encoding='utf-8')
    (out / 'manifest.json').write_text(json.dumps({'source': str(source), 'normalization': refs, 'records': records}, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'artifacts/review_20260906')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'artifacts/waveforms_20260907')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(); out = args.out_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    metadata_path = args.source / 'metadata.json'
    if metadata_path.is_file():
        refs = json.loads(metadata_path.read_text(encoding='utf-8'))['normalization']
    else:
        refs = json.loads((ROOT / 'configs/hybrid_event_matlab_v3/project.json').read_text(encoding='utf-8'))['normalization']
    records = []; tasks = []; split_worst = []; voltage_worst = []
    for split in SPLITS:
        with np.load(args.source / f'{split}_predictions.npz') as data:
            pred, target = data['prediction'], data['target']
            assert pred.shape == target.shape and np.isfinite(pred).all() and np.isfinite(target).all()
            assert (len(pred) - 1) % 100 == 0
            cycles = (len(pred) - 1) // 100
            error = pred.astype(np.float64) - target
            rmse = np.sqrt((error ** 2).mean(axis=(0, 2))) * 100
            local = []
            (out / split).mkdir(exist_ok=True)
            for i, (voltage, tid) in enumerate(zip(data['input_voltage_v'], data['trajectory_ids'])):
                tid = str(tid)
                assert '/' not in tid and '\\' not in tid
                record = {'split': split, 'voltage_v': float(voltage), 'trajectory_id': tid, 'cycles': cycles,
                          'nrmse_pct': float(rmse[i]), 'current_ref_a': refs['current_ref_a'], 'voltage_ref_v': refs['voltage_ref_v'],
                          'current_rmse_ma': float(np.sqrt(np.mean(error[:, i, 0] ** 2)) * refs['current_ref_a'] * 1000),
                          'voltage_rmse_mv': float(np.sqrt(np.mean(error[:, i, 1] ** 2)) * refs['voltage_ref_v'] * 1000),
                          'initial_current_a': float(target[0, i, 0] * refs['current_ref_a']),
                          'initial_voltage_v': float(target[0, i, 1] * refs['voltage_ref_v']),
                          'image': f'{split}/{tid}.png'}
                records.append(record); local.append((record, pred[:, i].copy(), target[:, i].copy()))
                tasks.append((*local[-1], str(out)))
            split_worst.append(local[int(np.argmax(rmse))])
            for voltage in np.unique(data['input_voltage_v']):
                indices = np.flatnonzero(data['input_voltage_v'] == voltage)
                voltage_worst.append(local[int(indices[np.argmax(rmse[indices])])])
    print(f'Plotting {len(tasks)} trajectories with {args.workers} workers', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, _ in enumerate(pool.map(plot_trajectory, tasks, chunksize=2), 1):
            if index % 20 == 0 or index == len(tasks): print(f'Plotted {index}/{len(tasks)}', flush=True)
    overview(split_worst, out)
    for split in SPLITS: overview(voltage_worst, out, split)
    make_gallery(records, out, args.source.resolve(), refs)
    assert all((out / r['image']).is_file() for r in records)
    print(f'Complete: {len(records)} trajectory figures + 7 overviews. Gallery: {out / "index.html"}', flush=True)


if __name__ == '__main__':
    main()
