"""CPU-only copied359 checkpoints: native replay, fixed probes and actor swaps."""
import argparse
from pathlib import Path

import numpy as np
import torch

from app.frozen_recipe_inputs import file_sha
from app.online_transient_forensics import functional_metrics, checkpoint_drift, swap_actor
from app.rl_subset import support_hash
from app.selected_input_market import market_for
from app.structural_refit import load_bundle, verify_files
from app.training_forensics import tree_digest, rng_state
from tools.run_discrete_curve_probe import read, write, configure
from tools.run_structural_sac_refit import assess_full
from tools.run_long_sac_continuation import checked_baseline, SUPPORT


def compact(metrics):
    keys = ('return_pct', 'sharpe', 'trades', 'orders_per_week', 'win_rate', 'signals',
            'close_reasons', 'max_drawdown_pct')
    return {p: {k: metrics[p][k] for k in keys} for p in ('TRAIN', 'VALIDATION')}


def plot(results, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 4, figsize=(17, 10), constrained_layout=True)
    for row, result in enumerate(results):
        history = result['history']
        points = result['points']
        for p, c in (('TRAIN', 'tab:blue'), ('VALIDATION', 'tab:red')):
            axes[row, 0].plot([0]+[h['epoch'] for h in history],
                [result['baseline'][p]['return_pct']]+[h['metrics'][p]['return_pct'] for h in history], label=p, color=c)
            axes[row, 1].plot([v['epoch'] for v in points], [v['probe'][p]['entropy'] for v in points], label=p, color=c)
            axes[row, 2].plot([v['epoch'] for v in points], [v['probe'][p]['action_flips_fraction'] for v in points], label=p, color=c)
        axes[row, 1].axhline(.6*np.log(3), color='gray', linestyle='--', label='replay entropy target')
        for key in ('q_common_abs', 'q_action_gap', 'twin_disagreement'):
            axes[row, 3].plot([v['epoch'] for v in points], [v['probe']['TRAIN'][key] for v in points], label=key)
        for col, title in enumerate(('Return %', 'Fixed-probe entropy', 'Action flips vs initial', 'Fixed TRAIN Q geometry')):
            ax = axes[row, col]
            ax.set_title(result['arm']+' / '+title)
            ax.set_xlabel('Epoch'); ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.savefig(output/'TRANSIENTS.png', dpi=130)
    plt.close(fig)


def run(a):
    if torch.cuda.is_available():
        raise ValueError('FORCE_CPU_FOR_NONINTERFERING_ANALYSIS')
    configure(7)
    if a.output.exists():
        raise ValueError('NEW_ANALYSIS_OUTPUT_REQUIRED')
    a.output.mkdir(parents=True)
    parent = read(a.parent/'SPEC.json')
    verify_files(a.parent, {k: v for k, v in parent['files'].items()
        if k.startswith(('data/', 'engine/')) or k in ('SUPPORT_RECIPE.json', 'RISK.json')})
    risk = read(a.parent/'RISK.json')
    results = []
    for seed in (7, 19, 43):
        root = a.snapshots/('seed_%d' % seed)
        history = read(root/'HISTORY.json')
        design = read(root/'DESIGN.json')
        effective = read(root/'EFFECTIVE.json')
        model, _ = load_bundle(root/'initial_bundle', 'cpu')
        initial = {n: v.detach().clone() for n, v in model.loss.state_dict().items()}
        market = market_for(a.parent, parent, model.recipe)
        if support_hash(market.inputs) != SUPPORT:
            raise ValueError('TRANSIENT_SUPPORT_CHANGED')
        probes = {}
        baseline_logits = {}
        for period in ('TRAIN', 'VALIDATION'):
            valid = np.flatnonzero(market.inputs.valid & (market.inputs.period == period))
            ix = valid[np.linspace(0, len(valid)-1, 256, dtype=int)]
            probes[period] = torch.as_tensor(market.inputs.windows[ix].copy())
            with torch.no_grad():
                baseline_logits[period] = model.surfaces(probes[period]).clone()
        baseline = assess_full(model, market, risk, design)
        checked_baseline(baseline)
        receipt = dict(seed=seed, arm=effective['arm'], baseline=compact(baseline),
            history=history, points=[], hybrids=[], input_sha256={
                n: file_sha(root/n) for n in ('HISTORY.json', 'DESIGN.json', 'EFFECTIVE.json', 'initial_bundle/model.pt')})
        paths = sorted((root/'weights').glob('epoch_*.pt'))
        epochs = [(0, None)] + [(int(p.stem.split('_')[-1]), p) for p in paths]
        for epoch, path in epochs:
            if path is None:
                state = initial
            else:
                saved = next(h for h in history if h['epoch'] == epoch)
                if file_sha(path) != saved['weight_snapshot']['sha256']:
                    raise ValueError('TRANSIENT_WEIGHT_DIGEST')
                state = torch.load(path, map_location='cpu', weights_only=True)
            model.loss.load_state_dict(state)
            before = tree_digest(model.loss.state_dict()); rng = tree_digest(rng_state())
            row = dict(epoch=epoch, drift=checkpoint_drift(state, initial), probe={})
            for period, x in probes.items():
                with torch.no_grad():
                    row['probe'][period] = functional_metrics(model.surfaces(x), model.q_surfaces(x),
                        baseline_logits[period], float(model.loss.log_alpha.detach().exp()))
            metrics = baseline if epoch == 0 else assess_full(model, market, risk, design)
            row['metrics'] = compact(metrics)
            reference = baseline if epoch == 0 else saved['metrics']
            row['native_parity'] = {p: dict(return_delta=metrics[p]['return_pct']-reference[p]['return_pct'],
                trades_equal=metrics[p]['trades']==reference[p]['trades']) for p in ('TRAIN', 'VALIDATION')}
            row['native_parity_pass'] = all(abs(v['return_delta'])<=1e-8 and v['trades_equal']
                for v in row['native_parity'].values())
            if not row['native_parity_pass']:
                write(a.output/f'PARITY_MISMATCH_{seed}_{epoch}.json', row)
                if not a.record_device_mismatches:
                    raise ValueError('TRANSIENT_CPU_NATIVE_PARITY')
            if before != tree_digest(model.loss.state_dict()) or rng != tree_digest(rng_state()):
                raise ValueError('TRANSIENT_PROBE_MUTATION')
            receipt['points'].append(row)
            if epoch in (1, epochs[-1][0]):
                for component in ('head', 'encoder'):
                    hybrid, replaced = swap_actor(state, initial, component)
                    model.loss.load_state_dict(hybrid)
                    hm = assess_full(model, market, risk, design)
                    receipt['hybrids'].append(dict(epoch=epoch, restored_component=component,
                        replaced_keys=replaced, metrics=compact(hm)))
                model.loss.load_state_dict(state)
            write(a.output/f'SEED_{seed}.json', receipt)
            print(f"seed={seed} arm={effective['arm']} epoch={epoch} native_parity={row['native_parity_pass']}", flush=True)
        results.append(receipt)
    plot(results, a.output)
    parity = all(p['native_parity_pass'] for r in results for p in r['points'])
    write(a.output/'RESULT.json', dict(status='COPIED_CHECKPOINT_DIAGNOSIS_COMPLETE', seeds=results,
        all_native_parity_pass=parity, device_mismatch_override=a.record_device_mismatches,
        trained=False, promoted=False, live_workers_modified=False, source_support=SUPPORT,
        analysis_source={n: file_sha(Path(__file__).resolve().parents[1]/n)
                         for n in ('tools/analyze_online_transients.py', 'app/online_transient_forensics.py')}))
    report = ['# Online Transient Forensics', '',
        '| Arm | Epoch | Configuration | TRAIN return % | VAL return % | VAL orders |',
        '|---|---:|---|---:|---:|---:|']
    for r in results:
        for point in r['points']:
            if point['epoch'] not in (0, 1, r['points'][-1]['epoch']):
                continue
            m = point['metrics']
            report.append(f"|{r['arm']}|{point['epoch']}|observed|{m['TRAIN']['return_pct']:.6f}|{m['VALIDATION']['return_pct']:.6f}|{m['VALIDATION']['trades']}|")
        for point in r['hybrids']:
            m = point['metrics']
            report.append(f"|{r['arm']}|{point['epoch']}|restore {point['restored_component']}|{m['TRAIN']['return_pct']:.6f}|{m['VALIDATION']['return_pct']:.6f}|{m['VALIDATION']['trades']}|")
    report += ['', 'Hybrid swaps are diagnostic, not promoted models or a validation optimization. '
        'Each treatment currently has a different seed; compare trajectories within runs. '
        'Critic agreement is not ground-truth calibration; probe entropy is not replay-batch entropy.']
    report += ['', f'All CPU/GPU native accounts agree: {parity}. '
        'Any PARITY_MISMATCH files are unresolved exceptions unless a separate device witness explains them. '
        'All hybrid comparisons use CPU accounts, not mixed-device deltas.']
    (a.output/'REPORT.md').write_text('\n'.join(report)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('snapshots', 'parent', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--record-device-mismatches', action='store_true',
        help='Record failed parity gates and continue descriptive CPU analysis; never mark them passed.')
    run(parser.parse_args())
