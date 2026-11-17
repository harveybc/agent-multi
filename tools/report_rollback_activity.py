"""Render completed read-only rollback diagnosis without selecting a candidate."""
import argparse
from pathlib import Path

from app.frozen_recipe_inputs import file_sha
from tools.run_discrete_curve_probe import read, write


def run(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = read(root / 'RESULT.json')
    if not data['complete'] or data['trained'] or data['live_workers_modified']:
        raise ValueError('INCOMPLETE_OR_MUTATING_DIAGNOSIS')
    summary = dict(result_sha256=file_sha(root / 'RESULT.json'),
        analysis_sources=data['analysis_sources'], all_native_parity_pass=data['all_native_parity_pass'],
        rows=[], swaps=[], sampling=[], rollbacks=0, native_replays=0)
    fig, axes = plt.subplots(3, 4, figsize=(18, 11), constrained_layout=True)
    for i, result in enumerate(data['results']):
        arm = result['arm']
        summary['rollbacks'] += len(result['rollback_receipts'])
        for point in result['points']:
            summary['native_replays'] += 1
            for period in ('TRAIN', 'VALIDATION'):
                m, p = point['metrics'][period], point['probe'][period]
                summary['rows'].append(dict(arm=arm, epoch=point['epoch'], period=period,
                    best_epoch=result['best_epoch'], latest_epoch=result['latest_epoch'],
                    return_pct=m['return_pct'], trades=m['trades'], orders_per_week=m['orders_per_week'],
                    buy=m['buy']['trades'], sell=m['sell']['trades'],
                    early_close=m['close_reasons'].get('AGENT_CLOSE', 0),
                    signals=m['signals'], native_counts=m['native_counts'],
                    nop_fraction=m['signals'].get('0', 0) / sum(m['signals'].values()),
                    entropy=p['entropy'], q_nop_fraction=p['q_greedy_share'][0],
                    q_advantage_over_nop=p['q_advantage_over_nop'],
                    mean_logit_gradient=p['mean_logit_gradient'],
                    mean_probability=p['mean_probability'], alpha=p['alpha'],
                    actor_drift_best=point['drift_vs_best']))
        for part in result['swaps']:
            summary['native_replays'] += 1
            summary['swaps'].append(dict(arm=arm, restored=part['restored_component'],
                metrics={p: {k: part['metrics'][p][k] for k in ('return_pct', 'trades', 'orders_per_week')}
                         for p in ('TRAIN', 'VALIDATION')}))
        for sample in result['sampling']:
            summary['native_replays'] += 4
            for period in ('TRAIN', 'VALIDATION'):
                summary['sampling'].append(dict(arm=arm, epoch=sample['epoch'], period=period,
                    greedy_return=sample['greedy'][period]['return_pct'],
                    greedy_trades=sample['greedy'][period]['trades'],
                    sampled_returns=[s['metrics'][period]['return_pct'] for s in sample['sampled']],
                    sampled_trades=[s['metrics'][period]['trades'] for s in sample['sampled']]))
        for period, color in (('TRAIN', 'tab:blue'), ('VALIDATION', 'tab:red')):
            rows = [r for r in summary['rows'] if r['arm'] == arm and r['period'] == period]
            x = [r['epoch'] for r in rows]
            for col, key in enumerate(('return_pct', 'nop_fraction', 'entropy', 'q_nop_fraction')):
                axes[i, col].plot(x, [r[key] for r in rows], 'o-', color=color, label=period)
        for col, title in enumerate(('Native return %', 'Native NOP fraction', 'Probe entropy', 'Probe critic NOP fraction')):
            ax = axes[i, col]
            ax.set_title(arm + ': ' + title)
            ax.set_xlabel('Epoch (pre-rollback snapshot)')
            ax.legend(fontsize=7)
            ax.grid(alpha=.2)
            for event in result['rollback_receipts']:
                ax.axvline(event['global_epoch'], linestyle=':', color='gray', alpha=.5)
        axes[i, 2].axhline(.6591674, color='gray', linestyle='--')
    fig.savefig(root / 'ROLLBACK_ACTIVITY.png', dpi=140)
    plt.close(fig)
    summary['plot_note'] = 'Dashed entropy line is replay target, not a per-probe hard constraint; dotted lines are rollback events.'
    write(root / 'SUMMARY.json', summary)
    print({k: summary[k] for k in ('all_native_parity_pass', 'native_replays', 'rollbacks')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
