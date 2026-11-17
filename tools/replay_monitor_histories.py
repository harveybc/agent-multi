"""Descriptive shadow monitors on sealed histories; never a training counterfactual."""
import argparse
from collections import Counter
from pathlib import Path

from app.frozen_recipe_inputs import file_sha
from app.structural_refit import EpochPatience
from tools.run_discrete_curve_probe import read, write


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    results = []
    for arm in ('A', 'D', 'control'):
        root = a.snapshots / arm
        history = read(root / 'HISTORY.json')
        initial = read(root / 'EPOCH_ZERO.json')
        rows = [dict(epoch=0, metrics=initial['metrics']), *history]
        if [r['epoch'] for r in rows] != list(range(len(rows))):
            raise ValueError('NONCONTIGUOUS_HISTORY')
        for mode in EpochPatience.MODES + EpochPatience.SHARED_MODES:
            monitor = EpochPatience(20, 1e-8, mode, 20, True)
            selected, trace, rejections = None, [], Counter()
            for row in rows:
                metrics = row['metrics']
                eligible = all(metrics[p]['orders_per_week'] > 2 for p in ('TRAIN', 'VALIDATION'))
                v = monitor.observe(metrics['TRAIN']['return_per_week'],
                    metrics['VALIDATION']['return_per_week'], eligible)
                if row['epoch'] == 0:
                    monitor.epoch = 0
                if v['improved']:
                    selected = row
                rejections.update(v['rejections'])
                trace.append(dict(epoch=row['epoch'], **{k:v[k] for k in
                    ('improved', 'score', 'best', 'bad', 'stop', 'rejections')}))
            results.append(dict(arm=arm, history_end=rows[-1]['epoch'], mode=mode,
                selected_epoch=None if selected is None else selected['epoch'],
                selected_metrics=None if selected is None else selected['metrics'],
                best_score=monitor.best, rejections=dict(rejections), trace=trace,
                history_sha256=file_sha(root / 'HISTORY.json'), initial_sha256=file_sha(root / 'EPOCH_ZERO.json')))
    write(a.output / 'RESULT.json', dict(complete=True, trained=False, results=results,
        scope='Shadow full observed prefix including observations after hypothetical stop; not counterfactual training',
        sources={p:file_sha(Path(__file__).resolve().parents[1]/p)
                 for p in ('app/structural_refit.py','tools/replay_monitor_histories.py')}))
    for r in results:
        print(r['arm'], r['mode'], r['selected_epoch'], r['best_score'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshots', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
