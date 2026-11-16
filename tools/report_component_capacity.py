"""Summarize measured capacity probes and distinguish primitive from full history."""
import argparse
import csv
from pathlib import Path

import numpy as np

from app.frozen_recipe_inputs import file_sha
from app.g3_producer_audit import dependency_inventory
from tools.run_discrete_curve_probe import read, write


def table(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(a):
    result = read(a.input/'RESULT.json')
    for name, checksum in result['files'].items():
        if file_sha(a.input/name) != checksum:
            raise ValueError('CAPACITY_REPORT_HASH')
    records = [read(a.input/p['artifact']) for p in result['points']]
    if len(records) != 9 or any(r['optimizer_steps'] or not r['restoration_exact'] for r in records):
        raise ValueError('CAPACITY_REPORT_INCOMPLETE')
    a.output.mkdir(parents=True, exist_ok=False)
    rows, sensitivity = [], []
    for record in records:
        for role, profile in record['roles'].items():
            for component, periods in profile['activations'].items():
                for period, values in periods.items():
                    rows.append(dict(seed=record['seed'], epoch=record['epoch'], role=role,
                        component=component, period=period,
                        **{k:v for k,v in values.items() if k != 'channel_std'}))
        for test in record['interventions']:
            for period, value in test['function'].items():
                sensitivity.append(dict(seed=record['seed'], epoch=record['epoch'],
                    component=test['component'], rank_fraction=test['fraction'], period=period,
                    relative_weight_l2=test['relative_weight_l2'], **value))
    table(a.output/'ACTIVATIONS.csv', rows)
    table(a.output/'SENSITIVITY.csv', sensitivity)
    recipe, producer, history = read(a.recipe), read(a.producer_receipt), read(a.history_receipt)
    inventory = dependency_inventory(recipe['list']['columns'], [v['name'] for v in producer['activity_predictors']])
    availability = []
    for entry in inventory['columns']:
        deps = entry['primitive_dependencies']
        availability.append(dict(column=entry['column'], producer=entry['producer'],
            dependencies=';'.join(deps),
            needs_learned_scope=any(x.startswith('FROZEN_') for x in deps),
            needs_calendar='ADMITTED_TIMESTAMPS' in deps,
            pre2020_complete_column_verified=False,
            status='CANDIDATE_PRIMITIVES_ONLY_NOT_RECONSTRUCTED71'))
    if len(availability) != 71:
        raise ValueError('CAPACITY_HISTORY_COLUMN_COUNT')
    table(a.output/'HISTORY_71_COLUMNS.csv', availability)
    write(a.output/'HISTORY_SCOPE.json', dict(recipe_identity=recipe['identity'],
        features=len(availability), inventory=inventory,
        primitive_candidate=history['candidate_before_train'],
        source_receipts={p.name:file_sha(p) for p in (a.recipe,a.producer_receipt,a.history_receipt)},
        all71_common_start_verified=False, real_history_extended=False,
        note='Dependency reconciliation only, not a new source download or earlier-feature reconstruction.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), constrained_layout=True)
    names = ['fusion', 'attention', 'core', 'hidden', 'flattened']
    for axis, seed in zip(axes, (7,19,43)):
        selected = sorted([r for r in records if r['seed'] == seed], key=lambda r:r['epoch'])
        for record in selected:
            values = record['roles']['actor']['activations']
            axis[0].plot(names, [values[n]['pooled']['participation_rank']/values[n]['pooled']['channels'] for n in names],
                         marker='o', label=f"epoch{record['epoch']}")
            tests = [v for v in record['interventions'] if v['fraction'] == .5 and v['component'] in names+['head']]
            axis[1].plot([v['component'] for v in tests],
                         [100*v['function']['pooled']['flip_fraction'] for v in tests], marker='o',
                         label=f"epoch{record['epoch']}")
        axis[0].set_title(f'Seed{seed}: centered activation rank / channels')
        axis[1].set_title(f'Seed{seed}: one-component 50% matrix-rank approximation')
        axis[1].set_ylabel('Changed TRAIN signals %')
        for ax in axis:
            ax.legend(); ax.grid(alpha=.2)
    fig.savefig(a.output/'CAPACITY.png', dpi=140)
    plt.close(fig)
    compact = []
    for record in records:
        actor = record['roles']['actor']['activations']
        branch_ranks = [v['pooled']['rank99'] for k,v in actor.items() if k.startswith('branch.')]
        core_ranks = {role:record['roles'][role]['activations']['core']['pooled']['participation_rank']
                      for role in record['roles']}
        half = [v for v in record['interventions'] if v['fraction'] == .5]
        compact.append(dict(seed=record['seed'], epoch=record['epoch'],
            branch_rank99_median=float(np.median(branch_ranks)),
            branch_rank99_min=min(branch_ranks), branch_rank99_max=max(branch_ranks),
            core_participation=core_ranks,
            half_rank_signal_flips={v['component']:v['function']['pooled']['flips'] for v in half},
            full_rank_identity_tests=sum(v['fraction']==1 and v['function']['pooled']['flips']==0
                                         for v in record['interventions'])))
    write(a.output/'SUMMARY.json', dict(points=compact, rows_per_point=512, years=[2020,2021,2022,2023],
        approximation_not_pruning=True, no_optimal_size_claim=True, no_data_years_estimate=True,
        code={p.name:file_sha(p) for p in (Path(__file__),Path(__file__).with_name('diagnose_component_capacity.py'),
                                            Path(__file__).resolve().parents[1]/'app/capacity_measurements.py')},
        raw_result_sha256=file_sha(a.input/'RESULT.json')))
    print(a.output/'SUMMARY.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input','output','recipe','producer-receipt','history-receipt'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(parser.parse_args())
