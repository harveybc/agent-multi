"""Authenticate the finite critic-only run and export compact diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(root, output, runtime_commit, bundle):
    receipt = read(root / 'DONE.json')
    for name, expected in receipt['files'].items():
        file = (root / name).resolve()
        if not file.is_relative_to(root.resolve()) or sha(file) != expected:
            raise ValueError('CALIBRATION_FILE_IDENTITY:' + name)
    config = read(root / 'CONFIG.json')
    result = read(root / 'RESULT.json')
    from app.rl_subset import digest
    effective_config = {k:v for k,v in config.items() if k != 'identity'}
    if digest(effective_config) != config['identity'] or result['identity'] != config['identity'] or receipt['identity'] != config['identity']:
        raise ValueError('CALIBRATION_IDENTITY')
    manifest = read(bundle / 'MANIFEST.json')
    for name, expected in manifest['files'].items():
        file = (bundle / name).resolve()
        if not file.is_relative_to(bundle.resolve()) or sha(file) != expected:
            raise ValueError('BUNDLE_IDENTITY')
    recipe = read(bundle / 'RECIPE.json')
    if (sha(bundle / 'model.pt') != config['source_sha256']
            or len(recipe['list']['columns']) != 71
            or recipe['identity'] != config['recipe_identity']):
        raise ValueError('CALIBRATION_SOURCE71')
    if (config['schema'] != 'selected_input_critic_calibration.v1'
            or config['modes'] or result['actor_updates'] != 0
            or not result['actor_cold_exact'] or result['model_promoted']
            or result['test_read'] or not result['source_unchanged']):
        raise ValueError('CALIBRATION_SCOPE')
    records = read(root / 'panel/RECORDS.json')
    import pandas as pd
    timestamps = pd.to_datetime([r['timestamp'] for r in records], utc=True)
    assert timestamps.min() >= pd.Timestamp('2020-01-01', tz='UTC')
    assert timestamps.max() < pd.Timestamp('2024-01-01', tz='UTC')
    for index, row in enumerate(records):
        expected = 'fit' if timestamps[index] < pd.Timestamp('2023-01-01', tz='UTC') else 'monitor'
        if row['role'] != expected:
            raise ValueError('CALIBRATION_SPLIT')
    curves = {}; checkpoint_checks = {}
    import torch
    source = torch.load(bundle / 'model.pt', map_location='cpu', weights_only=False)
    for seed, critic in result['critics'].items():
        if critic['curve'][-1]['updates'] != 4096 or not critic['actor_frozen']:
            raise ValueError('CALIBRATION_UPDATES')
        if set(critic['fit_rows']) & set(critic['monitor_rows']):
            raise ValueError('CALIBRATION_OVERLAP')
        assert critic['fit_rows'] == [i for i,r in enumerate(records) if r['role'] == 'fit']
        assert critic['monitor_rows'] == [i for i,r in enumerate(records) if r['role'] == 'monitor']
        state = torch.load(root / f'critic_{seed}/MODEL.pt', map_location='cpu', weights_only=False)
        for key in ('codec','recipe_identity','effective','trainability'):
            if state[key] != source[key]:
                raise ValueError('CALIBRATION_CHECKPOINT_CONTRACT')
        if state['loss'].keys() != source['loss'].keys():
            raise ValueError('CALIBRATION_STATE_KEYS')
        changed = [k for k,v in source['loss'].items() if not torch.equal(v,state['loss'][k])]
        if not changed or any(not k.startswith(('qvalue_network_params.','target_qvalue_network_params.')) or '.head.' not in k for k in changed):
            raise ValueError('CALIBRATION_CHANGED_FROZEN_STATE')
        steps = {int(v['step']) for v in state['optimizer']['state'].values()}
        if steps != {4096}:
            raise ValueError('CALIBRATION_OPTIMIZER_STEPS')
        checkpoint_checks[seed] = dict(changed_tensors=changed,
            frozen_tensors=len(source['loss'])-len(changed),actor_exact=True,optimizer_steps=4096,
            checkpoint_sha256=sha(root / f'critic_{seed}/MODEL.pt'))
        curves[seed] = critic['curve']
    summary = dict(identity=result['identity'], runtime_commit=runtime_commit,
        source_model_sha256=config['source_sha256'], verified_files=len(receipt['files']),
        features=71,groups=len(recipe['representation']['resolved']['names']),checkpoint_checks=checkpoint_checks,
        seconds=result['seconds'], reference=result['reference'],
        seeds=config['seeds'], critic_updates=4096, actor_updates=0,
        actor_cold_exact=True, source_unchanged=True, model_promoted=False, test_read=False,
        fit_rows=sum(r['role'] == 'fit' for r in records),
        monitor_rows=sum(r['role'] == 'monitor' for r in records),
        critics={s:{role:[p['metrics'][role]['minimum'] for p in (c[0],c[-1])]
                    for role in ('fit','monitor')} for s,c in curves.items()})
    output.mkdir(parents=True, exist_ok=True)
    for name, value in [('SUMMARY.json',summary),('CURVES.json',curves)]:
        (output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    for seed, curve in curves.items():
        for role in ('fit','monitor'):
            for axis, metric in zip(axes, ('rmse','pairwise_accuracy')):
                axis.plot([p['updates'] for p in curve],
                          [p['metrics'][role]['minimum'][metric] for p in curve],
                          label=f'{seed} {role}', linestyle='--' if role == 'monitor' else '-')
                axis.set(xlabel='Critic updates', ylabel=metric)
    axes[1].legend();fig.tight_layout();fig.savefig(output / 'CURVES.png');plt.close(fig)
    (output / 'MANIFEST.json').write_text(json.dumps(dict(
        source_receipt_sha256=sha(root / 'DONE.json'),
        files={name:sha(output / name) for name in ('SUMMARY.json','CURVES.json','CURVES.png')}), indent=2) + '\n')
    print(json.dumps({k:summary[k] for k in ('identity','verified_files','seconds','fit_rows','monitor_rows')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-commit', required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    collect(args.root, args.output, args.runtime_commit, args.bundle)
