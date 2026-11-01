"""Pinned feature_selector diagnostic with native SAC and sequential worker queues."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from feature_selector.plugins import resolve
from app.ablation_comparison import make_selectors, observe, prefix_select
from app.frozen_recipe_inputs import file_sha
from app.recipe_runtime import RecipeMarket
from app.rl_subset import digest, groups_for_selector, coherent_blocks, subset_recipe, project_inputs, support_hash
from app.three_signal_learning import environment, emit, evaluate
from tools.run_discrete_curve_probe import read, write, now, configure
from tools.run_stopping_contrast import authenticate as contrast_auth, verified as contrast_verified
from tools.run_three_signal_baseline import make_model, monitor, train_loop

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ['app/ablation_comparison.py', 'app/rl_subset.py', 'tools/run_paired_ablation.py']


def prepare(args):
    contrast, parent = contrast_auth(args.parent)
    cfg = read(args.config)
    if set(cfg) != {'schema', 'selector', 'selector_params', 'steps', 'guarded_patience', 'seeds'}:
        raise ValueError('ABLATION_CONFIG_FIELDS')
    if cfg['schema'] != 'paired_ablation.design.v1' or cfg['steps'] != 524288 or cfg['guarded_patience'] != 5 or cfg['seeds'] != [7, 19]:
        raise ValueError('ABLATION_FROZEN_DESIGN')
    plugin, witness = resolve('selectors', cfg['selector'])
    market = RecipeMarket(parent); recipe = market.recipe
    dataset = dict(schema='feature_selector.dataset.v1', scope='development',
        features=recipe['list']['columns'], groups=groups_for_selector(recipe),
        blocks=coherent_blocks(recipe['list']['columns']), support_hash=support_hash(market.inputs),
        scaler_hash=digest(recipe['scaler']), representation=recipe['representation'],
        splits=dict(train=recipe['split']['periods']['TRAIN'], selection=recipe['split']['periods']['INNER_A'],
                    test_start=recipe['split']['test_closed_from']),
        constraints=dict(agent_state='none', scaling='fixed_train', clipping=False, fit_scope='train_only', test_access=False))
    arms = plugin.plan(dataset, cfg['selector_params'])
    design = dict(contrast['design'], steps=cfg['steps'])
    spec = dict(schema='paired_ablation.v1', contrast_root=str(args.parent.resolve()),
        contrast_identity=contrast['identity'], baseline_root=contrast['baseline_root'],
        plugin=witness, config=cfg, design=design, dataset=dataset, arms=arms,
        sources={p:file_sha(ROOT/p) for p in SOURCES}, created_at=now(),
        scope='DEVELOPMENT_ONLY; upstream internal-period exposure; two previously qualified seeds; no final selection')
    spec['identity'] = digest(spec)
    args.root.mkdir(parents=True, exist_ok=False)
    write(args.root/'SPEC.private.json', spec)
    print(json.dumps(dict(identity=spec['identity'], arms=[(a['name'],len(a['features']),len(a['groups'])) for a in arms])))


def authenticate(root):
    spec = read(root/'SPEC.private.json')
    if spec['identity'] != digest({k:v for k,v in spec.items() if k != 'identity'}):
        raise ValueError('ABLATION_IDENTITY')
    contrast, parent = contrast_auth(Path(spec['contrast_root']))
    if contrast['identity'] != spec['contrast_identity']: raise ValueError('ABLATION_PARENT')
    _, witness = resolve('selectors', spec['config']['selector'])
    if witness != spec['plugin']: raise ValueError('ABLATION_PLUGIN_CHANGED')
    for path, sha in spec['sources'].items():
        if file_sha(ROOT/path) != sha: raise ValueError('ABLATION_SOURCE_CHANGED')
    return spec, parent


def verified(path, spec, unit=None, allow_smoke=False):
    if not path.exists(): return False
    r = read(path)
    if r['identity'] != spec['identity'] or (unit is not None and r['unit'] != unit):
        raise ValueError('ABLATION_RECEIPT_IDENTITY')
    if r.get('smoke') and not allow_smoke: raise ValueError('ABLATION_SMOKE_NOT_PRODUCTION')
    if r['support_hash'] != spec['dataset']['support_hash'] or set(r['selectors']) != {'train_best','guarded_p5'}:
        raise ValueError('ABLATION_RECEIPT_SUPPORT')
    required = {'HISTORY.json','RECIPE.json','SELECTORS.json','TRAINING.json','INITIAL_PROBE.json'}
    for state in r['selectors'].values():
        step = state['selected_step']
        if step is not None:
            required.update({f'checkpoints/{step}.pt', f'accounts/{step}/EMISSIONS.npz'})
            for period in ('TRAIN','INNER_A'):
                required.update(f'accounts/{step}/{period}/{n}' for n in
                    ('METRICS.json','TRADES.json','ACTIONS.json','EQUITY.npz','CLOSE_COUNTERFACTUALS.json'))
            if str(step) not in r['accounts']: raise ValueError('ABLATION_ACCOUNT_MISSING')
    if not required <= set(r['artifacts']): raise ValueError('ABLATION_RECEIPT_FILES')
    for n, sha in r['artifacts'].items():
        p = (path.parent/n).resolve()
        if not p.is_relative_to(path.parent.resolve()) or file_sha(p) != sha:
            raise ValueError('ABLATION_ARTIFACT_CHANGED')
    arm = next(a for a in spec['arms'] if a['name']==r['arm'])
    recipe = read(path.parent/'RECIPE.json')
    if (recipe['list']['columns']!=arm['features'] or r['recipe_identity']!=recipe['identity']
        or r['feature_count']!=len(arm['features']) or r['group_count']!=len(arm['groups'])):
        raise ValueError('ABLATION_RECEIPT_COLUMNS')
    if not r.get('smoke'):
        if read(path.parent/'TRAINING.json')['step']!=spec['design']['steps']:
            raise ValueError('ABLATION_TRAINING_BUDGET')
        selectors,_=prefix_select(read(path.parent/'HISTORY.json'),spec['design'])
        for s in selectors:
            if not s.stopped: s.stopped,s.stop_step,s.reason=True,spec['design']['steps'],'fixed_budget'
        if {s.name:s.state() for s in selectors}!=r['selectors']:
            raise ValueError('ABLATION_SELECTOR_REPLAY')
    return True


def execute(root, output, unit, device, reference=None, smoke=False):
    spec, parent = authenticate(root)
    arm_name, seed_text = unit.rsplit('__', 1); seed = int(seed_text)
    arm = next(a for a in spec['arms'] if a['name'] == arm_name)
    if seed not in spec['config']['seeds']: raise ValueError('ABLATION_SEED')
    design = copy.deepcopy(spec['design'])
    if smoke: design.update(steps=128, minimum_steps=32, warmup=16, evaluation_every=64, batch_size=16, replay_size=128)
    configure(seed)
    parent_recipe = read(parent['inputs']['recipe'])
    recipe = parent_recipe if arm_name == 'full' else subset_recipe(parent_recipe, arm['features'])
    model = make_model(recipe, design, seed, device)
    market = RecipeMarket(parent)
    expected_support = support_hash(market.inputs)
    market.inputs = project_inputs(market.inputs, recipe); market.recipe = recipe
    if support_hash(market.inputs) != expected_support or expected_support != spec['dataset']['support_hash']:
        raise ValueError('ABLATION_SUPPORT_CHANGED')
    risk = read(Path(spec['baseline_root'])/'RISK.json')
    output.mkdir(parents=True, exist_ok=False); (output/'checkpoints').mkdir()
    write(output/'RECIPE.json', recipe)
    valid = np.flatnonzero(market.inputs.valid)
    probe = torch.tensor(market.inputs.windows[valid[:8]], device=device)
    with torch.no_grad(): initial = model.surfaces(probe).cpu().tolist()
    write(output/'INITIAL_PROBE.json', dict(values=initial, fresh_weights=True))
    selectors = make_selectors(design)
    source_receipt = None
    if reference is not None:
        if arm_name != 'full' or smoke or not contrast_verified(reference, spec['contrast_identity']):
            raise ValueError('ABLATION_CONTROL_UNVERIFIED')
        source = read(reference)
        if source['seed'] != seed: raise ValueError('ABLATION_CONTROL_SEED')
        selectors, history = prefix_select(read(reference.parent/'HISTORY.json'), design)
        if read(reference.parent/'INITIAL_PROBE.json')['values'] != initial:
            raise ValueError('ABLATION_CONTROL_INITIAL_WEIGHTS')
        source_receipt = dict(identity=source['identity'], sha256=file_sha(reference))
        write(output/'SOURCE_RECEIPT.json', source)
        write(output/'HISTORY.json', history)
        training = dict(reused_prefix=True, step=design['steps'], source_receipt=source_receipt)
        write(output/'TRAINING.json', training)
        for step in {s.selected_step for s in selectors if s.selected_step is not None}:
            shutil.copy2(reference.parent/'checkpoints'/f'{step}.pt', output/'checkpoints'/f'{step}.pt')
    else:
        if arm_name == 'full' and not smoke: raise ValueError('CONTROL_MUST_REUSE_AUTHENTICATED_PREFIX')
        def assess(step, current):
            report = monitor(current, market, risk, design)
            current.save(output/'checkpoints'/f'{step}.pt')
            states = observe(selectors, report, step)
            write(output/'SELECTORS.json', states)
            return dict(report, stop=False, selectors=states)
        factory = lambda episode: environment(market, risk, 6, recipe['split']['periods']['TRAIN'], design['gamma'])
        training, history = train_loop(model, factory, design, seed, output, assess)
    for s in selectors:
        if not s.stopped:
            s.stopped, s.stop_step, s.reason = True, design['steps'], 'fixed_budget'
    states = {s.name:s.state() for s in selectors}; write(output/'SELECTORS.json', states)
    accounts = {}
    for step in sorted({s.selected_step for s in selectors if s.selected_step is not None}):
        checkpoint = output/'checkpoints'/f'{step}.pt'
        model.load(checkpoint)
        with torch.no_grad():
            before = model.surfaces(probe).clone(); model.load(checkpoint)
            if not torch.equal(before, model.surfaces(probe)): raise ValueError('ABLATION_RELOAD')
        logits, q = emit(model, market.inputs, twins=True)
        folder = output/'accounts'/str(step); folder.mkdir(parents=True)
        np.savez_compressed(folder/'EMISSIONS.npz', indices=valid, logits=logits[valid], qvalues=q[:,valid],
                            timestamp=market.inputs.row_ts_ns[valid], period=market.inputs.period[valid].astype('U16'))
        accounts[str(step)] = {p:evaluate(environment(market, risk, 6, recipe['split']['periods'][p], design['gamma']),
                                         logits, folder/p, diagnose=True) for p in ('TRAIN','INNER_A')}
    result = dict(identity=spec['identity'], unit=unit, seed=seed, arm=arm_name, smoke=smoke,
        feature_count=len(arm['features']), group_count=len(arm['groups']),
        actor_parameters=sum(p.numel() for p in model.actor_net.parameters()),
        support_hash=expected_support, recipe_identity=recipe['identity'], source_receipt=source_receipt,
        selectors=states, accounts=accounts, scope=spec['scope'], finished_at=now(),
        artifacts={str(p.relative_to(output)):file_sha(p) for p in output.rglob('*')
                   if p.is_file() and p.name not in ('PROGRESS.json','RESULT.json')})
    write(output/'RESULT.json', result)
    verified(output/'RESULT.json', spec, unit, allow_smoke=smoke)
    return result


def worker(args):
    execute(args.root, args.output, args.unit, args.device, args.reference, args.smoke)


def queue(args):
    spec, _ = authenticate(args.root); cfg = read(args.assignment)
    if cfg['identity'] != spec['identity']: raise ValueError('ABLATION_ASSIGNMENT')
    with (args.root/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        failed = []
        for item in cfg['units']:
            unit = item['unit']; success = False
            if unit not in [f"{a['name']}__{s}" for a in spec['arms'] for s in spec['config']['seeds']]:
                raise ValueError('ABLATION_UNKNOWN_UNIT')
            for attempt in (1, 2):
                folder = args.root/'runs'/unit/f'attempt{attempt}'
                if verified(folder/'RESULT.json', spec, unit): success = True; break
                if folder.exists(): continue
                waiting_since=time.monotonic()
                dependency_failed=False
                while True:
                    waits = cfg.get('wait_for', []) + item.get('wait_for', [])
                    states=[read(Path(p)).get('state') if Path(p).exists() else None for p in waits]
                    ready = all(s in ('COMPLETE','FAILED') for s in states)
                    ref = item.get('reference')
                    if ref and not Path(ref).exists():
                        ready = False
                        if 'FAILED' in states: dependency_failed=True
                    if dependency_failed or time.monotonic()-waiting_since > 86400:
                        write(args.root/f'EXIT_{unit}_{attempt}.json',dict(code='DEPENDENCY_OR_RESOURCE_TIMEOUT',time=now()))
                        dependency_failed=True
                        break
                    mem = {l.split(':')[0]:int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines() if ':' in l}
                    gpu = int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits',
                              '-i',os.environ['CUDA_VISIBLE_DEVICES']], text=True).strip())
                    if ready and mem['MemAvailable'] > 5*1024**2 and gpu > 3072: break
                    write(args.root/'STATUS.json', dict(state='WAIT_RESOURCES_OR_DEPENDENCY', unit=unit, time=now()))
                    time.sleep(60)
                if dependency_failed: break
                command = [sys.executable, '-m', 'tools.run_paired_ablation', 'worker', '--root', str(args.root),
                           '--output', str(folder), '--unit', unit, '--device', 'cuda']
                if item.get('reference'): command += ['--reference', item['reference']]
                write(args.root/'STATUS.json', dict(state='RUNNING', unit=unit, attempt=attempt, time=now()))
                with (args.root/f'{unit}_attempt{attempt}.log').open('a') as log:
                    try: code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=172800).returncode
                    except subprocess.TimeoutExpired: code = 'TIMEOUT'
                write(args.root/f'EXIT_{unit}_{attempt}.json', dict(code=code, time=now()))
                if code == 0 and verified(folder/'RESULT.json', spec, unit): success = True; break
            if not success: failed.append(unit)
        write(args.root/'STATUS.json', dict(state='COMPLETE' if not failed else 'FAILED', failed=failed, time=now()))


def summarize(spec, results, paths):
    rows = []
    for unit, r in sorted(results.items()):
        history = {h['step']:h['report'] for h in read(paths[unit].parent/'HISTORY.json')}
        for name, state in r['selectors'].items():
            step = state['selected_step']
            row = dict(unit=unit, selector=name, selected_step=step, n_features=r['feature_count'], n_groups=r['group_count'])
            if step is not None:
                row['metrics'] = r['accounts'][str(step)]
                report = history[step]
                row['monitor'] = {p:float(np.mean([x['return_per_week'] for x in report[k]]))
                                  for p,k in [('train','train'),('validation','internal_validation')]}
                control = results.get(f"full__{r['seed']}")
                if control and control['selectors'][name]['selected_step'] is not None:
                    cs = control['selectors'][name]['selected_step']
                    ch = {h['step']:h['report'] for h in read(paths[f"full__{r['seed']}"].parent/'HISTORY.json')}[cs]
                    row['paired_monitor_delta'] = {p:row['monitor'][p]-float(np.mean([x['return_per_week'] for x in ch[k]]))
                        for p,k in [('train','train'),('validation','internal_validation')]}
                    a = np.load(paths[unit].parent/'accounts'/str(step)/'EMISSIONS.npz')
                    b = np.load(paths[f"full__{r['seed']}"].parent/'accounts'/str(cs)/'EMISSIONS.npz')
                    if not np.array_equal(a['timestamp'],b['timestamp']): raise ValueError('ABLATION_EMISSION_SUPPORT')
                    row['signal_disagreement'] = {p:float(np.mean(a['logits'][a['period']==p].argmax(1) !=
                        b['logits'][a['period']==p].argmax(1))) for p in ('TRAIN','INNER_A')}
            rows.append(row)
    stability = []
    for name in ('train_best','guarded_p5'):
        for arm in spec['arms']:
            if arm['name'] == 'full': continue
            subset = [r for r in rows if r['selector']==name and r['unit'].startswith(arm['name']+'__')
                      and 'paired_monitor_delta' in r]
            if len(subset)==len(spec['config']['seeds']):
                stability.append(dict(arm=arm['name'], selector=name, descriptive_only=True,
                    delta_by_seed={r['unit'].rsplit('__',1)[1]:r['paired_monitor_delta'] for r in subset},
                    signs_agree={p:len({int(np.sign(r['paired_monitor_delta'][p])) for r in subset})==1
                                 for p in ('train','validation')}))
    return dict(identity=spec['identity'], rows=rows, stability=stability, final_selection=False,
                note='Two qualified seeds: descriptive paired deltas, not a significance claim. Internal period reused.')


def collect(args):
    cfg = read(args.assignment); spec, _ = authenticate(Path(cfg['source_root']))
    if cfg['identity'] != spec['identity']: raise ValueError('ABLATION_COLLECTION_IDENTITY')
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root/'collect.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.root/'CLOSE.json').exists():
            c=read(args.root/'CLOSE.json')
            if c['identity']!=spec['identity'] or c['aggregate_sha256']!=file_sha(args.root/'AGGREGATE.json'):
                raise ValueError('ABLATION_CLOSED_CHANGED')
            return
        while True:
            results = {}; paths = {}; statuses = {}; issues = []
            for node in cfg['nodes']:
                dest = args.root/node['role']; dest.mkdir(exist_ok=True)
                try:
                    subprocess.run(['rsync','-a','--exclude=*.log','--exclude=*.tmp','--exclude=queue.lock',
                        '-e','ssh -o BatchMode=yes -o ConnectTimeout=10',node['host']+':'+node['root']+'/',str(dest)+'/'],
                        check=True, timeout=180, capture_output=True)
                    for unit in node['units']:
                        for path in sorted((dest/'runs'/unit).glob('attempt*/RESULT.json')):
                            if verified(path, spec, unit): results[unit] = read(path); paths[unit] = path; break
                    if (dest/'STATUS.json').exists(): statuses[node['role']] = read(dest/'STATUS.json')
                except (OSError, subprocess.SubprocessError) as e: issues.append(dict(role=node['role'], error=type(e).__name__))
            report = summarize(spec, results, paths); report.update(statuses=statuses, issues=issues, time=now())
            write(args.root/'AGGREGATE.json', report)
            lines = ['# Paired ablation pilot', '', report['note'], '',
                     '| Unit | Selector | Step | Inputs/groups | TRAIN % | INNER_A % | VAL Sharpe | VAL DD % | Orders | Win % |',
                     '|---|---|---:|---|---:|---:|---:|---:|---:|---:|']
            for r in report['rows']:
                if 'metrics' not in r:
                    lines.append(f"| {r['unit']} | {r['selector']}: no eligible checkpoint | - | - | - | - | - | - | - | - |"); continue
                t,v = r['metrics']['TRAIN'],r['metrics']['INNER_A']
                win = None if v['win_rate'] is None else 100*v['win_rate']
                lines.append(f"| {r['unit']} | {r['selector']} | {r['selected_step']} | {r['n_features']}/{r['n_groups']} | {t['return_pct']:.3f} | {v['return_pct']:.3f} | {v['sharpe']} | {v['max_drawdown_pct']:.3f} | {v['trades']} | {win} |")
            (args.root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
            if len(statuses)==len(cfg['nodes']) and all(s['state'] in ('COMPLETE','FAILED') for s in statuses.values()):
                expected = {u for n in cfg['nodes'] for u in n['units']}
                write(args.root/'CLOSE.json', dict(identity=spec['identity'], completed=len(results), expected=len(expected),
                    missing=sorted(expected-set(results)), aggregate_sha256=file_sha(args.root/'AGGREGATE.json'), time=now()))
                return
            time.sleep(120)


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='mode', required=True)
    a = sub.add_parser('prepare'); a.add_argument('--root',type=Path,required=True); a.add_argument('--parent',type=Path,required=True)
    a.add_argument('--config',type=Path,required=True)
    a = sub.add_parser('worker'); a.add_argument('--root',type=Path,required=True); a.add_argument('--output',type=Path,required=True)
    a.add_argument('--unit',required=True); a.add_argument('--device',default='cpu'); a.add_argument('--reference',type=Path); a.add_argument('--smoke',action='store_true')
    for mode in ('queue','collect'):
        a = sub.add_parser(mode); a.add_argument('--root',type=Path,required=True); a.add_argument('--assignment',type=Path,required=True)
    args = p.parse_args()
    try:
        globals()[args.mode](args)
    except Exception as exc:
        if args.mode=='queue':
            write(args.root/'STATUS.json',dict(state='FAILED',cause=type(exc).__name__,detail=str(exc),time=now()))
        raise


if __name__ == '__main__': main()
